// transcriber — follows a meeting-recorder directory and writes each chunk's text to Neon.
//
//   node recorder/transcriber.mjs <recording-dir> <meeting-id>
//
// Reads <dir>/chunks.jsonl as it grows (one chunk per spoken sentence), sends each to the
// local whisper-server (language detected per phrase), checks the text agrees with the
// language it claims, and:
//   1. appends each line to <dir>/lines.jsonl AT ONCE — that is what the app shows, so a
//      line is on screen about a second after the sentence ends;
//   2. queues it for Neon, inserted in the background. A Neon failure retries the INSERT,
//      never the transcription, and never delays the screen (6 Oct: 25 Neon failures per
//      meeting used to re-run whisper and hold every later line back by up to 30 s).
// When <dir>/done exists, every chunk is transcribed and the queue is empty, it marks the
// meeting ended.
//
// ⛔ A 200 is not a correct result: whisper can return Hebrew speech as fluent English
// nonsense. Every result is checked by counting Hebrew letters and redone in the other
// language if they disagree. Whisper also loops on silence — near-silent chunks are
// skipped (and logged), and a repeated line is collapsed and marked, never deleted.
//
// Zero dependencies: Neon's HTTP SQL endpoint over fetch.

import fs from "fs";
import os from "os";
import path from "path";
import { spawnSync } from "child_process";
import net from "node:net";
import dns from "node:dns";
import { isEcho } from "./echo.mjs";
// ⛔ NETWORK: IPv4 first, and no address race. Erez's Mac runs NordVPN: IPv6 is
// unreachable through it and a connect takes longer than Node's 250 ms happy-eyeballs
// attempt, so EVERY fetch (Neon, Cloudflare) died "fetch failed"/ETIMEDOUT while curl
// worked — 25 Neon failures a meeting, then Record refused outright (6 Oct 2026).
net.setDefaultAutoSelectFamily(false);
dns.setDefaultResultOrder("ipv4first");

const [dir, meetingIdArg] = process.argv.slice(2);
if (!dir || !meetingIdArg) {
  console.error("usage: transcriber.mjs <recording-dir> <meeting-id>");
  process.exit(2);
}
const meetingId = Number(meetingIdArg);
const WHISPER = process.env.WHISPER_URL || "http://127.0.0.1:8178";
const SILENCE_RMS = 0.003;
const SPEAKER = { mic: "Erez", system: "Others" };

const dbFile = path.join(os.homedir(), "Workspace/jarvis/integrations/erez-database-url");
const dbUrl = (fs.readFileSync(dbFile, "utf8").match(/^DATABASE_URL=(.*)$/m) || [])[1];
if (!dbUrl) {
  console.error(`transcriber: no DATABASE_URL in ${dbFile} — FAILING CLOSED`);
  process.exit(1);
}

const log = (s) => console.error(`[transcriber ${new Date().toISOString().slice(11, 19)}] ${s}`);

async function sql(query, params = []) {
  const r = await fetch(`https://${new URL(dbUrl).hostname}/sql`, {
    method: "POST",
    headers: { "Neon-Connection-String": dbUrl, "Content-Type": "application/json" },
    body: JSON.stringify({ query, params }),
  });
  const body = await r.text();
  if (!r.ok) throw new Error(`neon ${r.status}: ${body.slice(0, 300)}`);
  return JSON.parse(body).rows;
}

const HEBREW = /[֐-׿]/g;
const LETTER = /\p{L}/gu;
function hebrewShare(text) {
  const letters = (text.match(LETTER) || []).length;
  return letters ? (text.match(HEBREW) || []).length / letters : 0;
}

/** Collapse a line repeated 3+ times in a row into one, marked. */
function collapseLoops(segments) {
  const out = [];
  for (const s of segments) {
    const prev = out[out.length - 1];
    if (prev && prev.text === s.text) { prev.repeat = (prev.repeat || 1) + 1; prev.end = s.end; continue; }
    out.push({ ...s });
  }
  return out.map((s) => (s.repeat >= 3 ? { ...s, text: `${s.text} [whisper repeated this ×${s.repeat} — likely silence]` } : s));
}

async function whisper(wav, language) {
  const form = new FormData();
  form.append("file", new Blob([wav]), "phrase.wav");
  form.append("response_format", "verbose_json");
  form.append("temperature", "0");
  form.append("language", language);
  const r = await fetch(`${WHISPER}/inference`, { method: "POST", body: form });
  if (!r.ok) {
    fs.writeFileSync("/tmp/whisper-fail.wav", wav);
    throw new Error(`whisper ${r.status} (lang ${language}, ${wav.length} bytes → /tmp/whisper-fail.wav): ${(await r.text()).slice(0, 200)}`);
  }
  return r.json();
}

const LANG_CODE = { hebrew: "he", english: "en" };
const VAD_BIN = "/opt/homebrew/bin/whisper-vad-speech-segments";
const VAD_MODEL = "/Applications/OpenSuperWhisper.app/Contents/Resources/ggml-silero-v5.1.2.bin";
const SR = 16000, HEADER = 44;

/** Speech regions (seconds), split at pauses — a language switch almost always sits on one. */
function speechRegions(file) {
  const r = spawnSync(VAD_BIN, ["-f", file, "-vm", VAD_MODEL, "-vsd", "300"], { encoding: "utf8" });
  const out = `${r.stdout}${r.stderr}`;
  // The tool prints each region twice — "VAD segment" in seconds, "Speech segment" in centiseconds.
  const regions = [...out.matchAll(/Speech segment \d+: start = ([\d.]+), end = ([\d.]+)/g)].map((m) => ({ start: +m[1] / 100, end: +m[2] / 100 }));
  // Merge fragments too short to carry a language into their neighbour.
  const merged = [];
  for (const g of regions) {
    const prev = merged[merged.length - 1];
    if (prev && (g.start - prev.end < 0.3 || g.end - g.start < 0.8)) prev.end = g.end;
    else merged.push({ ...g });
  }
  return merged;
}

/** A standalone WAV of [start,end) seconds from one of the recorder's 16 kHz mono chunks. */
function sliceWav(buf, start, end) {
  const total = (buf.length - HEADER) / 2;
  const a = Math.max(0, Math.floor((start - 0.15) * SR)), b = Math.min(total, Math.ceil((end + 0.15) * SR));
  const data = buf.subarray(HEADER + a * 2, HEADER + b * 2);
  const out = Buffer.alloc(HEADER + data.length);
  buf.copy(out, 0, 0, HEADER);
  out.writeUInt32LE(36 + data.length, 4);
  out.writeUInt32LE(data.length, 40);
  data.copy(out, HEADER);
  return { wav: out, offset: a / SR };
}

/** One phrase: language auto-detected on THIS phrase alone, then checked against its content. */
async function transcribePhrase(wav) {
  const first = await whisper(wav, "auto");
  let lang = LANG_CODE[first.language] || first.language;
  let result = first, redone = false;
  const share = hebrewShare(first.text || "");
  if (lang === "en" && share > 0.3) { result = await whisper(wav, "he"); lang = "he"; redone = true; }
  else if (lang === "he" && (first.text || "").trim() && share < 0.3) { result = await whisper(wav, "en"); lang = "en"; redone = true; }
  else if (lang !== "he" && lang !== "en") { result = await whisper(wav, "en"); lang = "en"; redone = true; }
  return { lang, redone, prob: first.detected_language_probability, segments: result.segments || [] };
}

function rmsOf(wav) {
  let sq = 0;
  const n = (wav.length - HEADER) / 2;
  for (let i = HEADER; i < wav.length; i += 2) { const v = wav.readInt16LE(i) / 32768; sq += v * v; }
  return n ? Math.sqrt(sq / n) : 0;
}

/** A chunk → phrases, each with its own language. Whisper picks ONE language per request and
 *  TRANSLATES the rest (107: "אחרי שבועיים" came back "after the week") — so never send it two.
 *  A phrase far quieter than the loudest in its chunk is room noise that whisper would turn into
 *  words (107: 0.0022 → "זה לא נכנס..." beside speech at 0.0287); it is dropped and logged. */
async function transcribe(file) {
  const buf = fs.readFileSync(file);
  // The recorder cuts at every pause, so a chunk up to its 12 s cap is ONE sentence:
  // send it whole and skip the separate VAD pass (0.65 s a chunk, measured 6 Oct). The
  // whisper-server runs its own VAD, and the language is still detected on this chunk.
  const seconds = (buf.length - HEADER) / 2 / SR;
  if (seconds <= 12.5) {
    const r = await transcribePhrase(buf);
    const segs = collapseLoops(r.segments.map((s) => ({ start: s.start, end: s.end, text: (s.text || "").trim() })).filter((s) => s.text));
    const text = segs.map((s) => s.text).join(" ").trim();
    return text ? [{ start: segs[0].start, end: segs[segs.length - 1].end, lang: r.lang, redone: r.redone, prob: r.prob, rms: rmsOf(buf), text }] : [];
  }
  let regions = speechRegions(file);
  if (!regions.length) regions = [{ start: 0, end: (buf.length - HEADER) / 2 / SR }];
  const slices = regions.map((g) => { const s = sliceWav(buf, g.start, g.end); return { ...s, rms: rmsOf(s.wav) }; });
  const floor = Math.max(SILENCE_RMS, 0.15 * Math.max(...slices.map((s) => s.rms)));
  const phrases = [];
  for (const { wav, offset, rms } of slices) {
    if (rms < floor) { log(`${path.basename(file)} @${offset.toFixed(1)}s dropped: rms ${rms.toFixed(4)} < ${floor.toFixed(4)}`); continue; }
    const r = await transcribePhrase(wav);
    const segs = collapseLoops(r.segments.map((s) => ({ start: s.start, end: s.end, text: (s.text || "").trim() })).filter((s) => s.text));
    const text = segs.map((s) => s.text).join(" ").trim();
    if (!text) continue;
    phrases.push({ start: offset + segs[0].start, end: offset + segs[segs.length - 1].end, lang: r.lang, redone: r.redone, prob: r.prob, rms, text });
  }
  return phrases;
}

if (process.env.DRY) {
  for (const f of process.argv.slice(2)) {
    for (const p of await transcribe(f)) console.log(`${path.basename(f)} ${p.start.toFixed(1)}-${p.end.toFixed(1)} ${p.lang}${p.redone ? "*" : ""} ${p.text}`);
  }
  process.exit(0);
}

// ── main loop ──
const journal = path.join(dir, "chunks.jsonl");
const doneFile = path.join(dir, "done");
const progressFile = path.join(dir, "transcribed.json");
const linesFile = path.join(dir, "lines.jsonl");
const echoFile = path.join(dir, "echo.jsonl");       // mic lines dropped as speaker echo, kept for audit
const speakingFile = path.join(dir, "system.speaking"); // the recorder holds it while the system track is mid-sentence
const seen = new Set(fs.existsSync(progressFile) ? JSON.parse(fs.readFileSync(progressFile, "utf8")) : []);
let seq = fs.existsSync(linesFile) ? fs.readFileSync(linesFile, "utf8").split("\n").filter(Boolean).length : 0;

// ── Echo: what the system track just played, so the mic's copy of it can be dropped ──
// Rebuilt from lines.jsonl on restart; walls come from the journal (the tracks' start_s drift).
const systemLines = [];
{
  const walls = Object.fromEntries((fs.existsSync(journal) ? fs.readFileSync(journal, "utf8").split("\n").filter(Boolean) : [])
    .map((l) => JSON.parse(l)).map((c) => [c.file, c.wall]));
  if (fs.existsSync(linesFile)) for (const l of fs.readFileSync(linesFile, "utf8").split("\n").filter(Boolean)) {
    const r = JSON.parse(l);
    if (r.track === "system" && walls[r.raw?.chunk]) systemLines.push({ wall: walls[r.raw.chunk], text: r.words });
  }
}
/** True while the system track is mid-sentence: its chunk is not cut yet, so a mic chunk must wait for it.
 *  A flag older than 30 s is stale (the recorder died holding it) and is ignored. */
function systemSpeaking() {
  try { return Date.now() - fs.statSync(speakingFile).mtimeMs < 30000; } catch { return false; }
}

// ── Neon, in the background ──
// Rows are queued with the local seq; on restart, lines not yet in Neon are re-queued from
// lines.jsonl by comparing against what the meeting already holds.
const queue = [];
async function requeueUnsent() {
  if (!seq) return;
  const [{ n }] = await sql(`SELECT count(*)::int AS n FROM meeting_transcript WHERE meeting_id = $1`, [meetingId]);
  const lines = fs.readFileSync(linesFile, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));
  queue.push(...lines.slice(n));
  if (lines.length > n) log(`requeued ${lines.length - n} line(s) not yet in Neon`);
}
let neonFailures = 0;
async function flushNeon() {
  for (;;) {
    if (queue.length) {
      const batch = queue.slice(0, 50);
      const params = [], values = [];
      for (const r of batch) {
        const i = params.length;
        values.push(`($${i + 1}, 'mjv-recorder', $${i + 2}, $${i + 3}, $${i + 4}, $${i + 5}, $${i + 6}, $${i + 7}, 'transcript.data', $${i + 8})`);
        params.push(meetingId, r.speaker_name, r.track, r.is_host, r.words, r.start_ts, r.end_ts, JSON.stringify(r.raw));
      }
      try {
        await sql(`INSERT INTO meeting_transcript (meeting_id, bot_id, speaker_name, speaker_id, is_host, words, start_ts, end_ts, event_type, raw)
                   VALUES ${values.join(", ")}`, params);
        queue.splice(0, batch.length);
        neonFailures = 0;
        continue;
      } catch (e) {
        neonFailures++;
        log(`neon insert of ${batch.length} FAILED (${neonFailures}), will retry: ${e.message}`);
        await new Promise((r) => setTimeout(r, Math.min(10000, 500 * 2 ** neonFailures)));
        continue;
      }
    }
    await new Promise((r) => setTimeout(r, 200));
  }
}

async function processChunk(c) {
  const file = path.join(dir, c.file);
  if (c.rms < SILENCE_RMS) { log(`${c.file} silent (rms ${c.rms.toFixed(4)}) — skipped`); return; }
  const t0 = Date.now();
  let phrases = await transcribe(file);
  if (!phrases.length) { log(`${c.file} no speech`); return; }
  if (c.track === "system") {
    for (const p of phrases) systemLines.push({ wall: c.wall, text: p.text });
    while (systemLines.length && systemLines[0].wall < c.wall - 180) systemLines.shift();
  } else if (c.track === "mic") {
    phrases = phrases.filter((p) => {
      const e = isEcho(p.text, c, systemLines);
      if (!e.echo) return true;
      log(`${c.file} ECHO dropped (score ${e.score.toFixed(2)}) → ${p.text.slice(0, 80)}`);
      fs.appendFileSync(echoFile, JSON.stringify({ chunk: c.file, score: e.score, text: p.text }) + "\n");
      return false;
    });
    if (!phrases.length) return;
  }
  const rows = phrases.map((p) => ({
    seq: ++seq, track: c.track, speaker_name: SPEAKER[c.track] || c.track, is_host: c.track === "mic",
    words: p.text, start_ts: +(c.start_s + p.start).toFixed(2), end_ts: +(c.start_s + p.end).toFixed(2), language: p.lang,
    raw: { language: p.lang, detected_prob: p.prob, redone: p.redone, rms: p.rms, chunk: c.file, source: "mjv-recorder" },
  }));
  // On screen first: one append, the server sees the file change and pushes it to the app.
  fs.appendFileSync(linesFile, rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
  queue.push(...rows);
  const lag = c.wall ? ((Date.now() / 1000 - c.wall)).toFixed(1) : "?";
  for (const p of phrases) log(`${c.file} ${p.lang}${p.redone ? " (redone)" : ""} → ${p.text.slice(0, 80)}`);
  log(`${c.file} ${c.dur_s?.toFixed?.(1) ?? "?"}s audio, ${phrases.length} phrase(s) in ${((Date.now() - t0) / 1000).toFixed(1)}s, on screen ${lag}s after the cut`);
}

await requeueUnsent().catch((e) => log(`requeue check failed: ${e.message}`));
flushNeon();

let failures = 0;
for (;;) {
  const lines = fs.existsSync(journal) ? fs.readFileSync(journal, "utf8").split("\n").filter(Boolean) : [];
  // System chunks first, so a mic chunk is always compared against what the speakers played;
  // and a mic chunk waits while the system track is mid-sentence (its chunk is not cut yet).
  const pending = lines.map((l) => JSON.parse(l)).filter((c) => !seen.has(c.file));
  pending.sort((a, b) => (a.track === "system" ? 0 : 1) - (b.track === "system" ? 0 : 1));
  const holdMic = systemSpeaking();
  for (const c of pending) {
    if (holdMic && c.track === "mic") break;
    try {
      await processChunk(c);
      seen.add(c.file);
      fs.writeFileSync(progressFile, JSON.stringify([...seen]));
      failures = 0;
    } catch (e) {
      // Only whisper can fail here now — Neon is the queue's problem.
      failures++;
      log(`${c.file} FAILED (${failures}): ${e.message}`);
      await new Promise((r) => setTimeout(r, Math.min(30000, 2000 * failures)));
      break; // retry from the top, in order
    }
  }
  const allDone = fs.existsSync(doneFile) && lines.every((l) => seen.has(JSON.parse(l).file));
  if (allDone && !queue.length) {
    for (let i = 1; ; i++) {
      try { await sql(`UPDATE meetings SET status = 'ended', ended_at = now() WHERE id = $1`, [meetingId]); break; }
      catch (e) { log(`marking ended FAILED (${i}): ${e.message}`); await new Promise((r) => setTimeout(r, Math.min(10000, 1000 * i))); }
    }
    log(`meeting ${meetingId} ended — ${seen.size} chunks, ${seq} lines`);
    process.exit(0);
  }
  await new Promise((r) => setTimeout(r, 150));
}
