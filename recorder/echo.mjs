// Echo filter (16 Oct 2026, Erez: "my voice and your voice … are appearing as duplicates").
//
// Without a headset the mic hears the speakers, so whatever the system track plays (an agent's
// voice reply, the other side of a call) comes back a few seconds later as a mic line — shown
// as Erez, and read by the meeting agent as his words. Meeting 139 measured it: every system
// sentence returned on the mic 0–16 s later, and whisper heard "your line" as "you're lying".
//
// A mic phrase is an echo when most of its word pairs were just played by the system track.
// Pairs, not single words: "I", "you", "the" are in everything; a run of two words in the same
// order is not. Short phrases (under three words) must appear whole.
//
// And by letters: the two tracks can spell the same audio differently — meeting 140 heard a
// spoken URL as "Vos slash ballpark slash 9656 Adenine dash …" on the speakers and
// "Bose/Ballpark/9656-Adenine-…" on the mic, so no word pair matched. Spoken punctuation
// dropped, spaces dropped, then the share of the mic's 6-letter runs found in the system text.
// Calibrated on meetings 139+140 (16 Oct): Erez's own lines scored at most 0.36 (a line that
// was half his, half echo), every echo at least 0.71.

/** Lowercase words, letters and digits only (any script — Hebrew too). */
export function words(text) {
  return (text.toLowerCase().normalize("NFKC").match(/[\p{L}\p{N}]+/gu) || []).filter((w) => w !== "s");
}

function pairs(ws) {
  const out = [];
  for (let i = 0; i + 1 < ws.length; i++) out.push(ws[i] + " " + ws[i + 1]);
  return out;
}

/** Share of the mic phrase's word pairs found in the recent system text (0..1). */
export function echoScore(micText, systemTexts) {
  const mw = words(micText);
  if (!mw.length) return 0;
  const sw = systemTexts.flatMap((t) => [...words(t), "|"]);   // "|" stops pairs across lines
  if (mw.length < 3) {
    const hay = " " + sw.join(" ") + " ", needle = " " + mw.join(" ") + " ";
    return hay.includes(needle) ? 1 : 0;
  }
  const have = new Set(pairs(sw));
  const mp = pairs(mw);
  return mp.filter((p) => have.has(p)).length / mp.length;
}

const SPOKEN = new Set(["slash", "dash", "dot", "colon", "underscore", "hyphen"]);
const letters = (t) => words(t).filter((w) => !SPOKEN.has(w)).join("");
const GRAM = 6;

/** Share of the mic phrase's 6-letter runs found in the recent system text (0..1). */
export function letterScore(micText, systemTexts) {
  const a = letters(micText);
  if (a.length < 12) return 0;                     // too short to say anything by letters
  const have = new Set();
  for (const t of systemTexts) { const b = letters(t); for (let i = 0; i + GRAM <= b.length; i++) have.add(b.slice(i, i + GRAM)); }
  let hit = 0, tot = 0;
  for (let i = 0; i + GRAM <= a.length; i++) { tot++; if (have.has(a.slice(i, i + GRAM))) hit++; }
  return hit / tot;
}

export const ECHO_THRESHOLD = 0.5;
// A system chunk is "recent" for a mic chunk when it was cut between 40 s before the mic chunk
// started and 15 s after it was cut. Generous on purpose: the two tracks' clocks drift apart
// (meeting 139: 9–16 s), and a false keep is the bug we are fixing.
export const WINDOW_BEFORE_S = 40, WINDOW_AFTER_S = 15;

export function isEcho(micText, micChunk, systemLines) {
  const lo = micChunk.wall - (micChunk.dur_s || 0) - WINDOW_BEFORE_S, hi = micChunk.wall + WINDOW_AFTER_S;
  const recent = systemLines.filter((s) => s.wall >= lo && s.wall <= hi).map((s) => s.text);
  if (!recent.length) return { echo: false, score: 0 };
  const score = Math.max(echoScore(micText, recent), letterScore(micText, recent));
  return { echo: score >= ECHO_THRESHOLD, score };
}
