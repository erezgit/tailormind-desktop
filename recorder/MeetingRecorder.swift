// meeting-recorder — records a meeting as two tracks, whatever app the meeting is in.
//
//   mic    = the default input device (Erez)
//   system = everything the Mac plays, via a Core Audio process tap (macOS 14.2+)
//            — Teams, Meet in Chrome, Zoom alike; the output device does not matter,
//            so speakers, AirPods and wired earphones all work.
//
// Usage:  meeting-recorder <out-dir> [max-chunk-seconds=12]
//
// Writes 16 kHz mono 16-bit WAV chunks, ONE PER SPOKEN SENTENCE: a chunk closes
// the moment the speaker pauses (0.5 s of quiet after speech), so a line can be
// transcribed and shown about a second after it is said (Erez, 6 Oct: "as fast as
// whisper can do it, but still precise"). Someone who never pauses is cut at the
// cap, at the quietest 100 ms of its last 4 s. Long silence is dropped, not written.
// It and appends one JSON line per chunk to
// <out-dir>/chunks.jsonl AFTER the file is renamed into place — a reader that
// follows chunks.jsonl never sees a half-written file.
// SIGINT/SIGTERM flushes both tracks and writes <out-dir>/done.

import AVFoundation
import CoreAudio
import Foundation
import IOKit.pwr_mgt

let SR: Double = 16_000
let args = CommandLine.arguments
guard args.count >= 2 else {
    FileHandle.standardError.write("usage: meeting-recorder <out-dir> [chunk-seconds]\n".data(using: .utf8)!)
    exit(2)
}
let outDir = URL(fileURLWithPath: args[1])
let chunkSeconds = args.count >= 3 ? Double(args[2]) ?? 12 : 12   // the cap, not the rhythm
try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
let startedAt = Date()
let targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: SR, channels: 1, interleaved: false)!
let journalLock = NSLock()

func log(_ s: String) {
    let t = String(format: "%6.2f", Date().timeIntervalSince(startedAt))
    FileHandle.standardError.write("[recorder +\(t)s] \(s)\n".data(using: .utf8)!)
}

func appendJournal(_ obj: [String: Any]) {
    journalLock.lock(); defer { journalLock.unlock() }
    let url = outDir.appendingPathComponent("chunks.jsonl")
    var line = try! JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys])
    line.append(0x0A)
    if let h = try? FileHandle(forWritingTo: url) {
        h.seekToEndOfFile(); h.write(line); try? h.close()
    } else {
        try? line.write(to: url)
    }
}

func writeWav(_ samples: ArraySlice<Float>, to url: URL) throws {
    var d = Data()
    func u32(_ v: UInt32) { var x = v.littleEndian; d.append(Data(bytes: &x, count: 4)) }
    func u16(_ v: UInt16) { var x = v.littleEndian; d.append(Data(bytes: &x, count: 2)) }
    let bytes = UInt32(samples.count * 2)
    d.append("RIFF".data(using: .ascii)!); u32(36 + bytes)
    d.append("WAVEfmt ".data(using: .ascii)!); u32(16); u16(1); u16(1)
    u32(UInt32(SR)); u32(UInt32(SR) * 2); u16(2); u16(16)
    d.append("data".data(using: .ascii)!); u32(bytes)
    d.reserveCapacity(d.count + samples.count * 2)
    for s in samples {
        var v = Int16(max(-1, min(1, s)) * 32767).littleEndian
        d.append(Data(bytes: &v, count: 2))
    }
    try d.write(to: url)
}

/// One track: converts whatever arrives to 16 kHz mono and cuts chunks.
final class Track {
    let name: String
    let lock = NSLock()
    var converter: AVAudioConverter?
    var sourceFormat: AVAudioFormat?
    var buf: [Float] = []
    var offset = 0          // samples already written out, from recording start
    var index = 0

    init(_ name: String) { self.name = name }

    func append(_ input: AVAudioPCMBuffer) {
        lock.lock(); defer { lock.unlock() }
        if sourceFormat != input.format {
            sourceFormat = input.format
            converter = AVAudioConverter(from: input.format, to: targetFormat)
            log("\(name): source format \(input.format)")
        }
        guard let conv = converter else { return }
        let cap = AVAudioFrameCount(Double(input.frameLength) * SR / input.format.sampleRate) + 64
        guard let out = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: cap) else { return }
        var fed = false
        var err: NSError?
        conv.convert(to: out, error: &err) { _, status in
            if fed { status.pointee = .noDataNow; return nil }
            fed = true; status.pointee = .haveData; return input
        }
        if let err { log("\(name): convert error \(err)"); return }
        if out.frameLength > 0, let ch = out.floatChannelData?[0] {
            let n = Int(out.frameLength)
            var peak: Float = 0, sq: Float = 0
            for i in 0..<n { let v = ch[i]; peak = max(peak, abs(v)); sq += v * v }
            buf.append(contentsOf: UnsafeBufferPointer(start: ch, count: n))
            beatBuffers += 1; beatSq += sq; beatN += n
            // A live microphone never delivers EXACT zeros — its noise floor does not allow it.
            // A run of them means the device behind the engine is gone (6 Oct, see restartMic).
            if peak == 0 { if zeroSince == nil { zeroSince = Date() } } else { zeroSince = nil }
        }
        lastAudio = Date()
        scan()
    }

    // ── Liveness, read by the watchdog (under `lock`) ──
    var zeroSince: Date?         // first buffer of the current run of exact digital silence
    var beatBuffers = 0, beatN = 0
    var beatSq: Float = 0
    /// One heartbeat line for the log, then reset the counters.
    func heartbeat() -> String {
        lock.lock(); defer { lock.unlock() }
        let rms = beatN > 0 ? (beatSq / Float(beatN)).squareRoot() : 0
        let line = String(format: "%@ alive: %d buffers, rms %.4f, %d chunks so far", name, beatBuffers, rms, index)
        beatBuffers = 0; beatN = 0; beatSq = 0
        return line
    }

    // ── Pause detection: plain energy on 30 ms frames against an adaptive floor ──
    let frame = Int(0.03 * SR)
    var scanned = 0              // samples of buf already classified
    var floor: Float = 0.002     // running noise floor (rises slowly, falls at once)
    var heardSpeech = false      // speech since the last cut
    var silentRun = 0            // samples of quiet since the last speech frame
    var lastAudio = Date()

    func scan() {
        while scanned + frame <= buf.count {
            var sq: Float = 0
            for j in scanned..<(scanned + frame) { sq += buf[j] * buf[j] }
            let rms = (sq / Float(frame)).squareRoot()
            floor = rms < floor ? rms : floor + (rms - floor) * 0.002
            let speech = rms > max(0.004, floor * 3)
            scanned += frame
            if speech { if !heardSpeech { setSpeaking(true) }; heardSpeech = true; silentRun = 0 } else { silentRun += frame }

            if heardSpeech && silentRun >= Int(0.5 * SR) && scanned >= Int(1.0 * SR) {
                // The sentence ended: cut 0.25 s into the pause, keeping a little tail.
                cut(at: scanned - silentRun + Int(0.25 * SR))
            } else if !heardSpeech && buf.count >= Int(10 * SR) {
                drop(Int(10 * SR) - Int(1 * SR))     // ten seconds of nothing: discard, keep 1 s lead-in
            } else if buf.count >= Int(chunkSeconds * SR) {
                cut(at: quietestCut(target: buf.count))
            }
        }
    }

    /// Advance past audio nobody needs (silence), keeping timestamps true.
    func drop(_ n: Int) {
        offset += n
        buf.removeFirst(n)
        scanned = max(0, scanned - n)
    }

    /// The quietest 100 ms in the last 4 s before the target, so a cut rarely splits a word.
    func quietestCut(target: Int) -> Int {
        let win = Int(0.1 * SR), lo = max(win, target - Int(4 * SR))
        var best = target, bestE = Float.greatestFiniteMagnitude
        var i = lo
        while i + win <= target {
            var e: Float = 0
            for j in i..<(i + win) { e += buf[j] * buf[j] }
            if e < bestE { bestE = e; best = i + win / 2 }
            i += win / 2
        }
        return best
    }

    func cut(at n: Int) {
        guard n > 0 else { return }
        let slice = buf[0..<n]
        var sq: Float = 0
        for s in slice { sq += s * s }
        let rms = (sq / Float(n)).squareRoot()
        index += 1
        let file = String(format: "%@-%06d.wav", name, index)
        let tmp = outDir.appendingPathComponent(file + ".tmp")
        let dst = outDir.appendingPathComponent(file)
        do {
            try writeWav(slice, to: tmp)
            try FileManager.default.moveItem(at: tmp, to: dst)
            appendJournal([
                "track": name, "index": index, "file": file,
                "start_s": Double(offset) / SR, "dur_s": Double(n) / SR,
                "rms": Double(rms), "wall": Date().timeIntervalSince1970,
            ])
            log(String(format: "%@ chunk %d  %.1fs  rms=%.4f", name, index, Double(n) / SR, rms))
        } catch { log("\(name): write failed \(error)") }
        offset += n
        buf.removeFirst(n)
        scanned = max(0, scanned - n)
        heardSpeech = false
        silentRun = 0
        setSpeaking(false)      // AFTER the journal line, so the transcriber finds the chunk it waited for
    }

    /// Echo filter (16 Oct 2026): `system.speaking` exists while the SYSTEM track is mid-sentence.
    /// The transcriber holds mic chunks until it is gone, so a mic line that is only the speakers
    /// coming back through the microphone is always compared against what the speakers played.
    func setSpeaking(_ on: Bool) {
        guard name == "system" else { return }
        let f = outDir.appendingPathComponent("system.speaking")
        if on { FileManager.default.createFile(atPath: f.path, contents: nil) }
        else { try? FileManager.default.removeItem(at: f) }
    }

    func flush() {
        lock.lock(); defer { lock.unlock() }
        if heardSpeech && buf.count > Int(0.5 * SR) { cut(at: buf.count) }
    }
}

let mic = Track("mic")
let sys = Track("system")

// ── mic: AVAudioEngine on the default input ──
//
// ⛔ A NEW ENGINE ON EVERY RESTART — never the old one restarted (6 Oct 2026). Recordings
// 125, 127 and 129 each logged "mic: device changed — restarting" then "mic: started",
// and NOT ONE mic chunk after it, for 5 to 8 minutes while Erez kept talking: the reused
// engine reports started and its tap never fires again once the input device has changed
// (his BC-80 headset coming and going). So the engine is thrown away and rebuilt, and the
// watchdog below rebuilds it too whenever audio stops arriving or arrives as pure zeros.
var engine = AVAudioEngine()
var micObserver: NSObjectProtocol?
// ⛔ THE MIC IS SET UP ON ITS OWN QUEUE, NEVER THE MAIN ONE (7 Oct 2026). `engine.inputNode`
// can block inside coreaudiod indefinitely (sampled: a rebuilt binary waiting on a microphone
// permission check while the screen was locked). On the main queue that froze the watchdog,
// the log AND the SIGTERM handler — Stop could not stop it, and it held the Mac awake forever.
// On its own queue, a stuck mic is just a mic the watchdog reports; Stop always works.
let micQueue = DispatchQueue(label: "recorder.mic")
let sysQueue = DispatchQueue(label: "recorder.system")
func startMic() { micQueue.async { startMicNow() } }
func startMicNow() {
    if let o = micObserver { NotificationCenter.default.removeObserver(o) }
    engine.inputNode.removeTap(onBus: 0)
    engine.stop()
    engine = AVAudioEngine()
    let input = engine.inputNode
    let fmt = input.outputFormat(forBus: 0)
    guard fmt.sampleRate > 0 else { log("mic: no input device — the watchdog will retry"); return }
    input.installTap(onBus: 0, bufferSize: 4096, format: fmt) { b, _ in mic.append(b) }
    do { try engine.start(); log("mic: started \(fmt)") } catch { log("mic: start failed \(error) — the watchdog will retry") }
    micObserver = NotificationCenter.default.addObserver(forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main) { _ in
        restartMic("device changed")
    }
}
var lastMicRestart = Date.distantPast
func restartMic(_ why: String) {
    guard Date().timeIntervalSince(lastMicRestart) > 2 else { return }
    lastMicRestart = Date()
    log("mic: \(why) — rebuilding the engine")
    mic.lock.lock(); mic.lastAudio = Date(); mic.zeroSince = nil; mic.lock.unlock()
    startMic()
}

// ── system: a global process tap wrapped in a private aggregate device ──
func prop<T>(_ obj: AudioObjectID, _ sel: AudioObjectPropertySelector, _ def: T) -> T {
    var addr = AudioObjectPropertyAddress(mSelector: sel, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
    var size = UInt32(MemoryLayout<T>.size)
    var v = def
    let st = AudioObjectGetPropertyData(obj, &addr, 0, nil, &size, &v)
    return st == noErr ? v : def
}

var tapID = AudioObjectID(kAudioObjectUnknown)
var aggID = AudioObjectID(kAudioObjectUnknown)
var procID: AudioDeviceIOProcID?
let ioQueue = DispatchQueue(label: "recorder.system", qos: .userInitiated)

func startSystem() -> Bool {
    let desc = CATapDescription(stereoGlobalTapButExcludeProcesses: [])
    desc.uuid = UUID()
    desc.muteBehavior = .unmuted
    desc.isPrivate = true
    var st = AudioHardwareCreateProcessTap(desc, &tapID)
    guard st == noErr else { log("system: create tap failed \(st)"); return false }

    var asbd: AudioStreamBasicDescription = prop(tapID, kAudioTapPropertyFormat, AudioStreamBasicDescription())
    guard let tapFormat = AVAudioFormat(streamDescription: &asbd) else { log("system: bad tap format"); return false }

    let outDev: AudioObjectID = prop(AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyDefaultSystemOutputDevice, AudioObjectID(0))
    let outUID: CFString = prop(outDev, kAudioDevicePropertyDeviceUID, "" as CFString)
    let agg: [String: Any] = [
        kAudioAggregateDeviceNameKey: "MyJarvis Meeting Tap",
        kAudioAggregateDeviceUIDKey: UUID().uuidString,
        kAudioAggregateDeviceMainSubDeviceKey: outUID as String,
        kAudioAggregateDeviceIsPrivateKey: true,
        kAudioAggregateDeviceIsStackedKey: false,
        kAudioAggregateDeviceTapAutoStartKey: true,
        kAudioAggregateDeviceSubDeviceListKey: [[kAudioSubDeviceUIDKey: outUID as String]],
        kAudioAggregateDeviceTapListKey: [[kAudioSubTapDriftCompensationKey: true,
                                           kAudioSubTapUIDKey: desc.uuid.uuidString]],
    ]
    st = AudioHardwareCreateAggregateDevice(agg as CFDictionary, &aggID)
    guard st == noErr else { log("system: create aggregate failed \(st)"); return false }

    st = AudioDeviceCreateIOProcIDWithBlock(&procID, aggID, ioQueue) { _, inData, _, _, _ in
        guard let b = AVAudioPCMBuffer(pcmFormat: tapFormat, bufferListNoCopy: inData, deallocator: nil) else { return }
        sys.append(b)
    }
    guard st == noErr, let procID else { log("system: ioproc failed \(st)"); return false }
    st = AudioDeviceStart(aggID, procID)
    guard st == noErr else { log("system: start failed \(st)"); return false }
    sys.lock.lock(); sys.lastAudio = Date(); sys.lock.unlock()   // the watchdog counts from now
    log("system: started \(tapFormat) via output \(outUID)")
    return true
}

func stopSystem() {
    if let procID { AudioDeviceStop(aggID, procID); AudioDeviceDestroyIOProcID(aggID, procID) }
    if aggID != kAudioObjectUnknown { AudioHardwareDestroyAggregateDevice(aggID) }
    if tapID != kAudioObjectUnknown { AudioHardwareDestroyProcessTap(tapID) }
    procID = nil; aggID = AudioObjectID(kAudioObjectUnknown); tapID = AudioObjectID(kAudioObjectUnknown)
}

// ⛔ The aggregate device is built on the output device that was current at start.
// Switch to headphones and that device goes away — the tap stops delivering and the
// other side vanishes from the transcript (meeting 110, 6 Oct: silent from 9:44 on,
// while the mic, which already restarted itself, carried on). So: rebuild the tap
// when the default output changes, and as a backstop whenever it has delivered
// nothing for 3 s (a running tap delivers silence, never nothing).
var lastSystemRestart = Date.distantPast
func restartSystem(_ why: String) {
    guard Date().timeIntervalSince(lastSystemRestart) > 2 else { return }
    lastSystemRestart = Date()
    log("system: \(why) — restarting tap")
    sys.lock.lock(); sys.lastAudio = Date(); sys.lock.unlock()
    // Off the main queue, like the mic: creating a process tap can block in coreaudiod.
    sysQueue.async {
        stopSystem()
        if !startSystem() { log("system: restart failed — will retry") }
    }
}
for sel in [kAudioHardwarePropertyDefaultOutputDevice, kAudioHardwarePropertyDefaultSystemOutputDevice] {
    var addr = AudioObjectPropertyAddress(mSelector: sel, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
    AudioObjectAddPropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &addr, DispatchQueue.main) { _, _ in
        restartSystem("output device changed")
    }
}
let systemWatchdog = DispatchSource.makeTimerSource(queue: .main)
systemWatchdog.schedule(deadline: .now() + 5, repeating: 1)
var lastBeat = Date()
var lastTick = Date()
systemWatchdog.setEventHandler {
    // A watchdog tick that arrives long after the last one means the whole process was frozen:
    // the Mac slept (recording 132, 7 Oct: lid closed 03:21:52 UTC, 15 min with nothing recorded).
    let gap = Date().timeIntervalSince(lastTick); lastTick = Date()
    if gap > 10 { log(String(format: "⛔ GAP: the Mac was asleep (or the recorder frozen) for %.0f s (%.1f min) — NOTHING was recorded in that time", gap, gap / 60)) }
    sys.lock.lock(); let quiet = Date().timeIntervalSince(sys.lastAudio); sys.lock.unlock()
    if quiet > 3 { restartSystem(String(format: "no audio for %.0f s", quiet)) }
    // The mic, by the same rule — plus exact digital silence, which a working mic never makes.
    mic.lock.lock()
    let micQuiet = Date().timeIntervalSince(mic.lastAudio)
    let micZero = mic.zeroSince.map { Date().timeIntervalSince($0) } ?? 0
    mic.lock.unlock()
    if micQuiet > 3 { restartMic(String(format: "no audio for %.0f s", micQuiet)) }
    else if micZero > 10 { restartMic(String(format: "pure digital silence for %.0f s", micZero)) }
    // One line a minute per track, so a dead track shows in the log the minute it dies.
    if Date().timeIntervalSince(lastBeat) >= 60 { lastBeat = Date(); log(mic.heartbeat()); log(sys.heartbeat()) }
}

func finish() {
    // ⛔ STOP NEVER TOUCHES AN AUDIO DEVICE. Stopping the engine or the tap waits on coreaudiod,
    // and that is exactly what can be stuck (7 Oct: a hung mic setup made Stop hang forever).
    // Flush what we have, write `done`, exit — the kernel releases the devices and the
    // power assertion. Runs on its own queue, so a blocked main queue cannot block it either.
    mic.flush(); sys.flush()
    let meta: [String: Any] = ["started": startedAt.timeIntervalSince1970, "ended": Date().timeIntervalSince1970]
    try? JSONSerialization.data(withJSONObject: meta).write(to: outDir.appendingPathComponent("done"))
    log("stopped")
    exit(0)
}

// SIGUSR1 = rebuild the mic engine now (a live repair, and how the rebuild path is tested).
signal(SIGUSR1, SIG_IGN)
let usr1 = DispatchSource.makeSignalSource(signal: SIGUSR1, queue: .main)
usr1.setEventHandler { lastMicRestart = .distantPast; restartMic("SIGUSR1") }
usr1.resume()

let stopQueue = DispatchQueue(label: "recorder.stop")
var signalSources: [DispatchSourceSignal] = []
for s in [SIGINT, SIGTERM] {
    signal(s, SIG_IGN)
    let src = DispatchSource.makeSignalSource(signal: s, queue: stopQueue)
    src.setEventHandler { finish() }
    src.resume()
    signalSources.append(src)
}

try? JSONSerialization.data(withJSONObject: ["started": startedAt.timeIntervalSince1970, "chunk_s": chunkSeconds, "pid": getpid()])
    .write(to: outDir.appendingPathComponent("started"))
// ⛔ KEEP THE MAC AWAKE WHILE RECORDING (7 Oct 2026). Recording 132 "stopped" for 15 minutes
// because the lid closed and the Mac slept: every process froze. PreventSystemSleep keeps it
// awake, lid closed included, ON MAINS POWER; on battery macOS sleeps on lid-close regardless
// (no app can stop that), and the GAP line above says so in the log. Released when we exit.
var sleepAssertion = IOPMAssertionID(0)
let asserted = IOPMAssertionCreateWithName(kIOPMAssertionTypePreventSystemSleep as CFString,
    IOPMAssertionLevel(kIOPMAssertionLevelOn), "Tailormind Desktop is recording a meeting" as CFString, &sleepAssertion)
log(asserted == kIOReturnSuccess ? "power: holding the Mac awake while recording (works on mains power, lid closed included)" : "power: could NOT hold the Mac awake (\(asserted))")

startMic()
sysQueue.async { if !startSystem() { log("system track unavailable — the watchdog will keep retrying") } }
systemWatchdog.resume()
dispatchMain()
