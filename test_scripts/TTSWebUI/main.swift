import Foundation
import Nightingale
#if canImport(Network)
import Network
#endif

// MARK: - TTS Web UI Server
// Simple HTTP server with web interface for TTS testing

let PORT: UInt16 = 8080
let VERSION = "1.0.0"

// Get current git commit hash
func getGitVersion() -> String {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
    process.arguments = ["rev-parse", "--short", "HEAD"]
    process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)

    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = FileHandle.nullDevice

    do {
        try process.run()
        process.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        if let hash = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines), !hash.isEmpty {
            return hash
        }
    } catch {}
    return "unknown"
}

// Check for updates from GitHub
func checkForUpdates() -> (isLatest: Bool, localHash: String, remoteHash: String?) {
    let localHash = getGitVersion()

    // Try to fetch latest from remote
    let fetchProcess = Process()
    fetchProcess.executableURL = URL(fileURLWithPath: "/usr/bin/git")
    fetchProcess.arguments = ["fetch", "--quiet"]
    fetchProcess.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    fetchProcess.standardOutput = FileHandle.nullDevice
    fetchProcess.standardError = FileHandle.nullDevice
    try? fetchProcess.run()
    fetchProcess.waitUntilExit()

    // Get remote HEAD
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
    process.arguments = ["rev-parse", "--short", "origin/main"]
    process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)

    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = FileHandle.nullDevice

    do {
        try process.run()
        process.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        if let remoteHash = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines), !remoteHash.isEmpty {
            return (localHash == remoteHash, localHash, remoteHash)
        }
    } catch {}
    return (true, localHash, nil)
}

let gitHash = getGitVersion()

print("🎤 Nightingale TTS Web UI v\(VERSION) (\(gitHash))")
print(String(repeating: "=", count: 60))

// Check for updates
print("🔄 Checking for updates...")
let (isLatest, localHash, remoteHash) = checkForUpdates()
if isLatest {
    print("✅ You are running the latest version")
} else if let remote = remoteHash {
    print("⚠️  Update available: \(localHash) → \(remote)")
    print("   Run: cd ~/Nightingale_Swift && git pull")
}

// MARK: - HTML Interface
func makeHtmlPage(version: String, gitHash: String, isWarmed: Bool) -> String {
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nightingale TTS</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }
        h1 { color: #00d4ff; margin-bottom: 5px; }
        .header-row { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; }
        .version { color: #666; font-size: 12px; }
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 500;
        }
        .status-badge.warm {
            background: rgba(0, 255, 136, 0.15);
            color: #00ff88;
            border: 1px solid rgba(0, 255, 136, 0.3);
        }
        .status-badge.cold {
            background: rgba(255, 170, 0, 0.15);
            color: #ffaa00;
            border: 1px solid rgba(255, 170, 0, 0.3);
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        .status-dot.warm { background: #00ff88; }
        .status-dot.cold { background: #ffaa00; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .subtitle { color: #888; margin-bottom: 30px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; color: #aaa; font-weight: 500; }
        textarea {
            width: 100%;
            height: 120px;
            padding: 12px;
            border: 2px solid #333;
            border-radius: 8px;
            background: #0f0f1a;
            color: #fff;
            font-size: 16px;
            resize: vertical;
        }
        textarea:focus { border-color: #00d4ff; outline: none; }
        select, input[type="checkbox"] { accent-color: #00d4ff; }
        select {
            padding: 10px;
            border: 2px solid #333;
            border-radius: 8px;
            background: #0f0f1a;
            color: #fff;
            font-size: 14px;
            min-width: 200px;
        }
        .checkbox-group { display: flex; gap: 20px; flex-wrap: wrap; }
        .checkbox-label {
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
        }
        .checkbox-label input { width: 18px; height: 18px; }
        button {
            background: linear-gradient(135deg, #00d4ff, #0099cc);
            color: #000;
            border: none;
            padding: 15px 40px;
            font-size: 18px;
            font-weight: bold;
            border-radius: 8px;
            cursor: pointer;
            transition: transform 0.1s, box-shadow 0.2s;
        }
        button:hover { transform: translateY(-2px); box-shadow: 0 4px 20px rgba(0, 212, 255, 0.4); }
        button:active { transform: translateY(0); }
        button:disabled { background: #555; cursor: not-allowed; transform: none; box-shadow: none; }
        .button-group { display: flex; gap: 10px; justify-content: center; }
        #stop { background: linear-gradient(135deg, #ff6b6b, #cc4444); }
        #stop:hover:not(:disabled) { box-shadow: 0 4px 20px rgba(255, 107, 107, 0.4); }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 25px 0;
        }
        .metric {
            background: #0f0f1a;
            border: 2px solid #333;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
        .metric-value { font-size: 28px; font-weight: bold; color: #00d4ff; }
        .metric-value.rtf-fast { color: #00ff88; }
        .metric-value.rtf-ok { color: #ffdd00; }
        .metric-value.rtf-slow { color: #ff6b6b; }
        .metric-label { color: #888; font-size: 12px; margin-top: 5px; }
        .rtf-key {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 10px;
            font-size: 11px;
            color: #888;
        }
        .rtf-key span { display: flex; align-items: center; gap: 5px; }
        .rtf-dot { width: 10px; height: 10px; border-radius: 50%; }
        .rtf-dot.fast { background: #00ff88; }
        .rtf-dot.ok { background: #ffdd00; }
        .rtf-dot.slow { background: #ff6b6b; }
        .status {
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            display: none;
        }
        .status.loading { display: block; background: #1a3a5c; border: 1px solid #2a5a8c; }
        .status.success { display: block; background: #1a3a2a; border: 1px solid #2a6a3a; }
        .status.error { display: block; background: #3a1a1a; border: 1px solid #6a2a2a; color: #ff6b6b; }
        audio { width: 100%; margin-top: 20px; }
        .audio-container { display: none; }
        .audio-container.visible { display: block; }
        .row { display: flex; gap: 20px; flex-wrap: wrap; }
        .row > .form-group { flex: 1; min-width: 200px; }
    </style>
</head>
<body>
    <div class="header-row">
        <div>
            <h1>🎤 Nightingale TTS</h1>
            <p class="subtitle">Swift/MLX Text-to-Speech Engine</p>
        </div>
        <div style="text-align: right;">
            <div class="status-badge \(isWarmed ? "warm" : "cold")">
                <span class="status-dot \(isWarmed ? "warm" : "cold")"></span>
                \(isWarmed ? "Model Warmed" : "Warming...")
            </div>
            <div class="version">v\(version) (\(gitHash))</div>
        </div>
    </div>

    <div class="form-group">
        <label for="text">Text to Synthesize</label>
        <textarea id="text" placeholder="Enter text to convert to speech (minimum 30 characters)...">Hello! This is Nightingale, a streaming text-to-speech engine running entirely on Apple Silicon.</textarea>
    </div>

    <div class="row">
        <div class="form-group">
            <label for="voice">Voice</label>
            <select id="voice">
                <option value="samantha" selected>Samantha (Female)</option>
                <option value="sujano">Sujano (Male)</option>
            </select>
        </div>
        <div class="form-group">
            <label for="temperature">Temperature</label>
            <select id="temperature">
                <option value="0.0001" selected>0.0001 (Deterministic)</option>
                <option value="0.3">0.3 (Low)</option>
                <option value="0.5">0.5 (Medium)</option>
                <option value="0.7">0.7 (High)</option>
                <option value="1.0">1.0 (Max)</option>
            </select>
        </div>
    </div>

    <div class="form-group">
        <label>Options</label>
        <div class="checkbox-group">
            <label class="checkbox-label">
                <input type="checkbox" id="quantized" checked>
                <span>INT8 Quantized (Faster)</span>
            </label>
            <label class="checkbox-label">
                <input type="checkbox" id="autoplay" checked>
                <span>Auto-play</span>
            </label>
            <label class="checkbox-label">
                <input type="checkbox" id="streaming" checked>
                <span>Streaming</span>
            </label>
        </div>
    </div>

    <div class="button-group">
        <button id="generate" onclick="generateSpeech()">▶ Generate Speech</button>
        <button id="stop" onclick="stopPlayback()" disabled>⏹ Stop</button>
    </div>

    <div id="status" class="status"></div>

    <div class="metrics">
        <div class="metric">
            <div class="metric-value" id="ttfa">--</div>
            <div class="metric-label">TTFA (s)</div>
        </div>
        <div class="metric">
            <div class="metric-value" id="genTime">--</div>
            <div class="metric-label">Gen Time (s)</div>
        </div>
        <div class="metric">
            <div class="metric-value" id="rtf">--</div>
            <div class="metric-label">RTF</div>
        </div>
        <div class="metric">
            <div class="metric-value" id="tokens">--</div>
            <div class="metric-label">Tokens</div>
        </div>
        <div class="metric">
            <div class="metric-value" id="duration">--</div>
            <div class="metric-label">Audio (s)</div>
        </div>
    </div>
    <div class="rtf-key">
        <span><div class="rtf-dot fast"></div> Fast (&lt;0.8x)</span>
        <span><div class="rtf-dot ok"></div> Real-time (0.8-1.2x)</span>
        <span><div class="rtf-dot slow"></div> Slow (&gt;1.2x)</span>
    </div>

    <div id="audioContainer" class="audio-container">
        <audio id="audio" controls></audio>
    </div>

    <script>
        let audioContext = null;
        let nextPlayTime = 0;
        let isPlaying = false;
        let abortController = null;
        let activeSources = [];

        function stopPlayback() {
            // Abort fetch request
            if (abortController) {
                abortController.abort();
                abortController = null;
            }
            // Stop all scheduled audio
            activeSources.forEach(source => {
                try { source.stop(); } catch(e) {}
            });
            activeSources = [];
            isPlaying = false;
            nextPlayTime = 0;

            document.getElementById('generate').disabled = false;
            document.getElementById('generate').textContent = '▶ Generate Speech';
            document.getElementById('stop').disabled = true;
            document.getElementById('status').textContent = 'Stopped';
            document.getElementById('status').className = 'status';
        }

        async function generateSpeech() {
            const btn = document.getElementById('generate');
            const stopBtn = document.getElementById('stop');
            const status = document.getElementById('status');
            const audioContainer = document.getElementById('audioContainer');
            const audio = document.getElementById('audio');

            // Stop any existing playback
            stopPlayback();

            btn.disabled = true;
            stopBtn.disabled = false;
            btn.textContent = '⏳ Generating...';
            status.className = 'status loading';
            status.textContent = 'Generating speech...';
            audioContainer.className = 'audio-container';

            // Reset metrics
            ['ttfa', 'genTime', 'rtf', 'tokens', 'duration'].forEach(id => {
                document.getElementById(id).textContent = '--';
            });
            document.getElementById('rtf').classList.remove('rtf-fast', 'rtf-ok', 'rtf-slow');

            const params = new URLSearchParams({
                text: document.getElementById('text').value,
                voice: document.getElementById('voice').value,
                temperature: document.getElementById('temperature').value,
                quantized: document.getElementById('quantized').checked
            });

            const autoplay = document.getElementById('autoplay').checked;
            const streaming = document.getElementById('streaming').checked;

            try {
                const startTime = performance.now();
                const sampleRate = 24000;
                let allSamples = [];

                abortController = new AbortController();

                if (streaming) {
                    // STREAMING MODE: Play audio as it's generated
                    if (!audioContext) {
                        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate });
                    }
                    if (audioContext.state === 'suspended') {
                        await audioContext.resume();
                    }

                    nextPlayTime = 0;
                    isPlaying = false;
                    activeSources = [];

                    const response = await fetch('/stream?' + params.toString(), { signal: abortController.signal });
                    if (!response.ok) throw new Error(await response.text());

                    const tokens = response.headers.get('X-Tokens');
                    if (tokens) document.getElementById('tokens').textContent = tokens;

                    const reader = response.body.getReader();

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;

                        if (value && value.length > 0) {
                            const int16Array = new Int16Array(value.buffer, value.byteOffset, value.byteLength / 2);
                            const chunkSamples = new Float32Array(int16Array.length);
                            for (let i = 0; i < int16Array.length; i++) {
                                chunkSamples[i] = int16Array[i] / 32768.0;
                                allSamples.push(chunkSamples[i]);
                            }

                            if (autoplay && chunkSamples.length > 0) {
                                const buffer = audioContext.createBuffer(1, chunkSamples.length, sampleRate);
                                buffer.getChannelData(0).set(chunkSamples);

                                const source = audioContext.createBufferSource();
                                source.buffer = buffer;
                                source.connect(audioContext.destination);
                                activeSources.push(source);

                                if (!isPlaying) {
                                    nextPlayTime = audioContext.currentTime;
                                    isPlaying = true;
                                    const ttfa = (performance.now() - startTime) / 1000;
                                    document.getElementById('ttfa').textContent = ttfa.toFixed(2);
                                    status.textContent = 'Playing audio...';
                                }

                                source.start(nextPlayTime);
                                nextPlayTime += buffer.duration;
                            }
                        }
                    }
                } else {
                    // BATCH MODE: Generate full audio, then play
                    document.getElementById('ttfa').textContent = 'N/A';

                    const response = await fetch('/generate?' + params.toString(), { signal: abortController.signal });
                    if (!response.ok) throw new Error(await response.text());

                    const tokens = response.headers.get('X-Tokens');
                    if (tokens) document.getElementById('tokens').textContent = tokens;

                    const arrayBuffer = await response.arrayBuffer();
                    const int16Array = new Int16Array(arrayBuffer);
                    allSamples = new Array(int16Array.length);
                    for (let i = 0; i < int16Array.length; i++) {
                        allSamples[i] = int16Array[i] / 32768.0;
                    }
                }

                const endTime = performance.now();
                const genTime = (endTime - startTime) / 1000;
                document.getElementById('genTime').textContent = genTime.toFixed(1);

                const duration = allSamples.length / sampleRate;
                document.getElementById('duration').textContent = duration.toFixed(2);

                const rtf = genTime / duration;
                const rtfEl = document.getElementById('rtf');
                rtfEl.textContent = rtf.toFixed(2);
                rtfEl.classList.remove('rtf-fast', 'rtf-ok', 'rtf-slow');
                if (rtf < 0.8) {
                    rtfEl.classList.add('rtf-fast');
                } else if (rtf <= 1.2) {
                    rtfEl.classList.add('rtf-ok');
                } else {
                    rtfEl.classList.add('rtf-slow');
                }

                const wavBlob = createWavBlob(allSamples, sampleRate);
                const wavUrl = URL.createObjectURL(wavBlob);
                audio.src = wavUrl;
                audioContainer.className = 'audio-container visible';

                // Auto-play in batch mode
                if (!streaming && autoplay) {
                    audio.play();
                }

                status.className = 'status success';
                status.textContent = 'Generation complete!';

            } catch (err) {
                if (err.name === 'AbortError') {
                    // User stopped - already handled by stopPlayback()
                    return;
                }
                status.className = 'status error';
                status.textContent = 'Error: ' + err.message;
            } finally {
                btn.disabled = false;
                btn.textContent = '▶ Generate Speech';
                stopBtn.disabled = true;
                abortController = null;
            }
        }

        function createWavBlob(samples, sampleRate) {
            const buffer = new ArrayBuffer(44 + samples.length * 2);
            const view = new DataView(buffer);

            const writeString = (offset, str) => {
                for (let i = 0; i < str.length; i++) {
                    view.setUint8(offset + i, str.charCodeAt(i));
                }
            };
            writeString(0, 'RIFF');
            view.setUint32(4, 36 + samples.length * 2, true);
            writeString(8, 'WAVE');
            writeString(12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true);
            view.setUint16(22, 1, true);
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, sampleRate * 2, true);
            view.setUint16(32, 2, true);
            view.setUint16(34, 16, true);
            writeString(36, 'data');
            view.setUint32(40, samples.length * 2, true);

            let offset = 44;
            for (let i = 0; i < samples.length; i++) {
                const s = Math.max(-1, Math.min(1, samples[i]));
                view.setInt16(offset, s * 32767, true);
                offset += 2;
            }

            return new Blob([buffer], { type: 'audio/wav' });
        }

        document.getElementById('text').addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                generateSpeech();
            }
        });
    </script>
</body>
</html>
"""
}

// MARK: - Simple HTTP Server

class TTSWebServer {
    let engine = ChatterboxEngine()
    var currentVoice: String = ""
    var isQuantized: Bool = true
    var isLoaded: Bool = false
    var isWarmed: Bool = false

    let projectRoot = FileManager.default.currentDirectoryPath
    lazy var modelDir = URL(fileURLWithPath: "\(projectRoot)/models/chatterbox")
    lazy var voicesDir = URL(fileURLWithPath: "\(projectRoot)/baked_voices")

    func loadModels(quantized: Bool) async throws {
        if isLoaded && isQuantized == quantized { return }

        print("⏳ Loading models (quantized: \(quantized))...")
        let start = Date()
        try await engine.loadModels(modelsURL: modelDir, useQuantization: quantized)
        isQuantized = quantized
        isLoaded = true
        print("✅ Models loaded in \(String(format: "%.2f", Date().timeIntervalSince(start)))s")
    }

    func loadVoice(_ voice: String) async throws {
        if currentVoice == voice { return }

        print("🎤 Loading voice: \(voice)")
        try await engine.loadVoice(voice, voicesURL: voicesDir)
        currentVoice = voice
    }

    struct GenerationMetrics {
        let audio: [Float]
        let totalMs: Int
        let t3Ms: Int
        let s3Ms: Int
        let ttfaMs: Int  // Theoretical time to first audio (streaming)
        let tokenCount: Int
    }

    func generateAudio(text: String, temperature: Float, streaming: Bool) async throws -> GenerationMetrics {
        let overallStart = Date()

        // Generate tokens first (T3)
        let t3Start = Date()
        let tokens = try await engine.runT3Only(text, temperature: temperature)
        let t3Ms = Int(Date().timeIntervalSince(t3Start) * 1000)

        // Synthesize audio (S3Gen)
        let s3Start = Date()
        let audio: [Float]

        if streaming {
            let (audioResult, _) = try await synthesizeStreamingWithTTFA(tokens: tokens)
            audio = audioResult
        } else {
            audio = try await engine.synthesizeFromTokens(tokens)
        }
        let s3Ms = Int(Date().timeIntervalSince(s3Start) * 1000)

        // Calculate theoretical TTFA for streaming:
        // Time to generate first 20 tokens + time to synthesize first chunk
        let firstChunkTokens = min(20, tokens.count)
        let t3PerToken = Double(t3Ms) / Double(tokens.count)
        let s3PerToken = Double(s3Ms) / Double(tokens.count)
        let ttfaMs = Int(t3PerToken * Double(firstChunkTokens) + s3PerToken * Double(firstChunkTokens))

        let totalMs = Int(Date().timeIntervalSince(overallStart) * 1000)
        return GenerationMetrics(
            audio: audio,
            totalMs: totalMs,
            t3Ms: t3Ms,
            s3Ms: s3Ms,
            ttfaMs: ttfaMs,
            tokenCount: tokens.count
        )
    }

    func synthesizeStreamingWithTTFA(tokens: [Int]) async throws -> (audio: [Float], firstChunkMs: Int) {
        let initialChunkSize = 20
        let subsequentChunkSize = 10
        let overlapTokens = 4
        let samplesPerToken = 960  // 40ms at 24kHz

        var allAudio: [Float] = []
        var tokenIndex = 0
        var chunkNum = 0
        var firstChunkMs = 0

        while tokenIndex < tokens.count {
            let chunkSize = (chunkNum == 0) ? initialChunkSize : subsequentChunkSize
            let endIndex = min(tokenIndex + chunkSize, tokens.count)

            var chunkTokens: [Int]
            if chunkNum == 0 {
                chunkTokens = Array(tokens[tokenIndex..<endIndex])
            } else {
                let overlapStart = max(0, tokenIndex - overlapTokens)
                chunkTokens = Array(tokens[overlapStart..<endIndex])
            }

            let chunkStart = Date()
            let chunkAudio = try await engine.synthesizeFromTokens(chunkTokens)
            let chunkTime = Date().timeIntervalSince(chunkStart)

            // Record S3Gen time for first chunk only
            if chunkNum == 0 {
                firstChunkMs = Int(chunkTime * 1000)
            }

            let samplesToSkip = (chunkNum == 0) ? 0 : overlapTokens * samplesPerToken
            if samplesToSkip < chunkAudio.count {
                let usableAudio = Array(chunkAudio[samplesToSkip...])
                allAudio.append(contentsOf: usableAudio)
            }

            tokenIndex = endIndex
            chunkNum += 1
        }

        return (allAudio, firstChunkMs)
    }

    func start() async {
        // Pre-load models
        do {
            try await loadModels(quantized: true)
            try await loadVoice("samantha")
        } catch {
            print("❌ Failed to load models: \(error)")
            return
        }

        // Warmup disabled for faster startup - first generation will be slower
        isWarmed = true
        print("\n⚡ Warmup disabled - first generation will prime GPU caches")

        // Create socket
        let server = try! Socket(port: PORT)
        print("\n🌐 Server running at http://localhost:\(PORT)")
        print("   Press Ctrl+C to stop\n")

        while true {
            guard let client = server.accept() else { continue }

            Task {
                await handleClient(client)
            }
        }
    }

    func handleClient(_ client: ClientSocket) async {
        guard let request = client.readRequest() else {
            client.close()
            return
        }

        let (_, path) = parseRequest(request)

        if path == "/" || path == "/index.html" {
            // Serve HTML page with current state
            let html = makeHtmlPage(version: VERSION, gitHash: gitHash, isWarmed: isWarmed)
            client.sendResponse(
                status: "200 OK",
                contentType: "text/html",
                body: html.data(using: .utf8)!
            )
        } else if path.hasPrefix("/stream") {
            await handleStreamingGenerate(client, path: path)
        } else if path.hasPrefix("/generate") {
            await handleGenerate(client, path: path)
        } else {
            client.sendResponse(
                status: "404 Not Found",
                contentType: "text/plain",
                body: "Not Found".data(using: .utf8)!
            )
        }

        client.close()
    }

    func handleGenerate(_ client: ClientSocket, path: String) async {
        // Parse query parameters
        let params = parseQueryParams(path)

        guard let text = params["text"], !text.isEmpty else {
            client.sendResponse(
                status: "400 Bad Request",
                contentType: "text/plain",
                body: "Missing 'text' parameter".data(using: .utf8)!
            )
            return
        }

        // Minimum text length to avoid model issues with short input
        guard text.count >= 30 else {
            client.sendResponse(
                status: "400 Bad Request",
                contentType: "text/plain",
                body: "Text too short. Minimum 30 characters required to avoid audio artifacts.".data(using: .utf8)!
            )
            return
        }

        let voice = params["voice"] ?? "samantha"
        let temperature = Float(params["temperature"] ?? "0.0001") ?? 0.0001
        let quantized = params["quantized"] == "true"
        let streaming = params["streaming"] == "true"

        print("\n📝 Request: \"\(text.prefix(50))...\"")
        print("   Voice: \(voice), Temp: \(temperature), Quantized: \(quantized), Streaming: \(streaming)")

        do {
            // Load models/voice if needed
            try await loadModels(quantized: quantized)
            try await loadVoice(voice)

            // Generate
            let metrics = try await generateAudio(
                text: text,
                temperature: temperature,
                streaming: streaming
            )

            // Calculate metrics
            let sampleRate = 24000
            let duration = Float(metrics.audio.count) / Float(sampleRate)
            let rtf = Double(metrics.totalMs) / 1000.0 / Double(duration)

            print("   ✅ Generated \(metrics.audio.count) samples (\(String(format: "%.2f", duration))s)")
            print("   ⏱  Total: \(metrics.totalMs)ms (T3: \(metrics.t3Ms)ms, S3: \(metrics.s3Ms)ms)")
            print("   ⏱  TTFA: \(metrics.ttfaMs)ms, RTF: \(String(format: "%.2f", rtf))x, Tokens: \(metrics.tokenCount)")

            // Convert to WAV
            let wavData = createWAV(samples: metrics.audio, sampleRate: sampleRate)

            // Send response with metrics in headers
            let headers = [
                "X-Time-Ms": "\(metrics.totalMs)",
                "X-TTFA-Ms": "\(metrics.ttfaMs)",
                "X-T3-Ms": "\(metrics.t3Ms)",
                "X-S3-Ms": "\(metrics.s3Ms)",
                "X-RTF": String(format: "%.2f", rtf),
                "X-Tokens": "\(metrics.tokenCount)",
                "X-Duration": String(format: "%.2f", duration)
            ]

            client.sendResponse(
                status: "200 OK",
                contentType: "audio/wav",
                body: wavData,
                extraHeaders: headers
            )

        } catch {
            print("   ❌ Error: \(error)")
            client.sendResponse(
                status: "500 Internal Server Error",
                contentType: "text/plain",
                body: "Generation failed: \(error)".data(using: .utf8)!
            )
        }
    }

    func handleStreamingGenerate(_ client: ClientSocket, path: String) async {
        let params = parseQueryParams(path)

        guard let text = params["text"], text.count >= 30 else {
            client.sendResponse(
                status: "400 Bad Request",
                contentType: "text/plain",
                body: "Text must be at least 30 characters".data(using: .utf8)!
            )
            return
        }

        let voice = params["voice"] ?? "samantha"
        let temperature = Float(params["temperature"] ?? "0.0001") ?? 0.0001
        let quantized = params["quantized"] == "true"

        print("\n🌊 Streaming request: \"\(text.prefix(50))...\"")

        do {
            try await loadModels(quantized: quantized)
            try await loadVoice(voice)

            let overallStart = Date()

            // Generate T3 tokens first (fast)
            let t3Start = Date()
            let tokens = try await engine.runT3Only(text, temperature: temperature)
            let t3Ms = Int(Date().timeIntervalSince(t3Start) * 1000)
            print("   T3: \(t3Ms)ms for \(tokens.count) tokens")

            // Send chunked header with metadata
            client.sendChunkedHeader(contentType: "application/octet-stream", extraHeaders: [
                "X-Sample-Rate": "24000",
                "X-Tokens": "\(tokens.count)",
                "X-T3-Ms": "\(t3Ms)"
            ])

            // Stream audio in chunks with crossfading
            let initialChunkSize = 20
            let subsequentChunkSize = 10
            let overlapTokens = 6  // Increased overlap for smoother transitions
            let samplesPerToken = 960
            let crossfadeSamples = overlapTokens * samplesPerToken

            var tokenIndex = 0
            var chunkNum = 0
            var ttfaMs = 0
            var previousTail: [Float] = []  // Last samples from previous chunk for crossfading

            while tokenIndex < tokens.count {
                let chunkSize = (chunkNum == 0) ? initialChunkSize : subsequentChunkSize
                let endIndex = min(tokenIndex + chunkSize, tokens.count)

                var chunkTokens: [Int]
                if chunkNum == 0 {
                    chunkTokens = Array(tokens[tokenIndex..<endIndex])
                } else {
                    let overlapStart = max(0, tokenIndex - overlapTokens)
                    chunkTokens = Array(tokens[overlapStart..<endIndex])
                }

                let chunkStart = Date()
                let chunkAudio = try await engine.synthesizeFromTokens(chunkTokens)
                let chunkTime = Int(Date().timeIntervalSince(chunkStart) * 1000)

                if chunkNum == 0 {
                    ttfaMs = t3Ms + chunkTime
                    print("   🎵 TTFA: \(ttfaMs)ms (T3: \(t3Ms)ms + S3: \(chunkTime)ms)")
                }

                var outputAudio: [Float]

                if chunkNum == 0 {
                    // First chunk - send most, keep tail for crossfading
                    let keepSamples = min(crossfadeSamples, chunkAudio.count)
                    let sendCount = chunkAudio.count - keepSamples
                    outputAudio = Array(chunkAudio[0..<sendCount])
                    previousTail = Array(chunkAudio[sendCount...])
                } else {
                    // Subsequent chunks - crossfade with previous tail
                    let overlapSamples = overlapTokens * samplesPerToken

                    // Current chunk starts with overlap region
                    var crossfaded: [Float] = []

                    // Crossfade the overlap region
                    let fadeLen = min(previousTail.count, min(overlapSamples, chunkAudio.count))
                    for i in 0..<fadeLen {
                        let fadeOut = Float(fadeLen - i) / Float(fadeLen)  // 1.0 -> 0.0
                        let fadeIn = Float(i) / Float(fadeLen)              // 0.0 -> 1.0
                        let mixed = previousTail[i] * fadeOut + chunkAudio[i] * fadeIn
                        crossfaded.append(mixed)
                    }

                    // Add remaining samples after overlap
                    if overlapSamples < chunkAudio.count {
                        let remaining = Array(chunkAudio[overlapSamples...])
                        // Keep tail for next crossfade
                        let keepSamples = min(crossfadeSamples, remaining.count)
                        let sendCount = remaining.count - keepSamples
                        crossfaded.append(contentsOf: remaining[0..<sendCount])
                        previousTail = Array(remaining[sendCount...])
                    } else {
                        previousTail = []
                    }

                    outputAudio = crossfaded
                }

                // Convert to Int16 PCM and send
                var pcmData = Data()
                for sample in outputAudio {
                    let clipped = max(-1.0, min(1.0, sample))
                    let scaled = Int16(clipped * 32767.0)
                    pcmData.append(scaled.littleEndianData)
                }

                if !pcmData.isEmpty {
                    client.sendChunk(pcmData)
                }

                tokenIndex = endIndex
                chunkNum += 1
            }

            // Send any remaining tail
            if !previousTail.isEmpty {
                var pcmData = Data()
                for sample in previousTail {
                    let clipped = max(-1.0, min(1.0, sample))
                    let scaled = Int16(clipped * 32767.0)
                    pcmData.append(scaled.littleEndianData)
                }
                client.sendChunk(pcmData)
            }

            client.sendChunkedEnd()

            let totalMs = Int(Date().timeIntervalSince(overallStart) * 1000)
            print("   ✅ Streamed \(chunkNum) chunks in \(totalMs)ms, TTFA: \(ttfaMs)ms")

        } catch {
            print("   ❌ Stream error: \(error)")
        }
    }

    func parseRequest(_ request: String) -> (method: String, path: String) {
        let lines = request.components(separatedBy: "\r\n")
        guard let firstLine = lines.first else { return ("GET", "/") }
        let parts = firstLine.components(separatedBy: " ")
        guard parts.count >= 2 else { return ("GET", "/") }
        return (parts[0], parts[1])
    }

    func parseQueryParams(_ path: String) -> [String: String] {
        var params: [String: String] = [:]
        guard let queryStart = path.firstIndex(of: "?") else { return params }

        let query = String(path[path.index(after: queryStart)...])
        for pair in query.components(separatedBy: "&") {
            let parts = pair.components(separatedBy: "=")
            if parts.count == 2 {
                let key = parts[0].removingPercentEncoding ?? parts[0]
                // URL query strings use + for space, then percent encoding
                let rawValue = parts[1].replacingOccurrences(of: "+", with: " ")
                let value = rawValue.removingPercentEncoding ?? rawValue
                params[key] = value
            }
        }
        return params
    }

    func createWAV(samples: [Float], sampleRate: Int) -> Data {
        let numSamples = samples.count
        let dataSize = numSamples * 2

        var wavData = Data()

        // RIFF header
        wavData.append("RIFF".data(using: .ascii)!)
        wavData.append(UInt32(36 + dataSize).littleEndianData)
        wavData.append("WAVE".data(using: .ascii)!)

        // fmt chunk
        wavData.append("fmt ".data(using: .ascii)!)
        wavData.append(UInt32(16).littleEndianData)
        wavData.append(UInt16(1).littleEndianData)   // PCM
        wavData.append(UInt16(1).littleEndianData)   // Mono
        wavData.append(UInt32(sampleRate).littleEndianData)
        wavData.append(UInt32(sampleRate * 2).littleEndianData)
        wavData.append(UInt16(2).littleEndianData)   // Block align
        wavData.append(UInt16(16).littleEndianData)  // Bits per sample

        // data chunk
        wavData.append("data".data(using: .ascii)!)
        wavData.append(UInt32(dataSize).littleEndianData)

        // Samples
        for sample in samples {
            let clipped = max(-1.0, min(1.0, sample))
            let scaled = Int16(clipped * 32767.0)
            wavData.append(scaled.littleEndianData)
        }

        return wavData
    }
}

// MARK: - Simple Socket Helpers

class Socket {
    let fd: Int32

    init(port: UInt16) throws {
        fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { throw NSError(domain: "Socket", code: 1) }

        var opt: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, socklen_t(MemoryLayout<Int32>.size))

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr.s_addr = INADDR_ANY

        let bindResult = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bindResult >= 0 else { throw NSError(domain: "Bind", code: 2) }

        guard listen(fd, 10) >= 0 else { throw NSError(domain: "Listen", code: 3) }
    }

    func accept() -> ClientSocket? {
        var clientAddr = sockaddr_in()
        var addrLen = socklen_t(MemoryLayout<sockaddr_in>.size)

        let clientFd = withUnsafeMutablePointer(to: &clientAddr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.accept(fd, $0, &addrLen)
            }
        }

        guard clientFd >= 0 else { return nil }
        return ClientSocket(fd: clientFd)
    }
}

class ClientSocket {
    let fd: Int32

    init(fd: Int32) {
        self.fd = fd
    }

    func readRequest() -> String? {
        var buffer = [UInt8](repeating: 0, count: 8192)
        let bytesRead = read(fd, &buffer, buffer.count)
        guard bytesRead > 0 else { return nil }
        return String(bytes: buffer[0..<bytesRead], encoding: .utf8)
    }

    func sendResponse(status: String, contentType: String, body: Data, extraHeaders: [String: String] = [:]) {
        var response = "HTTP/1.1 \(status)\r\n"
        response += "Content-Type: \(contentType)\r\n"
        response += "Content-Length: \(body.count)\r\n"
        response += "Access-Control-Allow-Origin: *\r\n"
        response += "Access-Control-Expose-Headers: X-Time-Ms, X-TTFA-Ms, X-T3-Ms, X-S3-Ms, X-RTF, X-Tokens, X-Duration\r\n"

        for (key, value) in extraHeaders {
            response += "\(key): \(value)\r\n"
        }

        response += "\r\n"

        _ = response.withCString { ptr in
            write(fd, ptr, strlen(ptr))
        }

        body.withUnsafeBytes { ptr in
            _ = write(fd, ptr.baseAddress!, body.count)
        }
    }

    func sendChunkedHeader(contentType: String, extraHeaders: [String: String] = [:]) {
        var response = "HTTP/1.1 200 OK\r\n"
        response += "Content-Type: \(contentType)\r\n"
        response += "Transfer-Encoding: chunked\r\n"
        response += "Access-Control-Allow-Origin: *\r\n"
        response += "Cache-Control: no-cache\r\n"

        for (key, value) in extraHeaders {
            response += "\(key): \(value)\r\n"
        }

        response += "\r\n"

        _ = response.withCString { ptr in
            write(fd, ptr, strlen(ptr))
        }
    }

    func sendChunk(_ data: Data) {
        // Send chunk size in hex
        let sizeHex = String(format: "%X\r\n", data.count)
        _ = sizeHex.withCString { ptr in
            write(fd, ptr, strlen(ptr))
        }
        // Send chunk data
        data.withUnsafeBytes { ptr in
            _ = write(fd, ptr.baseAddress!, data.count)
        }
        // Send CRLF after chunk
        _ = "\r\n".withCString { ptr in
            write(fd, ptr, 2)
        }
    }

    func sendChunkedEnd() {
        // Send final empty chunk
        _ = "0\r\n\r\n".withCString { ptr in
            write(fd, ptr, 5)
        }
    }

    func close() {
        Darwin.close(fd)
    }
}

// MARK: - Extensions

extension FixedWidthInteger {
    var littleEndianData: Data {
        var value = self.littleEndian
        return Data(bytes: &value, count: MemoryLayout<Self>.size)
    }
}

// MARK: - Main

let server = TTSWebServer()
await server.start()
