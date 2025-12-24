import Foundation
import Nightingale
#if canImport(Network)
import Network
#endif

// MARK: - TTS Web UI Server
// Simple HTTP server with web interface for TTS testing

let PORT: UInt16 = 8080

print("🎤 Nightingale TTS Web UI")
print(String(repeating: "=", count: 60))

// MARK: - HTML Interface
let htmlPage = """
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
        .metric-label { color: #888; font-size: 12px; margin-top: 5px; }
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
    <h1>🎤 Nightingale TTS</h1>
    <p class="subtitle">Swift/MLX Text-to-Speech Engine</p>

    <div class="form-group">
        <label for="text">Text to Synthesize</label>
        <textarea id="text" placeholder="Enter text to convert to speech (minimum 30 characters)...">Hello! This is Nightingale, a streaming text-to-speech engine running entirely on Apple Silicon.</textarea>
    </div>

    <div class="row">
        <div class="form-group">
            <label for="voice">Voice</label>
            <select id="voice">
                <option value="samantha">Samantha (Female)</option>
                <option value="sujano" selected>Sujano (Male)</option>
            </select>
        </div>
        <div class="form-group">
            <label for="temperature">Temperature</label>
            <select id="temperature">
                <option value="0.0001">0.0001 (Deterministic)</option>
                <option value="0.3">0.3 (Low)</option>
                <option value="0.5" selected>0.5 (Medium)</option>
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
                <input type="checkbox" id="streaming" checked>
                <span>Streaming Mode</span>
            </label>
            <label class="checkbox-label">
                <input type="checkbox" id="autoplay" checked>
                <span>Auto-play</span>
            </label>
        </div>
    </div>

    <button id="generate" onclick="generateSpeech()">▶ Generate Speech</button>

    <div id="status" class="status"></div>

    <div class="metrics">
        <div class="metric">
            <div class="metric-value" id="ttfa">--</div>
            <div class="metric-label">TTFA (ms)</div>
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
            <div class="metric-label">Duration (s)</div>
        </div>
    </div>

    <div id="audioContainer" class="audio-container">
        <audio id="audio" controls></audio>
    </div>

    <script>
        async function generateSpeech() {
            const btn = document.getElementById('generate');
            const status = document.getElementById('status');
            const audioContainer = document.getElementById('audioContainer');
            const audio = document.getElementById('audio');

            btn.disabled = true;
            btn.textContent = '⏳ Generating...';
            status.className = 'status loading';
            status.textContent = 'Generating speech...';
            audioContainer.className = 'audio-container';

            // Reset metrics
            ['ttfa', 'rtf', 'tokens', 'duration'].forEach(id => {
                document.getElementById(id).textContent = '--';
            });

            const params = new URLSearchParams({
                text: document.getElementById('text').value,
                voice: document.getElementById('voice').value,
                temperature: document.getElementById('temperature').value,
                quantized: document.getElementById('quantized').checked,
                streaming: document.getElementById('streaming').checked
            });

            try {
                const startTime = performance.now();
                const response = await fetch('/generate?' + params.toString());

                if (!response.ok) {
                    throw new Error(await response.text());
                }

                // Get metrics from headers
                const ttfa = response.headers.get('X-TTFA-Ms');
                const rtf = response.headers.get('X-RTF');
                const tokens = response.headers.get('X-Tokens');
                const audioDuration = response.headers.get('X-Duration');

                if (ttfa) document.getElementById('ttfa').textContent = ttfa;
                if (rtf) document.getElementById('rtf').textContent = rtf;
                if (tokens) document.getElementById('tokens').textContent = tokens;
                if (audioDuration) document.getElementById('duration').textContent = audioDuration;

                const blob = await response.blob();
                const url = URL.createObjectURL(blob);

                audio.src = url;
                audioContainer.className = 'audio-container visible';

                if (document.getElementById('autoplay').checked) {
                    audio.play();
                }

                status.className = 'status success';
                status.textContent = 'Generation complete!';

            } catch (err) {
                status.className = 'status error';
                status.textContent = 'Error: ' + err.message;
            } finally {
                btn.disabled = false;
                btn.textContent = '▶ Generate Speech';
            }
        }

        // Allow Ctrl+Enter to generate
        document.getElementById('text').addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                generateSpeech();
            }
        });
    </script>
</body>
</html>
"""

// MARK: - Simple HTTP Server

class TTSWebServer {
    let engine = ChatterboxEngine()
    var currentVoice: String = ""
    var isQuantized: Bool = true
    var isLoaded: Bool = false

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

    func generateAudio(text: String, temperature: Float, streaming: Bool) async throws -> (audio: [Float], ttfaMs: Int, tokens: Int) {
        let overallStart = Date()

        // Generate tokens first
        let tokens = try await engine.runT3Only(text, temperature: temperature)
        let t3Time = Date().timeIntervalSince(overallStart)

        // Synthesize audio with TTFA measurement
        let audio: [Float]
        var ttfaMs: Int

        if streaming {
            // TTFA = T3 time for first 20 tokens + S3Gen first chunk time
            // Estimate first chunk T3 time proportionally
            let firstChunkTokens = min(20, tokens.count)
            let t3FirstChunkMs = Int(t3Time * 1000.0 * Double(firstChunkTokens) / Double(tokens.count))

            let (audioResult, firstChunkS3Ms) = try await synthesizeStreamingWithTTFA(tokens: tokens)
            audio = audioResult
            ttfaMs = t3FirstChunkMs + firstChunkS3Ms
        } else {
            // Full synthesis - TTFA is total time
            let s3Start = Date()
            audio = try await engine.synthesizeFromTokens(tokens)
            let s3Time = Date().timeIntervalSince(s3Start)
            ttfaMs = Int((t3Time + s3Time) * 1000)
        }

        return (audio, ttfaMs, tokens.count)
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
            try await loadVoice("sujano")
        } catch {
            print("❌ Failed to load models: \(error)")
            return
        }

        // Warmup generation to prime GPU caches
        // Use longer text to avoid short-text issues
        print("\n🔥 Warming up GPU caches...")
        let warmupStart = Date()
        do {
            let warmupText = "Wow! I absolutely cannot believe that it worked on the first try! This is amazing technology that will change everything."
            let _ = try await generateAudio(text: warmupText, temperature: 0.5, streaming: true)
            let warmupTime = Date().timeIntervalSince(warmupStart)
            print("✅ Warmup complete in \(String(format: "%.2f", warmupTime))s")
        } catch {
            print("⚠️ Warmup failed: \(error)")
        }

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
            // Serve HTML page
            client.sendResponse(
                status: "200 OK",
                contentType: "text/html",
                body: htmlPage.data(using: .utf8)!
            )
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

        let voice = params["voice"] ?? "sujano"
        let temperature = Float(params["temperature"] ?? "0.5") ?? 0.5
        let quantized = params["quantized"] == "true"
        let streaming = params["streaming"] == "true"

        print("\n📝 Request: \"\(text.prefix(50))...\"")
        print("   Voice: \(voice), Temp: \(temperature), Quantized: \(quantized), Streaming: \(streaming)")

        do {
            // Load models/voice if needed
            try await loadModels(quantized: quantized)
            try await loadVoice(voice)

            // Generate
            let genStart = Date()
            let (audio, ttfaMs, tokenCount) = try await generateAudio(
                text: text,
                temperature: temperature,
                streaming: streaming
            )
            let totalTime = Date().timeIntervalSince(genStart)

            // Calculate metrics
            let sampleRate = 24000
            let duration = Float(audio.count) / Float(sampleRate)
            let rtf = totalTime / Double(duration)

            print("   ✅ Generated \(audio.count) samples (\(String(format: "%.2f", duration))s)")
            print("   ⏱  TTFA: \(ttfaMs)ms, RTF: \(String(format: "%.2f", rtf))x, Tokens: \(tokenCount)")

            // Convert to WAV
            let wavData = createWAV(samples: audio, sampleRate: sampleRate)

            // Send response with metrics in headers
            let headers = [
                "X-TTFA-Ms": "\(ttfaMs)",
                "X-RTF": String(format: "%.2f", rtf),
                "X-Tokens": "\(tokenCount)",
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
                let value = parts[1].removingPercentEncoding ?? parts[1]
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
        response += "Access-Control-Expose-Headers: X-TTFA-Ms, X-RTF, X-Tokens, X-Duration\r\n"

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
