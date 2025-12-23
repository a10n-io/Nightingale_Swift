import Foundation
import MLX
import MLXFFT
import Accelerate

/// Mel Spectrogram extractor matching Python's Chatterbox implementation
/// Used for extracting prompt_feat from reference audio
public class MelSpectrogram {

    // S3Gen mel parameters (from s3gen/utils/mel.py)
    public struct Config {
        public let nFFT: Int
        public let numMels: Int
        public let sampleRate: Int
        public let hopSize: Int
        public let winSize: Int
        public let fmin: Float
        public let fmax: Float
        public let center: Bool

        /// Default config for S3Gen (24kHz mel spectrogram)
        public static let s3gen = Config(
            nFFT: 1920,
            numMels: 80,
            sampleRate: 24000,
            hopSize: 480,
            winSize: 1920,
            fmin: 0,
            fmax: 8000,
            center: false
        )

        /// Config for VoiceEncoder (16kHz, 40 mels)
        public static let voiceEncoder = Config(
            nFFT: 400,
            numMels: 40,
            sampleRate: 16000,
            hopSize: 160,
            winSize: 400,
            fmin: 0,
            fmax: 8000,
            center: true
        )
    }

    private let config: Config
    private let melFilterbank: MLXArray  // [numMels, nFFT/2+1]
    private let window: MLXArray         // [winSize]

    public init(config: Config = .s3gen) {
        self.config = config

        // Create mel filterbank
        self.melFilterbank = Self.createMelFilterbank(
            sampleRate: config.sampleRate,
            nFFT: config.nFFT,
            numMels: config.numMels,
            fmin: config.fmin,
            fmax: config.fmax
        )

        // Create Hann window
        self.window = Self.createHannWindow(size: config.winSize)
    }

    /// Extract mel spectrogram from audio waveform
    /// - Parameter audio: Audio waveform [samples] or [batch, samples]
    /// - Returns: Mel spectrogram [batch, numMels, time] or [numMels, time]
    public func extract(from audio: MLXArray) -> MLXArray {
        var y = audio

        // Ensure 2D: [batch, samples]
        let wasBatched = y.ndim == 2
        if y.ndim == 1 {
            y = y.expandedDimensions(axis: 0)
        }

        // Pad for STFT (reflect padding like Python)
        let padAmount = (config.nFFT - config.hopSize) / 2
        y = padReflect(y, left: padAmount, right: padAmount)

        // Compute STFT magnitude
        let magnitude = stftMagnitude(y)  // [batch, nFFT/2+1, time]

        // Apply mel filterbank using matmul
        // magnitude is [batch, freq, time], we need [batch, time, freq] for matmul
        let magT = magnitude.transposed(0, 2, 1)  // [batch, time, freq]
        let melT = MLX.matmul(magT, melFilterbank.T)  // [batch, time, numMels]
        var mel = melT.transposed(0, 2, 1)  // [batch, numMels, time]

        // Apply dynamic range compression (log)
        mel = dynamicRangeCompression(mel)

        eval(mel)

        if !wasBatched {
            return mel.squeezed(axis: 0)  // [numMels, time]
        }
        return mel  // [batch, numMels, time]
    }

    // MARK: - STFT

    /// Compute STFT magnitude (avoids complex number handling issues)
    private func stftMagnitude(_ y: MLXArray) -> MLXArray {
        let numSamples = y.dim(1)

        // Number of frames
        let numFrames = (numSamples - config.winSize) / config.hopSize + 1

        // Frame the signal
        var frames: [MLXArray] = []
        for i in 0..<numFrames {
            let start = i * config.hopSize
            let end = start + config.winSize
            let frame = y[.ellipsis, start..<end]  // [batch, winSize]
            frames.append(frame)
        }

        // Stack frames: [numFrames, batch, winSize]
        let framed = MLX.stacked(frames, axis: 0)
        // Transpose to [batch, numFrames, winSize]
        let framedT = framed.transposed(1, 0, 2)

        // Apply window
        let windowed = framedT * window  // [batch, numFrames, winSize]

        // FFT - returns complex array
        let fftResult = MLXFFT.rfft(windowed, n: config.nFFT, axis: -1)  // [batch, numFrames, nFFT/2+1] complex

        // Compute magnitude: sqrt(real^2 + imag^2)
        // MLX complex arrays support abs() which computes magnitude
        let magnitude = MLX.abs(fftResult)  // [batch, numFrames, nFFT/2+1]

        // Transpose to [batch, nFFT/2+1, numFrames]
        return magnitude.transposed(0, 2, 1)
    }

    /// Dynamic range compression (log)
    private func dynamicRangeCompression(_ x: MLXArray, c: Float = 1.0, clipVal: Float = 1e-5) -> MLXArray {
        return MLX.log(MLX.maximum(x, MLXArray(clipVal)) * c)
    }

    /// Reflect padding for audio
    private func padReflect(_ x: MLXArray, left: Int, right: Int) -> MLXArray {
        // x is [batch, samples]
        let batch = x.dim(0)
        let samples = x.dim(1)

        // For reflect padding: mirror the edge values
        // Left padding: take x[..., 1:left+1] and reverse it
        // Right padding: take x[..., -right-1:-1] and reverse it

        var leftPad: MLXArray
        if left > 0 {
            let leftSlice = x[.ellipsis, 1..<(left+1)]  // [batch, left]
            leftPad = leftSlice[.ellipsis, .stride(by: -1)]  // reverse
        } else {
            leftPad = MLXArray.zeros([batch, 0])
        }

        var rightPad: MLXArray
        if right > 0 {
            let rightStart = samples - right - 1
            let rightSlice = x[.ellipsis, rightStart..<(samples-1)]  // [batch, right]
            rightPad = rightSlice[.ellipsis, .stride(by: -1)]  // reverse
        } else {
            rightPad = MLXArray.zeros([batch, 0])
        }

        // Concatenate: [leftPad, x, rightPad]
        return MLX.concatenated([leftPad, x, rightPad], axis: 1)
    }

    // MARK: - Mel Filterbank Creation

    /// Create mel filterbank matrix using librosa's formula
    private static func createMelFilterbank(
        sampleRate: Int,
        nFFT: Int,
        numMels: Int,
        fmin: Float,
        fmax: Float
    ) -> MLXArray {
        // Convert Hz to Mel
        func hzToMel(_ hz: Float) -> Float {
            return 2595.0 * log10(1.0 + hz / 700.0)
        }

        // Convert Mel to Hz
        func melToHz(_ mel: Float) -> Float {
            return 700.0 * (pow(10.0, mel / 2595.0) - 1.0)
        }

        let melMin = hzToMel(fmin)
        let melMax = hzToMel(fmax)

        // Create mel points (numMels + 2 points for triangular filters)
        var melPoints: [Float] = []
        for i in 0...(numMels + 1) {
            let mel = melMin + Float(i) * (melMax - melMin) / Float(numMels + 1)
            melPoints.append(melToHz(mel))
        }

        // Convert Hz to FFT bin indices
        let nFreqs = nFFT / 2 + 1
        var binPoints: [Int] = []
        for hz in melPoints {
            let bin = Int(floor(Float(nFFT + 1) * hz / Float(sampleRate)))
            binPoints.append(min(bin, nFreqs - 1))
        }

        // Create filterbank matrix
        var filterbank = [[Float]](repeating: [Float](repeating: 0, count: nFreqs), count: numMels)

        for m in 0..<numMels {
            let left = binPoints[m]
            let center = binPoints[m + 1]
            let right = binPoints[m + 2]

            // Rising slope
            for k in left..<center {
                if center != left {
                    filterbank[m][k] = Float(k - left) / Float(center - left)
                }
            }

            // Falling slope
            for k in center..<right {
                if right != center {
                    filterbank[m][k] = Float(right - k) / Float(right - center)
                }
            }
        }

        // Convert to MLXArray
        let flatData = filterbank.flatMap { $0 }
        return MLXArray(flatData).reshaped([numMels, nFreqs])
    }

    /// Create Hann window
    private static func createHannWindow(size: Int) -> MLXArray {
        var window = [Float](repeating: 0, count: size)
        for i in 0..<size {
            window[i] = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(i) / Float(size)))
        }
        return MLXArray(window)
    }
}
