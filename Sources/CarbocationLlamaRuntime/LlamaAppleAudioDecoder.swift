import AVFoundation
import CarbocationLocalLLM
import Foundation

enum LlamaAppleAudioDecoder {
    static func decodeFloat32Mono(
        data: Data,
        format: LLMAudioFormat,
        sampleRate: Int,
        location: LLMContentLocation
    ) throws -> [Float] {
        guard sampleRate > 0 else {
            throw LLMEngineError.invalidAudioData(
                "Audio decode requires a positive target sample rate.",
                location: location
            )
        }

        let fileExtension = Self.fileExtension(for: format)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("CarbocationLocalLLM-\(UUID().uuidString)")
            .appendingPathExtension(fileExtension)

        do {
            try data.write(to: url, options: [.atomic])
        } catch {
            throw LLMEngineError.audioTokenizationFailed(
                "Could not stage encoded audio for AVFoundation decode: \(error.localizedDescription)",
                location: location
            )
        }
        defer {
            try? FileManager.default.removeItem(at: url)
        }

        do {
            return try decodeFile(at: url, sampleRate: sampleRate, location: location)
        } catch let error as LLMEngineError {
            throw error
        } catch {
            throw LLMEngineError.audioTokenizationFailed(
                "AVFoundation could not decode \(format.mimeType): \(error.localizedDescription)",
                location: location
            )
        }
    }

    private static func fileExtension(for format: LLMAudioFormat) -> String {
        switch format {
        case .m4a:
            return "m4a"
        case .aac:
            return "aac"
        case .wav:
            return "wav"
        case .mp3:
            return "mp3"
        case .flac:
            return "flac"
        }
    }

    private static func decodeFile(
        at url: URL,
        sampleRate: Int,
        location: LLMContentLocation
    ) throws -> [Float] {
        let sourceFile = try AVAudioFile(forReading: url)
        let sourceFormat = sourceFile.processingFormat
        guard sourceFormat.sampleRate > 0 else {
            throw LLMEngineError.invalidAudioData(
                "Encoded audio has an invalid sample rate.",
                location: location
            )
        }
        guard sourceFile.length > 0 else {
            throw LLMEngineError.invalidAudioData(
                "Encoded audio decoded to no samples.",
                location: location
            )
        }

        let sourceDuration = TimeInterval(sourceFile.length) / sourceFormat.sampleRate
        guard sourceDuration <= LLMAudioInput.maximumDuration else {
            throw LLMEngineError.audioDurationExceeded(
                LLMAudioDurationLimit(
                    maxSeconds: LLMAudioInput.maximumDuration,
                    actualSeconds: sourceDuration
                ),
                location: location
            )
        }

        guard sourceFile.length <= Int64(UInt32.max) else {
            throw LLMEngineError.invalidAudioData(
                "Decoded audio frame count exceeds AVFoundation buffer limits.",
                location: location
            )
        }
        guard let inputBuffer = AVAudioPCMBuffer(
            pcmFormat: sourceFormat,
            frameCapacity: AVAudioFrameCount(sourceFile.length)
        ) else {
            throw LLMEngineError.audioTokenizationFailed(
                "AVFoundation could not allocate an input audio buffer.",
                location: location
            )
        }
        try sourceFile.read(into: inputBuffer)
        guard inputBuffer.frameLength > 0 else {
            throw LLMEngineError.invalidAudioData(
                "Encoded audio decoded to no samples.",
                location: location
            )
        }

        guard let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: 1,
            interleaved: false
        ) else {
            throw LLMEngineError.audioTokenizationFailed(
                "AVFoundation could not create a normalized audio format.",
                location: location
            )
        }
        guard let converter = AVAudioConverter(from: sourceFormat, to: outputFormat) else {
            throw LLMEngineError.audioTokenizationFailed(
                "AVFoundation could not create an audio converter.",
                location: location
            )
        }

        let estimatedFrames = ceil(
            Double(inputBuffer.frameLength) * outputFormat.sampleRate / sourceFormat.sampleRate
        )
        guard estimatedFrames > 0, estimatedFrames <= Double(UInt32.max - 4_096) else {
            throw LLMEngineError.invalidAudioData(
                "Decoded audio frame count exceeds AVFoundation buffer limits.",
                location: location
            )
        }

        let outputCapacity = max(AVAudioFrameCount(estimatedFrames) + 4_096, 4_096)
        var didProvideInput = false
        var samples: [Float] = []
        samples.reserveCapacity(Int(estimatedFrames))

        while true {
            guard let outputBuffer = AVAudioPCMBuffer(
                pcmFormat: outputFormat,
                frameCapacity: outputCapacity
            ) else {
                throw LLMEngineError.audioTokenizationFailed(
                    "AVFoundation could not allocate an output audio buffer.",
                    location: location
                )
            }

            var conversionError: NSError?
            let status = converter.convert(to: outputBuffer, error: &conversionError) { _, outStatus in
                if didProvideInput {
                    outStatus.pointee = .endOfStream
                    return nil
                }
                didProvideInput = true
                outStatus.pointee = .haveData
                return inputBuffer
            }

            if let conversionError {
                throw conversionError
            }

            if outputBuffer.frameLength > 0 {
                guard let channelData = outputBuffer.floatChannelData else {
                    throw LLMEngineError.audioTokenizationFailed(
                        "AVFoundation did not return Float32 audio samples.",
                        location: location
                    )
                }
                samples.append(
                    contentsOf: UnsafeBufferPointer(
                        start: channelData[0],
                        count: Int(outputBuffer.frameLength)
                    )
                )
            }

            switch status {
            case .haveData, .inputRanDry:
                continue
            case .endOfStream:
                try LLMAudioInput.validateSamples(samples, sampleRate: sampleRate, location: location)
                return samples
            case .error:
                throw LLMEngineError.audioTokenizationFailed(
                    "AVFoundation audio conversion failed.",
                    location: location
                )
            @unknown default:
                throw LLMEngineError.audioTokenizationFailed(
                    "AVFoundation returned an unknown audio conversion status.",
                    location: location
                )
            }
        }
    }
}
