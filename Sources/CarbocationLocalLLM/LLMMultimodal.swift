import CoreGraphics
import Foundation
import ImageIO

public enum LLMInputModality: String, Codable, Hashable, Sendable {
    case text
    case image
    case audio
}

public enum LLMChatRole: String, Codable, Hashable, Sendable {
    case system
    case user
    case assistant
}

public struct LLMContentLocation: Codable, Hashable, Sendable {
    public var messageIndex: Int
    public var partIndex: Int

    public init(messageIndex: Int, partIndex: Int) {
        self.messageIndex = messageIndex
        self.partIndex = partIndex
    }
}

public enum LLMImageFormat: String, Codable, Hashable, Sendable {
    case png
    case jpeg
    case heic
    case heif

    public var mimeType: String {
        switch self {
        case .png:
            return "image/png"
        case .jpeg:
            return "image/jpeg"
        case .heic:
            return "image/heic"
        case .heif:
            return "image/heif"
        }
    }
}

public enum LLMImageInput: Hashable, Sendable {
    case encoded(data: Data, mimeType: String? = nil)
    case rgb8(width: Int, height: Int, data: Data)
}

public enum LLMAudioFormat: String, Codable, Hashable, Sendable {
    case wav
    case mp3
    case flac

    public var mimeType: String {
        switch self {
        case .wav:
            return "audio/wav"
        case .mp3:
            return "audio/mpeg"
        case .flac:
            return "audio/flac"
        }
    }
}

public enum LLMAudioInput: Hashable, Sendable {
    case encoded(data: Data, mimeType: String? = nil)
    case pcmFloat32Mono(sampleRate: Int, data: Data)
}

public enum LLMContentPart: Hashable, Sendable {
    case text(String)
    case image(LLMImageInput)
    case audio(LLMAudioInput)
}

public struct LLMChatMessage: Hashable, Sendable {
    public var role: LLMChatRole
    public var content: [LLMContentPart]

    public init(role: LLMChatRole, content: [LLMContentPart]) {
        self.role = role
        self.content = content
    }

    public init(role: LLMChatRole, text: String) {
        self.init(role: role, content: [.text(text)])
    }

    public var inputModalities: Set<LLMInputModality> {
        Set(content.map { part in
            switch part {
            case .text:
                return .text
            case .image:
                return .image
            case .audio:
                return .audio
            }
        })
    }

    public var containsImage: Bool {
        inputModalities.contains(.image)
    }

    public var containsAudio: Bool {
        inputModalities.contains(.audio)
    }

    public static func inputModalities(in messages: [LLMChatMessage]) -> Set<LLMInputModality> {
        messages.reduce(into: Set<LLMInputModality>()) { result, message in
            result.formUnion(message.inputModalities)
        }
    }

    public static func firstImageLocation(in messages: [LLMChatMessage]) -> LLMContentLocation? {
        for (messageIndex, message) in messages.enumerated() {
            for (partIndex, part) in message.content.enumerated() {
                if case .image = part {
                    return LLMContentLocation(messageIndex: messageIndex, partIndex: partIndex)
                }
            }
        }
        return nil
    }

    public static func firstAudioLocation(in messages: [LLMChatMessage]) -> LLMContentLocation? {
        firstLocation(of: .audio, in: messages)
    }

    public static func firstNonTextLocation(
        in messages: [LLMChatMessage]
    ) -> (modality: LLMInputModality, location: LLMContentLocation)? {
        for (messageIndex, message) in messages.enumerated() {
            for (partIndex, part) in message.content.enumerated() {
                switch part {
                case .text:
                    continue
                case .image:
                    return (.image, LLMContentLocation(messageIndex: messageIndex, partIndex: partIndex))
                case .audio:
                    return (.audio, LLMContentLocation(messageIndex: messageIndex, partIndex: partIndex))
                }
            }
        }
        return nil
    }

    public static func containsMultimodalInput(in messages: [LLMChatMessage]) -> Bool {
        inputModalities(in: messages).contains { modality in
            modality != .text
        }
    }

    private static func firstLocation(
        of modality: LLMInputModality,
        in messages: [LLMChatMessage]
    ) -> LLMContentLocation? {
        for (messageIndex, message) in messages.enumerated() {
            for (partIndex, part) in message.content.enumerated() {
                switch (modality, part) {
                case (.image, .image), (.audio, .audio):
                    return LLMContentLocation(messageIndex: messageIndex, partIndex: partIndex)
                default:
                    continue
                }
            }
        }
        return nil
    }
}

public struct LLMRGBImage: Hashable, Sendable {
    public var width: Int
    public var height: Int
    public var data: Data

    public init(width: Int, height: Int, data: Data) {
        self.width = width
        self.height = height
        self.data = data
    }
}

public enum LLMChatTextRenderer {
    public static func textOnlySystemAndPrompt(
        from messages: [LLMChatMessage]
    ) throws -> (system: String, prompt: String) {
        if let unsupported = LLMChatMessage.firstNonTextLocation(in: messages) {
            throw LLMEngineError.unsupportedInputModality(
                unsupported.modality,
                location: unsupported.location
            )
        }

        let systemParts = messages
            .filter { $0.role == .system }
            .map(textContent)
            .filter { !$0.isEmpty }

        let promptParts = messages
            .filter { $0.role != .system }
            .map { message -> String in
                let text = textContent(for: message)
                guard !text.isEmpty else { return "" }
                switch message.role {
                case .user:
                    return text
                case .assistant:
                    return "Assistant:\n\(text)"
                case .system:
                    return text
                }
            }
            .filter { !$0.isEmpty }

        return (
            system: systemParts.joined(separator: "\n\n"),
            prompt: promptParts.joined(separator: "\n\n")
        )
    }

    public static func textContent(for message: LLMChatMessage) -> String {
        message.content.compactMap { part in
            if case .text(let text) = part {
                return text
            }
            return nil
        }
        .joined(separator: "")
    }
}

public extension LLMAudioInput {
    static let maximumDuration: TimeInterval = 30

    func encodedFormat(location: LLMContentLocation? = nil) throws -> LLMAudioFormat? {
        guard case .encoded(let data, let mimeType) = self else {
            return nil
        }
        return try Self.encodedFormat(data: data, mimeType: mimeType, location: location)
    }

    static func sniffEncodedFormat(_ data: Data) -> LLMAudioFormat? {
        if data.count >= 12,
           data.starts(with: [0x52, 0x49, 0x46, 0x46]),
           data[data.index(data.startIndex, offsetBy: 8)] == 0x57,
           data[data.index(data.startIndex, offsetBy: 9)] == 0x41,
           data[data.index(data.startIndex, offsetBy: 10)] == 0x56,
           data[data.index(data.startIndex, offsetBy: 11)] == 0x45 {
            return .wav
        }
        if data.starts(with: [0x66, 0x4C, 0x61, 0x43]) {
            return .flac
        }
        if data.starts(with: [0x49, 0x44, 0x33]) {
            return .mp3
        }
        if data.count >= 2 {
            let first = data[data.startIndex]
            let second = data[data.index(after: data.startIndex)]
            if first == 0xFF, (second & 0xE0) == 0xE0 {
                return .mp3
            }
        }
        return nil
    }

    static func audioFormat(forMIMEType mimeType: String) -> LLMAudioFormat? {
        let normalized = mimeType
            .split(separator: ";", maxSplits: 1)
            .first
            .map(String.init)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()

        switch normalized {
        case "audio/wav", "audio/wave", "audio/x-wav", "audio/vnd.wave":
            return .wav
        case "audio/mpeg", "audio/mp3":
            return .mp3
        case "audio/flac", "audio/x-flac":
            return .flac
        default:
            return nil
        }
    }

    static func encodedFormat(
        data: Data,
        mimeType: String?,
        location: LLMContentLocation?
    ) throws -> LLMAudioFormat {
        guard !data.isEmpty else {
            throw LLMEngineError.invalidAudioData("Encoded audio data is empty.", location: location)
        }

        let detected = sniffEncodedFormat(data)
        if let mimeType {
            guard let declared = audioFormat(forMIMEType: mimeType) else {
                throw LLMEngineError.unsupportedAudioFormat(mimeType, location: location)
            }
            if let detected, declared != detected {
                throw LLMEngineError.audioMIMEMismatch(
                    declared: declared.mimeType,
                    detected: detected.mimeType,
                    location: location
                )
            }
            return declared
        }

        guard let detected else {
            throw LLMEngineError.unsupportedAudioFormat("unknown", location: location)
        }
        return detected
    }

    func validatedPCMFloat32Mono(
        expectedSampleRate: Int,
        location: LLMContentLocation? = nil
    ) throws -> [Float] {
        guard case .pcmFloat32Mono(let sampleRate, let data) = self else {
            throw LLMEngineError.invalidAudioData("Expected raw Float32 PCM audio.", location: location)
        }
        guard sampleRate > 0 else {
            throw LLMEngineError.invalidAudioData("PCM audio requires a positive sample rate.", location: location)
        }
        guard sampleRate == expectedSampleRate else {
            throw LLMEngineError.audioSampleRateMismatch(
                LLMAudioSampleRateMismatch(expected: expectedSampleRate, actual: sampleRate),
                location: location
            )
        }
        guard !data.isEmpty else {
            throw LLMEngineError.invalidAudioData("PCM audio data is empty.", location: location)
        }
        guard data.count % MemoryLayout<Float>.size == 0 else {
            throw LLMEngineError.invalidAudioData(
                "PCM Float32 data byte count must be divisible by 4.",
                location: location
            )
        }

        let samples = data.withUnsafeBytes { rawBuffer -> [Float] in
            let bytes = rawBuffer.bindMemory(to: UInt8.self)
            var values: [Float] = []
            values.reserveCapacity(data.count / MemoryLayout<Float>.size)
            var index = 0
            while index + 3 < bytes.count {
                let bitPattern = UInt32(bytes[index])
                    | (UInt32(bytes[index + 1]) << 8)
                    | (UInt32(bytes[index + 2]) << 16)
                    | (UInt32(bytes[index + 3]) << 24)
                values.append(Float(bitPattern: bitPattern))
                index += 4
            }
            return values
        }

        try Self.validateSamples(samples, sampleRate: sampleRate, location: location)
        return samples
    }

    static func validateSamples(
        _ samples: [Float],
        sampleRate: Int,
        location: LLMContentLocation?
    ) throws {
        guard !samples.isEmpty else {
            throw LLMEngineError.invalidAudioData("PCM audio contains no samples.", location: location)
        }
        for sample in samples {
            guard sample.isFinite, sample >= -1, sample <= 1 else {
                throw LLMEngineError.invalidAudioData(
                    "PCM Float32 samples must be finite and normalized to [-1, 1].",
                    location: location
                )
            }
        }
        let duration = TimeInterval(samples.count) / TimeInterval(sampleRate)
        guard duration <= maximumDuration else {
            throw LLMEngineError.audioDurationExceeded(
                LLMAudioDurationLimit(maxSeconds: maximumDuration, actualSeconds: duration),
                location: location
            )
        }
    }
}

public extension LLMImageInput {
    func normalizedRGB8(location: LLMContentLocation? = nil) throws -> LLMRGBImage {
        switch self {
        case .rgb8(let width, let height, let data):
            return try Self.validatedRGB8(width: width, height: height, data: data, location: location)
        case .encoded(let data, let mimeType):
            return try Self.decodedRGB8(data: data, mimeType: mimeType, location: location)
        }
    }

    static func sniffEncodedFormat(_ data: Data) -> LLMImageFormat? {
        if data.starts(with: [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
            return .png
        }
        if data.count >= 3,
           data[data.startIndex] == 0xFF,
           data[data.index(after: data.startIndex)] == 0xD8,
           data[data.index(data.startIndex, offsetBy: 2)] == 0xFF {
            return .jpeg
        }
        guard data.count >= 12 else {
            return nil
        }
        let bytes = [UInt8](data.prefix(64))
        guard String(bytes: bytes[4..<8], encoding: .ascii) == "ftyp" else {
            return nil
        }
        var brands: [String] = []
        var index = 8
        while index + 4 <= bytes.count {
            if let brand = String(bytes: bytes[index..<(index + 4)], encoding: .ascii) {
                brands.append(brand)
            }
            index += 4
        }
        if brands.contains(where: { ["heic", "heix", "hevc", "hevx"].contains($0) }) {
            return .heic
        }
        if brands.contains(where: { ["mif1", "msf1"].contains($0) }) {
            return .heif
        }
        return nil
    }

    static func imageFormat(forMIMEType mimeType: String) -> LLMImageFormat? {
        let normalized = mimeType
            .split(separator: ";", maxSplits: 1)
            .first
            .map(String.init)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()

        switch normalized {
        case "image/png":
            return .png
        case "image/jpeg", "image/jpg":
            return .jpeg
        case "image/heic":
            return .heic
        case "image/heif":
            return .heif
        default:
            return nil
        }
    }

    private static func validatedRGB8(
        width: Int,
        height: Int,
        data: Data,
        location: LLMContentLocation?
    ) throws -> LLMRGBImage {
        guard width > 0, height > 0 else {
            throw LLMEngineError.invalidImageData(
                "RGB8 images require positive dimensions.",
                location: location
            )
        }
        guard width <= Int.max / height / 3 else {
            throw LLMEngineError.invalidImageData(
                "RGB8 image dimensions overflow the expected byte count.",
                location: location
            )
        }
        let expectedCount = width * height * 3
        guard data.count == expectedCount else {
            throw LLMEngineError.invalidImageData(
                "RGB8 data count was \(data.count), expected \(expectedCount) for \(width)x\(height).",
                location: location
            )
        }
        return LLMRGBImage(width: width, height: height, data: data)
    }

    private static func decodedRGB8(
        data: Data,
        mimeType: String?,
        location: LLMContentLocation?
    ) throws -> LLMRGBImage {
        guard !data.isEmpty else {
            throw LLMEngineError.invalidImageData("Encoded image data is empty.", location: location)
        }

        let sniffedFormat = sniffEncodedFormat(data)
        let declaredFormat: LLMImageFormat?
        if let mimeType {
            guard let format = imageFormat(forMIMEType: mimeType) else {
                throw LLMEngineError.unsupportedImageFormat(mimeType, location: location)
            }
            declaredFormat = format
            if let sniffedFormat, sniffedFormat != format {
                throw LLMEngineError.imageMIMEMismatch(
                    declared: mimeType,
                    detected: sniffedFormat.mimeType,
                    location: location
                )
            }
        } else {
            declaredFormat = nil
        }

        let effectiveFormat = declaredFormat ?? sniffedFormat
        guard effectiveFormat != nil else {
            throw LLMEngineError.unsupportedImageFormat("unknown", location: location)
        }

        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else {
            throw LLMEngineError.imageDecodeFailed("ImageIO could not create an image source.", location: location)
        }

        let pixelDimensions = Self.pixelDimensions(source: source)
        let maxPixelSize = max(pixelDimensions.width, pixelDimensions.height, 1)
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixelSize
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
            ?? CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw LLMEngineError.imageDecodeFailed("ImageIO could not decode the image.", location: location)
        }

        let width = image.width
        let height = image.height
        guard width > 0, height > 0 else {
            throw LLMEngineError.invalidImageData("Decoded image has zero size.", location: location)
        }
        guard width <= Int.max / height / 4, width <= Int.max / height / 3 else {
            throw LLMEngineError.invalidImageData("Decoded image dimensions are too large.", location: location)
        }

        let rgbaBytesPerRow = width * 4
        var rgba = Data(count: rgbaBytesPerRow * height)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
        let drewImage = rgba.withUnsafeMutableBytes { rawBuffer -> Bool in
            guard let baseAddress = rawBuffer.baseAddress,
                  let context = CGContext(
                    data: baseAddress,
                    width: width,
                    height: height,
                    bitsPerComponent: 8,
                    bytesPerRow: rgbaBytesPerRow,
                    space: colorSpace,
                    bitmapInfo: bitmapInfo
                  )
            else {
                return false
            }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drewImage else {
            throw LLMEngineError.imageDecodeFailed("Could not normalize decoded image to RGB8.", location: location)
        }

        var rgb = Data(count: width * height * 3)
        rgba.withUnsafeBytes { rgbaBuffer in
            rgb.withUnsafeMutableBytes { rgbBuffer in
                guard let rgbaBase = rgbaBuffer.bindMemory(to: UInt8.self).baseAddress,
                      let rgbBase = rgbBuffer.bindMemory(to: UInt8.self).baseAddress else {
                    return
                }
                let pixelCount = width * height
                for pixelIndex in 0..<pixelCount {
                    let rgbaIndex = pixelIndex * 4
                    let rgbIndex = pixelIndex * 3
                    rgbBase[rgbIndex] = rgbaBase[rgbaIndex]
                    rgbBase[rgbIndex + 1] = rgbaBase[rgbaIndex + 1]
                    rgbBase[rgbIndex + 2] = rgbaBase[rgbaIndex + 2]
                }
            }
        }

        return LLMRGBImage(width: width, height: height, data: rgb)
    }

    private static func pixelDimensions(source: CGImageSource) -> (width: Int, height: Int) {
        guard let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any] else {
            return (0, 0)
        }
        let width = (properties[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue ?? 0
        let height = (properties[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue ?? 0
        return (width, height)
    }
}
