import CarbocationLocalLLM
import Foundation

#if canImport(FoundationModels)
import FoundationModels
#endif

public enum AppleIntelligenceUnavailableReason: String, Codable, Hashable, Sendable {
    case sdkUnavailable
    case operatingSystemUnavailable
    case deviceNotEligible
    case appleIntelligenceNotEnabled
    case modelNotReady
    case unknown

    public var displayMessage: String {
        switch self {
        case .sdkUnavailable:
            return "This app was built without the Foundation Models SDK."
        case .operatingSystemUnavailable:
            #if os(iOS)
            return "Apple Intelligence model access requires iOS 26 or newer."
            #elseif os(macOS)
            return "Apple Intelligence model access requires macOS 26 or newer."
            #else
            return "Apple Intelligence model access requires a supported OS version."
            #endif
        case .deviceNotEligible:
            return "This device does not support Apple Intelligence."
        case .appleIntelligenceNotEnabled:
            return "Apple Intelligence is not enabled in System Settings."
        case .modelNotReady:
            return "The Apple Intelligence model is not ready yet."
        case .unknown:
            return "Apple Intelligence is not available."
        }
    }
}

public enum AppleIntelligenceAvailability: Hashable, Sendable {
    case available(contextSize: Int)
    case unavailable(AppleIntelligenceUnavailableReason)

    public var isAvailable: Bool {
        if case .available = self { return true }
        return false
    }

    /// Host apps can use this to hide the system model option until it can run.
    public var shouldOfferModelOption: Bool {
        isAvailable
    }

    public var contextSize: Int {
        if case .available(let contextSize) = self { return contextSize }
        return 0
    }

    public var unavailableReason: AppleIntelligenceUnavailableReason? {
        if case .unavailable(let reason) = self { return reason }
        return nil
    }

    public var displayMessage: String {
        switch self {
        case .available(let contextSize):
            return "Apple Intelligence is available with a \(contextSize.formatted())-token context."
        case .unavailable(let reason):
            return reason.displayMessage
        }
    }
}

public enum AppleIntelligenceUnsupportedFeature: String, Codable, Hashable, Sendable {
    /// llama.cpp GBNF grammars are not accepted by Foundation Models text responses.
    case grammar
}

public enum AppleIntelligenceUnsupportedFeatureBehavior: String, Codable, Hashable, Sendable {
    /// Keep going and rely on prompting plus response post-processing.
    case ignore
    /// Throw before generation when an option cannot be represented.
    case fail
}

public struct AppleIntelligenceEngineConfiguration: Hashable, Sendable {
    public var unsupportedFeatureBehavior: AppleIntelligenceUnsupportedFeatureBehavior

    public init(
        unsupportedFeatureBehavior: AppleIntelligenceUnsupportedFeatureBehavior = .ignore
    ) {
        self.unsupportedFeatureBehavior = unsupportedFeatureBehavior
    }
}

public enum AppleIntelligenceEngineError: Error, LocalizedError, Sendable {
    case unavailable(AppleIntelligenceAvailability)
    case unsupportedFeatures(Set<AppleIntelligenceUnsupportedFeature>)

    public var errorDescription: String? {
        switch self {
        case .unavailable(let availability):
            return availability.displayMessage
        case .unsupportedFeatures(let features):
            let labels = features
                .map(\.rawValue)
                .sorted()
                .joined(separator: ", ")
            return "Apple Intelligence does not support these generation options: \(labels)."
        }
    }
}

public enum AppleIntelligenceSamplingPolicy: Hashable, Sendable {
    case systemDefault
    case greedy
    case randomTopK(Int, seed: UInt64?)
    case randomProbabilityThreshold(Double, seed: UInt64?)
}

public struct AppleIntelligenceResolvedOptions: Hashable, Sendable {
    public var sampling: AppleIntelligenceSamplingPolicy
    public var temperature: Double?
    public var maximumResponseTokens: Int?
    public var unsupportedFeatures: Set<AppleIntelligenceUnsupportedFeature>

    public init(
        sampling: AppleIntelligenceSamplingPolicy,
        temperature: Double?,
        maximumResponseTokens: Int?,
        unsupportedFeatures: Set<AppleIntelligenceUnsupportedFeature>
    ) {
        self.sampling = sampling
        self.temperature = temperature
        self.maximumResponseTokens = maximumResponseTokens
        self.unsupportedFeatures = unsupportedFeatures
    }
}

public enum AppleIntelligenceOptionsMapper {
    public static func resolve(
        _ options: CarbocationLocalLLM.GenerationOptions
    ) -> AppleIntelligenceResolvedOptions {
        let seed = options.seed.map(UInt64.init)
        let sampling: AppleIntelligenceSamplingPolicy
        let temperature: Double?

        if let requestedTemperature = options.temperature, requestedTemperature <= 0 {
            sampling = .greedy
            temperature = nil
        } else if let topK = options.topK, topK > 0 {
            sampling = .randomTopK(topK, seed: seed)
            temperature = options.temperature
        } else if let topP = normalizedProbabilityThreshold(options.topP) {
            sampling = .randomProbabilityThreshold(topP, seed: seed)
            temperature = options.temperature
        } else {
            sampling = .systemDefault
            temperature = options.temperature
        }

        var unsupportedFeatures = Set<AppleIntelligenceUnsupportedFeature>()
        if options.grammar != nil {
            unsupportedFeatures.insert(.grammar)
        }

        return AppleIntelligenceResolvedOptions(
            sampling: sampling,
            temperature: temperature,
            maximumResponseTokens: normalizedMaximumResponseTokens(options.maxOutputTokens),
            unsupportedFeatures: unsupportedFeatures
        )
    }

    private static func normalizedProbabilityThreshold(_ value: Double?) -> Double? {
        guard let value, value > 0 else { return nil }
        return min(1, value)
    }

    private static func normalizedMaximumResponseTokens(_ value: Int?) -> Int? {
        guard let value, value > 0 else { return nil }
        return value
    }
}

public struct AppleIntelligenceProcessedResponse: Hashable, Sendable {
    public var text: String
    public var stopReason: String?

    public init(text: String, stopReason: String?) {
        self.text = text
        self.stopReason = stopReason
    }
}

public enum AppleIntelligenceResponsePostProcessor {
    public static func process(
        _ text: String,
        options: CarbocationLocalLLM.GenerationOptions
    ) -> AppleIntelligenceProcessedResponse {
        var boundaryIndex: String.Index?
        var processedText: String?
        var stopReason: String?

        if let range = firstStopSequenceRange(in: text, stopSequences: options.stopSequences) {
            boundaryIndex = range.lowerBound
            processedText = String(text[..<range.lowerBound])
            stopReason = "stop-sequence"
        }

        if options.stopAtBalancedJSON,
           let jsonRange = balancedJSONValueRange(in: text),
           boundaryIndex.map({ jsonRange.upperBound < $0 }) ?? true {
            boundaryIndex = jsonRange.upperBound
            processedText = String(text[jsonRange])
            stopReason = "json-complete"
        }

        guard boundaryIndex != nil, let processedText else {
            return AppleIntelligenceProcessedResponse(text: text, stopReason: nil)
        }
        return AppleIntelligenceProcessedResponse(text: processedText, stopReason: stopReason)
    }

    public static func firstStopSequenceRange(
        in text: String,
        stopSequences: [String]
    ) -> Range<String.Index>? {
        stopSequences
            .filter { !$0.isEmpty }
            .compactMap { text.range(of: $0) }
            .min { $0.lowerBound < $1.lowerBound }
    }

    public static func endIndexAfterBalancedJSONObject(in text: String) -> String.Index? {
        balancedJSONValueRange(in: text)?.upperBound
    }

    public static func balancedJSONValueRange(in text: String) -> Range<String.Index>? {
        var startIndex: String.Index?
        var expectedClosers: [Character] = []
        var inString = false
        var escaped = false

        var index = text.startIndex
        while index < text.endIndex {
            let character = text[index]

            if startIndex == nil {
                if character == "{" {
                    startIndex = index
                    expectedClosers = ["}"]
                } else if character == "[" {
                    startIndex = index
                    expectedClosers = ["]"]
                }
                index = text.index(after: index)
                continue
            }

            if escaped {
                escaped = false
                index = text.index(after: index)
                continue
            }

            if character == "\\" && inString {
                escaped = true
                index = text.index(after: index)
                continue
            }

            if character == "\"" {
                inString.toggle()
                index = text.index(after: index)
                continue
            }

            if !inString {
                if character == "{" {
                    expectedClosers.append("}")
                } else if character == "[" {
                    expectedClosers.append("]")
                } else if character == "}" {
                    guard expectedClosers.last == "}" else {
                        return nil
                    }
                    expectedClosers.removeLast()
                    if expectedClosers.isEmpty, let startIndex {
                        return startIndex..<text.index(after: index)
                    }
                } else if character == "]" {
                    guard expectedClosers.last == "]" else {
                        return nil
                    }
                    expectedClosers.removeLast()
                    if expectedClosers.isEmpty, let startIndex {
                        return startIndex..<text.index(after: index)
                    }
                }
            }

            index = text.index(after: index)
        }

        return nil
    }
}

public actor AppleIntelligenceEngine: LLMEngine {
    public static let shared = AppleIntelligenceEngine()
    public static let systemModelID = LLMSystemModelID.appleIntelligence
    public static let displayName = "Apple Intelligence"

    private let configuration: AppleIntelligenceEngineConfiguration

    public init(configuration: AppleIntelligenceEngineConfiguration = AppleIntelligenceEngineConfiguration()) {
        self.configuration = configuration
    }

    public nonisolated static var isBuiltWithFoundationModelsSDK: Bool {
        #if canImport(FoundationModels)
        true
        #else
        false
        #endif
    }

    public nonisolated static func availability() -> AppleIntelligenceAvailability {
        #if canImport(FoundationModels)
        guard #available(iOS 26.0, macOS 26.0, *) else {
            return .unavailable(.operatingSystemUnavailable)
        }
        return foundationModelsAvailability()
        #else
        return .unavailable(.sdkUnavailable)
        #endif
    }

    public nonisolated static func systemModelOption() -> LLMSystemModelOption? {
        let availability = availability()
        guard availability.shouldOfferModelOption else { return nil }
        return LLMSystemModelOption(
            selection: .system(systemModelID),
            displayName: displayName,
            subtitle: "Built-in on-device Apple Intelligence model",
            contextLength: availability.contextSize,
            systemImageName: "sparkles"
        )
    }

    public func currentModelID() -> UUID? {
        nil
    }

    public func currentContextSize() -> Int {
        Self.availability().contextSize
    }

    public func generate(
        system: String,
        prompt: String,
        options: CarbocationLocalLLM.GenerationOptions,
        onEvent: @Sendable (LLMStreamEvent) -> Void
    ) async throws -> String {
        let availability = Self.availability()
        guard availability.isAvailable else {
            throw AppleIntelligenceEngineError.unavailable(availability)
        }

        let resolvedOptions = AppleIntelligenceOptionsMapper.resolve(options)
        if configuration.unsupportedFeatureBehavior == .fail,
           !resolvedOptions.unsupportedFeatures.isEmpty {
            throw AppleIntelligenceEngineError.unsupportedFeatures(resolvedOptions.unsupportedFeatures)
        }

        #if canImport(FoundationModels)
        guard #available(iOS 26.0, macOS 26.0, *) else {
            throw AppleIntelligenceEngineError.unavailable(.unavailable(.operatingSystemUnavailable))
        }
        return try await generateWithFoundationModels(
            system: system,
            prompt: prompt,
            options: options,
            resolvedOptions: resolvedOptions,
            onEvent: onEvent
        )
        #else
        throw AppleIntelligenceEngineError.unavailable(.unavailable(.sdkUnavailable))
        #endif
    }
}

public actor AppleIntelligenceSession {
    private let system: String
    private let configuration: AppleIntelligenceEngineConfiguration
    private var foundationSessionStorage: Any?

    public init(
        system: String = "",
        configuration: AppleIntelligenceEngineConfiguration = AppleIntelligenceEngineConfiguration()
    ) {
        self.system = system
        self.configuration = configuration
    }

    public func currentContextSize() -> Int {
        AppleIntelligenceEngine.availability().contextSize
    }

    public func generate(
        prompt: String,
        options: CarbocationLocalLLM.GenerationOptions,
        onEvent: @Sendable (LLMStreamEvent) -> Void
    ) async throws -> String {
        let availability = AppleIntelligenceEngine.availability()
        guard availability.isAvailable else {
            throw AppleIntelligenceEngineError.unavailable(availability)
        }

        let resolvedOptions = AppleIntelligenceOptionsMapper.resolve(options)
        if configuration.unsupportedFeatureBehavior == .fail,
           !resolvedOptions.unsupportedFeatures.isEmpty {
            throw AppleIntelligenceEngineError.unsupportedFeatures(resolvedOptions.unsupportedFeatures)
        }

        #if canImport(FoundationModels)
        guard #available(iOS 26.0, macOS 26.0, *) else {
            throw AppleIntelligenceEngineError.unavailable(.unavailable(.operatingSystemUnavailable))
        }
        return try await generateWithFoundationModels(
            prompt: prompt,
            options: options,
            resolvedOptions: resolvedOptions,
            onEvent: onEvent
        )
        #else
        throw AppleIntelligenceEngineError.unavailable(.unavailable(.sdkUnavailable))
        #endif
    }
}

#if canImport(FoundationModels)
extension AppleIntelligenceEngine {
    @available(iOS 26.0, macOS 26.0, *)
    private nonisolated static func foundationModelsAvailability() -> AppleIntelligenceAvailability {
        let model = SystemLanguageModel.default
        switch model.availability {
        case .available:
            return .available(contextSize: model.contextSize)
        case .unavailable(.deviceNotEligible):
            return .unavailable(.deviceNotEligible)
        case .unavailable(.appleIntelligenceNotEnabled):
            return .unavailable(.appleIntelligenceNotEnabled)
        case .unavailable(.modelNotReady):
            return .unavailable(.modelNotReady)
        case .unavailable:
            return .unavailable(.unknown)
        }
    }

    @available(iOS 26.0, macOS 26.0, *)
    private func generateWithFoundationModels(
        system: String,
        prompt: String,
        options: CarbocationLocalLLM.GenerationOptions,
        resolvedOptions: AppleIntelligenceResolvedOptions,
        onEvent: @Sendable (LLMStreamEvent) -> Void
    ) async throws -> String {
        onEvent(.requestSent)
        let startedAt = Date()

        let instructions = Self.instructions(system: system, options: options)
        let session: LanguageModelSession
        if instructions.isEmpty {
            session = LanguageModelSession()
        } else {
            session = LanguageModelSession(instructions: instructions)
        }

        let response = try await session.respond(
            to: prompt,
            options: Self.foundationModelsOptions(from: resolvedOptions)
        )
        let processed = AppleIntelligenceResponsePostProcessor.process(response.content, options: options)
        let result = processed.text
        let duration = Date().timeIntervalSince(startedAt)
        let promptTokens = TokenEstimator.estimate(text: system) + TokenEstimator.estimate(text: prompt)
        let generatedTokens = TokenEstimator.estimate(text: result)
        let stopReason = processed.stopReason ?? stopReasonForCompletedResponse(
            result: result,
            generatedTokens: generatedTokens,
            options: options
        )

        onEvent(.firstByteReceived(after: duration))
        if !result.isEmpty {
            onEvent(.tokenChunk(preview: String(result.suffix(60)), bytesSoFar: result.utf8.count))
        }
        onEvent(.generationStats(
            promptTokens: promptTokens,
            generatedTokens: generatedTokens,
            stopReason: stopReason,
            templateMode: .unavailable
        ))
        onEvent(.done(totalBytes: result.utf8.count, duration: duration))

        return result
    }

    @available(iOS 26.0, macOS 26.0, *)
    private nonisolated static func foundationModelsOptions(
        from options: AppleIntelligenceResolvedOptions
    ) -> FoundationModels.GenerationOptions {
        FoundationModels.GenerationOptions(
            sampling: foundationModelsSampling(from: options.sampling),
            temperature: options.temperature,
            maximumResponseTokens: options.maximumResponseTokens
        )
    }

    @available(iOS 26.0, macOS 26.0, *)
    private nonisolated static func foundationModelsSampling(
        from policy: AppleIntelligenceSamplingPolicy
    ) -> FoundationModels.GenerationOptions.SamplingMode? {
        switch policy {
        case .systemDefault:
            return nil
        case .greedy:
            return .greedy
        case .randomTopK(let top, let seed):
            return .random(top: top, seed: seed)
        case .randomProbabilityThreshold(let probabilityThreshold, let seed):
            return .random(probabilityThreshold: probabilityThreshold, seed: seed)
        }
    }

    private nonisolated static func instructions(
        system: String,
        options: CarbocationLocalLLM.GenerationOptions
    ) -> String {
        var parts: [String] = []
        let trimmedSystem = system.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedSystem.isEmpty {
            parts.append(trimmedSystem)
        }
        if options.grammar != nil || options.stopAtBalancedJSON {
            parts.append("Follow the requested output format exactly. If the prompt asks for JSON, return only valid JSON with no surrounding prose.")
        }
        return parts.joined(separator: "\n\n")
    }

    private nonisolated func stopReasonForCompletedResponse(
        result: String,
        generatedTokens: Int,
        options: CarbocationLocalLLM.GenerationOptions
    ) -> String {
        if let maxOutputTokens = options.maxOutputTokens,
           maxOutputTokens > 0,
           generatedTokens >= maxOutputTokens {
            return "max-tokens"
        }
        return "complete"
    }
}

extension AppleIntelligenceSession {
    @available(iOS 26.0, macOS 26.0, *)
    private func generateWithFoundationModels(
        prompt: String,
        options: CarbocationLocalLLM.GenerationOptions,
        resolvedOptions: AppleIntelligenceResolvedOptions,
        onEvent: @Sendable (LLMStreamEvent) -> Void
    ) async throws -> String {
        onEvent(.requestSent)
        let startedAt = Date()
        let session = foundationModelsSession(options: options)

        let response = try await session.respond(
            to: prompt,
            options: Self.foundationModelsOptions(from: resolvedOptions)
        )
        let processed = AppleIntelligenceResponsePostProcessor.process(response.content, options: options)
        let result = processed.text
        let duration = Date().timeIntervalSince(startedAt)
        let promptTokens = TokenEstimator.estimate(text: system) + TokenEstimator.estimate(text: prompt)
        let generatedTokens = TokenEstimator.estimate(text: result)
        let stopReason = processed.stopReason ?? Self.stopReasonForCompletedResponse(
            result: result,
            generatedTokens: generatedTokens,
            options: options
        )

        onEvent(.firstByteReceived(after: duration))
        if !result.isEmpty {
            onEvent(.tokenChunk(preview: String(result.suffix(60)), bytesSoFar: result.utf8.count))
        }
        onEvent(.generationStats(
            promptTokens: promptTokens,
            generatedTokens: generatedTokens,
            stopReason: stopReason,
            templateMode: .unavailable
        ))
        onEvent(.done(totalBytes: result.utf8.count, duration: duration))

        return result
    }

    @available(iOS 26.0, macOS 26.0, *)
    private func foundationModelsSession(options: CarbocationLocalLLM.GenerationOptions) -> LanguageModelSession {
        if let session = foundationSessionStorage as? LanguageModelSession {
            return session
        }

        let instructions = Self.instructions(system: system, options: options)
        let session: LanguageModelSession
        if instructions.isEmpty {
            session = LanguageModelSession()
        } else {
            session = LanguageModelSession(instructions: instructions)
        }
        foundationSessionStorage = session
        return session
    }

    @available(iOS 26.0, macOS 26.0, *)
    private nonisolated static func foundationModelsOptions(
        from options: AppleIntelligenceResolvedOptions
    ) -> FoundationModels.GenerationOptions {
        FoundationModels.GenerationOptions(
            sampling: foundationModelsSampling(from: options.sampling),
            temperature: options.temperature,
            maximumResponseTokens: options.maximumResponseTokens
        )
    }

    @available(iOS 26.0, macOS 26.0, *)
    private nonisolated static func foundationModelsSampling(
        from policy: AppleIntelligenceSamplingPolicy
    ) -> FoundationModels.GenerationOptions.SamplingMode? {
        switch policy {
        case .systemDefault:
            return nil
        case .greedy:
            return .greedy
        case .randomTopK(let top, let seed):
            return .random(top: top, seed: seed)
        case .randomProbabilityThreshold(let probabilityThreshold, let seed):
            return .random(probabilityThreshold: probabilityThreshold, seed: seed)
        }
    }

    private nonisolated static func instructions(
        system: String,
        options: CarbocationLocalLLM.GenerationOptions
    ) -> String {
        var parts: [String] = []
        let trimmedSystem = system.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedSystem.isEmpty {
            parts.append(trimmedSystem)
        }
        if options.grammar != nil || options.stopAtBalancedJSON {
            parts.append("Follow the requested output format exactly. If the prompt asks for JSON, return only valid JSON with no surrounding prose.")
        }
        return parts.joined(separator: "\n\n")
    }

    private nonisolated static func stopReasonForCompletedResponse(
        result: String,
        generatedTokens: Int,
        options: CarbocationLocalLLM.GenerationOptions
    ) -> String {
        if let maxOutputTokens = options.maxOutputTokens,
           maxOutputTokens > 0,
           generatedTokens >= maxOutputTokens {
            return "max-tokens"
        }
        return "complete"
    }
}
#endif
