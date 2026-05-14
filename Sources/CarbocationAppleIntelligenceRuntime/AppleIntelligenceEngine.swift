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
    /// Foundation Models exposes top-k and top-p choices separately, not as a combined sampler chain.
    case combinedSamplingFilters
    /// Foundation Models does not expose llama.cpp-style min-p sampling.
    case minP
    /// Foundation Models does not expose llama.cpp-style presence penalties.
    case presencePenalty
    /// Foundation Models does not expose llama.cpp-style repetition penalties.
    case repetitionPenalty
}

public enum AppleIntelligenceUnsupportedFeatureBehavior: String, Codable, Hashable, Sendable {
    /// Keep going and rely on prompting plus response post-processing.
    case ignore
    /// Throw before generation when an option cannot be represented.
    case fail
}

public struct AppleIntelligenceEngineConfiguration: Hashable, Sendable {
    public var unsupportedFeatureBehavior: AppleIntelligenceUnsupportedFeatureBehavior
    public var promptReserveTokens: Int

    public init(
        unsupportedFeatureBehavior: AppleIntelligenceUnsupportedFeatureBehavior = .ignore,
        promptReserveTokens: Int = LLMGenerationBudget.outputTokenReserve
    ) {
        self.unsupportedFeatureBehavior = unsupportedFeatureBehavior
        self.promptReserveTokens = promptReserveTokens
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
        let isGreedy = options.temperature.map { $0 <= 0 } ?? false
        if !isGreedy,
           let topK = options.topK, topK > 0,
           let topP = options.topP, topP > 0, topP < 1 {
            unsupportedFeatures.insert(.combinedSamplingFilters)
        }
        if !isGreedy, let minP = options.minP, minP > 0 {
            unsupportedFeatures.insert(.minP)
        }
        if let presencePenalty = options.presencePenalty, presencePenalty != 0 {
            unsupportedFeatures.insert(.presencePenalty)
        }
        if let repetitionPenalty = options.repetitionPenalty, repetitionPenalty != 1 {
            unsupportedFeatures.insert(.repetitionPenalty)
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

fileprivate enum AppleIntelligencePromptBudget {
    static func instructions(
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

    static func estimatedPromptTokens(
        system: String,
        prompt: String,
        options: CarbocationLocalLLM.GenerationOptions
    ) -> Int {
        TokenEstimator.estimate(text: instructions(system: system, options: options))
            + TokenEstimator.estimate(text: prompt)
    }

    static func preflight(
        system: String,
        prompt: String,
        options: CarbocationLocalLLM.GenerationOptions,
        contextSize: Int,
        promptReserveTokens: Int
    ) -> LLMGenerationPreflight {
        LLMGenerationPreflight(
            loadedContextSize: contextSize,
            modelTrainingContextSize: contextSize,
            promptTokens: estimatedPromptTokens(system: system, prompt: prompt, options: options),
            reservedOutputTokens: promptReserveTokens,
            requestedMaxOutputTokens: options.maxOutputTokens,
            usesExactTokenCounts: false,
            templateMode: .unavailable
        )
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

    public func preflight(
        system: String,
        prompt: String,
        options: CarbocationLocalLLM.GenerationOptions
    ) async throws -> LLMGenerationPreflight {
        let availability = Self.availability()
        guard availability.isAvailable else {
            throw AppleIntelligenceEngineError.unavailable(availability)
        }

        let resolvedOptions = AppleIntelligenceOptionsMapper.resolve(options)
        if configuration.unsupportedFeatureBehavior == .fail,
           !resolvedOptions.unsupportedFeatures.isEmpty {
            throw AppleIntelligenceEngineError.unsupportedFeatures(resolvedOptions.unsupportedFeatures)
        }

        return AppleIntelligencePromptBudget.preflight(
            system: system,
            prompt: prompt,
            options: options,
            contextSize: availability.contextSize,
            promptReserveTokens: configuration.promptReserveTokens
        )
    }

    public func generate(
        system: String,
        prompt: String,
        options: CarbocationLocalLLM.GenerationOptions,
        onEvent: @Sendable (LLMStreamEvent) -> Void
    ) async throws -> String {
        try await generate(
            system: system,
            prompt: prompt,
            options: options,
            control: nil,
            onEvent: onEvent
        )
    }

    public func generate(
        system: String,
        prompt: String,
        options: CarbocationLocalLLM.GenerationOptions,
        control: LLMGenerationControl? = nil,
        onEvent: @Sendable (LLMStreamEvent) -> Void
    ) async throws -> String {
        try await generate(
            system: system,
            prompt: prompt,
            options: options,
            control: control,
            onPhaseAwareEvent: { event in
                if let streamEvent = event.streamEvent {
                    onEvent(streamEvent)
                }
            }
        )
    }

    public func generate(
        system: String,
        prompt: String,
        options: CarbocationLocalLLM.GenerationOptions,
        onPhaseAwareEvent: @Sendable (LLMPhaseAwareStreamEvent) -> Void,
        _ phaseAwareOverload: Void = ()
    ) async throws -> String {
        try await generate(
            system: system,
            prompt: prompt,
            options: options,
            control: nil,
            onPhaseAwareEvent: onPhaseAwareEvent,
            phaseAwareOverload
        )
    }

    public func generate(
        system: String,
        prompt: String,
        options: CarbocationLocalLLM.GenerationOptions,
        control: LLMGenerationControl? = nil,
        onPhaseAwareEvent: @Sendable (LLMPhaseAwareStreamEvent) -> Void,
        _ phaseAwareOverload: Void = ()
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
            onPhaseAwareEvent: onPhaseAwareEvent
        )
        #else
        throw AppleIntelligenceEngineError.unavailable(.unavailable(.sdkUnavailable))
        #endif
    }

    public func generateWithTools(
        _ request: LLMToolGenerationRequest,
        onPhaseAwareEvent: @escaping @Sendable (LLMToolPhaseAwareStreamEvent) -> Void = { _ in }
    ) async throws -> LLMToolGenerationResult {
        try await generateWithTools(
            request,
            control: nil,
            onPhaseAwareEvent: onPhaseAwareEvent
        )
    }

    public func generateWithTools(
        _ request: LLMToolGenerationRequest,
        control: LLMGenerationControl? = nil,
        onPhaseAwareEvent: @escaping @Sendable (LLMToolPhaseAwareStreamEvent) -> Void = { _ in }
    ) async throws -> LLMToolGenerationResult {
        try LLMToolRuntime.validate(request)
        let stats = AppleToolGenerationStatsAccumulator()

        func emitAggregateStats(stopReason: String) {
            let snapshot = stats.snapshot(fallbackStopReason: stopReason)
            onPhaseAwareEvent(.aggregateGenerationStats(
                promptTokens: snapshot.promptTokens,
                generatedTokens: snapshot.generatedTokens,
                stopReason: snapshot.stopReason
            ))
        }

        guard !request.tools.isEmpty, request.toolChoice != .none else {
            let text = try await generate(
                system: request.system,
                prompt: request.prompt,
                options: request.options,
                control: control,
                onPhaseAwareEvent: { event in
                    stats.record(event)
                    onPhaseAwareEvent(.finalAnswerEvent(event))
                }
            )
            let snapshot = stats.snapshot(fallbackStopReason: "complete")
            emitAggregateStats(stopReason: snapshot.stopReason)
            return LLMToolGenerationResult(finalText: text, stopReason: snapshot.stopReason)
        }

        let availability = Self.availability()
        guard availability.isAvailable else {
            throw AppleIntelligenceEngineError.unavailable(availability)
        }

        let resolvedOptions = AppleIntelligenceOptionsMapper.resolve(request.options)
        if configuration.unsupportedFeatureBehavior == .fail,
           !resolvedOptions.unsupportedFeatures.isEmpty {
            throw AppleIntelligenceEngineError.unsupportedFeatures(resolvedOptions.unsupportedFeatures)
        }

        #if canImport(FoundationModels)
        guard #available(iOS 26.0, macOS 26.0, *) else {
            throw AppleIntelligenceEngineError.unavailable(.unavailable(.operatingSystemUnavailable))
        }
        let result = try await generateWithFoundationModelsTools(
            request: request,
            resolvedOptions: resolvedOptions,
            stats: stats,
            onPhaseAwareEvent: onPhaseAwareEvent
        )
        emitAggregateStats(stopReason: result.stopReason)
        return result
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

    public func preflight(
        prompt: String,
        options: CarbocationLocalLLM.GenerationOptions
    ) async throws -> LLMGenerationPreflight {
        let availability = AppleIntelligenceEngine.availability()
        guard availability.isAvailable else {
            throw AppleIntelligenceEngineError.unavailable(availability)
        }

        let resolvedOptions = AppleIntelligenceOptionsMapper.resolve(options)
        if configuration.unsupportedFeatureBehavior == .fail,
           !resolvedOptions.unsupportedFeatures.isEmpty {
            throw AppleIntelligenceEngineError.unsupportedFeatures(resolvedOptions.unsupportedFeatures)
        }

        return AppleIntelligencePromptBudget.preflight(
            system: system,
            prompt: prompt,
            options: options,
            contextSize: availability.contextSize,
            promptReserveTokens: configuration.promptReserveTokens
        )
    }

    public func generate(
        prompt: String,
        options: CarbocationLocalLLM.GenerationOptions,
        onEvent: @Sendable (LLMStreamEvent) -> Void
    ) async throws -> String {
        try await generate(
            prompt: prompt,
            options: options,
            control: nil,
            onEvent: onEvent
        )
    }

    public func generate(
        prompt: String,
        options: CarbocationLocalLLM.GenerationOptions,
        control: LLMGenerationControl? = nil,
        onEvent: @Sendable (LLMStreamEvent) -> Void
    ) async throws -> String {
        try await generate(
            prompt: prompt,
            options: options,
            control: control,
            onPhaseAwareEvent: { event in
                if let streamEvent = event.streamEvent {
                    onEvent(streamEvent)
                }
            }
        )
    }

    public func generate(
        prompt: String,
        options: CarbocationLocalLLM.GenerationOptions,
        onPhaseAwareEvent: @Sendable (LLMPhaseAwareStreamEvent) -> Void,
        _ phaseAwareOverload: Void = ()
    ) async throws -> String {
        try await generate(
            prompt: prompt,
            options: options,
            control: nil,
            onPhaseAwareEvent: onPhaseAwareEvent,
            phaseAwareOverload
        )
    }

    public func generate(
        prompt: String,
        options: CarbocationLocalLLM.GenerationOptions,
        control: LLMGenerationControl? = nil,
        onPhaseAwareEvent: @Sendable (LLMPhaseAwareStreamEvent) -> Void,
        _ phaseAwareOverload: Void = ()
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
            onPhaseAwareEvent: onPhaseAwareEvent
        )
        #else
        throw AppleIntelligenceEngineError.unavailable(.unavailable(.sdkUnavailable))
        #endif
    }
}

private final class AppleToolGenerationStatsAccumulator: @unchecked Sendable {
    private let lock = NSLock()
    private var promptTokens = 0
    private var generatedTokens = 0
    private var stopReason: String?

    func record(_ event: LLMPhaseAwareStreamEvent) {
        guard case .generationStats(let promptTokens, let generatedTokens, let stopReason, _, _) = event else {
            return
        }
        lock.lock()
        defer { lock.unlock() }
        self.promptTokens += promptTokens
        self.generatedTokens += generatedTokens
        self.stopReason = stopReason
    }

    func snapshot(fallbackStopReason: String) -> (promptTokens: Int, generatedTokens: Int, stopReason: String) {
        lock.lock()
        defer { lock.unlock() }
        return (promptTokens, generatedTokens, stopReason ?? fallbackStopReason)
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
    private func generateWithFoundationModelsTools(
        request: LLMToolGenerationRequest,
        resolvedOptions: AppleIntelligenceResolvedOptions,
        stats: AppleToolGenerationStatsAccumulator,
        onPhaseAwareEvent: @escaping @Sendable (LLMToolPhaseAwareStreamEvent) -> Void
    ) async throws -> LLMToolGenerationResult {
        let phase = LLMStreamContentPhase.final
        func emit(_ event: LLMPhaseAwareStreamEvent) {
            stats.record(event)
            onPhaseAwareEvent(.finalAnswerEvent(event))
        }

        emit(.requestSent(phase: phase))
        let startedAt = Date()

        let recorder = AppleNativeToolRecorder(
            maxToolCalls: request.maxToolRounds,
            onPhaseAwareEvent: onPhaseAwareEvent
        )
        let effectiveTools = Self.effectiveTools(
            request.tools,
            toolChoice: request.toolChoice
        )
        let adapters: [any Tool] = try effectiveTools.map {
            try AppleNativeToolAdapter(tool: $0, recorder: recorder)
        }
        let instructions = Self.toolInstructions(
            system: request.system,
            options: request.options,
            toolChoice: request.toolChoice
        )
        let session = LanguageModelSession(
            model: .default,
            tools: adapters,
            instructions: instructions.isEmpty ? nil : instructions
        )

        var streamedText = ""
        var sawFirstSnapshot = false
        let stream = session.streamResponse(
            to: request.prompt,
            options: Self.foundationModelsOptions(from: resolvedOptions)
        )

        for try await snapshot in stream {
            let nextText = snapshot.content
            if !sawFirstSnapshot {
                sawFirstSnapshot = true
                emit(.firstByteReceived(
                    after: Date().timeIntervalSince(startedAt),
                    phase: phase
                ))
            }
            guard nextText != streamedText else { continue }
            if nextText.hasPrefix(streamedText) {
                let delta = String(nextText.dropFirst(streamedText.count))
                if !delta.isEmpty {
                    emit(.finalAnswerDelta(text: delta, bytesSoFar: nextText.utf8.count))
                }
            } else {
                emit(.finalAnswerSnapshot(
                    text: nextText,
                    bytesSoFar: nextText.utf8.count,
                    reason: .streamCorrection
                ))
            }
            streamedText = nextText
            emit(.tokenChunk(
                preview: String(streamedText.suffix(60)),
                bytesSoFar: streamedText.utf8.count,
                phase: phase
            ))
        }

        if !sawFirstSnapshot {
            emit(.firstByteReceived(
                after: Date().timeIntervalSince(startedAt),
                phase: phase
            ))
        }

        let processed = AppleIntelligenceResponsePostProcessor.process(
            streamedText,
            options: request.options
        )
        if processed.text != streamedText {
            streamedText = processed.text
            emit(.finalAnswerSnapshot(
                text: streamedText,
                bytesSoFar: streamedText.utf8.count,
                reason: .completed
            ))
        }

        let duration = Date().timeIntervalSince(startedAt)
        let promptTokens = AppleIntelligencePromptBudget.estimatedPromptTokens(
            system: instructions,
            prompt: request.prompt,
            options: request.options
        )
        let generatedTokens = TokenEstimator.estimate(text: streamedText)
        let stopReason = recorder.reachedToolLimit
            ? "max-tool-rounds"
            : (processed.stopReason ?? stopReasonForCompletedResponse(
                result: streamedText,
                generatedTokens: generatedTokens,
                options: request.options
            ))

        emit(.generationStats(
            promptTokens: promptTokens,
            generatedTokens: generatedTokens,
            stopReason: stopReason,
            templateMode: .unavailable,
            phase: phase
        ))
        emit(.done(
            totalBytes: streamedText.utf8.count,
            duration: duration,
            phase: phase
        ))

        return LLMToolGenerationResult(
            finalText: streamedText,
            toolCalls: recorder.calls,
            toolOutputs: recorder.outputs,
            roundsCompleted: recorder.executedToolCallCount,
            stopReason: stopReason
        )
    }

    @available(iOS 26.0, macOS 26.0, *)
    private func generateWithFoundationModels(
        system: String,
        prompt: String,
        options: CarbocationLocalLLM.GenerationOptions,
        resolvedOptions: AppleIntelligenceResolvedOptions,
        onPhaseAwareEvent: @Sendable (LLMPhaseAwareStreamEvent) -> Void
    ) async throws -> String {
        let phase = LLMStreamContentPhase.final
        onPhaseAwareEvent(.requestSent(phase: phase))
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
        let promptTokens = AppleIntelligencePromptBudget.estimatedPromptTokens(
            system: system,
            prompt: prompt,
            options: options
        )
        let generatedTokens = TokenEstimator.estimate(text: result)
        let stopReason = processed.stopReason ?? stopReasonForCompletedResponse(
            result: result,
            generatedTokens: generatedTokens,
            options: options
        )

        onPhaseAwareEvent(.firstByteReceived(after: duration, phase: phase))
        if !result.isEmpty {
            onPhaseAwareEvent(.finalAnswerDelta(
                text: result,
                bytesSoFar: result.utf8.count
            ))
            onPhaseAwareEvent(.tokenChunk(
                preview: String(result.suffix(60)),
                bytesSoFar: result.utf8.count,
                phase: phase
            ))
        }
        onPhaseAwareEvent(.generationStats(
            promptTokens: promptTokens,
            generatedTokens: generatedTokens,
            stopReason: stopReason,
            templateMode: .unavailable,
            phase: phase
        ))
        onPhaseAwareEvent(.done(totalBytes: result.utf8.count, duration: duration, phase: phase))

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
        AppleIntelligencePromptBudget.instructions(system: system, options: options)
    }

    private nonisolated static func toolInstructions(
        system: String,
        options: CarbocationLocalLLM.GenerationOptions,
        toolChoice: LLMToolChoice
    ) -> String {
        var parts: [String] = []
        let base = instructions(system: system, options: options)
        if !base.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            parts.append(base)
        }

        switch toolChoice {
        case .auto:
            parts.append("Use available tools only when they help answer the user.")
        case .none:
            parts.append("Do not call tools.")
        case .required:
            parts.append("Call at least one available tool before giving a final answer.")
        case .named(let name):
            parts.append("If a tool is needed, call only \(name).")
        }
        return parts.joined(separator: "\n\n")
    }

    private nonisolated static func effectiveTools(
        _ tools: [LLMTool],
        toolChoice: LLMToolChoice
    ) -> [LLMTool] {
        if case .named(let name) = toolChoice {
            return tools.filter { $0.definition.name == name }
        }
        return tools
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

@available(iOS 26.0, macOS 26.0, *)
struct AppleNativeToolAdapter: Tool {
    typealias Arguments = GeneratedContent
    typealias Output = GeneratedContent

    let tool: LLMTool
    let recorder: AppleNativeToolRecorder
    let parameters: GenerationSchema

    var name: String { tool.definition.name }
    var description: String { tool.definition.description }
    var includesSchemaInInstructions: Bool { true }

    init(tool: LLMTool, recorder: AppleNativeToolRecorder) throws {
        self.tool = tool
        self.recorder = recorder
        self.parameters = try AppleNativeToolSchemaMapper.generationSchema(for: tool.definition)
    }

    func call(arguments: GeneratedContent) async throws -> GeneratedContent {
        try await recorder.execute(tool: tool, arguments: arguments)
    }
}

@available(iOS 26.0, macOS 26.0, *)
final class AppleNativeToolRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private let maxToolCalls: Int
    private let onPhaseAwareEvent: @Sendable (LLMToolPhaseAwareStreamEvent) -> Void
    private var nextToolCallNumber = 1
    private var storedCalls: [LLMToolCall] = []
    private var storedOutputs: [LLMToolOutput] = []
    private var executedCount = 0
    private var hitToolLimit = false

    init(
        maxToolCalls: Int,
        onPhaseAwareEvent: @escaping @Sendable (LLMToolPhaseAwareStreamEvent) -> Void
    ) {
        self.maxToolCalls = maxToolCalls
        self.onPhaseAwareEvent = onPhaseAwareEvent
    }

    var calls: [LLMToolCall] {
        lock.lock()
        defer { lock.unlock() }
        return storedCalls
    }

    var outputs: [LLMToolOutput] {
        lock.lock()
        defer { lock.unlock() }
        return storedOutputs
    }

    var executedToolCallCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return executedCount
    }

    var reachedToolLimit: Bool {
        lock.lock()
        defer { lock.unlock() }
        return hitToolLimit
    }

    func execute(tool: LLMTool, arguments: GeneratedContent) async throws -> GeneratedContent {
        let call = makeCall(tool: tool, arguments: arguments)
        onPhaseAwareEvent(.toolCallStarted(call))

        guard reserveExecutionSlotIfAvailable() else {
            let output = LLMToolOutput(
                callID: call.executionID,
                name: call.name,
                content: LLMToolRuntime.toolErrorContent(
                    message: "Maximum tool round limit reached.",
                    code: "max_tool_rounds"
                ),
                isError: true
            )
            store(output: output)
            onPhaseAwareEvent(.toolCallFailed(output))
            return try Self.generatedContent(from: output.content)
        }

        do {
            let content = try await tool.call(arguments: call.arguments)
            let output = LLMToolOutput(
                callID: call.executionID,
                name: call.name,
                content: content,
                isError: false
            )
            store(output: output)
            onPhaseAwareEvent(.toolCallCompleted(output))
            return try Self.generatedContent(from: content)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            let output = LLMToolOutput(
                callID: call.executionID,
                name: call.name,
                content: LLMToolRuntime.toolErrorContent(
                    message: error.localizedDescription,
                    code: "tool_execution_failed"
                ),
                isError: true
            )
            store(output: output)
            onPhaseAwareEvent(.toolCallFailed(output))
            return try Self.generatedContent(from: output.content)
        }
    }

    private func makeCall(tool: LLMTool, arguments: GeneratedContent) -> LLMToolCall {
        let parsedArguments = (try? LLMJSONValue(jsonString: arguments.jsonString))
            ?? .object(["value": .string(arguments.debugDescription)])
        lock.lock()
        let executionID = "call_\(nextToolCallNumber)"
        nextToolCallNumber += 1
        let call = LLMToolCall(
            executionID: executionID,
            name: tool.definition.name,
            arguments: parsedArguments
        )
        storedCalls.append(call)
        lock.unlock()
        return call
    }

    private func reserveExecutionSlotIfAvailable() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard executedCount < maxToolCalls else {
            hitToolLimit = true
            return false
        }
        executedCount += 1
        return true
    }

    private func store(output: LLMToolOutput) {
        lock.lock()
        storedOutputs.append(output)
        lock.unlock()
    }

    private static func generatedContent(from value: LLMJSONValue) throws -> GeneratedContent {
        try GeneratedContent(json: value.jsonString(prettyPrinted: false))
    }
}

@available(iOS 26.0, macOS 26.0, *)
enum AppleNativeToolSchemaMapper {
    static func generationSchema(for tool: LLMToolDefinition) throws -> GenerationSchema {
        let root = dynamicSchema(
            name: safeSchemaName("\(tool.name)_arguments"),
            description: nil,
            jsonSchema: tool.parameters
        )
        return try GenerationSchema(root: root, dependencies: [])
    }

    private static func dynamicSchema(
        name: String,
        description: String?,
        jsonSchema: LLMJSONValue
    ) -> DynamicGenerationSchema {
        guard let object = jsonSchema.objectValue else {
            return DynamicGenerationSchema(type: String.self)
        }

        if let enumValues = object["enum"]?.arrayValue?.compactMap(\.stringValue),
           !enumValues.isEmpty {
            return DynamicGenerationSchema(
                name: name,
                description: description ?? object["description"]?.stringValue,
                anyOf: enumValues
            )
        }

        let type = object["type"]?.stringValue?.lowercased()
        switch type {
        case "object", nil:
            let required = Set(object["required"]?.arrayValue?.compactMap(\.stringValue) ?? [])
            let propertiesObject = object["properties"]?.objectValue ?? [:]
            let properties = propertiesObject.keys.sorted().map { propertyName in
                let propertySchema = propertiesObject[propertyName] ?? .object([:])
                return DynamicGenerationSchema.Property(
                    name: propertyName,
                    description: propertySchema.objectValue?["description"]?.stringValue,
                    schema: dynamicSchema(
                        name: safeSchemaName("\(name)_\(propertyName)"),
                        description: propertySchema.objectValue?["description"]?.stringValue,
                        jsonSchema: propertySchema
                    ),
                    isOptional: !required.contains(propertyName)
                )
            }
            return DynamicGenerationSchema(
                name: name,
                description: description ?? object["description"]?.stringValue,
                properties: properties
            )
        case "array":
            let itemSchema = object["items"] ?? .object(["type": .string("string")])
            return DynamicGenerationSchema(arrayOf: dynamicSchema(
                name: safeSchemaName("\(name)_item"),
                description: nil,
                jsonSchema: itemSchema
            ))
        case "integer":
            return DynamicGenerationSchema(type: Int.self)
        case "number":
            return DynamicGenerationSchema(type: Double.self)
        case "boolean":
            return DynamicGenerationSchema(type: Bool.self)
        case "string":
            fallthrough
        default:
            return DynamicGenerationSchema(type: String.self)
        }
    }

    private static func safeSchemaName(_ raw: String) -> String {
        var result = ""
        for scalar in raw.unicodeScalars {
            if CharacterSet.alphanumerics.contains(scalar) || scalar == "_" {
                result.unicodeScalars.append(scalar)
            } else {
                result.append("_")
            }
        }
        if result.isEmpty || result.first?.isNumber == true {
            result = "Tool_\(result)"
        }
        return result
    }
}

extension AppleIntelligenceSession {
    @available(iOS 26.0, macOS 26.0, *)
    private func generateWithFoundationModels(
        prompt: String,
        options: CarbocationLocalLLM.GenerationOptions,
        resolvedOptions: AppleIntelligenceResolvedOptions,
        onPhaseAwareEvent: @Sendable (LLMPhaseAwareStreamEvent) -> Void
    ) async throws -> String {
        let phase = LLMStreamContentPhase.final
        onPhaseAwareEvent(.requestSent(phase: phase))
        let startedAt = Date()
        let session = foundationModelsSession(options: options)

        let response = try await session.respond(
            to: prompt,
            options: Self.foundationModelsOptions(from: resolvedOptions)
        )
        let processed = AppleIntelligenceResponsePostProcessor.process(response.content, options: options)
        let result = processed.text
        let duration = Date().timeIntervalSince(startedAt)
        let promptTokens = AppleIntelligencePromptBudget.estimatedPromptTokens(
            system: system,
            prompt: prompt,
            options: options
        )
        let generatedTokens = TokenEstimator.estimate(text: result)
        let stopReason = processed.stopReason ?? Self.stopReasonForCompletedResponse(
            result: result,
            generatedTokens: generatedTokens,
            options: options
        )

        onPhaseAwareEvent(.firstByteReceived(after: duration, phase: phase))
        if !result.isEmpty {
            onPhaseAwareEvent(.finalAnswerDelta(
                text: result,
                bytesSoFar: result.utf8.count
            ))
            onPhaseAwareEvent(.tokenChunk(
                preview: String(result.suffix(60)),
                bytesSoFar: result.utf8.count,
                phase: phase
            ))
        }
        onPhaseAwareEvent(.generationStats(
            promptTokens: promptTokens,
            generatedTokens: generatedTokens,
            stopReason: stopReason,
            templateMode: .unavailable,
            phase: phase
        ))
        onPhaseAwareEvent(.done(totalBytes: result.utf8.count, duration: duration, phase: phase))

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
        AppleIntelligencePromptBudget.instructions(system: system, options: options)
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
