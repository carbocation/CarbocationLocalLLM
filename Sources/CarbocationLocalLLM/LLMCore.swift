import Foundation

public struct GenerationOptions: Codable, Hashable, Sendable {
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    public var maxOutputTokens: Int?
    public var seed: UInt32?
    public var stopSequences: [String]
    public var stopAtBalancedJSON: Bool
    /// Optional GBNF grammar for token-constrained generation.
    public var grammar: String?

    public init(
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        maxOutputTokens: Int? = nil,
        seed: UInt32? = nil,
        stopSequences: [String] = [],
        stopAtBalancedJSON: Bool = false,
        grammar: String? = nil
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.maxOutputTokens = maxOutputTokens
        self.seed = seed
        self.stopSequences = stopSequences
        self.stopAtBalancedJSON = stopAtBalancedJSON
        self.grammar = grammar
    }

    public static var extractionSafe: GenerationOptions {
        GenerationOptions(temperature: 0, topP: 0.9, topK: 40)
    }

    public func with(grammar: String?) -> GenerationOptions {
        var copy = self
        copy.grammar = grammar
        return copy
    }

    private enum CodingKeys: String, CodingKey {
        case temperature
        case topP
        case topK
        case maxOutputTokens
        case seed
        case stopSequences
        case stopAtBalancedJSON
        case grammar
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        temperature = try container.decodeIfPresent(Double.self, forKey: .temperature)
        topP = try container.decodeIfPresent(Double.self, forKey: .topP)
        topK = try container.decodeIfPresent(Int.self, forKey: .topK)
        maxOutputTokens = try container.decodeIfPresent(Int.self, forKey: .maxOutputTokens)
        seed = try container.decodeIfPresent(UInt32.self, forKey: .seed)
        stopSequences = try container.decodeIfPresent([String].self, forKey: .stopSequences) ?? []
        stopAtBalancedJSON = try container.decodeIfPresent(Bool.self, forKey: .stopAtBalancedJSON) ?? false
        grammar = try container.decodeIfPresent(String.self, forKey: .grammar)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(temperature, forKey: .temperature)
        try container.encodeIfPresent(topP, forKey: .topP)
        try container.encodeIfPresent(topK, forKey: .topK)
        try container.encodeIfPresent(maxOutputTokens, forKey: .maxOutputTokens)
        try container.encodeIfPresent(seed, forKey: .seed)
        if !stopSequences.isEmpty {
            try container.encode(stopSequences, forKey: .stopSequences)
        }
        if stopAtBalancedJSON {
            try container.encode(stopAtBalancedJSON, forKey: .stopAtBalancedJSON)
        }
        try container.encodeIfPresent(grammar, forKey: .grammar)
    }
}

public enum GenerationOptionsMode: String, CaseIterable, Codable, Sendable {
    case extractionSafe
    case custom
}

public struct GenerationOptionsPreferenceKeys: Sendable {
    public var mode: String
    public var temperature: String
    public var topP: String
    public var topK: String

    public init(
        mode: String = "llama.optionsMode",
        temperature: String = "llama.temperature",
        topP: String = "llama.topP",
        topK: String = "llama.topK"
    ) {
        self.mode = mode
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
    }
}

public enum GenerationOptionsResolver {
    public static func configuredExtractionOptions(
        defaults: UserDefaults = .standard,
        keys: GenerationOptionsPreferenceKeys = GenerationOptionsPreferenceKeys()
    ) -> GenerationOptions {
        let rawMode = defaults.string(forKey: keys.mode) ?? GenerationOptionsMode.extractionSafe.rawValue
        guard GenerationOptionsMode(rawValue: rawMode) == .custom else {
            return .extractionSafe
        }

        let topPRaw = defaults.double(forKey: keys.topP)
        let topKRaw = defaults.integer(forKey: keys.topK)
        return GenerationOptions(
            temperature: defaults.double(forKey: keys.temperature),
            topP: topPRaw > 0 ? topPRaw : 0.9,
            topK: topKRaw > 0 ? topKRaw : 40
        )
    }
}

public enum LLMGenerationBudget {
    public static let outputTokenReserve = 1_024
    public static let promptSafetyTokens = 256
}

public enum LLMSystemModelID: String, Codable, Hashable, Sendable {
    case appleIntelligence = "system.apple-intelligence"
}

public enum LLMModelSelection: Codable, Hashable, Sendable {
    case installed(UUID)
    case system(LLMSystemModelID)

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let value = try container.decode(String.self)
        guard let selection = LLMModelSelection(storageValue: value) else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Invalid model selection: \(value)"
            )
        }
        self = selection
    }

    public init?(storageValue: String) {
        if let systemModel = LLMSystemModelID(rawValue: storageValue) {
            self = .system(systemModel)
            return
        }
        guard let uuid = UUID(uuidString: storageValue) else {
            return nil
        }
        self = .installed(uuid)
    }

    public var storageValue: String {
        switch self {
        case .installed(let id):
            return id.uuidString
        case .system(let id):
            return id.rawValue
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(storageValue)
    }
}

public struct LLMSystemModelOption: Identifiable, Hashable, Sendable {
    public var selection: LLMModelSelection
    public var displayName: String
    public var subtitle: String
    public var contextLength: Int
    public var systemImageName: String

    public var id: String {
        selection.storageValue
    }

    public init(
        selection: LLMModelSelection,
        displayName: String,
        subtitle: String,
        contextLength: Int,
        systemImageName: String
    ) {
        self.selection = selection
        self.displayName = displayName
        self.subtitle = subtitle
        self.contextLength = contextLength
        self.systemImageName = systemImageName
    }
}

public enum LlamaContextMode: String, CaseIterable, Codable, Sendable {
    case auto
    case manual
}

public struct LlamaContextPreferenceKeys: Sendable {
    public var contextMode: String
    public var numCtx: String

    public init(
        contextMode: String = "llama.contextMode",
        numCtx: String = "llama.numCtx"
    ) {
        self.contextMode = contextMode
        self.numCtx = numCtx
    }
}

public enum LlamaContextPolicy {
#if os(iOS)
    public static let defaultAutoCap = 4_096
    public static let unknownTrainingFallback = 4_096
#else
    public static let defaultAutoCap = 16_384
    public static let unknownTrainingFallback = 8_192
#endif
    public static let legacyDefaultNumCtx = 8_192
    public static let minimumContext = 512

    public static func currentMode(
        defaults: UserDefaults = .standard,
        keys: LlamaContextPreferenceKeys = LlamaContextPreferenceKeys()
    ) -> LlamaContextMode {
        if let raw = defaults.string(forKey: keys.contextMode),
           let mode = LlamaContextMode(rawValue: raw) {
            return mode
        }

        let hasLegacyOverride = defaults.object(forKey: keys.numCtx) != nil
        let legacyValue = defaults.integer(forKey: keys.numCtx)
        if hasLegacyOverride, legacyValue > 0, legacyValue != legacyDefaultNumCtx {
            return .manual
        }
        return .auto
    }

    public static func manualContext(
        defaults: UserDefaults = .standard,
        keys: LlamaContextPreferenceKeys = LlamaContextPreferenceKeys()
    ) -> Int {
        sanitizedContext(defaults.integer(forKey: keys.numCtx))
    }

    public static func autoContext(
        for trainingContext: Int,
        autoCap: Int = defaultAutoCap
    ) -> Int {
        guard trainingContext > 0 else { return unknownTrainingFallback }
        return max(minimumContext, min(trainingContext, max(minimumContext, autoCap)))
    }

    public static func resolvedRequestedContext(
        trainingContext: Int,
        mode: LlamaContextMode,
        manualContext: Int,
        autoCap: Int = defaultAutoCap
    ) -> Int {
        switch mode {
        case .auto:
            return autoContext(for: trainingContext, autoCap: autoCap)
        case .manual:
            return sanitizedContext(manualContext)
        }
    }

    public static func resolvedRequestedContext(
        trainingContext: Int,
        defaults: UserDefaults = .standard,
        keys: LlamaContextPreferenceKeys = LlamaContextPreferenceKeys(),
        autoCap: Int = defaultAutoCap
    ) -> Int {
        resolvedRequestedContext(
            trainingContext: trainingContext,
            mode: currentMode(defaults: defaults, keys: keys),
            manualContext: manualContext(defaults: defaults, keys: keys),
            autoCap: autoCap
        )
    }

    public static func resolvedRequestedContext(
        for model: InstalledModel,
        defaults: UserDefaults = .standard,
        keys: LlamaContextPreferenceKeys = LlamaContextPreferenceKeys(),
        autoCap: Int = defaultAutoCap
    ) -> Int {
        resolvedRequestedContext(
            trainingContext: model.contextLength,
            defaults: defaults,
            keys: keys,
            autoCap: autoCap
        )
    }

    public static func sanitizedContext(_ value: Int) -> Int {
        max(minimumContext, value > 0 ? value : legacyDefaultNumCtx)
    }
}

public enum LLMEngineError: Error, LocalizedError, Sendable {
    case noModelLoaded
    case modelLoadFailed(String)
    case contextInitFailed(String)
    case tokenizationFailed
    case insufficientGenerationBudget(contextSize: Int, promptTokens: Int, reserve: Int)
    case decodeFailed
    case samplerInitFailed
    case grammarParseFailed
    case chatTemplateUnavailable(String)

    public var errorDescription: String? {
        switch self {
        case .noModelLoaded:
            return "No model is loaded. Pick a model in Settings."
        case .modelLoadFailed(let detail):
            return "Failed to load model: \(detail)"
        case .contextInitFailed(let detail):
            return "Failed to create inference context: \(detail)"
        case .tokenizationFailed:
            return "Failed to tokenize the prompt."
        case .insufficientGenerationBudget(let contextSize, let promptTokens, let reserve):
            return "Prompt used \(promptTokens) tokens in a \(contextSize)-token context, leaving fewer than \(reserve) tokens to generate a response."
        case .decodeFailed:
            return "llama_decode failed."
        case .samplerInitFailed:
            return "Failed to initialize the sampler chain."
        case .grammarParseFailed:
            return "Failed to parse the JSON grammar."
        case .chatTemplateUnavailable(let detail):
            return "Loaded model has no supported chat template. \(detail)"
        }
    }
}

public enum LLMChatTemplateMode: String, Codable, Sendable {
    case embedded
    case gemmaFallback = "gemma-fallback"
    case chatMLFallback = "chatml-fallback"
    case unavailable

    public var displayLabel: String {
        switch self {
        case .embedded:
            return "embedded"
        case .gemmaFallback:
            return "Gemma fallback"
        case .chatMLFallback:
            return "ChatML fallback"
        case .unavailable:
            return "unavailable"
        }
    }
}

public enum LLMStreamEvent: Sendable {
    case requestSent
    case firstByteReceived(after: TimeInterval)
    case tokenChunk(preview: String, bytesSoFar: Int)
    case generationStats(promptTokens: Int, generatedTokens: Int, stopReason: String, templateMode: LLMChatTemplateMode)
    case done(totalBytes: Int, duration: TimeInterval)
}

public func shouldRethrowLLMError(_ error: Error) -> Bool {
    error is LLMEngineError || error is CancellationError
}

public protocol LLMEngine: Sendable {
    func currentModelID() async -> UUID?
    func currentContextSize() async -> Int

    func generate(
        system: String,
        prompt: String,
        options: GenerationOptions,
        onEvent: @Sendable (LLMStreamEvent) -> Void
    ) async throws -> String
}

public enum TokenEstimator {
    public static func estimate(utf8Count: Int) -> Int {
        (utf8Count + 2) / 3
    }

    public static func estimate(text: String) -> Int {
        estimate(utf8Count: text.utf8.count)
    }
}

public enum DurationFormatter {
    public static func format(_ seconds: TimeInterval) -> String {
        let clamped = max(0, Int(seconds))
        if clamped < 60 { return "\(clamped)s" }
        return "\(clamped / 60)m \(clamped % 60)s"
    }
}
