import Foundation

public enum LLMStreamContentPhase: String, Codable, Hashable, Sendable {
    case unknown
    case thinking
    case final
}

public struct LLMStreamPhaseConfiguration: Codable, Hashable, Sendable {
    public var thinkingPairs: [OutputDelimiterPair]
    public var finalMarkers: [String]
    /// nil keeps automatic model/template behavior, true starts generated content in thinking, false starts in final.
    public var startsInThinking: Bool?

    public init(
        thinkingPairs: [OutputDelimiterPair] = [],
        finalMarkers: [String] = [],
        startsInThinking: Bool? = nil
    ) {
        self.thinkingPairs = thinkingPairs
        self.finalMarkers = finalMarkers
        self.startsInThinking = startsInThinking
    }

    public static let automatic = LLMStreamPhaseConfiguration()

    public var isEmpty: Bool {
        thinkingPairs.isEmpty
            && finalMarkers.isEmpty
            && startsInThinking == nil
    }
}

public struct LLMSamplingDefaults: Codable, Hashable, Sendable {
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    public var minP: Double?
    public var presencePenalty: Double?
    public var repetitionPenalty: Double?
    public var seed: UInt32?

    public init(
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        minP: Double? = nil,
        presencePenalty: Double? = nil,
        repetitionPenalty: Double? = nil,
        seed: UInt32? = nil
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.presencePenalty = presencePenalty
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
    }

    public static let providerDefault = LLMSamplingDefaults()
    public static let extractionSafe = LLMSamplingDefaults(temperature: 0, topP: 0.9, topK: 40)

    public func merged(with override: LLMSamplingDefaults?) -> LLMSamplingDefaults {
        guard let override else { return self }
        return LLMSamplingDefaults(
            temperature: override.temperature ?? temperature,
            topP: override.topP ?? topP,
            topK: override.topK ?? topK,
            minP: override.minP ?? minP,
            presencePenalty: override.presencePenalty ?? presencePenalty,
            repetitionPenalty: override.repetitionPenalty ?? repetitionPenalty,
            seed: override.seed ?? seed
        )
    }

    public func applying(to options: GenerationOptions) -> GenerationOptions {
        var copy = options
        if copy.temperature == nil {
            copy.temperature = temperature
        }
        if copy.topP == nil {
            copy.topP = topP
        }
        if copy.topK == nil {
            copy.topK = topK
        }
        if copy.minP == nil {
            copy.minP = minP
        }
        if copy.presencePenalty == nil {
            copy.presencePenalty = presencePenalty
        }
        if copy.repetitionPenalty == nil {
            copy.repetitionPenalty = repetitionPenalty
        }
        if copy.seed == nil {
            copy.seed = seed
        }
        return copy
    }
}

public struct GenerationOptions: Codable, Hashable, Sendable {
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    public var minP: Double?
    public var presencePenalty: Double?
    public var repetitionPenalty: Double?
    public var maxOutputTokens: Int?
    public var seed: UInt32?
    public var stopSequences: [String]
    public var stopAtBalancedJSON: Bool
    /// Optional GBNF grammar for token-constrained generation.
    public var grammar: String?
    /// Enables model-native thinking/reasoning channels when the chat template supports them.
    public var enableThinking: Bool
    /// Optional token budget for generated thinking/reasoning content.
    public var thinkingBudgetTokens: Int? {
        didSet {
            precondition(
                thinkingBudgetTokens.map { $0 >= 0 } ?? true,
                "thinkingBudgetTokens must be nil or nonnegative."
            )
        }
    }
    /// Optional text inserted before the model-native end-of-thinking tag when the thinking budget is exhausted.
    public var thinkingBudgetMessage: String
    /// Optional per-request hints for phase-aware streaming when prompt markers are not discoverable from the model template.
    public var streamPhaseConfiguration: LLMStreamPhaseConfiguration

    public init(
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        minP: Double? = nil,
        presencePenalty: Double? = nil,
        repetitionPenalty: Double? = nil,
        maxOutputTokens: Int? = nil,
        seed: UInt32? = nil,
        stopSequences: [String] = [],
        stopAtBalancedJSON: Bool = false,
        grammar: String? = nil,
        enableThinking: Bool = false,
        thinkingBudgetTokens: Int? = nil,
        thinkingBudgetMessage: String = "",
        streamPhaseConfiguration: LLMStreamPhaseConfiguration = .automatic
    ) {
        precondition(
            thinkingBudgetTokens.map { $0 >= 0 } ?? true,
            "thinkingBudgetTokens must be nil or nonnegative."
        )
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.presencePenalty = presencePenalty
        self.repetitionPenalty = repetitionPenalty
        self.maxOutputTokens = maxOutputTokens
        self.seed = seed
        self.stopSequences = stopSequences
        self.stopAtBalancedJSON = stopAtBalancedJSON
        self.grammar = grammar
        self.enableThinking = enableThinking
        self.thinkingBudgetTokens = thinkingBudgetTokens
        self.thinkingBudgetMessage = thinkingBudgetMessage
        self.streamPhaseConfiguration = streamPhaseConfiguration
    }

    public init(
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        maxOutputTokens: Int? = nil,
        seed: UInt32? = nil,
        stopSequences: [String] = [],
        stopAtBalancedJSON: Bool = false,
        grammar: String? = nil,
        enableThinking: Bool = false,
        thinkingBudgetTokens: Int? = nil,
        thinkingBudgetMessage: String = "",
        streamPhaseConfiguration: LLMStreamPhaseConfiguration = .automatic
    ) {
        self.init(
            temperature: temperature,
            topP: topP,
            topK: topK,
            minP: nil,
            presencePenalty: nil,
            repetitionPenalty: nil,
            maxOutputTokens: maxOutputTokens,
            seed: seed,
            stopSequences: stopSequences,
            stopAtBalancedJSON: stopAtBalancedJSON,
            grammar: grammar,
            enableThinking: enableThinking,
            thinkingBudgetTokens: thinkingBudgetTokens,
            thinkingBudgetMessage: thinkingBudgetMessage,
            streamPhaseConfiguration: streamPhaseConfiguration
        )
    }

    public init(
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        minP: Double? = nil,
        presencePenalty: Double? = nil,
        repetitionPenalty: Double? = nil,
        maxOutputTokens: Int? = nil,
        seed: UInt32? = nil,
        stopSequences: [String] = [],
        stopAtBalancedJSON: Bool = false,
        grammar: String? = nil,
        enableThinking: Bool = false,
        streamPhaseConfiguration: LLMStreamPhaseConfiguration = .automatic
    ) {
        self.init(
            temperature: temperature,
            topP: topP,
            topK: topK,
            minP: minP,
            presencePenalty: presencePenalty,
            repetitionPenalty: repetitionPenalty,
            maxOutputTokens: maxOutputTokens,
            seed: seed,
            stopSequences: stopSequences,
            stopAtBalancedJSON: stopAtBalancedJSON,
            grammar: grammar,
            enableThinking: enableThinking,
            thinkingBudgetTokens: nil,
            thinkingBudgetMessage: "",
            streamPhaseConfiguration: streamPhaseConfiguration
        )
    }

    public init(
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        maxOutputTokens: Int? = nil,
        seed: UInt32? = nil,
        stopSequences: [String] = [],
        stopAtBalancedJSON: Bool = false,
        grammar: String? = nil,
        enableThinking: Bool = false,
        streamPhaseConfiguration: LLMStreamPhaseConfiguration = .automatic
    ) {
        self.init(
            temperature: temperature,
            topP: topP,
            topK: topK,
            minP: nil,
            presencePenalty: nil,
            repetitionPenalty: nil,
            maxOutputTokens: maxOutputTokens,
            seed: seed,
            stopSequences: stopSequences,
            stopAtBalancedJSON: stopAtBalancedJSON,
            grammar: grammar,
            enableThinking: enableThinking,
            streamPhaseConfiguration: streamPhaseConfiguration
        )
    }

    public static var extractionSafe: GenerationOptions {
        LLMSamplingDefaults.extractionSafe.applying(to: GenerationOptions())
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
        case minP
        case presencePenalty
        case repetitionPenalty
        case maxOutputTokens
        case seed
        case stopSequences
        case stopAtBalancedJSON
        case grammar
        case enableThinking
        case thinkingBudgetTokens
        case thinkingBudgetMessage
        case streamPhaseConfiguration
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        temperature = try container.decodeIfPresent(Double.self, forKey: .temperature)
        topP = try container.decodeIfPresent(Double.self, forKey: .topP)
        topK = try container.decodeIfPresent(Int.self, forKey: .topK)
        minP = try container.decodeIfPresent(Double.self, forKey: .minP)
        presencePenalty = try container.decodeIfPresent(Double.self, forKey: .presencePenalty)
        repetitionPenalty = try container.decodeIfPresent(Double.self, forKey: .repetitionPenalty)
        maxOutputTokens = try container.decodeIfPresent(Int.self, forKey: .maxOutputTokens)
        seed = try container.decodeIfPresent(UInt32.self, forKey: .seed)
        stopSequences = try container.decodeIfPresent([String].self, forKey: .stopSequences) ?? []
        stopAtBalancedJSON = try container.decodeIfPresent(Bool.self, forKey: .stopAtBalancedJSON) ?? false
        grammar = try container.decodeIfPresent(String.self, forKey: .grammar)
        enableThinking = try container.decodeIfPresent(Bool.self, forKey: .enableThinking) ?? false
        let decodedThinkingBudgetTokens = try container.decodeIfPresent(Int.self, forKey: .thinkingBudgetTokens)
        if let decodedThinkingBudgetTokens, decodedThinkingBudgetTokens < 0 {
            throw DecodingError.dataCorruptedError(
                forKey: .thinkingBudgetTokens,
                in: container,
                debugDescription: "thinkingBudgetTokens must be nil or nonnegative."
            )
        }
        thinkingBudgetTokens = decodedThinkingBudgetTokens
        thinkingBudgetMessage = try container.decodeIfPresent(String.self, forKey: .thinkingBudgetMessage) ?? ""
        streamPhaseConfiguration = try container.decodeIfPresent(
            LLMStreamPhaseConfiguration.self,
            forKey: .streamPhaseConfiguration
        ) ?? .automatic
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(temperature, forKey: .temperature)
        try container.encodeIfPresent(topP, forKey: .topP)
        try container.encodeIfPresent(topK, forKey: .topK)
        try container.encodeIfPresent(minP, forKey: .minP)
        try container.encodeIfPresent(presencePenalty, forKey: .presencePenalty)
        try container.encodeIfPresent(repetitionPenalty, forKey: .repetitionPenalty)
        try container.encodeIfPresent(maxOutputTokens, forKey: .maxOutputTokens)
        try container.encodeIfPresent(seed, forKey: .seed)
        if !stopSequences.isEmpty {
            try container.encode(stopSequences, forKey: .stopSequences)
        }
        if stopAtBalancedJSON {
            try container.encode(stopAtBalancedJSON, forKey: .stopAtBalancedJSON)
        }
        try container.encodeIfPresent(grammar, forKey: .grammar)
        if enableThinking {
            try container.encode(enableThinking, forKey: .enableThinking)
        }
        try container.encodeIfPresent(thinkingBudgetTokens, forKey: .thinkingBudgetTokens)
        if !thinkingBudgetMessage.isEmpty {
            try container.encode(thinkingBudgetMessage, forKey: .thinkingBudgetMessage)
        }
        if !streamPhaseConfiguration.isEmpty {
            try container.encode(streamPhaseConfiguration, forKey: .streamPhaseConfiguration)
        }
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
    public var minP: String
    public var presencePenalty: String
    public var repetitionPenalty: String
    public var seed: String

    public init(
        mode: String = "llama.optionsMode",
        temperature: String = "llama.temperature",
        topP: String = "llama.topP",
        topK: String = "llama.topK",
        minP: String = "llama.minP",
        presencePenalty: String = "llama.presencePenalty",
        repetitionPenalty: String = "llama.repetitionPenalty",
        seed: String = "llama.seed"
    ) {
        self.mode = mode
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.presencePenalty = presencePenalty
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
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
            topK: topKRaw > 0 ? topKRaw : 40,
            minP: defaults.object(forKey: keys.minP) == nil ? nil : defaults.double(forKey: keys.minP),
            presencePenalty: defaults.object(forKey: keys.presencePenalty) == nil
                ? nil
                : defaults.double(forKey: keys.presencePenalty),
            repetitionPenalty: defaults.object(forKey: keys.repetitionPenalty) == nil
                ? nil
                : defaults.double(forKey: keys.repetitionPenalty),
            seed: seedValue(from: defaults, key: keys.seed)
        )
    }

    private static func seedValue(from defaults: UserDefaults, key: String) -> UInt32? {
        guard let object = defaults.object(forKey: key) else { return nil }

        if let text = object as? String {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty,
                  let value = UInt64(trimmed),
                  value <= UInt64(UInt32.max)
            else { return nil }
            return UInt32(value)
        }

        if let number = object as? NSNumber {
            let value = number.doubleValue
            guard value.isFinite,
                  value.rounded(.towardZero) == value,
                  value >= 0,
                  value <= Double(UInt32.max)
            else { return nil }
            return UInt32(value)
        }

        return nil
    }
}
