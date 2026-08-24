import Foundation

public enum LLMStreamContentPhase: String, Codable, Hashable, Sendable {
    case unknown
    case thinking
    case final
}

/// A model-native reasoning-effort value forwarded to a compatible chat template.
///
/// Templates define which values they accept. Qwen3.8 accepts `low`, `medium`, and
/// `xhigh` (and treats `high` as `xhigh`). Other model families may use the other
/// common values represented here, including explicit non-thinking effort values.
public enum LLMReasoningEffort: String, CaseIterable, Codable, Hashable, Sendable {
    case minimal
    case low
    case medium
    case high
    case xhigh
    case max
    case disabled = "none"
    case noThink = "no_think"
}

/// Thinking controls detected in a model's embedded chat template.
public struct LLMThinkingCapabilities: Codable, Hashable, Sendable {
    public var supportsThinking: Bool
    public var supportsReasoningEffort: Bool
    public var supportsPreserveThinking: Bool
    /// Whether the template preserves historical assistant thinking when the
    /// request leaves `preserve_thinking` undefined.
    /// `nil` means the template's default could not be inferred safely.
    public var defaultPreserveThinking: Bool?
    /// The effort values explicitly handled or accepted by the template.
    ///
    /// `nil` means the template accepts `reasoning_effort` but does not declare a
    /// finite vocabulary that can be inferred safely. It does not mean every value
    /// in `LLMReasoningEffort` is supported.
    public var supportedReasoningEfforts: [LLMReasoningEffort]?
    /// The named effort selected by the template when the request omits one.
    /// `nil` means no named default could be inferred safely.
    public var defaultReasoningEffort: LLMReasoningEffort?

    public init(
        supportsThinking: Bool = false,
        supportsReasoningEffort: Bool = false,
        supportsPreserveThinking: Bool = false,
        defaultPreserveThinking: Bool? = nil,
        supportedReasoningEfforts: [LLMReasoningEffort]? = nil,
        defaultReasoningEffort: LLMReasoningEffort? = nil
    ) {
        self.supportsThinking = supportsThinking
        self.supportsReasoningEffort = supportsReasoningEffort
        self.supportsPreserveThinking = supportsPreserveThinking
        self.defaultPreserveThinking = defaultPreserveThinking
        self.supportedReasoningEfforts = supportedReasoningEfforts
        self.defaultReasoningEffort = defaultReasoningEffort
    }

    public static let none = LLMThinkingCapabilities()

    public static func derived(fromChatTemplate template: String?) -> LLMThinkingCapabilities {
        guard let template else { return .none }
        let reasoningEffortMetadata = reasoningEffortMetadata(in: template)
        return LLMThinkingCapabilities(
            supportsThinking: template.contains("enable_thinking"),
            supportsReasoningEffort: template.contains("reasoning_effort"),
            supportsPreserveThinking: template.contains("preserve_thinking")
                && template.contains("reasoning_content"),
            defaultPreserveThinking: inferredDefaultPreserveThinking(in: template),
            supportedReasoningEfforts: reasoningEffortMetadata.supported,
            defaultReasoningEffort: reasoningEffortMetadata.defaultValue
        )
    }

    private static func reasoningEffortMetadata(
        in template: String
    ) -> (supported: [LLMReasoningEffort]?, defaultValue: LLMReasoningEffort?) {
        guard template.contains("reasoning_effort") else { return (nil, nil) }

        var rawSupported = Set<String>()
        let membershipPattern = #"\b(?:[A-Za-z_][A-Za-z0-9_]*_)?reasoning_effort\b\s+(?:not\s+)?in\s*[\(\[]([^\)\]]+)[\)\]]"#
        for list in captureValues(matching: membershipPattern, group: 1, in: template) {
            rawSupported.formUnion(quotedValues(in: list))
        }

        let equalityPattern = #"\b(?:[A-Za-z_][A-Za-z0-9_]*_)?reasoning_effort\b(?:\s*\|\s*(?:lower|upper))*\s*==\s*([\"'])([^\"']+)\1"#
        rawSupported.formUnion(captureValues(matching: equalityPattern, group: 2, in: template))

        let defaultValue = inferredDefaultReasoningEffort(in: template)
        if !rawSupported.isEmpty, let defaultValue {
            rawSupported.insert(defaultValue.rawValue)
        }

        let recognizedSupported = LLMReasoningEffort.allCases.filter {
            rawSupported.contains($0.rawValue)
        }
        let supported: [LLMReasoningEffort]? = recognizedSupported.isEmpty
            ? nil
            : recognizedSupported

        return (supported, defaultValue)
    }

    private static func inferredDefaultReasoningEffort(
        in template: String
    ) -> LLMReasoningEffort? {
        let patterns: [(pattern: String, group: Int)] = [
            (#"\breasoning_effort\b\s*\|\s*default\s*\(\s*([\"'])([^\"']+)\1"#, 2),
            (#"\bset\s+reasoning_effort\s*=\s*reasoning_effort\s+if\s+reasoning_effort\s+is\s+defined\s+else\s+([\"'])([^\"']+)\1"#, 2),
            (#"(?s)\bif\s+(?:not\s+reasoning_effort\s+is\s+defined|reasoning_effort\s+is\s+(?:not\s+defined|undefined)).{0,300}?\bset\s+reasoning_effort\s*=\s*([\"'])([^\"']+)\1"#, 2)
        ]

        for entry in patterns {
            guard let rawValue = captureValues(
                matching: entry.pattern,
                group: entry.group,
                in: template
            ).first else { continue }
            if let effort = LLMReasoningEffort(rawValue: rawValue) {
                return effort
            }
        }
        return nil
    }

    private static func inferredDefaultPreserveThinking(in template: String) -> Bool? {
        guard template.contains("preserve_thinking") else { return nil }

        let patterns: [(pattern: String, group: Int)] = [
            (#"\bpreserve_thinking\b\s*\|\s*default\s*\(?\s*((?i:true|false))"#, 1),
            (#"\bset\s+preserve_thinking\s*=\s*preserve_thinking\s+if\s+preserve_thinking\s+is\s+defined\s+else\s+((?i:true|false))"#, 1),
            (#"(?s)\bif\s+(?:not\s+preserve_thinking\s+is\s+defined|preserve_thinking\s+is\s+(?:not\s+defined|undefined)).{0,300}?\bset\s+preserve_thinking\s*=\s*((?i:true|false))"#, 1),
            (#"\b(?:not\s+preserve_thinking\s+is\s+defined|preserve_thinking\s+is\s+(?:not\s+defined|undefined))\s+or\s+preserve_thinking\s+(?:is|==)\s+((?i:true|false))"#, 1)
        ]

        for entry in patterns {
            guard let rawValue = captureValues(
                matching: entry.pattern,
                group: entry.group,
                in: template
            ).first else { continue }
            if rawValue.lowercased() == "true" { return true }
            if rawValue.lowercased() == "false" { return false }
        }

        let undefinedActsAsTruePattern = #"\b(?:not\s+preserve_thinking\s+is\s+defined|preserve_thinking\s+is\s+(?:not\s+defined|undefined))\s+or\s+preserve_thinking\b(?!\s*(?:is\b|==|!=))"#
        if !captureValues(
            matching: undefinedActsAsTruePattern,
            group: 0,
            in: template
        ).isEmpty {
            return true
        }

        return nil
    }

    private static func quotedValues(in text: String) -> [String] {
        captureValues(
            matching: #"([\"'])([^\"']+)\1"#,
            group: 2,
            in: text
        )
    }

    private static func captureValues(
        matching pattern: String,
        group: Int,
        in text: String
    ) -> [String] {
        guard let expression = try? NSRegularExpression(pattern: pattern) else { return [] }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return expression.matches(in: text, range: range).compactMap { match in
            guard group < match.numberOfRanges,
                  let captureRange = Range(match.range(at: group), in: text)
            else { return nil }
            return String(text[captureRange])
        }
    }
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

/// A sampling parameter accepted by ``GenerationOptions`` and validated before
/// a provider forwards it to a native inference backend.
public enum GenerationOptionParameter: String, Hashable, Sendable {
    case temperature
    case topP
    case topK
    case minP
    case presencePenalty
    case repetitionPenalty
    case thinkingBudgetTokens
}

/// Describes an invalid generation option without relying on a native backend
/// assertion or a trapping Swift numeric conversion.
public enum GenerationOptionsValidationError: Error, LocalizedError, Hashable, Sendable {
    case mustBeFinite(GenerationOptionParameter)
    case mustBeWithinUnitInterval(GenerationOptionParameter)
    case mustNotExceedOne(GenerationOptionParameter)
    case mustBePositive(GenerationOptionParameter)
    case mustBeNonnegative(GenerationOptionParameter)
    case exceedsBackendIntegerLimit(GenerationOptionParameter, maximum: Int)
    case unsafeForBackendFloatingPoint(GenerationOptionParameter)

    public var errorDescription: String? {
        switch self {
        case .mustBeFinite(let parameter):
            return "Generation option '\(parameter.rawValue)' must be finite."
        case .mustBeWithinUnitInterval(let parameter):
            return "Generation option '\(parameter.rawValue)' must be between 0 and 1."
        case .mustNotExceedOne(let parameter):
            return "Generation option '\(parameter.rawValue)' must not exceed 1."
        case .mustBePositive(let parameter):
            return "Generation option '\(parameter.rawValue)' must be greater than 0."
        case .mustBeNonnegative(let parameter):
            return "Generation option '\(parameter.rawValue)' must be nonnegative."
        case .exceedsBackendIntegerLimit(let parameter, let maximum):
            return "Generation option '\(parameter.rawValue)' must not exceed \(maximum)."
        case .unsafeForBackendFloatingPoint(let parameter):
            return "Generation option '\(parameter.rawValue)' cannot be represented or used safely by the native floating-point backend."
        }
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
    /// Optional model-native effort level. nil preserves the chat template's default.
    public var reasoningEffort: LLMReasoningEffort?
    /// Controls whether compatible templates include historical assistant reasoning. nil preserves the template default.
    public var preserveThinking: Bool?
    /// Optional token budget for generated thinking/reasoning content.
    public var thinkingBudgetTokens: Int?
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
        reasoningEffort: LLMReasoningEffort?,
        preserveThinking: Bool? = nil,
        thinkingBudgetTokens: Int? = nil,
        thinkingBudgetMessage: String = "",
        streamPhaseConfiguration: LLMStreamPhaseConfiguration = .automatic
    ) {
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
        self.reasoningEffort = reasoningEffort
        self.preserveThinking = preserveThinking
        self.thinkingBudgetTokens = thinkingBudgetTokens
        self.thinkingBudgetMessage = thinkingBudgetMessage
        self.streamPhaseConfiguration = streamPhaseConfiguration
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
        thinkingBudgetTokens: Int? = nil,
        thinkingBudgetMessage: String = "",
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
            reasoningEffort: nil,
            preserveThinking: nil,
            thinkingBudgetTokens: thinkingBudgetTokens,
            thinkingBudgetMessage: thinkingBudgetMessage,
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
            reasoningEffort: nil,
            preserveThinking: nil,
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
            reasoningEffort: nil,
            preserveThinking: nil,
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

    /// Validates provider-independent sampling semantics.
    ///
    /// Nonpositive temperatures continue to select greedy decoding, nonpositive
    /// `topK` values continue to disable top-k sampling, and nonpositive `minP`
    /// values continue to disable min-p sampling. Those established sentinel
    /// values are therefore accepted as long as they are finite.
    public func validate() throws {
        if let temperature {
            try Self.requireFinite(temperature, parameter: .temperature)
        }

        if let topP {
            try Self.requireFinite(topP, parameter: .topP)
            guard (0...1).contains(topP) else {
                throw GenerationOptionsValidationError.mustBeWithinUnitInterval(.topP)
            }
        }

        if let minP {
            try Self.requireFinite(minP, parameter: .minP)
            if minP > 1 {
                throw GenerationOptionsValidationError.mustNotExceedOne(.minP)
            }
        }

        if let presencePenalty {
            try Self.requireFinite(presencePenalty, parameter: .presencePenalty)
        }

        if let repetitionPenalty {
            try Self.requireFinite(repetitionPenalty, parameter: .repetitionPenalty)
            guard repetitionPenalty > 0 else {
                throw GenerationOptionsValidationError.mustBePositive(.repetitionPenalty)
            }
        }

        if let thinkingBudgetTokens, thinkingBudgetTokens < 0 {
            throw GenerationOptionsValidationError.mustBeNonnegative(.thinkingBudgetTokens)
        }
    }

    /// Adds bounds required by native Float32 sampler APIs. Providers with a
    /// different numeric ABI keep using ``validate()`` alone.
    package func validateForFloat32Backend(maximumTopK: Int) throws {
        try validate()

        if let temperature, temperature > 0 {
            try Self.requireBackendFloat(temperature, parameter: .temperature)
            try Self.requireFiniteBackendReciprocal(temperature, parameter: .temperature)
        }
        if let topP, topP > 0 {
            try Self.requireBackendFloat(topP, parameter: .topP)
        }
        if let topK, topK > 0, topK > maximumTopK {
            throw GenerationOptionsValidationError.exceedsBackendIntegerLimit(
                .topK,
                maximum: maximumTopK
            )
        }
        if let minP, minP > 0 {
            try Self.requireBackendFloat(minP, parameter: .minP)
        }
        if let presencePenalty {
            try Self.requireBackendFloat(presencePenalty, parameter: .presencePenalty)
        }
        if let repetitionPenalty {
            try Self.requireBackendFloat(repetitionPenalty, parameter: .repetitionPenalty)
            try Self.requireFiniteBackendReciprocal(
                repetitionPenalty,
                parameter: .repetitionPenalty
            )
        }
    }

    private static func requireFinite(
        _ value: Double,
        parameter: GenerationOptionParameter
    ) throws {
        guard value.isFinite else {
            throw GenerationOptionsValidationError.mustBeFinite(parameter)
        }
    }

    private static func requireBackendFloat(
        _ value: Double,
        parameter: GenerationOptionParameter
    ) throws {
        let backendValue = Float(value)
        guard backendValue.isFinite,
              value == 0 || backendValue != 0 else {
            throw GenerationOptionsValidationError.unsafeForBackendFloatingPoint(parameter)
        }
    }

    private static func requireFiniteBackendReciprocal(
        _ value: Double,
        parameter: GenerationOptionParameter
    ) throws {
        guard (1 / Float(value)).isFinite else {
            throw GenerationOptionsValidationError.unsafeForBackendFloatingPoint(parameter)
        }
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
        case reasoningEffort
        case preserveThinking
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
        reasoningEffort = try container.decodeIfPresent(LLMReasoningEffort.self, forKey: .reasoningEffort)
        preserveThinking = try container.decodeIfPresent(Bool.self, forKey: .preserveThinking)
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
        try container.encodeIfPresent(reasoningEffort, forKey: .reasoningEffort)
        try container.encodeIfPresent(preserveThinking, forKey: .preserveThinking)
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
