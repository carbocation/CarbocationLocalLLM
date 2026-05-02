public struct OutputDelimiterPair: Equatable, Hashable, Sendable {
    public var open: String
    public var close: String

    public init(open: String, close: String) {
        self.open = open
        self.close = close
    }
}

/// Per-model output handling hints derived from an embedded GGUF chat template.
public struct OutputSanitizationProfile: Equatable, Hashable, Sendable {
    public var thinkingPairs: [OutputDelimiterPair]
    public var sliceAfterMarker: String?
    public var extraStopStrings: [String]
    public var scrubTokens: [String]

    public init(
        thinkingPairs: [OutputDelimiterPair] = [],
        sliceAfterMarker: String? = nil,
        extraStopStrings: [String] = [],
        scrubTokens: [String] = []
    ) {
        self.thinkingPairs = thinkingPairs
        self.sliceAfterMarker = sliceAfterMarker
        self.extraStopStrings = extraStopStrings
        self.scrubTokens = scrubTokens
    }

    public static let empty = OutputSanitizationProfile()

    public var isEmpty: Bool {
        thinkingPairs.isEmpty
            && sliceAfterMarker == nil
            && extraStopStrings.isEmpty
            && scrubTokens.isEmpty
    }

    public static func derived(fromChatTemplate template: String?) -> OutputSanitizationProfile {
        guard let template, !template.isEmpty else { return .empty }

        var profile = OutputSanitizationProfile.empty

        for pair in [
            OutputDelimiterPair(open: "<think>", close: "</think>"),
            OutputDelimiterPair(open: "<|channel>thought", close: "<channel|>"),
            OutputDelimiterPair(open: "<|channel|>thought", close: "<|channel|>"),
            OutputDelimiterPair(open: "<|START_THINKING|>", close: "<|END_THINKING|>")
        ] where template.contains(pair.open) && template.contains(pair.close) {
            profile.appendThinkingPair(pair)
        }

        if template.contains("<|channel|>final<|message|>") {
            profile.sliceAfterMarker = "<|channel|>final<|message|>"
        }

        for token in ["<|return|>", "<|end|>"] where template.contains(token) {
            profile.appendScrubToken(token)
        }

        for stop in ["<turn|>", "<|turn>", "<|im_end|>", "<|im_start|>", "<end_of_turn>", "<start_of_turn>"]
            where template.contains(stop) {
            profile.appendStopString(stop)
        }

        return profile
    }

    private mutating func appendThinkingPair(_ pair: OutputDelimiterPair) {
        guard !thinkingPairs.contains(pair) else { return }
        thinkingPairs.append(pair)
    }

    private mutating func appendStopString(_ stop: String) {
        guard !extraStopStrings.contains(stop) else { return }
        extraStopStrings.append(stop)
    }

    private mutating func appendScrubToken(_ token: String) {
        guard !scrubTokens.contains(token) else { return }
        scrubTokens.append(token)
    }
}
