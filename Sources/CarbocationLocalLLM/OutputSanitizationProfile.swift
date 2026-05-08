public struct OutputDelimiterPair: Codable, Equatable, Hashable, Sendable {
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
    public var finalMarkers: [String]

    public init(
        thinkingPairs: [OutputDelimiterPair] = [],
        sliceAfterMarker: String? = nil,
        extraStopStrings: [String] = [],
        scrubTokens: [String] = [],
        finalMarkers: [String] = []
    ) {
        self.thinkingPairs = thinkingPairs
        self.sliceAfterMarker = sliceAfterMarker
        self.extraStopStrings = extraStopStrings
        self.scrubTokens = scrubTokens
        self.finalMarkers = finalMarkers
    }

    public static let empty = OutputSanitizationProfile()

    public var isEmpty: Bool {
        thinkingPairs.isEmpty
            && sliceAfterMarker == nil
            && extraStopStrings.isEmpty
            && scrubTokens.isEmpty
            && finalMarkers.isEmpty
    }

    public var allFinalMarkers: [String] {
        var markers: [String] = []
        if let sliceAfterMarker, !sliceAfterMarker.isEmpty {
            markers.append(sliceAfterMarker)
        }
        for marker in finalMarkers where !marker.isEmpty && !markers.contains(marker) {
            markers.append(marker)
        }
        return markers
    }

    public func merging(_ configuration: LLMStreamPhaseConfiguration) -> OutputSanitizationProfile {
        var profile = self
        for pair in configuration.thinkingPairs {
            profile.appendThinkingPair(pair)
        }
        for marker in configuration.finalMarkers {
            profile.appendFinalMarker(marker)
        }
        return profile
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
            profile.appendFinalMarker("<|channel|>final<|message|>")
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

    private mutating func appendFinalMarker(_ marker: String) {
        guard !marker.isEmpty, !finalMarkers.contains(marker) else { return }
        finalMarkers.append(marker)
    }
}
