import CarbocationLocalLLM
import Foundation

extension LlamaEngine {
    static func generationGrammarMode(
        for options: GenerationOptions,
        profile: OutputSanitizationProfile,
        continuingOpenThinkingPairs: [OutputDelimiterPair]
    ) -> GenerationGrammarMode {
        guard let grammar = options.grammar else { return .none }

        let canStageStructuredOutput = !profile.thinkingPairs.isEmpty
            || !profile.allFinalMarkers.isEmpty
            || !continuingOpenThinkingPairs.isEmpty
        guard options.enableThinking, canStageStructuredOutput else {
            return .eager(grammar: grammar)
        }

        let triggerPatterns = lazyGrammarTriggerPatterns(
            profile: profile,
            continuingOpenThinkingPairs: continuingOpenThinkingPairs
        )
        guard !triggerPatterns.isEmpty else {
            return .eager(grammar: grammar)
        }
        return .lazy(grammar: grammar, triggerPatterns: triggerPatterns)
    }

    static func reasoningBudgetPlan(
        for options: GenerationOptions,
        profile: OutputSanitizationProfile,
        continuingOpenThinkingPairs: [OutputDelimiterPair],
        startsInThinking: Bool = false
    ) -> ReasoningBudgetPlan? {
        guard options.enableThinking,
              let budgetTokens = options.thinkingBudgetTokens,
              budgetTokens >= 0 else {
            return nil
        }

        if let continuingPair = continuingOpenThinkingPairs.first,
           !continuingPair.close.isEmpty {
            return ReasoningBudgetPlan(
                pair: continuingPair,
                budgetTokens: budgetTokens,
                message: options.thinkingBudgetMessage,
                initialState: .counting
            )
        }

        guard let pair = profile.thinkingPairs.first,
              !pair.open.isEmpty,
              !pair.close.isEmpty else {
            return nil
        }

        if startsInThinking {
            return ReasoningBudgetPlan(
                pair: pair,
                budgetTokens: budgetTokens,
                message: options.thinkingBudgetMessage,
                initialState: .counting
            )
        }

        return ReasoningBudgetPlan(
            pair: pair,
            budgetTokens: budgetTokens,
            message: options.thinkingBudgetMessage,
            initialState: .idle
        )
    }

    static func lazyGrammarTriggerPatterns(
        profile: OutputSanitizationProfile,
        continuingOpenThinkingPairs: [OutputDelimiterPair]
    ) -> [String] {
        var patterns: [String] = []
        let jsonStartCapture = #"\s*(\{|\[)"#

        for pair in continuingOpenThinkingPairs {
            patterns.append(regexEscaped(pair.close) + jsonStartCapture)
        }

        for pair in profile.thinkingPairs {
            patterns.append(regexEscaped(pair.close) + jsonStartCapture)
        }

        for marker in profile.allFinalMarkers {
            patterns.append(regexEscaped(marker) + jsonStartCapture)
        }

        patterns.append(#"^\s*(\{|\[)"#)

        var deduplicated: [String] = []
        for pattern in patterns where !deduplicated.contains(pattern) {
            deduplicated.append(pattern)
        }
        return deduplicated
    }

    static func regexEscaped(_ literal: String) -> String {
        var escaped = ""
        for character in literal {
            if #"\\.^$|?*+()[]{}"#.contains(character) {
                escaped.append("\\")
            }
            escaped.append(character)
        }
        return escaped
    }

    static func mergingStopSequences(_ callerStops: [String], _ templateStops: [String]) -> [String] {
        var merged: [String] = []
        for stop in callerStops + templateStops where !stop.isEmpty && !merged.contains(stop) {
            merged.append(stop)
        }
        return merged
    }

    static func continuingOpenThinkingPairs(
        in renderedPrompt: String,
        profile: OutputSanitizationProfile
    ) -> [OutputDelimiterPair] {
        profile.thinkingPairs.filter { pair in
            guard let openRange = renderedPrompt.range(of: pair.open, options: .backwards) else {
                return false
            }
            guard let closeRange = renderedPrompt.range(of: pair.close, options: .backwards) else {
                return renderedPrompt[openRange.upperBound...]
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .isEmpty
            }
            return closeRange.lowerBound < openRange.lowerBound
                && renderedPrompt[openRange.upperBound...]
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .isEmpty
        }
    }

    static func streamContentPhase(
        in text: String,
        plan: StreamPhasePlan
    ) -> LLMStreamContentPhase {
        guard plan.hasPhaseMarkers else {
            return .final
        }

        if text.isEmpty {
            if plan.startsInThinking == true || !plan.continuingOpenThinkingPairs.isEmpty {
                return .thinking
            }
            if plan.startsInThinking == false {
                return .final
            }
            return .unknown
        }

        var lowerBound = text.startIndex
        let initialThinkingCloseMarkers: [String]
        if !plan.continuingOpenThinkingPairs.isEmpty {
            initialThinkingCloseMarkers = plan.continuingOpenThinkingPairs.map(\.close)
        } else if plan.startsInThinking == true,
                  let close = plan.profile.thinkingPairs.first?.close,
                  !close.isEmpty {
            initialThinkingCloseMarkers = [close]
        } else {
            initialThinkingCloseMarkers = []
        }

        for close in initialThinkingCloseMarkers where !close.isEmpty {
            guard let closeRange = text.range(of: close, range: lowerBound..<text.endIndex) else {
                return .thinking
            }
            lowerBound = closeRange.upperBound
        }

        return streamContentPhaseAfterInitialThinking(in: text, from: lowerBound, plan: plan)
    }

    private static func streamContentPhaseAfterInitialThinking(
        in text: String,
        from start: String.Index,
        plan: StreamPhasePlan
    ) -> LLMStreamContentPhase {
        let lowerBound = indexAfterWhitespace(in: text, from: start)
        guard lowerBound < text.endIndex else {
            return plan.profile.allFinalMarkers.isEmpty ? .final : .unknown
        }

        for pair in plan.profile.thinkingPairs where !pair.open.isEmpty {
            if text[lowerBound...].hasPrefix(pair.open) {
                guard !pair.close.isEmpty,
                      let closeRange = text.range(
                        of: pair.close,
                        range: lowerBound..<text.endIndex
                      ) else {
                    return .thinking
                }
                return streamContentPhaseAfterInitialThinking(
                    in: text,
                    from: closeRange.upperBound,
                    plan: plan
                )
            }
        }

        let markers = plan.profile.thinkingPairs.map(\.open) + plan.profile.allFinalMarkers
        if isPossibleMarkerPrefix(in: text, from: lowerBound, markers: markers) {
            return .unknown
        }

        if firstFinalMarkerRange(
            in: text,
            markers: plan.profile.allFinalMarkers,
            range: lowerBound..<text.endIndex
        ) != nil {
            return .final
        }

        return plan.profile.allFinalMarkers.isEmpty ? .final : .thinking
    }

    private static func isPossibleMarkerPrefix(
        in text: String,
        from start: String.Index,
        markers: [String]
    ) -> Bool {
        let suffix = String(text[start...])
        return markers.contains { marker in
            !suffix.isEmpty
                && suffix.count < marker.count
                && marker.hasPrefix(suffix)
        }
    }

    static func structuredOutputPhase(
        in text: String,
        plan: StructuredOutputPlan
    ) -> StructuredOutputPhase {
        guard structuredFinalOutputSearchRange(in: text, plan: plan) == nil else {
            return .final
        }
        return structuredOutputIsInsideThinking(in: text, plan: plan)
            ? .thinking
            : .awaitingFinal
    }

    private static func structuredOutputIsInsideThinking(
        in text: String,
        plan: StructuredOutputPlan
    ) -> Bool {
        var lowerBound = text.startIndex
        for pair in plan.continuingOpenThinkingPairs {
            guard let closeRange = text.range(of: pair.close, range: lowerBound..<text.endIndex) else {
                return true
            }
            lowerBound = closeRange.upperBound
        }

        let trimmedLowerBound = indexAfterWhitespace(in: text, from: lowerBound)
        for pair in plan.profile.thinkingPairs where text[trimmedLowerBound...].hasPrefix(pair.open) {
            return text.range(of: pair.close, range: trimmedLowerBound..<text.endIndex) == nil
        }
        return false
    }

    static func firstStructuredGenerationBoundary(
        in text: String,
        stopSequences: [String],
        stopAtBalancedJSON: Bool,
        plan: StructuredOutputPlan
    ) -> GenerationBoundary? {
        let phase = structuredOutputPhase(in: text, plan: plan)
        let activeStopSequences = stopSequencesForStructuredPhase(
            stopSequences,
            phase: phase,
            plan: plan
        )
        var boundaryIndex: String.Index?
        var boundaryText: String?
        var reason: String?

        if let stopRange = firstStopSequenceRange(in: text, stopSequences: activeStopSequences) {
            boundaryIndex = stopRange.lowerBound
            boundaryText = String(text[..<stopRange.lowerBound])
            reason = "stop-sequence"
        }

        if stopAtBalancedJSON,
           let finalSearchRange = structuredFinalOutputSearchRange(in: text, plan: plan),
           let jsonRange = balancedJSONValueRange(in: text, searchRange: finalSearchRange),
           boundaryIndex.map({ jsonRange.upperBound < $0 }) ?? true {
            boundaryIndex = jsonRange.upperBound
            boundaryText = String(text[..<jsonRange.upperBound])
            reason = "json-complete"
        }

        guard boundaryIndex != nil, let boundaryText, let reason else {
            return nil
        }
        return GenerationBoundary(text: boundaryText, reason: reason)
    }

    private static func stopSequencesForStructuredPhase(
        _ stopSequences: [String],
        phase: StructuredOutputPhase,
        plan: StructuredOutputPlan
    ) -> [String] {
        guard phase == .thinking || phase == .awaitingFinal else {
            return stopSequences
        }

        let structuralStops = Set(
            plan.profile.thinkingPairs.map(\.close)
                + plan.continuingOpenThinkingPairs.map(\.close)
                + plan.profile.allFinalMarkers
        )
        return stopSequences.filter { !structuralStops.contains($0) }
    }

    static func structuredFinalOutputSearchRange(
        in text: String,
        plan: StructuredOutputPlan
    ) -> Range<String.Index>? {
        guard !text.isEmpty else { return nil }

        var lowerBound = text.startIndex
        for pair in plan.continuingOpenThinkingPairs {
            guard let closeRange = text.range(of: pair.close, range: lowerBound..<text.endIndex) else {
                return nil
            }
            lowerBound = closeRange.upperBound
        }

        guard let afterThinkingBlocks = indexAfterGeneratedThinkingPrefix(
            in: text,
            from: lowerBound,
            pairs: plan.profile.thinkingPairs
        ) else {
            return nil
        }
        lowerBound = afterThinkingBlocks

        if let jsonStart = immediateJSONStartIndex(in: text, from: lowerBound) {
            return jsonStart..<text.endIndex
        }

        if let markerRange = firstFinalMarkerRange(
            in: text,
            markers: plan.profile.allFinalMarkers,
            range: lowerBound..<text.endIndex
        ) {
            lowerBound = markerRange.upperBound
            guard let afterMarkerThinkingBlocks = indexAfterGeneratedThinkingPrefix(
                in: text,
                from: lowerBound,
                pairs: plan.profile.thinkingPairs
            ) else {
                return nil
            }
            lowerBound = afterMarkerThinkingBlocks

            if let jsonStart = immediateJSONStartIndex(in: text, from: lowerBound) {
                return jsonStart..<text.endIndex
            }
        }

        return nil
    }

    private static func firstFinalMarkerRange(
        in text: String,
        markers: [String],
        range: Range<String.Index>
    ) -> Range<String.Index>? {
        markers.compactMap { marker in
            text.range(of: marker, range: range)
        }
        .min { lhs, rhs in lhs.lowerBound < rhs.lowerBound }
    }

    private static func indexAfterGeneratedThinkingPrefix(
        in text: String,
        from start: String.Index,
        pairs: [OutputDelimiterPair]
    ) -> String.Index? {
        var lowerBound = indexAfterWhitespace(in: text, from: start)
        var strippedBlock = true

        while strippedBlock {
            strippedBlock = false
            for pair in pairs where text[lowerBound...].hasPrefix(pair.open) {
                guard let closeRange = text.range(of: pair.close, range: lowerBound..<text.endIndex) else {
                    return nil
                }
                lowerBound = indexAfterWhitespace(in: text, from: closeRange.upperBound)
                strippedBlock = true
                break
            }
        }

        return lowerBound
    }

    private static func immediateJSONStartIndex(
        in text: String,
        from start: String.Index
    ) -> String.Index? {
        let lowerBound = indexAfterWhitespace(in: text, from: start)
        guard lowerBound < text.endIndex else { return nil }
        return text[lowerBound] == "{" || text[lowerBound] == "["
            ? lowerBound
            : nil
    }

    private static func indexAfterWhitespace(
        in text: String,
        from start: String.Index
    ) -> String.Index {
        var index = start
        while index < text.endIndex, text[index].isWhitespace {
            index = text.index(after: index)
        }
        return index
    }

    static func sanitizedGeneratedText(
        _ accumulatedText: String,
        profile: OutputSanitizationProfile,
        continuingOpenThinkingPairs: [OutputDelimiterPair],
        requiresNonEmptyStructuredOutput: Bool
    ) throws -> String {
        let returnedText = profile.isEmpty
            ? accumulatedText
            : LLMResponseSanitizer.unwrapStructuredOutput(
                accumulatedText,
                using: profile,
                continuingOpenThinkingPairs: continuingOpenThinkingPairs
            )

        if requiresNonEmptyStructuredOutput,
           !accumulatedText.isEmpty,
           returnedText.isEmpty {
            throw LLMEngineError.structuredOutputPhaseFailed(
                "Sanitization removed all generated structured output."
            )
        }

        return returnedText
    }

    static func trimmingAtFirstStopSequence(
        _ text: String,
        stopSequences: [String]
    ) -> String? {
        guard let earliest = firstStopSequenceRange(in: text, stopSequences: stopSequences) else {
            return nil
        }
        return String(text[..<earliest.lowerBound])
    }

    static func firstStopSequenceRange(
        in text: String,
        stopSequences: [String]
    ) -> Range<String.Index>? {
        stopSequences
            .filter { !$0.isEmpty }
            .compactMap { sequence -> Range<String.Index>? in
                text.range(of: sequence)
            }
            .min(by: { $0.lowerBound < $1.lowerBound })
    }

    struct GenerationBoundary: Equatable {
        var text: String
        var reason: String
    }

    static func firstGenerationBoundary(
        in text: String,
        stopSequences: [String],
        stopAtBalancedJSON: Bool
    ) -> GenerationBoundary? {
        var boundaryIndex: String.Index?
        var boundaryText: String?
        var reason: String?

        if let stopRange = firstStopSequenceRange(in: text, stopSequences: stopSequences) {
            boundaryIndex = stopRange.lowerBound
            boundaryText = String(text[..<stopRange.lowerBound])
            reason = "stop-sequence"
        }

        if stopAtBalancedJSON,
           let jsonRange = balancedJSONValueRange(in: text),
           boundaryIndex.map({ jsonRange.upperBound < $0 }) ?? true {
            boundaryIndex = jsonRange.upperBound
            boundaryText = String(text[jsonRange])
            reason = "json-complete"
        }

        guard boundaryIndex != nil, let boundaryText, let reason else {
            return nil
        }
        return GenerationBoundary(text: boundaryText, reason: reason)
    }

    static func balancedJSONValueRange(in text: String) -> Range<String.Index>? {
        balancedJSONValueRange(in: text, searchRange: text.startIndex..<text.endIndex)
    }

    static func balancedJSONValueRange(
        in text: String,
        searchRange: Range<String.Index>
    ) -> Range<String.Index>? {
        var startIndex: String.Index?
        var expectedClosers: [Character] = []
        var inString = false
        var escaped = false

        var index = searchRange.lowerBound
        while index < searchRange.upperBound {
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
                } else if character == "}" || character == "]" {
                    guard expectedClosers.last == character else {
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
