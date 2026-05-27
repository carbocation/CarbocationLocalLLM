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
        startsInThinking: Bool = false,
        requiresSampler: Bool = false
    ) -> ReasoningBudgetPlan? {
        guard options.enableThinking,
              options.thinkingBudgetTokens.map({ $0 >= 0 }) ?? true else {
            return nil
        }
        let budgetTokens = options.thinkingBudgetTokens
            ?? (requiresSampler ? Int(Int32.max) : nil)
        guard let budgetTokens else { return nil }

        if let continuingPair = continuingOpenThinkingPairs.first,
           !continuingPair.close.isEmpty {
            return ReasoningBudgetPlan(
                pair: continuingPair,
                budgetTokens: budgetTokens,
                message: options.thinkingBudgetMessage,
                initialState: .counting
            )
        }

        if let pair = profile.thinkingPairs.first,
           !pair.open.isEmpty,
           !pair.close.isEmpty {
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

        guard let finalMarker = profile.allFinalMarkers.first,
              !finalMarker.isEmpty else {
            return nil
        }
        return ReasoningBudgetPlan(
            pair: OutputDelimiterPair(open: "", close: finalMarker),
            budgetTokens: budgetTokens,
            message: options.thinkingBudgetMessage,
            initialState: .counting
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
        let explicitPairs = profile.thinkingPairs.filter { pair in
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

        if !explicitPairs.isEmpty {
            return explicitPairs
        }

        if let implicitPair = implicitPromptDrivenThinkingPair(
            in: renderedPrompt,
            profile: profile
        ) {
            return [implicitPair]
        }

        return []
    }

    private static func implicitPromptDrivenThinkingPair(
        in renderedPrompt: String,
        profile: OutputSanitizationProfile
    ) -> OutputDelimiterPair? {
        guard renderedPrompt.contains("<|turn>system\n<|think|>\n"),
              renderedPrompt.hasSuffix("<|turn>model\n"),
              let pair = profile.thinkingPairs.first(where: {
                $0.open == "<|channel>thought" && $0.close == "<channel|>"
              }) else {
            return nil
        }

        return OutputDelimiterPair(open: "", close: pair.close)
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

    static func shouldEmitFinalAnswerProgress(
        currentPhase: LLMStreamContentPhase,
        structuredPhase: StructuredOutputPhase?
    ) -> Bool {
        guard currentPhase == .final else {
            return false
        }
        guard let structuredPhase else {
            return true
        }

        switch structuredPhase {
        case .thinking, .awaitingFinal:
            return false
        case .final, .complete:
            return true
        }
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

    struct PhasedGenerationText: Equatable {
        var thinkingText: String
        var finalText: String
        var phaseSegments: [LLMGenerationPhaseSegment]
        var rawGeneratedText: String

        func replacingPhaseText(
            _ phase: LLMStreamContentPhase,
            with text: String
        ) -> PhasedGenerationText {
            guard phase == .thinking || phase == .final else { return self }
            let currentText = joinedText(for: phase)
            guard text != currentText else { return self }

            let segments = replacingSegments(for: phase, with: text)
            return PhasedGenerationText(
                thinkingText: phase == .thinking ? text : thinkingText,
                finalText: phase == .final ? text : finalText,
                phaseSegments: segments,
                rawGeneratedText: rawGeneratedText
            )
        }

        private func replacingSegments(
            for phase: LLMStreamContentPhase,
            with text: String
        ) -> [LLMGenerationPhaseSegment] {
            let phaseSegmentsWithIndices = phaseSegments.enumerated().filter { _, segment in
                segment.phase == phase
            }

            guard !text.isEmpty else {
                return phaseSegments.filter { $0.phase != phase }
            }

            let replacementTexts = replacementTextsPreservingBoundaries(
                originalTexts: phaseSegmentsWithIndices.map { $0.element.text },
                replacementText: text
            ) ?? [text]

            guard let firstPhaseIndex = phaseSegmentsWithIndices.first?.offset else {
                var segments = phaseSegments
                let insertionIndex = phase == .thinking
                    ? segments.firstIndex { $0.phase == .final } ?? segments.endIndex
                    : segments.endIndex
                segments.insert(contentsOf: replacementTexts.map {
                    LLMGenerationPhaseSegment(phase: phase, text: $0)
                }, at: insertionIndex)
                return segments
            }

            if replacementTexts.count == phaseSegmentsWithIndices.count {
                var replacementIndex = 0
                return phaseSegments.map { segment in
                    guard segment.phase == phase else { return segment }
                    defer { replacementIndex += 1 }
                    return LLMGenerationPhaseSegment(
                        phase: phase,
                        text: replacementTexts[replacementIndex]
                    )
                }
            }

            var insertedReplacement = false
            var segments: [LLMGenerationPhaseSegment] = []
            for (index, segment) in phaseSegments.enumerated() {
                guard segment.phase == phase else {
                    segments.append(segment)
                    continue
                }

                if index == firstPhaseIndex {
                    segments.append(contentsOf: replacementTexts.map {
                        LLMGenerationPhaseSegment(phase: phase, text: $0)
                    })
                    insertedReplacement = true
                } else if insertedReplacement {
                    continue
                }
            }

            return segments
        }

        private func replacementTextsPreservingBoundaries(
            originalTexts: [String],
            replacementText: String
        ) -> [String]? {
            let originalTexts = originalTexts.filter { !$0.isEmpty }
            guard !originalTexts.isEmpty else { return nil }

            let joinedOriginal = originalTexts.joined(separator: "\n")
            if replacementText == joinedOriginal {
                return originalTexts
            }

            let newlineParts = replacementText.components(separatedBy: "\n")
            if newlineParts.count == originalTexts.count,
               newlineParts.allSatisfy({ !$0.isEmpty }) {
                return newlineParts
            }

            guard let aligned = replacementTextsByExactAlignment(
                originalTexts: originalTexts,
                replacementText: replacementText
            ) else {
                return nil
            }

            return aligned.joined(separator: "\n") == replacementText
                ? aligned
                : nil
        }

        private func replacementTextsByExactAlignment(
            originalTexts: [String],
            replacementText: String
        ) -> [String]? {
            var cursor = replacementText.startIndex
            var aligned: [String] = []

            for originalText in originalTexts {
                guard let range = replacementText.range(
                    of: originalText,
                    range: cursor..<replacementText.endIndex
                ) else {
                    return nil
                }

                if range.lowerBound > cursor {
                    let interstitial = String(replacementText[cursor..<range.lowerBound])
                    if interstitial == "\n", !aligned.isEmpty {
                        // The result joiner already represents segment boundaries with a newline.
                    } else if aligned.isEmpty {
                        aligned.append(interstitial + originalText)
                    } else {
                        aligned[aligned.count - 1] += interstitial
                        aligned.append(originalText)
                    }
                } else {
                    aligned.append(originalText)
                }

                cursor = range.upperBound
            }

            if cursor < replacementText.endIndex {
                guard !aligned.isEmpty else { return nil }
                aligned[aligned.count - 1] += replacementText[cursor..<replacementText.endIndex]
            }

            return aligned
        }

        private func joinedText(for phase: LLMStreamContentPhase) -> String {
            phaseSegments
                .filter { $0.phase == phase }
                .map(\.text)
                .joined(separator: "\n")
        }

        func generationResult(
            stopReason: String,
            promptTokens: Int,
            generatedTokens: Int,
            templateMode: LLMChatTemplateMode,
            accelerationStats: LLMGenerationAccelerationStats?
        ) -> LLMGenerationResult {
            LLMGenerationResult(
                thinkingText: thinkingText,
                finalText: finalText,
                phaseSegments: phaseSegments,
                stopReason: stopReason,
                promptTokens: promptTokens,
                generatedTokens: generatedTokens,
                templateMode: templateMode,
                accelerationStats: accelerationStats,
                rawGeneratedText: rawGeneratedText
            )
        }
    }

    static func phasedGeneratedText(
        _ text: String,
        plan: StreamPhasePlan
    ) -> PhasedGenerationText {
        var parser = PhasedGenerationTextParser(text: text, plan: plan)
        return parser.parse()
    }

    private struct PhasedGenerationTextParser {
        let text: String
        let plan: StreamPhasePlan
        var index: String.Index
        var segments: [LLMGenerationPhaseSegment] = []
        var hasSeenFinalMarker = false

        init(text: String, plan: StreamPhasePlan) {
            self.text = text
            self.plan = plan
            self.index = text.startIndex
        }

        mutating func parse() -> PhasedGenerationText {
            parseContinuingThinkingPrefix()
            guard index < text.endIndex else {
                return snapshot()
            }

            if plan.startsInThinking == true,
               plan.continuingOpenThinkingPairs.isEmpty,
               let pair = plan.profile.thinkingPairs.first(where: { !$0.close.isEmpty }) {
                guard parseThinkingUntilClose(pair.close) else {
                    return snapshot()
                }
            }

            while index < text.endIndex {
                let contentStart = LlamaEngine.indexAfterWhitespace(in: text, from: index)

                if let pair = prefixedThinkingPair(at: contentStart) {
                    if contentStart > index {
                        index = contentStart
                    }
                    index = text.index(index, offsetBy: pair.open.count)
                    guard parseThinkingUntilClose(pair.close) else {
                        return snapshot()
                    }
                    continue
                }

                if let markerRange = LlamaEngine.firstFinalMarkerRange(
                    in: text,
                    markers: plan.profile.allFinalMarkers,
                    range: index..<text.endIndex
                ) {
                    appendSegment(defaultPhase, text[index..<markerRange.lowerBound])
                    index = markerRange.upperBound
                    hasSeenFinalMarker = true
                    continue
                }

                let markers = plan.profile.thinkingPairs.map(\.open) + plan.profile.allFinalMarkers
                if let partialMarkerRange = trailingPartialMarkerPrefixRange(
                    in: index..<text.endIndex,
                    markers: markers
                ) {
                    appendSegment(defaultPhase, text[index..<partialMarkerRange.lowerBound])
                    index = text.endIndex
                    continue
                }

                appendSegment(defaultPhase, text[index..<text.endIndex])
                index = text.endIndex
            }

            return snapshot()
        }

        private mutating func parseContinuingThinkingPrefix() {
            for pair in plan.continuingOpenThinkingPairs where index < text.endIndex {
                guard !pair.close.isEmpty,
                      let closeRange = text.range(of: pair.close, range: index..<text.endIndex) else {
                    appendSegment(.thinking, text[index..<text.endIndex])
                    index = text.endIndex
                    return
                }

                appendSegment(.thinking, text[index..<closeRange.lowerBound])
                index = closeRange.upperBound
            }
        }

        private mutating func parseThinkingUntilClose(_ close: String) -> Bool {
            guard !close.isEmpty,
                  let closeRange = text.range(of: close, range: index..<text.endIndex) else {
                appendSegment(.thinking, text[index..<text.endIndex])
                index = text.endIndex
                return false
            }

            appendSegment(.thinking, text[index..<closeRange.lowerBound])
            index = closeRange.upperBound
            return true
        }

        private func prefixedThinkingPair(at start: String.Index) -> OutputDelimiterPair? {
            guard start < text.endIndex else { return nil }
            return plan.profile.thinkingPairs.first { pair in
                !pair.open.isEmpty && text[start...].hasPrefix(pair.open)
            }
        }

        private var defaultPhase: LLMStreamContentPhase {
            (hasSeenFinalMarker || plan.profile.allFinalMarkers.isEmpty) ? .final : .thinking
        }

        private func trailingPartialMarkerPrefixRange(
            in range: Range<String.Index>,
            markers: [String]
        ) -> Range<String.Index>? {
            guard range.lowerBound < range.upperBound else { return nil }
            let candidate = String(text[range])
            var bestRange: Range<String.Index>?
            var bestLength = 0

            for marker in markers where !marker.isEmpty {
                let maxLength = min(marker.count - 1, candidate.count)
                guard maxLength > 0 else { continue }

                for length in stride(from: maxLength, through: 1, by: -1) {
                    let prefix = String(marker.prefix(length))
                    guard candidate.hasSuffix(prefix), length > bestLength else {
                        continue
                    }

                    let lowerBound = text.index(range.upperBound, offsetBy: -length)
                    bestRange = lowerBound..<range.upperBound
                    bestLength = length
                    break
                }
            }

            return bestRange
        }

        private mutating func appendSegment(
            _ phase: LLMStreamContentPhase,
            _ substring: Substring
        ) {
            guard phase == .thinking || phase == .final else { return }
            let sanitized = sanitizedPhaseText(String(substring))
            guard !sanitized.isEmpty else { return }
            segments.append(LLMGenerationPhaseSegment(phase: phase, text: sanitized))
        }

        private func sanitizedPhaseText(_ value: String) -> String {
            var output = value
            for token in plan.profile.scrubTokens {
                output = output.replacingOccurrences(of: token, with: "")
            }
            return output.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        private func snapshot() -> PhasedGenerationText {
            let thinkingText = joinedText(for: .thinking)
            let finalText = joinedText(for: .final)
            return PhasedGenerationText(
                thinkingText: thinkingText,
                finalText: finalText,
                phaseSegments: segments,
                rawGeneratedText: text
            )
        }

        private func joinedText(for phase: LLMStreamContentPhase) -> String {
            segments
                .filter { $0.phase == phase }
                .map(\.text)
                .joined(separator: "\n")
        }
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
        let nativeToolRange = stopAtBalancedJSON
            ? nativeToolCallEnvelopeRange(in: text)
            : nil

        if let stopRange = firstStopSequenceRange(in: text, stopSequences: stopSequences) {
            boundaryIndex = stopRange.lowerBound
            boundaryText = String(text[..<stopRange.lowerBound])
            reason = "stop-sequence"
        }

        if stopAtBalancedJSON,
           let nativeToolRange,
           boundaryIndex.map({ nativeToolRange.upperBound < $0 }) ?? true {
            boundaryIndex = nativeToolRange.upperBound
            boundaryText = String(text[..<nativeToolRange.upperBound])
            reason = "tool-call-complete"
        }

        if stopAtBalancedJSON,
           !hasUnclosedNativeToolCallEnvelope(in: text),
           let jsonRange = balancedJSONValueRange(in: text),
           nativeToolRange.map({ !$0.contains(jsonRange.lowerBound) }) ?? true,
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

    static func nativeToolCallEnvelopeRange(in text: String) -> Range<String.Index>? {
        let startMarker = "<|tool_call>call:"
        let endMarker = "<tool_call|>"
        guard let firstStart = text.range(of: startMarker)?.lowerBound else {
            return nil
        }

        var searchStart = firstStart
        var lastCompleteEnd: String.Index?
        while let startRange = text.range(of: startMarker, range: searchStart..<text.endIndex) {
            guard let endRange = text.range(of: endMarker, range: startRange.upperBound..<text.endIndex) else {
                break
            }

            lastCompleteEnd = endRange.upperBound
            let nextNonWhitespace = indexAfterWhitespace(in: text, from: endRange.upperBound)
            guard nextNonWhitespace < text.endIndex,
                  text[nextNonWhitespace..<text.endIndex].hasPrefix(startMarker) else {
                break
            }
            searchStart = nextNonWhitespace
        }

        guard let lastCompleteEnd else { return nil }
        return firstStart..<lastCompleteEnd
    }

    static func hasUnclosedNativeToolCallEnvelope(in text: String) -> Bool {
        let startMarker = "<|tool_call>call:"
        let endMarker = "<tool_call|>"
        guard let lastStart = text.range(of: startMarker, options: .backwards) else {
            return false
        }
        return text.range(of: endMarker, range: lastStart.upperBound..<text.endIndex) == nil
    }

}
