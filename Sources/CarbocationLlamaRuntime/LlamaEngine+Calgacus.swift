import CarbocationLocalLLM
import Foundation
import llama

extension LlamaEngine {
    public func encodeCalgacus(
        _ request: CalgacusEncodeRequest,
        onEvent: @Sendable (CalgacusEvent) -> Void
    ) async throws -> CalgacusEncodeResult {
        guard let context, let vocabulary else {
            throw LLMEngineError.noModelLoaded
        }
        clearPromptCaches()

        let startedAt = Date()
        onEvent(.started(operation: "encode"))

        let secretTokens = try tokenize(vocab: vocabulary, text: request.secretText, addSpecial: false)
        guard !secretTokens.isEmpty else {
            throw CalgacusError.emptySecretText
        }
        onEvent(.tokensPrepared(operation: "secret", count: secretTokens.count))

        let secretContextTokens = try calgacusInitialTokens(vocab: vocabulary)
        try Self.calgacusValidateBudget(
            operation: "Secret ranking",
            contextSize: currentContextSize(),
            contextTokenCount: secretContextTokens.count,
            payloadTokenCount: secretTokens.count
        )

        let secretTrace = try calgacusTrace(
            tokens: secretTokens,
            initialContextTokens: secretContextTokens,
            context: context,
            vocabulary: vocabulary,
            stage: .secretRanking,
            onEvent: onEvent
        )
        let ranks = secretTrace.map(\.rank)

        let coverPromptTokens = try calgacusPromptTokens(vocab: vocabulary, text: request.coverPrompt)
        try Self.calgacusValidateBudget(
            operation: "Cover generation",
            contextSize: currentContextSize(),
            contextTokenCount: coverPromptTokens.count,
            payloadTokenCount: ranks.count
        )

        let coverPayload = try calgacusSelectTokens(
            ranks: ranks,
            initialContextTokens: coverPromptTokens,
            context: context,
            vocabulary: vocabulary,
            stage: .coverGeneration,
            operation: "Cover generation",
            rejectsControlTokens: true,
            onEvent: onEvent
        )

        guard let coverText = String(data: coverPayload.data, encoding: .utf8) else {
            throw CalgacusError.textRenderingFailed(operation: "Cover generation")
        }

        let verification = try calgacusDecodePayload(
            coverText: coverText,
            coverPrompt: request.coverPrompt,
            context: context,
            vocabulary: vocabulary,
            emitsEvents: false,
            onEvent: onEvent
        )
        guard verification.recoveredTokens == secretTokens else {
            throw CalgacusError.verificationFailed(
                expectedTokenCount: secretTokens.count,
                recoveredTokenCount: verification.recoveredTokens.count
            )
        }

        onEvent(.completed(operation: "encode", duration: Date().timeIntervalSince(startedAt)))
        return CalgacusEncodeResult(
            coverText: coverText,
            secretTokenCount: secretTokens.count,
            coverTokenCount: coverPayload.tokens.count,
            stats: Self.calgacusStats(for: secretTrace),
            trace: secretTrace
        )
    }

    public func decodeCalgacus(
        _ request: CalgacusDecodeRequest,
        onEvent: @Sendable (CalgacusEvent) -> Void
    ) async throws -> CalgacusDecodeResult {
        guard let context, let vocabulary else {
            throw LLMEngineError.noModelLoaded
        }
        clearPromptCaches()

        let startedAt = Date()
        onEvent(.started(operation: "decode"))
        let payload = try calgacusDecodePayload(
            coverText: request.coverText,
            coverPrompt: request.coverPrompt,
            context: context,
            vocabulary: vocabulary,
            emitsEvents: true,
            onEvent: onEvent
        )
        onEvent(.completed(operation: "decode", duration: Date().timeIntervalSince(startedAt)))
        return payload.result
    }


    static func calgacusRank(of tokenID: Int32, in logits: [Float]) throws -> Int {
        guard tokenID >= 0, Int(tokenID) < logits.count else {
            throw CalgacusError.invalidTokenID(tokenID, vocabularySize: logits.count)
        }

        let targetIndex = Int(tokenID)
        let targetLogit = calgacusComparableLogit(logits[targetIndex])
        var rank = 1
        for index in logits.indices where index != targetIndex {
            let logit = calgacusComparableLogit(logits[index])
            if logit > targetLogit || (logit == targetLogit && index < targetIndex) {
                rank += 1
            }
        }
        return rank
    }

    static func calgacusToken(atRank rank: Int, in logits: [Float]) throws -> Int32 {
        guard rank >= 1, rank <= logits.count else {
            throw CalgacusError.invalidRank(rank, vocabularySize: logits.count)
        }

        let sorted = logits.indices.sorted { lhs, rhs in
            let lhsLogit = calgacusComparableLogit(logits[lhs])
            let rhsLogit = calgacusComparableLogit(logits[rhs])
            if lhsLogit == rhsLogit {
                return lhs < rhs
            }
            return lhsLogit > rhsLogit
        }
        return Int32(sorted[rank - 1])
    }

    static func calgacusNegativeLogProbability(of tokenID: Int32, in logits: [Float]) throws -> Double {
        guard tokenID >= 0, Int(tokenID) < logits.count else {
            throw CalgacusError.invalidTokenID(tokenID, vocabularySize: logits.count)
        }

        let finiteLogits = logits
            .map { Double($0) }
            .filter { $0.isFinite }
        guard let maxLogit = finiteLogits.max() else {
            return .infinity
        }

        let normalizer = finiteLogits.reduce(0.0) { partial, logit in
            partial + exp(logit - maxLogit)
        }
        let targetLogit = Double(calgacusComparableLogit(logits[Int(tokenID)]))
        guard targetLogit.isFinite, normalizer > 0 else {
            return .infinity
        }

        return maxLogit + log(normalizer) - targetLogit
    }

    static func calgacusStats(for trace: [CalgacusTraceEntry]) -> CalgacusRankStats {
        guard !trace.isEmpty else {
            return CalgacusRankStats(
                tokenCount: 0,
                maxRank: 0,
                meanRank: 0,
                medianRank: 0,
                cumulativeNegativeLogProbability: 0,
                averageNegativeLogProbability: 0
            )
        }

        let ranks = trace.map(\.rank).sorted()
        let rankSum = ranks.reduce(0, +)
        let nll = trace.reduce(0.0) { $0 + $1.negativeLogProbability }
        let median: Double
        if ranks.count.isMultiple(of: 2) {
            median = Double(ranks[(ranks.count / 2) - 1] + ranks[ranks.count / 2]) / 2.0
        } else {
            median = Double(ranks[ranks.count / 2])
        }

        return CalgacusRankStats(
            tokenCount: trace.count,
            maxRank: ranks.last ?? 0,
            meanRank: Double(rankSum) / Double(ranks.count),
            medianRank: median,
            cumulativeNegativeLogProbability: nll,
            averageNegativeLogProbability: nll / Double(trace.count)
        )
    }

    static func calgacusValidateBudget(
        operation: String,
        contextSize: Int,
        contextTokenCount: Int,
        payloadTokenCount: Int
    ) throws {
        let requiredTokens = contextTokenCount + payloadTokenCount
        guard requiredTokens <= contextSize else {
            throw CalgacusError.contextBudgetExceeded(
                operation: operation,
                contextSize: contextSize,
                requiredTokens: requiredTokens
            )
        }
    }


    private struct CalgacusSelectedPayload {
        var tokens: [llama_token]
        var data: Data
    }

    private struct CalgacusDecodedPayload {
        var result: CalgacusDecodeResult
        var recoveredTokens: [llama_token]
    }

    private static func calgacusComparableLogit(_ logit: Float) -> Float {
        logit.isFinite ? logit : -.infinity
    }

    private func calgacusDecodePayload(
        coverText: String,
        coverPrompt: String,
        context: OpaquePointer,
        vocabulary: OpaquePointer,
        emitsEvents: Bool,
        onEvent: @Sendable (CalgacusEvent) -> Void
    ) throws -> CalgacusDecodedPayload {
        let coverTokens = try tokenize(vocab: vocabulary, text: coverText, addSpecial: false)
        guard !coverTokens.isEmpty else {
            throw CalgacusError.emptyCoverText
        }
        if emitsEvents {
            onEvent(.tokensPrepared(operation: "cover", count: coverTokens.count))
        }

        let coverPromptTokens = try calgacusPromptTokens(vocab: vocabulary, text: coverPrompt)
        try Self.calgacusValidateBudget(
            operation: "Cover ranking",
            contextSize: currentContextSize(),
            contextTokenCount: coverPromptTokens.count,
            payloadTokenCount: coverTokens.count
        )
        let coverTrace: [CalgacusTraceEntry]
        if emitsEvents {
            coverTrace = try calgacusTrace(
                tokens: coverTokens,
                initialContextTokens: coverPromptTokens,
                context: context,
                vocabulary: vocabulary,
                stage: .coverRanking,
                onEvent: onEvent
            )
        } else {
            coverTrace = try calgacusTrace(
                tokens: coverTokens,
                initialContextTokens: coverPromptTokens,
                context: context,
                vocabulary: vocabulary,
                stage: .verification,
                onEvent: { (_: CalgacusEvent) in }
            )
        }
        let ranks = coverTrace.map(\.rank)

        let secretContextTokens = try calgacusInitialTokens(vocab: vocabulary)
        try Self.calgacusValidateBudget(
            operation: "Secret recovery",
            contextSize: currentContextSize(),
            contextTokenCount: secretContextTokens.count,
            payloadTokenCount: ranks.count
        )
        let recovered: CalgacusSelectedPayload
        if emitsEvents {
            recovered = try calgacusSelectTokens(
                ranks: ranks,
                initialContextTokens: secretContextTokens,
                context: context,
                vocabulary: vocabulary,
                stage: .secretRecovery,
                operation: "Secret recovery",
                rejectsControlTokens: false,
                onEvent: onEvent
            )
        } else {
            recovered = try calgacusSelectTokens(
                ranks: ranks,
                initialContextTokens: secretContextTokens,
                context: context,
                vocabulary: vocabulary,
                stage: .verification,
                operation: "Secret recovery",
                rejectsControlTokens: false,
                onEvent: { (_: CalgacusEvent) in }
            )
        }
        guard let secretText = String(data: recovered.data, encoding: .utf8) else {
            throw CalgacusError.textRenderingFailed(operation: "Secret recovery")
        }

        return CalgacusDecodedPayload(
            result: CalgacusDecodeResult(
                secretText: secretText,
                coverTokenCount: coverTokens.count,
                recoveredTokenCount: recovered.tokens.count,
                stats: Self.calgacusStats(for: coverTrace),
                trace: coverTrace
            ),
            recoveredTokens: recovered.tokens
        )
    }

    private func calgacusTrace(
        tokens: [llama_token],
        initialContextTokens: [llama_token],
        context: OpaquePointer,
        vocabulary: OpaquePointer,
        stage: CalgacusStage,
        onEvent: @Sendable (CalgacusEvent) -> Void
    ) throws -> [CalgacusTraceEntry] {
        try prefillCalgacusContext(initialContextTokens, context: context)

        var trace: [CalgacusTraceEntry] = []
        trace.reserveCapacity(tokens.count)

        for (index, token) in tokens.enumerated() {
            try Task.checkCancellation()
            let logits = try calgacusLogits(context: context, vocabulary: vocabulary)
            let rank = try Self.calgacusRank(of: token, in: logits)
            let nll = try Self.calgacusNegativeLogProbability(of: token, in: logits)
            trace.append(CalgacusTraceEntry(
                index: index,
                tokenID: token,
                tokenText: calgacusDisplayText(for: token, vocab: vocabulary),
                rank: rank,
                negativeLogProbability: nll
            ))
            onEvent(.tokenProcessed(stage: stage, index: index + 1, total: tokens.count, rank: rank))
            try decodeCalgacusToken(token, context: context)
        }

        return trace
    }

    private func calgacusSelectTokens(
        ranks: [Int],
        initialContextTokens: [llama_token],
        context: OpaquePointer,
        vocabulary: OpaquePointer,
        stage: CalgacusStage,
        operation: String,
        rejectsControlTokens: Bool,
        onEvent: @Sendable (CalgacusEvent) -> Void
    ) throws -> CalgacusSelectedPayload {
        try prefillCalgacusContext(initialContextTokens, context: context)

        var tokens: [llama_token] = []
        var data = Data()
        tokens.reserveCapacity(ranks.count)

        for (index, rank) in ranks.enumerated() {
            try Task.checkCancellation()
            let logits = try calgacusLogits(context: context, vocabulary: vocabulary)
            let token = try Self.calgacusToken(atRank: rank, in: logits)
            if rejectsControlTokens {
                if llama_vocab_is_eog(vocabulary, token) {
                    throw CalgacusError.invalidGeneratedCoverToken(rank: rank, tokenID: token, reason: "end-of-generation token")
                }
                if llama_vocab_is_control(vocabulary, token) {
                    throw CalgacusError.invalidGeneratedCoverToken(rank: rank, tokenID: token, reason: "control token")
                }
            }

            let piece = tokenToPiece(vocab: vocabulary, token: token)
            guard !piece.isEmpty else {
                if rejectsControlTokens {
                    throw CalgacusError.invalidGeneratedCoverToken(rank: rank, tokenID: token, reason: "empty rendered token")
                }
                throw CalgacusError.textRenderingFailed(operation: operation)
            }

            tokens.append(token)
            data.append(piece)
            onEvent(.tokenProcessed(stage: stage, index: index + 1, total: ranks.count, rank: rank))
            try decodeCalgacusToken(token, context: context)
        }

        return CalgacusSelectedPayload(tokens: tokens, data: data)
    }

    private func calgacusInitialTokens(vocab: OpaquePointer) throws -> [llama_token] {
        let tokens = try tokenize(vocab: vocab, text: "", addSpecial: true)
        if !tokens.isEmpty {
            return tokens
        }

        let bos = llama_vocab_bos(vocab)
        guard bos >= 0 else {
            throw CalgacusError.noInitialContext
        }
        return [bos]
    }

    private func calgacusPromptTokens(vocab: OpaquePointer, text: String) throws -> [llama_token] {
        let tokens = try tokenize(vocab: vocab, text: text, addSpecial: true)
        if !tokens.isEmpty {
            return tokens
        }
        return try calgacusInitialTokens(vocab: vocab)
    }

    private func prefillCalgacusContext(_ tokens: [llama_token], context: OpaquePointer) throws {
        llama_memory_clear(llama_get_memory(context), false)

        let maxBatchSize = max(1, Int(llama_n_batch(context)))
        for range in Self.prefillRanges(tokenCount: tokens.count, maxBatchSize: maxBatchSize) {
            var chunk = Array(tokens[range])
            let decodeResult = chunk.withUnsafeMutableBufferPointer { buffer -> Int32 in
                let batch = llama_batch_get_one(buffer.baseAddress, Int32(buffer.count))
                return llama_decode(context, batch)
            }
            if decodeResult != 0 {
                throw LLMEngineError.decodeFailed
            }
        }
    }

    private func decodeCalgacusToken(_ token: llama_token, context: OpaquePointer) throws {
        var oneToken: [llama_token] = [token]
        let decodeResult = oneToken.withUnsafeMutableBufferPointer { buffer -> Int32 in
            let batch = llama_batch_get_one(buffer.baseAddress, 1)
            return llama_decode(context, batch)
        }
        if decodeResult != 0 {
            throw LLMEngineError.decodeFailed
        }
    }

    private func calgacusLogits(context: OpaquePointer, vocabulary: OpaquePointer) throws -> [Float] {
        let vocabularySize = Int(llama_vocab_n_tokens(vocabulary))
        guard vocabularySize > 0,
              let logits = llama_get_logits_ith(context, -1)
        else {
            throw CalgacusError.logitsUnavailable
        }

        let buffer = UnsafeBufferPointer(start: logits, count: vocabularySize)
        return Array(buffer)
    }

    private func calgacusDisplayText(for token: llama_token, vocab: OpaquePointer) -> String {
        let data = tokenToPiece(vocab: vocab, token: token)
        if let text = String(data: data, encoding: .utf8), !text.isEmpty {
            return text
        }
        return "<token:\(token)>"
    }
}
