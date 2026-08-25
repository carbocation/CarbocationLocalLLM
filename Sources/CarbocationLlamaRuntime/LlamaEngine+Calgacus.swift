import CarbocationLocalLLM
import Foundation
import llama

extension LlamaEngine {
    /// Scores each model token in `text` against the raw next-token logits
    /// conditioned on every preceding token. The text must fit in the loaded
    /// context; no truncation or sliding window is applied.
    public func tokenLogLikelihoods(
        for text: String,
        onProgress: @escaping @Sendable (LlamaTokenLikelihoodProgress) async -> Void = { _ in }
    ) async throws -> [LlamaTokenLikelihood] {
        try await tokenLogLikelihoods(
            for: text,
            options: LlamaTokenLikelihoodOptions(),
            maximumBatchTokenCount: nil,
            onProgress: onProgress
        )
    }

    /// Scores known text and optionally returns compact statistics from each
    /// full next-token vocabulary distribution. The default overload above
    /// performs no full-vocabulary summary work beyond ordinary likelihood and
    /// rank calculation.
    public func tokenLogLikelihoods(
        for text: String,
        options: LlamaTokenLikelihoodOptions,
        onProgress: @escaping @Sendable (LlamaTokenLikelihoodProgress) async -> Void = { _ in }
    ) async throws -> [LlamaTokenLikelihood] {
        try await tokenLogLikelihoods(
            for: text,
            options: options,
            maximumBatchTokenCount: nil,
            onProgress: onProgress
        )
    }

    func tokenLogLikelihoods(
        for text: String,
        maximumBatchTokenCount: Int?,
        onProgress: @escaping @Sendable (LlamaTokenLikelihoodProgress) async -> Void = { _ in }
    ) async throws -> [LlamaTokenLikelihood] {
        try await tokenLogLikelihoods(
            for: text,
            options: LlamaTokenLikelihoodOptions(),
            maximumBatchTokenCount: maximumBatchTokenCount,
            onProgress: onProgress
        )
    }

    func tokenLogLikelihoods(
        for text: String,
        options: LlamaTokenLikelihoodOptions,
        maximumBatchTokenCount: Int?,
        onProgress: @escaping @Sendable (LlamaTokenLikelihoodProgress) async -> Void = { _ in }
    ) async throws -> [LlamaTokenLikelihood] {
        guard let context, let vocabulary else {
            throw LLMEngineError.noModelLoaded
        }
        let vocabularyStatisticsOptions = options.vocabularyStatistics
        let vocabularyTemperatures = try vocabularyStatisticsOptions.map {
            try Self.calgacusValidatedTemperatures(
                $0.temperatureNormalizationTemperatures
            )
        }
        try acquireGenerationLease()
        defer { endGenerationLease() }
        clearPromptCaches()

        let tokens = try tokenize(vocab: vocabulary, text: text, addSpecial: false)
        await onProgress(LlamaTokenLikelihoodProgress(
            completedTokenCount: 0,
            totalTokenCount: tokens.count
        ))
        guard !tokens.isEmpty else { return [] }

        let initialContextTokens: [llama_token]
        do {
            initialContextTokens = try calgacusInitialTokens(vocab: vocabulary)
        } catch CalgacusError.noInitialContext {
            throw LlamaTokenLikelihoodError.noInitialContext
        }

        // The final observed token does not need to be decoded: the preceding
        // context already provides the logits used to score it.
        let requiredTokens = initialContextTokens.count + max(0, tokens.count - 1)
        let contextSize = currentContextSize()
        guard requiredTokens <= contextSize else {
            throw LlamaTokenLikelihoodError.contextBudgetExceeded(
                contextSize: contextSize,
                requiredTokens: requiredTokens
            )
        }

        guard let scoringInitialToken = initialContextTokens.last else {
            throw LlamaTokenLikelihoodError.noInitialContext
        }

        // Every observed token is already known, so score with teacher forcing:
        // [last initial token, x0, ..., x(n-2)] predicts [x0, ..., x(n-1)].
        // Earlier initial tokens are prefetched once and retained in the KV cache.
        try prefillCalgacusContext(Array(initialContextTokens.dropLast()), context: context)

        let scoringInputs = [scoringInitialToken] + tokens.dropLast()
        let vocabularySize = Int(llama_vocab_n_tokens(vocabulary))
        guard vocabularySize > 0 else {
            throw LlamaTokenLikelihoodError.logitsUnavailable
        }
        let scoringBatchSize = Self.calgacusLikelihoodBatchSize(
            vocabularySize: vocabularySize,
            contextBatchSize: Int(llama_n_batch(context)),
            maximumBatchTokenCount: maximumBatchTokenCount
        )

        var results: [LlamaTokenLikelihood] = []
        results.reserveCapacity(tokens.count)
        var renderedText = Data()

        for range in Self.prefillRanges(
            tokenCount: scoringInputs.count,
            maxBatchSize: scoringBatchSize
        ) {
            try Task.checkCancellation()

            var inputChunk = Array(scoringInputs[range])
            var logitsRequests = [Int8](repeating: 1, count: inputChunk.count)
            let decodeResult = inputChunk.withUnsafeMutableBufferPointer { inputBuffer in
                logitsRequests.withUnsafeMutableBufferPointer { logitsBuffer in
                    var batch = llama_batch_get_one(
                        inputBuffer.baseAddress,
                        Int32(inputBuffer.count)
                    )
                    batch.logits = logitsBuffer.baseAddress
                    return llama_decode(context, batch)
                }
            }
            guard decodeResult == 0 else {
                throw LLMEngineError.decodeFailed
            }

            for localIndex in inputChunk.indices {
                try Task.checkCancellation()

                let index = range.lowerBound + localIndex
                let token = tokens[index]
                guard let logits = llama_get_logits_ith(context, Int32(localIndex)) else {
                    throw LlamaTokenLikelihoodError.logitsUnavailable
                }
                let logitsBuffer = UnsafeBufferPointer(
                    start: logits,
                    count: vocabularySize
                )

                let rank: Int
                let negativeLogProbability: Double
                let vocabularyStatistics: LlamaTokenVocabularyStatistics?
                do {
                    if let vocabularyTemperatures {
                        let metrics = try Self.calgacusTokenMetrics(
                            of: token,
                            in: logitsBuffer,
                            vocabularyTemperatures: vocabularyTemperatures,
                            includesProbabilityMassRank:
                                vocabularyStatisticsOptions?.includesProbabilityMassRank ?? false
                        )
                        rank = metrics.rank
                        negativeLogProbability = metrics.negativeLogProbability
                        vocabularyStatistics = metrics.vocabularyStatistics
                    } else {
                        let metrics = try Self.calgacusTokenMetrics(
                            of: token,
                            in: logitsBuffer
                        )
                        rank = metrics.rank
                        negativeLogProbability = metrics.negativeLogProbability
                        vocabularyStatistics = nil
                    }
                } catch CalgacusError.invalidTokenID(let tokenID, let vocabularySize) {
                    throw LlamaTokenLikelihoodError.invalidTokenID(
                        tokenID,
                        vocabularySize: vocabularySize
                    )
                }

                let tokenBytes = tokenToPiece(vocab: vocabulary, token: token)
                let byteRange = renderedText.count..<(renderedText.count + tokenBytes.count)
                renderedText.append(tokenBytes)
                results.append(LlamaTokenLikelihood(
                    index: index,
                    tokenID: token,
                    tokenBytes: tokenBytes,
                    byteRange: byteRange,
                    logProbability: -negativeLogProbability,
                    rank: rank,
                    vocabularyStatistics: vocabularyStatistics
                ))
            }

            await onProgress(LlamaTokenLikelihoodProgress(
                completedTokenCount: range.upperBound,
                totalTokenCount: tokens.count
            ))
        }

        guard renderedText == Data(text.utf8) else {
            throw LlamaTokenLikelihoodError.tokenRenderingMismatch
        }
        return results
    }

    static func calgacusLikelihoodBatchSize(
        vocabularySize: Int,
        contextBatchSize: Int,
        maximumBatchTokenCount: Int? = nil,
        logitsMemoryBudget: Int = 64 * 1_024 * 1_024
    ) -> Int {
        let validVocabularySize = max(1, vocabularySize)
        let (bytesPerRow, overflowed) = validVocabularySize.multipliedReportingOverflow(
            by: MemoryLayout<Float>.stride
        )
        let memoryBound = overflowed
            ? 1
            : max(1, max(1, logitsMemoryBudget) / max(1, bytesPerRow))
        let requestedBound = maximumBatchTokenCount.map { max(1, $0) } ?? Int.max
        return max(1, min(max(1, contextBatchSize), memoryBound, requestedBound))
    }

    public func encodeCalgacus(
        _ request: CalgacusEncodeRequest,
        onEvent: @Sendable (CalgacusEvent) -> Void
    ) async throws -> CalgacusEncodeResult {
        guard let context, let vocabulary else {
            throw LLMEngineError.noModelLoaded
        }
        try acquireGenerationLease()
        defer { endGenerationLease() }
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
        try acquireGenerationLease()
        defer { endGenerationLease() }
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

    static func calgacusTokenMetrics(
        of tokenID: Int32,
        in logits: UnsafeBufferPointer<Float>
    ) throws -> (rank: Int, negativeLogProbability: Double) {
        guard tokenID >= 0, Int(tokenID) < logits.count else {
            throw CalgacusError.invalidTokenID(tokenID, vocabularySize: logits.count)
        }

        let targetIndex = Int(tokenID)
        let targetLogit = calgacusComparableLogit(logits[targetIndex])
        var rank = 1
        var maxLogit = -Double.infinity

        for index in logits.indices {
            let logit = calgacusComparableLogit(logits[index])
            if index != targetIndex,
               logit > targetLogit || (logit == targetLogit && index < targetIndex) {
                rank += 1
            }
            if logit.isFinite {
                maxLogit = max(maxLogit, Double(logit))
            }
        }

        guard maxLogit.isFinite, targetLogit.isFinite else {
            return (rank, .infinity)
        }

        var normalizer = 0.0
        for rawLogit in logits {
            let logit = calgacusComparableLogit(rawLogit)
            if logit.isFinite {
                normalizer += exp(Double(logit) - maxLogit)
            }
        }
        guard normalizer > 0 else {
            return (rank, .infinity)
        }

        return (
            rank,
            maxLogit + log(normalizer) - Double(targetLogit)
        )
    }

    static func calgacusTokenMetrics(
        of tokenID: Int32,
        in logits: UnsafeBufferPointer<Float>,
        vocabularyTemperatures: [Double],
        includesProbabilityMassRank: Bool = false
    ) throws -> (
        rank: Int,
        negativeLogProbability: Double,
        vocabularyStatistics: LlamaTokenVocabularyStatistics?
    ) {
        guard tokenID >= 0, Int(tokenID) < logits.count else {
            throw CalgacusError.invalidTokenID(tokenID, vocabularySize: logits.count)
        }

        let targetIndex = Int(tokenID)
        let targetLogit = calgacusComparableLogit(logits[targetIndex])
        var rank = 1
        var maxLogit = -Double.infinity

        for index in logits.indices {
            let logit = calgacusComparableLogit(logits[index])
            if index != targetIndex,
               logit > targetLogit || (logit == targetLogit && index < targetIndex) {
                rank += 1
            }
            if logit.isFinite {
                maxLogit = max(maxLogit, Double(logit))
            }
        }

        guard maxLogit.isFinite else {
            return (rank, .infinity, nil)
        }

        var normalizer = 0.0
        var weightedShiftedLogit = 0.0
        var weightedSquaredShiftedLogit = 0.0
        var moreLikelyWeight = 0.0
        var tiedWeight = 0.0
        var temperatureNormalizers = [Double](
            repeating: 0,
            count: vocabularyTemperatures.count
        )

        for rawLogit in logits {
            let logit = calgacusComparableLogit(rawLogit)
            guard logit.isFinite else { continue }

            let shiftedLogit = Double(logit) - maxLogit
            let weight = exp(shiftedLogit)
            normalizer += weight
            weightedShiftedLogit += weight * shiftedLogit
            weightedSquaredShiftedLogit += weight * shiftedLogit * shiftedLogit
            if includesProbabilityMassRank {
                if logit > targetLogit {
                    moreLikelyWeight += weight
                } else if logit == targetLogit {
                    tiedWeight += weight
                }
            }
            for index in vocabularyTemperatures.indices
                where vocabularyTemperatures[index] != 1
            {
                temperatureNormalizers[index] += exp(
                    shiftedLogit / vocabularyTemperatures[index]
                )
            }
        }
        guard normalizer > 0 else {
            return (rank, .infinity, nil)
        }

        let logNormalizer = log(normalizer)
        let expectedShiftedLogit = weightedShiftedLogit / normalizer
        let expectedSquaredShiftedLogit = weightedSquaredShiftedLogit / normalizer
        let expectedLogProbability = expectedShiftedLogit - logNormalizer
        let logProbabilityVariance = max(
            0,
            expectedSquaredShiftedLogit - expectedShiftedLogit * expectedShiftedLogit
        )
        let probabilityMassRank = includesProbabilityMassRank
            ? min(1, max(0, (moreLikelyWeight + tiedWeight / 2) / normalizer))
            : nil
        let temperatureNormalizations = zip(
            vocabularyTemperatures,
            temperatureNormalizers
        ).map { temperature, sum in
            LlamaTemperatureNormalization(
                temperature: temperature,
                logNormalizer: temperature == 1
                    ? 0
                    : log(sum) - logNormalizer / temperature
            )
        }
        let negativeLogProbability: Double
        if targetLogit.isFinite {
            negativeLogProbability = maxLogit + logNormalizer - Double(targetLogit)
        } else {
            negativeLogProbability = .infinity
        }

        return (
            rank,
            negativeLogProbability,
            LlamaTokenVocabularyStatistics(
                expectedLogProbability: expectedLogProbability,
                logProbabilityVariance: logProbabilityVariance,
                probabilityMassRank: probabilityMassRank,
                temperatureNormalizations: temperatureNormalizations
            )
        )
    }

    static func calgacusValidatedTemperatures(_ temperatures: [Double]) throws -> [Double] {
        var uniqueTemperatures = Set<Double>()
        for temperature in temperatures {
            guard temperature.isFinite, temperature > 0 else {
                throw LlamaVocabularyStatisticsError.invalidTemperature(temperature)
            }
            uniqueTemperatures.insert(temperature)
        }
        return uniqueTemperatures.sorted()
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
