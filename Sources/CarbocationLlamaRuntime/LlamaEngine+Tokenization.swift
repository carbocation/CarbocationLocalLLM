import CarbocationLocalLLM
import Foundation
import llama

extension LlamaEngine {
    struct PromptCacheCommitPlan: Equatable {
        var cachedTokens: [llama_token]
        var mtpCachedTokens: [llama_token]?
    }

    struct PromptPrefillChunkPlan: Equatable {
        var range: Range<Int>
        var requestsLogits: Bool
    }

    enum PromptCheckpointRestoreDisposition: Equatable {
        case restored
        case unavailable
        case requiresFullReset
    }

    enum PromptRuntimeResetPath: Equatable {
        case llamaMemory
        case mtpBridge
    }

    struct PromptRuntimeResetPlan: Equatable {
        var clearsPromptCaches: Bool
        var resetPath: PromptRuntimeResetPath
    }

    func clearPromptCaches() {
        cachedPromptTokens = nil
        mtpCachedPromptTokens = nil
        promptCheckpointTokens = nil
        carbocation_llama_prompt_checkpoint_clear_bridge(promptCheckpoint)
    }

    func clearPromptTokenCaches() {
        cachedPromptTokens = nil
        mtpCachedPromptTokens = nil
    }

    static func promptRuntimeResetPlan(
        mtpContext: UnsafeMutableRawPointer?
    ) -> PromptRuntimeResetPlan {
        PromptRuntimeResetPlan(
            clearsPromptCaches: true,
            resetPath: mtpContext == nil ? .llamaMemory : .mtpBridge
        )
    }

    func clearPromptRuntimeState(
        context: OpaquePointer,
        mtpContext: UnsafeMutableRawPointer?
    ) {
        let plan = Self.promptRuntimeResetPlan(mtpContext: mtpContext)
        if plan.clearsPromptCaches {
            clearPromptCaches()
        }
        switch plan.resetPath {
        case .llamaMemory:
            llama_memory_clear(llama_get_memory(context), false)
        case .mtpBridge:
            carbocation_llama_mtp_clear_bridge(mtpContext)
        }
    }

    func commitPromptCache(
        _ promptTokens: [llama_token],
        decodedGeneratedTokens: [llama_token],
        mtpSynchronized: Bool
    ) {
        let plan = Self.promptCacheCommitPlan(
            promptTokens: promptTokens,
            decodedGeneratedTokens: decodedGeneratedTokens,
            mtpSynchronized: mtpSynchronized
        )
        cachedPromptTokens = plan.cachedTokens
        mtpCachedPromptTokens = plan.mtpCachedTokens
    }

    func preparePromptContext(
        _ promptTokens: [llama_token],
        context: OpaquePointer,
        mtpContext: UnsafeMutableRawPointer? = nil,
        checkpointTokenCount: Int? = nil
    ) throws {
        let memory = llama_get_memory(context)
        let reusableCache: [llama_token]?
        if mtpContext != nil, mtpCachedPromptTokens != cachedPromptTokens {
            reusableCache = nil
        } else {
            reusableCache = cachedPromptTokens
        }
        let plan = Self.promptPrefillPlan(
            cachedPromptTokens: reusableCache,
            newPromptTokens: promptTokens
        )
        clearPromptTokenCaches()

        func decodeFromEmptyContext() throws {
            clearPromptCaches()
            if let mtpContext {
                carbocation_llama_mtp_clear_bridge(mtpContext)
            } else {
                llama_memory_clear(memory, false)
            }
            try decodePromptTokens(
                promptTokens,
                startingAt: 0,
                context: context,
                mtpContext: mtpContext,
                checkpointTokenCount: checkpointTokenCount
            )
        }

        if plan.shouldClearMemory {
            try decodeFromEmptyContext()
            return
        }

        var decodeStartIndex = plan.decodeStartIndex
        var restoredCheckpoint = false
        if mtpContext == nil,
           let promptCheckpoint,
           let savedTokens = promptCheckpointTokens,
           let restoreTokenCount = Self.promptCheckpointRestoreTokenCount(
               plan: plan,
               checkpointTokens: savedTokens,
               newPromptTokens: promptTokens
           ) {
            let restoreStartedAt = Date()
            let restored = carbocation_llama_prompt_checkpoint_restore_bridge(
                promptCheckpoint,
                Int32(restoreTokenCount)
            )
            switch Self.promptCheckpointRestoreDisposition(
                bridgeResult: restored,
                expectedTokenCount: restoreTokenCount
            ) {
            case .restored:
                decodeStartIndex = restoreTokenCount
                restoredCheckpoint = true
                llamaRuntimeLog.info(
                    "Restored recurrent prompt checkpoint: tokens=\(restoreTokenCount, privacy: .public) replayTokens=\(promptTokens.count - restoreTokenCount, privacy: .public) restoreMilliseconds=\(Date().timeIntervalSince(restoreStartedAt) * 1_000, privacy: .public)"
                )
            case .requiresFullReset:
                llamaRuntimeLog.info(
                    "Recurrent prompt checkpoint loaded but trailing-memory trim failed; falling back to full prompt prefill."
                )
                try decodeFromEmptyContext()
                return
            case .unavailable:
                break
            }
        }

        if !restoredCheckpoint, let removeStartPosition = plan.removeStartPosition {
            let removed = if let mtpContext {
                carbocation_llama_mtp_rollback_bridge(mtpContext, Int32(removeStartPosition)) != 0
            } else {
                llama_memory_seq_rm(memory, 0, Int32(removeStartPosition), -1)
            }
            if !removed {
                llamaRuntimeLog.info(
                    "Prompt prefix cache removal failed; falling back to full prompt prefill."
                )
                try decodeFromEmptyContext()
                return
            }
        }

        if let savedTokens = promptCheckpointTokens,
           Self.commonTokenPrefixCount(savedTokens, promptTokens) != savedTokens.count {
            promptCheckpointTokens = nil
            carbocation_llama_prompt_checkpoint_clear_bridge(promptCheckpoint)
        }

        do {
            try decodePromptTokens(
                promptTokens,
                startingAt: decodeStartIndex,
                context: context,
                mtpContext: mtpContext,
                checkpointTokenCount: checkpointTokenCount
            )
        } catch {
            llamaRuntimeLog.info(
                "Prompt prefix cache decode failed; falling back to full prompt prefill."
            )
            try decodeFromEmptyContext()
        }
    }

    func decodePromptTokens(
        _ tokens: [llama_token],
        startingAt startIndex: Int,
        context: OpaquePointer,
        mtpContext: UnsafeMutableRawPointer? = nil,
        checkpointTokenCount: Int? = nil
    ) throws {
        guard startIndex < tokens.count else { return }

        let maxBatchSize = max(1, Int(llama_n_batch(context)))
        let canRetainExistingCheckpoint = if let checkpointTokenCount,
                                             let savedTokens = promptCheckpointTokens,
                                             Self.commonTokenPrefixCount(savedTokens, tokens) == savedTokens.count {
            Self.shouldRetainPromptCheckpoint(
                savedTokenCount: savedTokens.count,
                requestedTokenCount: checkpointTokenCount
            )
        } else {
            false
        }
        let capturePosition: Int? = if mtpContext == nil,
                                 promptCheckpoint != nil,
                                 let checkpointTokenCount,
                                 !canRetainExistingCheckpoint,
                                 checkpointTokenCount >= startIndex,
                                 checkpointTokenCount < tokens.count {
            checkpointTokenCount
        } else {
            nil
        }
        if capturePosition == startIndex {
            capturePromptCheckpoint(tokens: tokens, tokenCount: startIndex, context: context)
        }

        var segmentBounds: [Int] = [startIndex]
        if let capturePosition, capturePosition > startIndex {
            segmentBounds.append(capturePosition)
        }
        segmentBounds.append(tokens.count)

        var logits = [Int8](repeating: 0, count: maxBatchSize)
        for segmentIndex in 0..<(segmentBounds.count - 1) {
            let segmentStart = segmentBounds[segmentIndex]
            let segmentEnd = segmentBounds[segmentIndex + 1]
            let isFinalSegment = segmentEnd == tokens.count
            let chunkPlans = Self.promptPrefillChunkPlans(
                tokenCount: segmentEnd - segmentStart,
                maxBatchSize: maxBatchSize,
                requestsFinalLogits: isFinalSegment
            )
            for chunkPlan in chunkPlans {
                let lower = segmentStart + chunkPlan.range.lowerBound
                let upper = segmentStart + chunkPlan.range.upperBound
                var chunk = Array(tokens[lower..<upper])
                try chunk.withUnsafeMutableBufferPointer { buffer in
                    let result: Int32
                    if let mtpContext {
                        result = carbocation_llama_mtp_decode_target_tokens_bridge(
                            mtpContext,
                            buffer.baseAddress,
                            Int32(buffer.count),
                            Int32(lower),
                            chunkPlan.requestsLogits ? 1 : 0
                        )
                    } else {
                        result = logits.withUnsafeMutableBufferPointer { logitsBuffer in
                            logitsBuffer.initialize(repeating: 0)
                            if chunkPlan.requestsLogits {
                                logitsBuffer[buffer.count - 1] = 1
                            }
                            var batch = llama_batch_get_one(buffer.baseAddress, Int32(buffer.count))
                            batch.logits = logitsBuffer.baseAddress
                            return llama_decode(context, batch)
                        }
                    }
                    if result != 0 {
                        throw LLMEngineError.decodeFailed
                    }
                }
            }
            if segmentEnd == capturePosition {
                capturePromptCheckpoint(tokens: tokens, tokenCount: segmentEnd, context: context)
            }
        }
    }

    func capturePromptCheckpoint(
        tokens: [llama_token],
        tokenCount: Int,
        context: OpaquePointer
    ) {
        guard let promptCheckpoint, tokenCount > 0, tokenCount < tokens.count else {
            return
        }
        let syncStartedAt = Date()
        llama_synchronize(context)
        let pendingDecodeSyncMilliseconds = Date().timeIntervalSince(syncStartedAt) * 1_000
        let startedAt = Date()
        let captured = carbocation_llama_prompt_checkpoint_capture_bridge(
            promptCheckpoint,
            Int32(tokenCount)
        ) != 0
        guard captured else {
            promptCheckpointTokens = nil
            return
        }
        promptCheckpointTokens = Array(tokens.prefix(tokenCount))
        let metadataBytes = carbocation_llama_prompt_checkpoint_size_bridge(promptCheckpoint)
        llamaRuntimeLog.info(
            "Captured recurrent prompt checkpoint: tokens=\(tokenCount, privacy: .public) replayTailTokens=\(tokens.count - tokenCount, privacy: .public) metadataBytes=\(metadataBytes, privacy: .public) pendingDecodeSyncMilliseconds=\(pendingDecodeSyncMilliseconds, privacy: .public) snapshotCopyMilliseconds=\(Date().timeIntervalSince(startedAt) * 1_000, privacy: .public)"
        )
    }

    func tokenize(vocab: OpaquePointer, text: String, addSpecial: Bool) throws -> [llama_token] {
        let utf8 = Array(text.utf8CString)
        let characterCount = Int32(utf8.count - 1)

        var probe = [llama_token](repeating: 0, count: max(8, Int(characterCount)))
        let probeCount = utf8.withUnsafeBufferPointer { cBuffer in
            probe.withUnsafeMutableBufferPointer { tokenBuffer in
                llama_tokenize(
                    vocab,
                    cBuffer.baseAddress,
                    characterCount,
                    tokenBuffer.baseAddress,
                    Int32(tokenBuffer.count),
                    addSpecial,
                    true
                )
            }
        }

        if probeCount >= 0 {
            return Array(probe.prefix(Int(probeCount)))
        }

        let neededCount = Int(-probeCount)
        var tokens = [llama_token](repeating: 0, count: neededCount)
        let tokenCount = utf8.withUnsafeBufferPointer { cBuffer in
            tokens.withUnsafeMutableBufferPointer { tokenBuffer in
                llama_tokenize(
                    vocab,
                    cBuffer.baseAddress,
                    characterCount,
                    tokenBuffer.baseAddress,
                    Int32(tokenBuffer.count),
                    addSpecial,
                    true
                )
            }
        }

        guard tokenCount > 0 else {
            throw LLMEngineError.tokenizationFailed
        }
        return Array(tokens.prefix(Int(tokenCount)))
    }

    func promptCheckpointTokenCount(
        promptFormatting: PromptFormattingResult,
        promptTokens: [llama_token],
        vocab: OpaquePointer
    ) -> Int? {
        guard promptCheckpoint != nil,
              let anchorText = promptFormatting.checkpointAnchorText
        else {
            return nil
        }
        let anchorForTokenization = promptWithAutoAddedSpecialTokensStripped(
            anchorText,
            vocab: vocab
        )
        guard let anchorTokens = try? tokenize(
            vocab: vocab,
            text: anchorForTokenization,
            addSpecial: true
        ) else {
            return nil
        }
        return Self.promptCheckpointTokenCount(
            promptTokens: promptTokens,
            anchorTokens: anchorTokens
        )
    }

    func promptWithAutoAddedSpecialTokensStripped(
        _ prompt: String,
        vocab: OpaquePointer
    ) -> String {
        var output = prompt
        if llama_vocab_get_add_bos(vocab),
           let bosToken = specialTokenString(vocab: vocab, token: llama_vocab_bos(vocab)),
           !bosToken.isEmpty,
           output.hasPrefix(bosToken) {
            output = String(output.dropFirst(bosToken.count))
        }
        if llama_vocab_get_add_eos(vocab),
           let eosToken = specialTokenString(vocab: vocab, token: llama_vocab_eos(vocab)),
           !eosToken.isEmpty,
           output.hasSuffix(eosToken) {
            output = String(output.dropLast(eosToken.count))
        }
        return output
    }

    func specialTokenString(vocab: OpaquePointer, token: llama_token) -> String? {
        guard token >= 0 else { return nil }
        let data = tokenToPiece(vocab: vocab, token: token, special: true)
        guard !data.isEmpty else { return nil }
        return String(data: data, encoding: .utf8)
    }

    func tokenToPiece(vocab: OpaquePointer, token: llama_token, special: Bool = false) -> Data {
        var probe = [CChar](repeating: 0, count: 32)
        let probeCount = probe.withUnsafeMutableBufferPointer { buffer in
            llama_token_to_piece(vocab, token, buffer.baseAddress, Int32(buffer.count), 0, special)
        }

        if probeCount >= 0 {
            return probe.withUnsafeBytes { rawBuffer in
                Data(rawBuffer.prefix(Int(probeCount)))
            }
        }

        let neededCount = Int(-probeCount)
        var bytes = [CChar](repeating: 0, count: neededCount)
        let byteCount = bytes.withUnsafeMutableBufferPointer { buffer in
            llama_token_to_piece(vocab, token, buffer.baseAddress, Int32(buffer.count), 0, special)
        }

        guard byteCount > 0 else {
            return Data()
        }
        return bytes.withUnsafeBytes { rawBuffer in
            Data(rawBuffer.prefix(Int(byteCount)))
        }
    }

    func diagnosticTokenDescription(_ token: llama_token, vocab: OpaquePointer) -> String {
        let rawPiece = tokenToPiece(vocab: vocab, token: token)
        let pieceData = rawPiece.isEmpty
            ? tokenToPiece(vocab: vocab, token: token, special: true)
            : rawPiece
        let piece = String(decoding: pieceData, as: UTF8.self)
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\n", with: "\\n")
            .replacingOccurrences(of: "\r", with: "\\r")
            .replacingOccurrences(of: "\t", with: "\\t")
            .replacingOccurrences(of: "\"", with: "\\\"")
        return "\(token):\"\(piece)\""
    }

    func diagnosticTokenList(_ tokens: [llama_token], vocab: OpaquePointer) -> String {
        "[" + tokens.map { diagnosticTokenDescription($0, vocab: vocab) }.joined(separator: ",") + "]"
    }

    func diagnosticTokenIDList(_ tokens: [llama_token]) -> String {
        tokens.map(String.init).joined(separator: ",")
    }

    func diagnosticShellQuoted(_ value: String) -> String {
        "'\(value.replacingOccurrences(of: "'", with: "'\"'\"'"))'"
    }

    func diagnosticBatchEquivalenceCommand(
        modelPath: String?,
        prefixTokens: [llama_token],
        windowTokens: [llama_token],
        recurrentStateSnapshots: Int,
        samplerPrefixSkip: Int
    ) -> String? {
        guard let modelPath else { return nil }
        return [
            "/private/tmp/llama-batch-equivalence-build/bin/llama-batch-equivalence",
            "-m",
            diagnosticShellQuoted(modelPath),
            "--prefix-token-ids",
            diagnosticShellQuoted(diagnosticTokenIDList(prefixTokens)),
            "--window-token-ids",
            diagnosticShellQuoted(diagnosticTokenIDList(windowTokens)),
            "--sampler-check",
            "--sampler-prefix-skip",
            String(max(0, samplerPrefixSkip)),
            "--baseline-n-rs-seq",
            "0",
            "--candidate-pre-norm",
            "--n-rs-seq",
            String(recurrentStateSnapshots)
        ].joined(separator: " ")
    }

    static func promptPrefillPlan(
        cachedPromptTokens: [llama_token]?,
        newPromptTokens: [llama_token]
    ) -> PromptPrefillPlan {
        guard let cachedPromptTokens,
              !cachedPromptTokens.isEmpty,
              !newPromptTokens.isEmpty
        else {
            return PromptPrefillPlan(
                commonPrefixCount: 0,
                retainedPrefixCount: 0,
                shouldClearMemory: true,
                removeStartPosition: nil,
                decodeStartIndex: 0
            )
        }

        let common = commonTokenPrefixCount(cachedPromptTokens, newPromptTokens)
        guard common > 0 else {
            return PromptPrefillPlan(
                commonPrefixCount: 0,
                retainedPrefixCount: 0,
                shouldClearMemory: true,
                removeStartPosition: nil,
                decodeStartIndex: 0
            )
        }

        let isAppendOnly = common == cachedPromptTokens.count && common < newPromptTokens.count
        let decodeStart = common == newPromptTokens.count ? max(0, common - 1) : common
        guard decodeStart > 0 else {
            return PromptPrefillPlan(
                commonPrefixCount: common,
                retainedPrefixCount: 0,
                shouldClearMemory: true,
                removeStartPosition: nil,
                decodeStartIndex: 0
            )
        }

        return PromptPrefillPlan(
            commonPrefixCount: common,
            retainedPrefixCount: decodeStart,
            shouldClearMemory: false,
            removeStartPosition: isAppendOnly ? nil : decodeStart,
            decodeStartIndex: decodeStart
        )
    }

    static func promptCheckpointTokenCount(
        promptTokens: [llama_token],
        anchorTokens: [llama_token],
        boundaryBackoff: Int = 4
    ) -> Int? {
        guard promptTokens.count > 1, !anchorTokens.isEmpty else { return nil }
        let sharedAnchorCount = commonTokenPrefixCount(promptTokens, anchorTokens)
        let checkpointCount = min(
            promptTokens.count - 1,
            sharedAnchorCount - max(0, boundaryBackoff)
        )
        return checkpointCount > 0 ? checkpointCount : nil
    }

    static func promptCheckpointRestoreTokenCount(
        plan: PromptPrefillPlan,
        checkpointTokens: [llama_token],
        newPromptTokens: [llama_token]
    ) -> Int? {
        guard plan.removeStartPosition != nil,
              !checkpointTokens.isEmpty,
              checkpointTokens.count <= plan.decodeStartIndex,
              commonTokenPrefixCount(checkpointTokens, newPromptTokens) == checkpointTokens.count
        else {
            return nil
        }
        return checkpointTokens.count
    }

    static func promptCheckpointRestoreDisposition(
        bridgeResult: Int32,
        expectedTokenCount: Int
    ) -> PromptCheckpointRestoreDisposition {
        if bridgeResult == Int32(expectedTokenCount) {
            return .restored
        }
        return bridgeResult == -2 ? .requiresFullReset : .unavailable
    }

    static func shouldRetainPromptCheckpoint(
        savedTokenCount: Int,
        requestedTokenCount: Int,
        maximumAdvance: Int = 64
    ) -> Bool {
        requestedTokenCount <= savedTokenCount ||
            requestedTokenCount - savedTokenCount <= max(0, maximumAdvance)
    }

    static func promptCacheCommitPlan(
        promptTokens: [llama_token],
        decodedGeneratedTokens: [llama_token],
        mtpSynchronized: Bool
    ) -> PromptCacheCommitPlan {
        var cachedTokens = promptTokens
        cachedTokens.append(contentsOf: decodedGeneratedTokens)
        return PromptCacheCommitPlan(
            cachedTokens: cachedTokens,
            mtpCachedTokens: mtpSynchronized ? cachedTokens : nil
        )
    }

    static func commonTokenPrefixCount(_ lhs: [llama_token], _ rhs: [llama_token]) -> Int {
        let count = min(lhs.count, rhs.count)
        var index = 0
        while index < count, lhs[index] == rhs[index] {
            index += 1
        }
        return index
    }

    static func prefillRanges(tokenCount: Int, maxBatchSize: Int) -> [Range<Int>] {
        promptPrefillChunkPlans(tokenCount: tokenCount, maxBatchSize: maxBatchSize).map(\.range)
    }

    static func promptPrefillChunkPlans(
        tokenCount: Int,
        maxBatchSize: Int,
        requestsFinalLogits: Bool = true
    ) -> [PromptPrefillChunkPlan] {
        guard tokenCount > 0, maxBatchSize > 0 else { return [] }

        var chunks: [PromptPrefillChunkPlan] = []
        var start = 0
        while start < tokenCount {
            let end = min(start + maxBatchSize, tokenCount)
            chunks.append(PromptPrefillChunkPlan(
                range: start..<end,
                requestsLogits: requestsFinalLogits && end == tokenCount
            ))
            start = end
        }
        return chunks
    }

    static func maxGenerationTokens(contextSize: Int, promptTokenCount: Int, reserve: Int) -> Int {
        max(0, contextSize - promptTokenCount - reserve)
    }

}
