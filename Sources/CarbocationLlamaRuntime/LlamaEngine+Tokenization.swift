import CarbocationLocalLLM
import Foundation
import llama

extension LlamaEngine {
    func preparePromptContext(_ promptTokens: [llama_token], context: OpaquePointer) throws {
        let memory = llama_get_memory(context)
        let plan = Self.promptPrefillPlan(
            cachedPromptTokens: cachedPromptTokens,
            newPromptTokens: promptTokens
        )
        cachedPromptTokens = nil

        if plan.shouldClearMemory {
            llama_memory_clear(memory, false)
            try decodePromptTokens(promptTokens, startingAt: 0, context: context)
            return
        }

        if let removeStartPosition = plan.removeStartPosition {
            let removed = llama_memory_seq_rm(memory, 0, Int32(removeStartPosition), -1)
            if !removed {
                llamaRuntimeLog.info(
                    "Prompt prefix cache removal failed; falling back to full prompt prefill."
                )
                llama_memory_clear(memory, false)
                try decodePromptTokens(promptTokens, startingAt: 0, context: context)
                return
            }
        }

        do {
            try decodePromptTokens(promptTokens, startingAt: plan.decodeStartIndex, context: context)
        } catch {
            llamaRuntimeLog.info(
                "Prompt prefix cache decode failed; falling back to full prompt prefill."
            )
            llama_memory_clear(memory, false)
            try decodePromptTokens(promptTokens, startingAt: 0, context: context)
        }
    }

    func decodePromptTokens(
        _ tokens: [llama_token],
        startingAt startIndex: Int,
        context: OpaquePointer
    ) throws {
        guard startIndex < tokens.count else { return }

        let maxBatchSize = max(1, Int(llama_n_batch(context)))
        for range in Self.prefillRanges(tokenCount: tokens.count - startIndex, maxBatchSize: maxBatchSize) {
            let lower = startIndex + range.lowerBound
            let upper = startIndex + range.upperBound
            var chunk = Array(tokens[lower..<upper])
            try chunk.withUnsafeMutableBufferPointer { buffer in
                let batch = llama_batch_get_one(buffer.baseAddress, Int32(buffer.count))
                if llama_decode(context, batch) != 0 {
                    throw LLMEngineError.decodeFailed
                }
            }
        }
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
            removeStartPosition: decodeStart,
            decodeStartIndex: decodeStart
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
        guard tokenCount > 0, maxBatchSize > 0 else { return [] }

        var ranges: [Range<Int>] = []
        var start = 0
        while start < tokenCount {
            let end = min(start + maxBatchSize, tokenCount)
            ranges.append(start..<end)
            start = end
        }
        return ranges
    }

    static func maxGenerationTokens(contextSize: Int, promptTokenCount: Int, reserve: Int) -> Int {
        max(0, contextSize - promptTokenCount - reserve)
    }

}
