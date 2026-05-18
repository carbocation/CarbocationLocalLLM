import CarbocationLocalLLM
import Foundation
import llama

extension LlamaEngine {
    static func clampedContextSize(requestedContext: Int, trainingContext: Int) -> Int {
        let normalizedTrainingContext = max(0, trainingContext)
        let requested = requestedContext > 0
            ? requestedContext
            : (normalizedTrainingContext > 0 ? normalizedTrainingContext : LlamaContextPolicy.unknownTrainingFallback)
        let modelUpperBound = normalizedTrainingContext > 0 ? normalizedTrainingContext : requested
        let upperBound = min(modelUpperBound, platformMaximumContextSize)
        return max(LlamaContextPolicy.minimumContext, min(requested, upperBound))
    }

    static func contextBatchCandidates(contextSize: Int, batchSizeLimit: Int) -> [Int] {
        let first = min(max(1, contextSize), max(1, batchSizeLimit))
        var candidates = [first]
        var next = first
        while next > 1 {
            next = max(1, next / 2)
            if candidates.last != next {
                candidates.append(next)
            }
        }
        return candidates
    }

    static func contextParams(
        contextSize: Int,
        batchSize: Int,
        threads: Int32,
        recurrentStateSnapshots: Int = 0
    ) -> llama_context_params {
        var params = llama_context_default_params()
        let clampedContext = max(1, contextSize)
        let clampedBatch = min(clampedContext, max(1, batchSize))
        params.n_ctx = UInt32(clampedContext)
        params.n_batch = UInt32(clampedBatch)
        params.n_ubatch = UInt32(clampedBatch)
        params.n_threads = threads
        params.n_threads_batch = threads
        params.n_rs_seq = UInt32(max(0, recurrentStateSnapshots))
        return params
    }

}

extension llama_model_params {
    mutating func configureForCPUOnly() {
        n_gpu_layers = 0
        split_mode = LLAMA_SPLIT_MODE_NONE
        main_gpu = -1
    }
}
