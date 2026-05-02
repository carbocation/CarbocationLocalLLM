import CarbocationLocalLLM
import Foundation
import llama

public enum LlamaRuntimeSmoke {
    public static func defaultModelParameterSummary() -> String {
        let params = llama_model_default_params()
        return "use_mmap=\(params.use_mmap);n_gpu_layers=\(params.n_gpu_layers)"
    }

    public static func defaultContextBatchSize() -> UInt32 {
        llama_context_default_params().n_batch
    }
}

public enum LlamaRuntimeModelProbe {
    public static func probeTrainingContext(at url: URL) -> Int? {
        GGUFMetadata.trainingContextLength(at: url) ?? probeTrainingContextByLoadingModel(atPath: url.path)
    }

    public static func probeTrainingContext(atPath path: String) -> Int? {
        GGUFMetadata.trainingContextLength(atPath: path) ?? probeTrainingContextByLoadingModel(atPath: path)
    }

    private static func probeTrainingContextByLoadingModel(atPath path: String) -> Int? {
#if os(iOS)
        return nil
#else
        LlamaBackend.ensureInitialized()

        var params = llama_model_default_params()
        params.configureForCPUOnly()
        params.use_mmap = true

        guard let model = path.withCString({ cPath in
            llama_model_load_from_file(cPath, params)
        }) else {
            return nil
        }
        defer { llama_model_free(model) }

        let trainingContext = Int(llama_model_n_ctx_train(model))
        return trainingContext > 0 ? trainingContext : nil
#endif
    }
}
