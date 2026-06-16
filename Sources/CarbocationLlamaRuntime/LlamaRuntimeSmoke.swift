import CarbocationLocalLLM
import CarbocationLlamaMTMDBridge
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

    public static func defaultMediaMarker() -> String {
        String(cString: carbocation_mtmd_default_marker_bridge())
    }

    public static func audioBitmapBridgeSummary() -> String {
        var sample: Float = 0
        guard let bitmap = carbocation_mtmd_bitmap_init_from_audio_bridge(1, &sample) else {
            return "audio_bitmap=nil"
        }
        defer { carbocation_mtmd_bitmap_free_bridge(bitmap) }
        let isAudio = carbocation_mtmd_bitmap_is_audio_bridge(bitmap)
        let samples = carbocation_mtmd_bitmap_get_nx_bridge(bitmap)
        return "is_audio=\(isAudio);samples=\(samples)"
    }

    public static func imageBufferBitmapBridgeSummary() -> String {
        let pngBytes: [UInt8] = [
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
            0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41,
            0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
            0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00,
            0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,
            0x42, 0x60, 0x82
        ]

        return pngBytes.withUnsafeBytes { rawBytes in
            guard let baseAddress = rawBytes.bindMemory(to: UInt8.self).baseAddress else {
                return "image_bitmap=nil"
            }
            guard let bitmap = carbocation_mtmd_helper_bitmap_init_from_buf_bridge(
                nil,
                baseAddress,
                rawBytes.count
            ) else {
                return "image_bitmap=nil"
            }
            defer { carbocation_mtmd_bitmap_free_bridge(bitmap) }
            let isAudio = carbocation_mtmd_bitmap_is_audio_bridge(bitmap)
            let width = carbocation_mtmd_bitmap_get_nx_bridge(bitmap)
            return "is_audio=\(isAudio);width=\(width)"
        }
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
