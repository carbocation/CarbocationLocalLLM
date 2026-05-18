import llama

@_silgen_name("carbocation_llama_mtp_create")
func carbocation_llama_mtp_create_bridge(
    _ model: OpaquePointer?,
    _ targetContext: OpaquePointer?,
    _ contextSize: UInt32,
    _ batchSize: UInt32,
    _ threadCount: Int32,
    _ maxDraftTokens: Int32,
    _ minDraftTokens: Int32
) -> UnsafeMutableRawPointer?

@_silgen_name("carbocation_llama_mtp_free")
func carbocation_llama_mtp_free_bridge(_ context: UnsafeMutableRawPointer?)

@_silgen_name("carbocation_llama_mtp_clear")
func carbocation_llama_mtp_clear_bridge(_ context: UnsafeMutableRawPointer?)

@_silgen_name("carbocation_llama_mtp_decode_target_tokens")
func carbocation_llama_mtp_decode_target_tokens_bridge(
    _ context: UnsafeMutableRawPointer?,
    _ tokens: UnsafePointer<llama_token>?,
    _ tokenCount: Int32,
    _ startPosition: Int32
) -> Int32

@_silgen_name("carbocation_llama_mtp_draft")
func carbocation_llama_mtp_draft_bridge(
    _ context: UnsafeMutableRawPointer?,
    _ lastToken: llama_token,
    _ lastTokenPosition: Int32,
    _ outputTokens: UnsafeMutablePointer<llama_token>?,
    _ outputTokenCapacity: Int32
) -> Int32

@_silgen_name("carbocation_llama_mtp_accept")
func carbocation_llama_mtp_accept_bridge(
    _ context: UnsafeMutableRawPointer?,
    _ acceptedDraftTokens: Int32
) -> Int32

@_silgen_name("carbocation_llama_mtp_rollback")
func carbocation_llama_mtp_rollback_bridge(
    _ context: UnsafeMutableRawPointer?,
    _ startPosition: Int32
) -> Int32

@_silgen_name("carbocation_llama_sampler_clone")
func carbocation_llama_sampler_clone_bridge(
    _ sampler: UnsafePointer<llama_sampler>?
) -> UnsafeMutablePointer<llama_sampler>?
