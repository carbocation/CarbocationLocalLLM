import llama

enum LlamaBackend {
    static let initializer: Void = {
        llama_backend_init()
    }()

    static func ensureInitialized() {
        _ = initializer
    }
}
