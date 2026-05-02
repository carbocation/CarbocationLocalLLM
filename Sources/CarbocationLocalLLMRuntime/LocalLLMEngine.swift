import CarbocationAppleIntelligenceRuntime
import CarbocationLlamaRuntime
import CarbocationLocalLLM
import Foundation

public struct LocalLLMEngineConfiguration: Hashable, Sendable {
    public var llamaGPULayerCount: Int32
    public var llamaUseMemoryMap: Bool
    public var llamaBatchSizeLimit: Int
    public var llamaThreadCount: Int32?
    public var promptReserveTokens: Int
    public var heartbeatInterval: TimeInterval

    public init(
        llamaGPULayerCount: Int32 = LlamaEngineConfiguration.defaultGPULayerCount,
        llamaUseMemoryMap: Bool = true,
        llamaBatchSizeLimit: Int = LlamaEngineConfiguration.defaultBatchSizeLimit,
        llamaThreadCount: Int32? = nil,
        promptReserveTokens: Int = LLMGenerationBudget.outputTokenReserve,
        heartbeatInterval: TimeInterval = 2
    ) {
        self.llamaGPULayerCount = llamaGPULayerCount
        self.llamaUseMemoryMap = llamaUseMemoryMap
        self.llamaBatchSizeLimit = llamaBatchSizeLimit
        self.llamaThreadCount = llamaThreadCount
        self.promptReserveTokens = promptReserveTokens
        self.heartbeatInterval = heartbeatInterval
    }

    func makeLlamaConfiguration() -> LlamaEngineConfiguration {
        LlamaEngineConfiguration(
            gpuLayerCount: llamaGPULayerCount,
            useMemoryMap: llamaUseMemoryMap,
            batchSizeLimit: llamaBatchSizeLimit,
            threadCount: llamaThreadCount,
            promptReserveTokens: promptReserveTokens,
            heartbeatInterval: heartbeatInterval
        )
    }

    func makeAppleIntelligenceConfiguration() -> AppleIntelligenceEngineConfiguration {
        AppleIntelligenceEngineConfiguration()
    }
}

public struct LocalLLMLoadedModelInfo: Hashable, Sendable {
    public var selection: LLMModelSelection
    public var displayName: String
    public var contextSize: Int
    public var trainingContextSize: Int
    public var supportsGrammar: Bool
    public var usesExactTokenCounts: Bool

    public init(
        selection: LLMModelSelection,
        displayName: String,
        contextSize: Int,
        trainingContextSize: Int,
        supportsGrammar: Bool,
        usesExactTokenCounts: Bool
    ) {
        self.selection = selection
        self.displayName = displayName
        self.contextSize = contextSize
        self.trainingContextSize = trainingContextSize
        self.supportsGrammar = supportsGrammar
        self.usesExactTokenCounts = usesExactTokenCounts
    }
}

public struct LocalLLMModelCapabilities: Hashable, Sendable {
    public var supportsGrammar: Bool
    public var usesExactTokenCounts: Bool
    public var contextSize: Int

    public init(
        supportsGrammar: Bool,
        usesExactTokenCounts: Bool,
        contextSize: Int
    ) {
        self.supportsGrammar = supportsGrammar
        self.usesExactTokenCounts = usesExactTokenCounts
        self.contextSize = contextSize
    }
}

public struct LocalLLMLoadPlan: Hashable, Sendable {
    public var selection: LLMModelSelection
    public var displayName: String
    public var requestedContext: Int
    public var capabilities: LocalLLMModelCapabilities

    public init(
        selection: LLMModelSelection,
        displayName: String,
        requestedContext: Int,
        capabilities: LocalLLMModelCapabilities
    ) {
        self.selection = selection
        self.displayName = displayName
        self.requestedContext = requestedContext
        self.capabilities = capabilities
    }
}

public enum LocalLLMEngineError: Error, LocalizedError, Sendable {
    case invalidSelection(String)
    case installedModelNotFound(UUID)
    case noSelectionLoaded
    case unavailableSystemModel(LLMSystemModelID)
    case systemModelGenerationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidSelection(let value):
            return "Selected model identifier is not valid: \(value)"
        case .installedModelNotFound(let id):
            return "Installed model was not found: \(id.uuidString)"
        case .noSelectionLoaded:
            return "No model selection is loaded."
        case .unavailableSystemModel(let id):
            return "System model is unavailable: \(id.rawValue)"
        case .systemModelGenerationFailed(let detail):
            return "System model generation failed: \(detail)"
        }
    }
}

public actor LocalLLMEngine: LLMEngine {
    public static let shared = LocalLLMEngine()

    private let llamaEngine: LlamaEngine
    private let appleIntelligenceEngine: AppleIntelligenceEngine
    private var loadedInfo: LocalLLMLoadedModelInfo?

    public init(configuration: LocalLLMEngineConfiguration = LocalLLMEngineConfiguration()) {
        self.llamaEngine = LlamaEngine(configuration: configuration.makeLlamaConfiguration())
        self.appleIntelligenceEngine = AppleIntelligenceEngine(configuration: configuration.makeAppleIntelligenceConfiguration())
    }

    public nonisolated static func availableSystemModels() -> [LLMSystemModelOption] {
        AppleIntelligenceEngine.systemModelOption().map { [$0] } ?? []
    }

    public nonisolated static func selection(from storageValue: String) throws -> LLMModelSelection {
        guard let selection = LLMModelSelection(storageValue: storageValue) else {
            throw LocalLLMEngineError.invalidSelection(storageValue)
        }
        return selection
    }

    public nonisolated static func probeTrainingContext(at url: URL) -> Int? {
        LlamaRuntimeModelProbe.probeTrainingContext(at: url)
    }

    public nonisolated static func probeTrainingContext(atPath path: String) -> Int? {
        LlamaRuntimeModelProbe.probeTrainingContext(atPath: path)
    }

    @MainActor
    public static func capabilities(
        for selection: LLMModelSelection,
        in library: ModelLibrary? = nil
    ) -> LocalLLMModelCapabilities {
        switch selection {
        case .installed(let id):
            let contextSize = library?.model(id: id)?.contextLength ?? 0
            return LocalLLMModelCapabilities(
                supportsGrammar: true,
                usesExactTokenCounts: true,
                contextSize: contextSize
            )
        case .system(.appleIntelligence):
            return LocalLLMModelCapabilities(
                supportsGrammar: false,
                usesExactTokenCounts: false,
                contextSize: AppleIntelligenceEngine.availability().contextSize
            )
        }
    }

    @MainActor
    public static func loadPlan(
        from storageValue: String,
        in library: ModelLibrary,
        defaults: UserDefaults = .standard,
        contextKeys: LlamaContextPreferenceKeys = .init(),
        refreshingLibrary: Bool = true
    ) async -> LocalLLMLoadPlan? {
        if refreshingLibrary {
            await library.refresh()
        }

        guard let selection = LLMModelSelection(storageValue: storageValue) else {
            return nil
        }

        switch selection {
        case .installed(let id):
            guard let model = library.model(id: id) else {
                return nil
            }
            let capabilities = capabilities(for: selection, in: library)
            return LocalLLMLoadPlan(
                selection: selection,
                displayName: model.displayName,
                requestedContext: LlamaContextPolicy.resolvedRequestedContext(
                    for: model,
                    defaults: defaults,
                    keys: contextKeys
                ),
                capabilities: capabilities
            )
        case .system:
            guard let option = availableSystemModels().first(where: { $0.selection == selection }) else {
                return nil
            }
            let capabilities = capabilities(for: selection, in: library)
            return LocalLLMLoadPlan(
                selection: selection,
                displayName: option.displayName,
                requestedContext: capabilities.contextSize,
                capabilities: capabilities
            )
        }
    }

    @discardableResult
    public func load(
        selection: LLMModelSelection,
        from library: ModelLibrary,
        requestedContext: Int
    ) async throws -> LocalLLMLoadedModelInfo {
        switch selection {
        case .installed(let id):
            let model = await MainActor.run {
                library.model(id: id)
            }
            let root = await MainActor.run {
                library.root
            }
            guard let model else {
                throw LocalLLMEngineError.installedModelNotFound(id)
            }
            let loaded = try await llamaEngine.load(
                model: model,
                from: root,
                requestedContext: requestedContext
            )
            let info = LocalLLMLoadedModelInfo(
                selection: selection,
                displayName: loaded.displayName ?? loaded.filename,
                contextSize: loaded.contextSize,
                trainingContextSize: loaded.trainingContextSize,
                supportsGrammar: true,
                usesExactTokenCounts: true
            )
            loadedInfo = info
            return info
        case .system(.appleIntelligence):
            let availability = AppleIntelligenceEngine.availability()
            guard availability.isAvailable else {
                throw LocalLLMEngineError.unavailableSystemModel(.appleIntelligence)
            }
            let info = LocalLLMLoadedModelInfo(
                selection: selection,
                displayName: AppleIntelligenceEngine.displayName,
                contextSize: availability.contextSize,
                trainingContextSize: availability.contextSize,
                supportsGrammar: false,
                usesExactTokenCounts: false
            )
            loadedInfo = info
            await llamaEngine.unload()
            return info
        }
    }

    public func unload() async {
        loadedInfo = nil
        await llamaEngine.unload()
    }

    public func currentSelection() -> LLMModelSelection? {
        loadedInfo?.selection
    }

    public func currentModelID() -> UUID? {
        guard case .installed(let id) = loadedInfo?.selection else { return nil }
        return id
    }

    public func currentContextSize() -> Int {
        loadedInfo?.contextSize ?? 0
    }

    public func currentLoadedModelInfo() -> LocalLLMLoadedModelInfo? {
        loadedInfo
    }

    public func generate(
        system: String,
        prompt: String,
        options: GenerationOptions,
        onEvent: @Sendable (LLMStreamEvent) -> Void
    ) async throws -> String {
        guard let loadedInfo else {
            throw LocalLLMEngineError.noSelectionLoaded
        }

        switch loadedInfo.selection {
        case .installed:
            return try await llamaEngine.generate(
                system: system,
                prompt: prompt,
                options: options,
                onEvent: onEvent
            )
        case .system(.appleIntelligence):
            do {
                return try await appleIntelligenceEngine.generate(
                    system: system,
                    prompt: prompt,
                    options: options,
                    onEvent: onEvent
                )
            } catch {
                throw LocalLLMEngineError.systemModelGenerationFailed(error.localizedDescription)
            }
        }
    }
}

public actor LocalLLMSession {
    private let system: String
    private var loadedInfo: LocalLLMLoadedModelInfo?
    private var llamaEngine: LlamaEngine?
    private var appleIntelligenceSession: AppleIntelligenceSession?

    public init(
        selection: LLMModelSelection,
        system: String = "",
        from library: ModelLibrary,
        requestedContext: Int,
        configuration: LocalLLMEngineConfiguration = LocalLLMEngineConfiguration()
    ) async throws {
        self.system = system

        switch selection {
        case .installed(let id):
            let model = await MainActor.run {
                library.model(id: id)
            }
            let root = await MainActor.run {
                library.root
            }
            guard let model else {
                throw LocalLLMEngineError.installedModelNotFound(id)
            }

            let engine = LlamaEngine(configuration: configuration.makeLlamaConfiguration())
            let loaded = try await engine.load(
                model: model,
                from: root,
                requestedContext: requestedContext
            )
            self.llamaEngine = engine
            self.loadedInfo = LocalLLMLoadedModelInfo(
                selection: selection,
                displayName: loaded.displayName ?? loaded.filename,
                contextSize: loaded.contextSize,
                trainingContextSize: loaded.trainingContextSize,
                supportsGrammar: true,
                usesExactTokenCounts: true
            )

        case .system(.appleIntelligence):
            let availability = AppleIntelligenceEngine.availability()
            guard availability.isAvailable else {
                throw LocalLLMEngineError.unavailableSystemModel(.appleIntelligence)
            }

            self.appleIntelligenceSession = AppleIntelligenceSession(
                system: system,
                configuration: configuration.makeAppleIntelligenceConfiguration()
            )
            self.loadedInfo = LocalLLMLoadedModelInfo(
                selection: selection,
                displayName: AppleIntelligenceEngine.displayName,
                contextSize: availability.contextSize,
                trainingContextSize: availability.contextSize,
                supportsGrammar: false,
                usesExactTokenCounts: false
            )
        }
    }

    public func currentLoadedModelInfo() -> LocalLLMLoadedModelInfo? {
        loadedInfo
    }

    public func currentSelection() -> LLMModelSelection? {
        loadedInfo?.selection
    }

    public func currentModelID() -> UUID? {
        guard case .installed(let id) = loadedInfo?.selection else { return nil }
        return id
    }

    public func currentContextSize() -> Int {
        loadedInfo?.contextSize ?? 0
    }

    public func generate(
        prompt: String,
        options: GenerationOptions,
        onEvent: @Sendable (LLMStreamEvent) -> Void
    ) async throws -> String {
        guard let loadedInfo else {
            throw LocalLLMEngineError.noSelectionLoaded
        }

        switch loadedInfo.selection {
        case .installed:
            guard let llamaEngine else {
                throw LocalLLMEngineError.noSelectionLoaded
            }
            return try await llamaEngine.generate(
                system: system,
                prompt: prompt,
                options: options,
                onEvent: onEvent
            )
        case .system(.appleIntelligence):
            guard let appleIntelligenceSession else {
                throw LocalLLMEngineError.noSelectionLoaded
            }
            do {
                return try await appleIntelligenceSession.generate(
                    prompt: prompt,
                    options: options,
                    onEvent: onEvent
                )
            } catch {
                throw LocalLLMEngineError.systemModelGenerationFailed(error.localizedDescription)
            }
        }
    }

    public func unload() async {
        loadedInfo = nil
        await llamaEngine?.unload()
        llamaEngine = nil
        appleIntelligenceSession = nil
    }
}

public enum LocalLLMRuntimeSmoke {
    public static func defaultModelParameterSummary() -> String {
        LlamaRuntimeSmoke.defaultModelParameterSummary()
    }

    public static func defaultContextBatchSize() -> UInt32 {
        LlamaRuntimeSmoke.defaultContextBatchSize()
    }
}
