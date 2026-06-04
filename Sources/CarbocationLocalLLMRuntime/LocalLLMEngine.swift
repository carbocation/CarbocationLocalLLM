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
    public var accelerationPolicy: LLMAccelerationPolicy
    public var mtpMaxDraftTokens: Int

    public init(
        llamaGPULayerCount: Int32 = LlamaEngineConfiguration.defaultGPULayerCount,
        llamaUseMemoryMap: Bool = true,
        llamaBatchSizeLimit: Int = LlamaEngineConfiguration.defaultBatchSizeLimit,
        llamaThreadCount: Int32? = nil,
        promptReserveTokens: Int = LLMGenerationBudget.outputTokenReserve,
        heartbeatInterval: TimeInterval = 2,
        accelerationPolicy: LLMAccelerationPolicy = .automatic,
        mtpMaxDraftTokens: Int = LlamaEngineConfiguration.defaultMTPMaxDraftTokens
    ) {
        self.llamaGPULayerCount = llamaGPULayerCount
        self.llamaUseMemoryMap = llamaUseMemoryMap
        self.llamaBatchSizeLimit = llamaBatchSizeLimit
        self.llamaThreadCount = llamaThreadCount
        self.promptReserveTokens = promptReserveTokens
        self.heartbeatInterval = heartbeatInterval
        self.accelerationPolicy = accelerationPolicy
        self.mtpMaxDraftTokens = mtpMaxDraftTokens
    }

    func makeLlamaConfiguration() -> LlamaEngineConfiguration {
        LlamaEngineConfiguration(
            gpuLayerCount: llamaGPULayerCount,
            useMemoryMap: llamaUseMemoryMap,
            batchSizeLimit: llamaBatchSizeLimit,
            threadCount: llamaThreadCount,
            promptReserveTokens: promptReserveTokens,
            heartbeatInterval: heartbeatInterval,
            accelerationPolicy: accelerationPolicy,
            mtpMaxDraftTokens: mtpMaxDraftTokens
        )
    }

    func makeAppleIntelligenceConfiguration() -> AppleIntelligenceEngineConfiguration {
        AppleIntelligenceEngineConfiguration(promptReserveTokens: promptReserveTokens)
    }
}

public struct LocalLLMLoadedModelInfo: Hashable, Sendable {
    public var selection: LLMModelSelection
    public var displayName: String
    public var contextSize: Int
    public var trainingContextSize: Int
    public var supportsGrammar: Bool
    public var usesExactTokenCounts: Bool
    public var supportsMTPAcceleration: Bool
    public var supportedInputModalities: Set<LLMInputModality>

    public init(
        selection: LLMModelSelection,
        displayName: String,
        contextSize: Int,
        trainingContextSize: Int,
        supportsGrammar: Bool,
        usesExactTokenCounts: Bool,
        supportsMTPAcceleration: Bool = false,
        supportedInputModalities: Set<LLMInputModality> = [.text]
    ) {
        self.selection = selection
        self.displayName = displayName
        self.contextSize = contextSize
        self.trainingContextSize = trainingContextSize
        self.supportsGrammar = supportsGrammar
        self.usesExactTokenCounts = usesExactTokenCounts
        self.supportsMTPAcceleration = supportsMTPAcceleration
        self.supportedInputModalities = supportedInputModalities
    }

    public var supportsVision: Bool {
        supportedInputModalities.contains(.image)
    }

    public var supportsAudio: Bool {
        supportedInputModalities.contains(.audio)
    }
}

public struct LocalLLMModelCapabilities: Hashable, Sendable {
    public var supportsGrammar: Bool
    public var usesExactTokenCounts: Bool
    public var contextSize: Int
    public var supportsMTPAcceleration: Bool
    public var supportedInputModalities: Set<LLMInputModality>

    public init(
        supportsGrammar: Bool,
        usesExactTokenCounts: Bool,
        contextSize: Int,
        supportsMTPAcceleration: Bool = false,
        supportedInputModalities: Set<LLMInputModality> = [.text]
    ) {
        self.supportsGrammar = supportsGrammar
        self.usesExactTokenCounts = usesExactTokenCounts
        self.contextSize = contextSize
        self.supportsMTPAcceleration = supportsMTPAcceleration
        self.supportedInputModalities = supportedInputModalities
    }

    public var supportsVision: Bool {
        supportedInputModalities.contains(.image)
    }

    public var supportsAudio: Bool {
        supportedInputModalities.contains(.audio)
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

public actor LocalLLMEngine: LLMEngine, LLMPhasedGenerationProvider, LLMMultimodalGenerationProvider, LLMToolPhasedGenerationProvider {
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

    public nonisolated static func contextCalibrationRuntimeFingerprint(
        configuration: LocalLLMEngineConfiguration = LocalLLMEngineConfiguration()
    ) -> LlamaContextCalibrationRuntimeFingerprint {
        LlamaEngine.contextCalibrationRuntimeFingerprint(
            configuration: configuration.makeLlamaConfiguration()
        )
    }

    @MainActor
    public static func capabilities(
        for selection: LLMModelSelection,
        in library: ModelLibrary? = nil
    ) -> LocalLLMModelCapabilities {
        switch selection {
        case .installed(let id):
            let installedModel = library?.model(id: id)
            let contextSize = installedModel?.contextLength ?? 0
            let supportsMTPAcceleration = if let library, let installedModel {
                GGUFMetadata.supportsMTPAcceleration(at: installedModel.weightsURL(in: library.root))
            } else {
                false
            }
            let supportedInputModalities: Set<LLMInputModality> = if let library,
                                                                     let installedModel,
                                                                     let mmprojURL = installedModel.mmprojURL(in: library.root) {
                Set<LLMInputModality>([.text])
                    .union(LlamaEngine.projectorSupportedInputModalities(at: mmprojURL))
            } else {
                [.text]
            }
            return LocalLLMModelCapabilities(
                supportsGrammar: true,
                usesExactTokenCounts: true,
                contextSize: contextSize,
                supportsMTPAcceleration: supportsMTPAcceleration,
                supportedInputModalities: supportedInputModalities
            )
        case .system(.appleIntelligence):
            return LocalLLMModelCapabilities(
                supportsGrammar: false,
                usesExactTokenCounts: false,
                contextSize: AppleIntelligenceEngine.availability().contextSize,
                supportsMTPAcceleration: false,
                supportedInputModalities: [.text]
            )
        }
    }

    @MainActor
    public static func loadPlan(
        from storageValue: String,
        in library: ModelLibrary,
        defaults: UserDefaults = .standard,
        contextKeys: LlamaContextPreferenceKeys = .init(),
        refreshingLibrary: Bool = true,
        calibrationStore: LlamaContextCalibrationStore? = .shared,
        configuration: LocalLLMEngineConfiguration = LocalLLMEngineConfiguration()
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
            let runtimeFingerprint = contextCalibrationRuntimeFingerprint(configuration: configuration)
            let calibratedContext = calibrationStore?
                .record(for: model, runtime: runtimeFingerprint)?
                .maximumSupportedContext
            return LocalLLMLoadPlan(
                selection: selection,
                displayName: model.displayName,
                requestedContext: LlamaContextPolicy.resolvedRequestedContext(
                    for: model,
                    defaults: defaults,
                    keys: contextKeys,
                    maximumSupportedContext: calibratedContext
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
                usesExactTokenCounts: true,
                supportsMTPAcceleration: loaded.supportsMTPAcceleration,
                supportedInputModalities: loaded.supportedInputModalities
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
                usesExactTokenCounts: false,
                supportsMTPAcceleration: false,
                supportedInputModalities: [.text]
            )
            loadedInfo = info
            await llamaEngine.unload()
            return info
        }
    }

    public static func calibrateContext(
        for model: InstalledModel,
        in library: ModelLibrary,
        store: LlamaContextCalibrationStore = .shared,
        configuration: LocalLLMEngineConfiguration = LocalLLMEngineConfiguration(),
        onProgress: @escaping @Sendable (LlamaContextCalibrationProgress) async -> Void = { _ in }
    ) async throws -> LlamaContextCalibrationRecord {
        let root = await MainActor.run { library.root }
        let engine = LlamaEngine(configuration: configuration.makeLlamaConfiguration())
        return try await engine.calibrateContext(
            model: model,
            from: root,
            store: store,
            onProgress: onProgress
        )
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

    public func preflight(
        system: String,
        prompt: String,
        options: GenerationOptions
    ) async throws -> LLMGenerationPreflight {
        guard let loadedInfo else {
            throw LocalLLMEngineError.noSelectionLoaded
        }

        switch loadedInfo.selection {
        case .installed:
            return try await llamaEngine.preflight(
                system: system,
                prompt: prompt,
                options: options
            )
        case .system(.appleIntelligence):
            do {
                return try await appleIntelligenceEngine.preflight(
                    system: system,
                    prompt: prompt,
                    options: options
                )
            } catch {
                throw LocalLLMEngineError.systemModelGenerationFailed(error.localizedDescription)
            }
        }
    }

    public func preflight(
        messages: [LLMChatMessage],
        options: GenerationOptions
    ) async throws -> LLMGenerationPreflight {
        guard let loadedInfo else {
            throw LocalLLMEngineError.noSelectionLoaded
        }

        switch loadedInfo.selection {
        case .installed:
            return try await llamaEngine.preflight(messages: messages, options: options)
        case .system(.appleIntelligence):
            do {
                return try await appleIntelligenceEngine.preflight(messages: messages, options: options)
            } catch let error as LLMEngineError {
                throw error
            } catch {
                throw LocalLLMEngineError.systemModelGenerationFailed(error.localizedDescription)
            }
        }
    }

    public func generate(
        system: String,
        prompt: String,
        options: GenerationOptions,
        onEvent: @Sendable (LLMStreamEvent) -> Void
    ) async throws -> String {
        try await generate(
            system: system,
            prompt: prompt,
            options: options,
            control: nil,
            onEvent: onEvent
        )
    }

    public func generate(
        system: String,
        prompt: String,
        options: GenerationOptions,
        control: LLMGenerationControl? = nil,
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
                control: control,
                onEvent: onEvent
            )
        case .system(.appleIntelligence):
            do {
                return try await appleIntelligenceEngine.generate(
                    system: system,
                    prompt: prompt,
                    options: options,
                    control: control,
                    onEvent: onEvent
                )
            } catch {
                throw LocalLLMEngineError.systemModelGenerationFailed(error.localizedDescription)
            }
        }
    }

    public func generate(
        messages: [LLMChatMessage],
        options: GenerationOptions,
        onEvent: @escaping @Sendable (LLMStreamEvent) -> Void
    ) async throws -> String {
        try await generate(
            messages: messages,
            options: options,
            control: nil,
            onEvent: onEvent
        )
    }

    public func generate(
        messages: [LLMChatMessage],
        options: GenerationOptions,
        control: LLMGenerationControl? = nil,
        onEvent: @escaping @Sendable (LLMStreamEvent) -> Void
    ) async throws -> String {
        try await generate(
            messages: messages,
            options: options,
            control: control,
            onPhaseAwareEvent: { event in
                if let streamEvent = event.streamEvent {
                    onEvent(streamEvent)
                }
            }
        )
    }

    public func generate(
        messages: [LLMChatMessage],
        options: GenerationOptions,
        onPhaseAwareEvent: @escaping @Sendable (LLMPhaseAwareStreamEvent) -> Void,
        _ phaseAwareOverload: Void = ()
    ) async throws -> String {
        try await generate(
            messages: messages,
            options: options,
            control: nil,
            onPhaseAwareEvent: onPhaseAwareEvent,
            phaseAwareOverload
        )
    }

    public func generate(
        messages: [LLMChatMessage],
        options: GenerationOptions,
        control: LLMGenerationControl? = nil,
        onPhaseAwareEvent: @escaping @Sendable (LLMPhaseAwareStreamEvent) -> Void,
        _ phaseAwareOverload: Void = ()
    ) async throws -> String {
        try await generate(
            messages: messages,
            options: options,
            control: control,
            onPhaseAwareEvent: onPhaseAwareEvent
        )
    }

    public func generate(
        messages: [LLMChatMessage],
        options: GenerationOptions,
        control: LLMGenerationControl?,
        onPhaseAwareEvent: @escaping @Sendable (LLMPhaseAwareStreamEvent) -> Void
    ) async throws -> String {
        guard let loadedInfo else {
            throw LocalLLMEngineError.noSelectionLoaded
        }

        switch loadedInfo.selection {
        case .installed:
            return try await llamaEngine.generate(
                messages: messages,
                options: options,
                control: control,
                onPhaseAwareEvent: onPhaseAwareEvent
            )
        case .system(.appleIntelligence):
            do {
                return try await appleIntelligenceEngine.generate(
                    messages: messages,
                    options: options,
                    control: control,
                    onPhaseAwareEvent: onPhaseAwareEvent
                )
            } catch let error as LLMEngineError {
                throw error
            } catch {
                throw LocalLLMEngineError.systemModelGenerationFailed(error.localizedDescription)
            }
        }
    }

    public func generate(
        system: String,
        prompt: String,
        options: GenerationOptions,
        onPhaseAwareEvent: @Sendable (LLMPhaseAwareStreamEvent) -> Void,
        _ phaseAwareOverload: Void = ()
    ) async throws -> String {
        try await generate(
            system: system,
            prompt: prompt,
            options: options,
            control: nil,
            onPhaseAwareEvent: onPhaseAwareEvent,
            phaseAwareOverload
        )
    }

    public func generate(
        system: String,
        prompt: String,
        options: GenerationOptions,
        control: LLMGenerationControl? = nil,
        onPhaseAwareEvent: @Sendable (LLMPhaseAwareStreamEvent) -> Void,
        _ phaseAwareOverload: Void = ()
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
                control: control,
                onPhaseAwareEvent: onPhaseAwareEvent
            )
        case .system(.appleIntelligence):
            do {
                return try await appleIntelligenceEngine.generate(
                    system: system,
                    prompt: prompt,
                    options: options,
                    control: control,
                    onPhaseAwareEvent: onPhaseAwareEvent
                )
            } catch {
                throw LocalLLMEngineError.systemModelGenerationFailed(error.localizedDescription)
            }
        }
    }

    public func generatePhased(
        system: String,
        prompt: String,
        options: GenerationOptions,
        control: LLMGenerationControl? = nil,
        onEvent: @escaping @Sendable (LLMGenerationStreamEvent) -> Void = { _ in }
    ) async throws -> LLMGenerationResult {
        guard let loadedInfo else {
            throw LocalLLMEngineError.noSelectionLoaded
        }

        switch loadedInfo.selection {
        case .installed:
            return try await llamaEngine.generatePhased(
                system: system,
                prompt: prompt,
                options: options,
                control: control,
                onEvent: onEvent
            )
        case .system(.appleIntelligence):
            do {
                return try await appleIntelligenceEngine.generatePhased(
                    system: system,
                    prompt: prompt,
                    options: options,
                    control: control,
                    onEvent: onEvent
                )
            } catch {
                throw LocalLLMEngineError.systemModelGenerationFailed(error.localizedDescription)
            }
        }
    }

    public func generatePhased(
        messages: [LLMChatMessage],
        options: GenerationOptions,
        control: LLMGenerationControl? = nil,
        onEvent: @escaping @Sendable (LLMGenerationStreamEvent) -> Void = { _ in }
    ) async throws -> LLMGenerationResult {
        guard let loadedInfo else {
            throw LocalLLMEngineError.noSelectionLoaded
        }

        switch loadedInfo.selection {
        case .installed:
            return try await llamaEngine.generatePhased(
                messages: messages,
                options: options,
                control: control,
                onEvent: onEvent
            )
        case .system(.appleIntelligence):
            do {
                return try await appleIntelligenceEngine.generatePhased(
                    messages: messages,
                    options: options,
                    control: control,
                    onEvent: onEvent
                )
            } catch let error as LLMEngineError {
                throw error
            } catch {
                throw LocalLLMEngineError.systemModelGenerationFailed(error.localizedDescription)
            }
        }
    }

    public func generateWithTools(
        _ request: LLMToolGenerationRequest,
        onPhaseAwareEvent: @escaping @Sendable (LLMToolPhaseAwareStreamEvent) -> Void = { _ in }
    ) async throws -> LLMToolGenerationResult {
        try await generateWithTools(
            request,
            control: nil,
            onPhaseAwareEvent: onPhaseAwareEvent
        )
    }

    public func generateWithTools(
        _ request: LLMToolGenerationRequest,
        control: LLMGenerationControl? = nil,
        onPhaseAwareEvent: @escaping @Sendable (LLMToolPhaseAwareStreamEvent) -> Void = { _ in }
    ) async throws -> LLMToolGenerationResult {
        guard let loadedInfo else {
            throw LocalLLMEngineError.noSelectionLoaded
        }

        switch loadedInfo.selection {
        case .installed:
            return try await llamaEngine.generateWithTools(
                request,
                control: control,
                onPhaseAwareEvent: onPhaseAwareEvent
            )
        case .system(.appleIntelligence):
            do {
                return try await appleIntelligenceEngine.generateWithTools(
                    request,
                    control: control,
                    onPhaseAwareEvent: onPhaseAwareEvent
                )
            } catch {
                throw LocalLLMEngineError.systemModelGenerationFailed(error.localizedDescription)
            }
        }
    }

    public func generateWithToolsPhased(
        _ request: LLMToolGenerationRequest,
        control: LLMGenerationControl? = nil,
        onEvent: @escaping @Sendable (LLMToolPhasedStreamEvent) -> Void = { _ in }
    ) async throws -> LLMToolPhasedGenerationResult {
        guard let loadedInfo else {
            throw LocalLLMEngineError.noSelectionLoaded
        }

        switch loadedInfo.selection {
        case .installed:
            return try await llamaEngine.generateWithToolsPhased(
                request,
                control: control,
                onEvent: onEvent
            )
        case .system(.appleIntelligence):
            do {
                return try await appleIntelligenceEngine.generateWithToolsPhased(
                    request,
                    control: control,
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
                usesExactTokenCounts: true,
                supportsMTPAcceleration: loaded.supportsMTPAcceleration,
                supportedInputModalities: loaded.supportedInputModalities
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
                usesExactTokenCounts: false,
                supportedInputModalities: [.text]
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

    public func preflight(
        prompt: String,
        options: GenerationOptions
    ) async throws -> LLMGenerationPreflight {
        guard let loadedInfo else {
            throw LocalLLMEngineError.noSelectionLoaded
        }

        switch loadedInfo.selection {
        case .installed:
            guard let llamaEngine else {
                throw LocalLLMEngineError.noSelectionLoaded
            }
            return try await llamaEngine.preflight(
                system: system,
                prompt: prompt,
                options: options
            )
        case .system(.appleIntelligence):
            guard let appleIntelligenceSession else {
                throw LocalLLMEngineError.noSelectionLoaded
            }
            do {
                return try await appleIntelligenceSession.preflight(
                    prompt: prompt,
                    options: options
                )
            } catch {
                throw LocalLLMEngineError.systemModelGenerationFailed(error.localizedDescription)
            }
        }
    }

    public func generate(
        prompt: String,
        options: GenerationOptions,
        onEvent: @Sendable (LLMStreamEvent) -> Void
    ) async throws -> String {
        try await generate(
            prompt: prompt,
            options: options,
            control: nil,
            onEvent: onEvent
        )
    }

    public func generate(
        prompt: String,
        options: GenerationOptions,
        control: LLMGenerationControl? = nil,
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
                control: control,
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
                    control: control,
                    onEvent: onEvent
                )
            } catch {
                throw LocalLLMEngineError.systemModelGenerationFailed(error.localizedDescription)
            }
        }
    }

    public func generate(
        prompt: String,
        options: GenerationOptions,
        onPhaseAwareEvent: @Sendable (LLMPhaseAwareStreamEvent) -> Void,
        _ phaseAwareOverload: Void = ()
    ) async throws -> String {
        try await generate(
            prompt: prompt,
            options: options,
            control: nil,
            onPhaseAwareEvent: onPhaseAwareEvent,
            phaseAwareOverload
        )
    }

    public func generate(
        prompt: String,
        options: GenerationOptions,
        control: LLMGenerationControl? = nil,
        onPhaseAwareEvent: @Sendable (LLMPhaseAwareStreamEvent) -> Void,
        _ phaseAwareOverload: Void = ()
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
                control: control,
                onPhaseAwareEvent: onPhaseAwareEvent
            )
        case .system(.appleIntelligence):
            guard let appleIntelligenceSession else {
                throw LocalLLMEngineError.noSelectionLoaded
            }
            do {
                return try await appleIntelligenceSession.generate(
                    prompt: prompt,
                    options: options,
                    control: control,
                    onPhaseAwareEvent: onPhaseAwareEvent
                )
            } catch {
                throw LocalLLMEngineError.systemModelGenerationFailed(error.localizedDescription)
            }
        }
    }

    public func generatePhased(
        prompt: String,
        options: GenerationOptions,
        control: LLMGenerationControl? = nil,
        onEvent: @escaping @Sendable (LLMGenerationStreamEvent) -> Void = { _ in }
    ) async throws -> LLMGenerationResult {
        guard let loadedInfo else {
            throw LocalLLMEngineError.noSelectionLoaded
        }

        switch loadedInfo.selection {
        case .installed:
            guard let llamaEngine else {
                throw LocalLLMEngineError.noSelectionLoaded
            }
            return try await llamaEngine.generatePhased(
                system: system,
                prompt: prompt,
                options: options,
                control: control,
                onEvent: onEvent
            )
        case .system(.appleIntelligence):
            guard let appleIntelligenceSession else {
                throw LocalLLMEngineError.noSelectionLoaded
            }
            do {
                let finalText = try await appleIntelligenceSession.generate(
                    prompt: prompt,
                    options: options,
                    control: control,
                    onPhaseAwareEvent: { event in
                        onEvent(LLMGenerationStreamEvent(adapting: event))
                    }
                )
                return LLMGenerationResult(
                    finalText: finalText,
                    phaseSegments: finalText.isEmpty
                        ? []
                        : [LLMGenerationPhaseSegment(phase: .final, text: finalText)]
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
