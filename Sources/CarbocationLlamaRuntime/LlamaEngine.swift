import CarbocationLocalLLM
import CarbocationLlamaCommonBridge
import CarbocationLlamaMTMDBridge
import Foundation
import OSLog
import llama

let llamaRuntimeLog = Logger(
    subsystem: "com.carbocation.CarbocationLocalLLM",
    category: "LlamaRuntime"
)

/// Opaque identity for one physical model/runtime instance. It changes only
/// when the lower llama.cpp runtime is actually replaced.
package struct LlamaRuntimeIdentity: Hashable, Sendable {
    fileprivate let value: UUID

    fileprivate init() {
        self.value = UUID()
    }
}

package struct LlamaManagedRuntimeSnapshot: Sendable {
    package var info: LlamaLoadedModelInfo
    package var identity: LlamaRuntimeIdentity
}

package enum LlamaManagedRuntimeError: Error, Sendable {
    case lifecycleOperationSuperseded
}

/// Binds wrapper-routed work to the physical runtime selected before the
/// cross-actor hop. Child tasks inherit this value automatically.
package enum LlamaManagedRuntimeContext {
    @TaskLocal package static var expectedIdentity: LlamaRuntimeIdentity?
}

#if os(iOS)
private let platformDefaultGPULayerCount: Int32 = 0
// Large mobile GGUFs can fit model + KV memory but fail llama.cpp's 512-token graph reservation.
private let platformDefaultBatchSizeLimit = 64
let platformMaximumContextSize = Int.max
#else
private let platformDefaultGPULayerCount: Int32 = 999
private let platformDefaultBatchSizeLimit = 2_048
let platformMaximumContextSize = Int.max
#endif

public struct LlamaEngineConfiguration: Hashable, Sendable {
    public static var defaultGPULayerCount: Int32 { platformDefaultGPULayerCount }
    public static var defaultBatchSizeLimit: Int { platformDefaultBatchSizeLimit }
    public static var defaultMicroBatchSizeLimit: Int { LlamaEngine.defaultMicroBatchSizeLimit }
    public static let defaultMTPMaxDraftTokens = 1

    public var gpuLayerCount: Int32
    public var useMemoryMap: Bool
    public var batchSizeLimit: Int
    /// Optional physical llama.cpp micro-batch (`n_ubatch`) ceiling. `nil` preserves
    /// the deployment-safe default while allowing benchmarks and advanced callers
    /// to trade additional memory for prompt-prefill throughput.
    public var microBatchSizeLimit: Int?
    public var threadCount: Int32?
    public var promptReserveTokens: Int
    public var heartbeatInterval: TimeInterval
    public var accelerationPolicy: LLMAccelerationPolicy
    public var mtpMaxDraftTokens: Int
    public var emitsMTPDiagnostics: Bool

    public init(
        gpuLayerCount: Int32 = LlamaEngineConfiguration.defaultGPULayerCount,
        useMemoryMap: Bool = true,
        batchSizeLimit: Int = LlamaEngineConfiguration.defaultBatchSizeLimit,
        microBatchSizeLimit: Int? = nil,
        threadCount: Int32? = nil,
        promptReserveTokens: Int = LLMGenerationBudget.outputTokenReserve,
        heartbeatInterval: TimeInterval = 2,
        accelerationPolicy: LLMAccelerationPolicy = .disabled,
        mtpMaxDraftTokens: Int = LlamaEngineConfiguration.defaultMTPMaxDraftTokens,
        emitsMTPDiagnostics: Bool = false
    ) {
        self.gpuLayerCount = gpuLayerCount
        self.useMemoryMap = useMemoryMap
        self.batchSizeLimit = batchSizeLimit
        self.microBatchSizeLimit = microBatchSizeLimit
        self.threadCount = threadCount
        self.promptReserveTokens = promptReserveTokens
        self.heartbeatInterval = heartbeatInterval
        self.accelerationPolicy = accelerationPolicy
        self.mtpMaxDraftTokens = mtpMaxDraftTokens
        self.emitsMTPDiagnostics = emitsMTPDiagnostics
    }
}

public struct LlamaModelDescriptor: Hashable, Sendable {
    public var id: UUID?
    public var url: URL
    public var mmprojURL: URL?
    public var displayName: String?
    public var filename: String
    public var hfRepo: String?
    public var hfFilename: String?

    public init(
        id: UUID? = nil,
        url: URL,
        mmprojURL: URL? = nil,
        displayName: String? = nil,
        filename: String? = nil,
        hfRepo: String? = nil,
        hfFilename: String? = nil
    ) {
        self.id = id
        self.url = url
        self.mmprojURL = mmprojURL
        self.displayName = displayName
        self.filename = filename ?? url.lastPathComponent
        self.hfRepo = hfRepo
        self.hfFilename = hfFilename
    }

    public init(model: InstalledModel, root: URL) {
        self.init(
            id: model.id,
            url: model.weightsURL(in: root),
            mmprojURL: model.mmprojURL(in: root),
            displayName: model.displayName,
            filename: model.filename,
            hfRepo: model.hfRepo,
            hfFilename: model.hfFilename
        )
    }
}

public struct LlamaLoadedModelInfo: Hashable, Sendable {
    public var modelID: UUID?
    public var modelPath: String
    public var displayName: String?
    public var filename: String
    public var contextSize: Int
    public var trainingContextSize: Int
    public var hasEmbeddedChatTemplate: Bool
    public var thinkingCapabilities: LLMThinkingCapabilities
    public var supportsMTPAcceleration: Bool
    /// Metadata-declared at load time when a projector is deferred, then narrowed
    /// to its validated capabilities after the first multimedia request. Query
    /// `LlamaEngine.currentLoadedModelInfo()` again after first use for current state.
    public var supportedInputModalities: Set<LLMInputModality>
    public var architecture: String?
    public var nextNPredictLayers: Int?
    public var logicalBatchSize: Int?
    public var physicalMicroBatchSize: Int?

    public init(
        modelID: UUID?,
        modelPath: String,
        displayName: String?,
        filename: String,
        contextSize: Int,
        trainingContextSize: Int,
        hasEmbeddedChatTemplate: Bool,
        thinkingCapabilities: LLMThinkingCapabilities = .none,
        supportsMTPAcceleration: Bool = false,
        supportedInputModalities: Set<LLMInputModality> = [.text],
        architecture: String? = nil,
        nextNPredictLayers: Int? = nil,
        logicalBatchSize: Int? = nil,
        physicalMicroBatchSize: Int? = nil
    ) {
        self.modelID = modelID
        self.modelPath = modelPath
        self.displayName = displayName
        self.filename = filename
        self.contextSize = contextSize
        self.trainingContextSize = trainingContextSize
        self.hasEmbeddedChatTemplate = hasEmbeddedChatTemplate
        self.thinkingCapabilities = thinkingCapabilities
        self.supportsMTPAcceleration = supportsMTPAcceleration
        self.supportedInputModalities = supportedInputModalities
        self.architecture = architecture
        self.nextNPredictLayers = nextNPredictLayers
        self.logicalBatchSize = logicalBatchSize
        self.physicalMicroBatchSize = physicalMicroBatchSize
    }

    public var supportsVision: Bool {
        supportedInputModalities.contains(.image)
    }

    public var supportsAudio: Bool {
        supportedInputModalities.contains(.audio)
    }
}

public actor LlamaEngine: LLMEngine, LLMPhasedGenerationProvider, LLMMultimodalGenerationProvider, LLMToolPhasedGenerationProvider {
    public static let shared = LlamaEngine()

    struct PromptPrefillPlan: Equatable {
        var commonPrefixCount: Int
        var retainedPrefixCount: Int
        var shouldClearMemory: Bool
        var removeStartPosition: Int?
        var decodeStartIndex: Int
    }

    enum GenerationGrammarMode: Equatable {
        case none
        case eager(grammar: String)
        case lazy(grammar: String, triggerPatterns: [String])

        var logLabel: String {
            switch self {
            case .none:
                return "none"
            case .eager:
                return "eager"
            case .lazy:
                return "lazy"
            }
        }

        var usesLazyGrammar: Bool {
            if case .lazy = self { return true }
            return false
        }
    }

    enum StructuredOutputPhase: String, Equatable {
        case thinking
        case awaitingFinal = "awaiting-final"
        case final
        case complete
    }

    struct StructuredOutputPlan: Equatable {
        var profile: OutputSanitizationProfile
        var continuingOpenThinkingPairs: [OutputDelimiterPair]
        var grammarMode: GenerationGrammarMode

        var usesLazyGrammar: Bool {
            grammarMode.usesLazyGrammar
        }
    }

    struct StreamPhasePlan: Equatable {
        var profile: OutputSanitizationProfile
        var continuingOpenThinkingPairs: [OutputDelimiterPair]
        var startsInThinking: Bool?

        var hasPhaseMarkers: Bool {
            startsInThinking != nil
                || !continuingOpenThinkingPairs.isEmpty
                || !profile.thinkingPairs.isEmpty
                || !profile.allFinalMarkers.isEmpty
        }
    }

    enum ReasoningBudgetInitialState: Equatable {
        case idle
        case counting
    }

    struct ReasoningBudgetPlan: Equatable {
        var pair: OutputDelimiterPair
        var budgetTokens: Int
        var message: String
        var initialState: ReasoningBudgetInitialState
    }

    struct PromptFormattingResult {
        var text: String
        var mode: LLMChatTemplateMode
        var outputProfile: OutputSanitizationProfile
        var checkpointUserContent: String? = nil
    }

    struct PromptTemplateOptions: Equatable {
        var enableThinking: Bool
        var reasoningEffort: LLMReasoningEffort?
        var preserveThinking: Bool?

        init(_ options: GenerationOptions) {
            enableThinking = options.enableThinking
            reasoningEffort = options.reasoningEffort
            preserveThinking = options.preserveThinking
        }
    }

    enum PromptFormattingCacheKey: Equatable {
        case systemUser(
            system: String,
            user: String,
            options: PromptTemplateOptions
        )
        case messages(
            messages: [ChatTemplateMessage],
            tools: [LLMToolDefinition],
            options: PromptTemplateOptions
        )
    }

    struct PromptTokenizationCache {
        var normalizedPrompt: String
        var tokens: [llama_token]
    }

    struct ToolAwareGenerationSegmentOverrideInput: Sendable {
        var renderedPrompt: String
        var templateMode: LLMChatTemplateMode
        var isInternalContinuation: Bool
    }

    struct ToolAwareGenerationSegmentOverrideOutput: Sendable {
        var finalText: String?
        var reasoningContent: String?
        var toolCalls: [LLMToolCall]
        var stopReason: String
        var triggerPhase: LLMStreamContentPhase?
        var remainingThinkingBudgetTokens: Int?
        var generatedTokens: Int

        init(
            finalText: String? = nil,
            reasoningContent: String? = nil,
            toolCalls: [LLMToolCall] = [],
            stopReason: String,
            triggerPhase: LLMStreamContentPhase? = nil,
            remainingThinkingBudgetTokens: Int? = nil,
            generatedTokens: Int = 0
        ) {
            self.finalText = finalText
            self.reasoningContent = reasoningContent
            self.toolCalls = toolCalls
            self.stopReason = stopReason
            self.triggerPhase = triggerPhase
            self.remainingThinkingBudgetTokens = remainingThinkingBudgetTokens
            self.generatedTokens = generatedTokens
        }
    }

    typealias ToolAwareGenerationSegmentOverride = @Sendable (
        ToolAwareGenerationSegmentOverrideInput
    ) async throws -> ToolAwareGenerationSegmentOverrideOutput

    private static let mtpAcceleratorName = "mtp"

    enum PreparedChatTemplate {
        case swiftJinja(ChatTemplatePromptFormatter)
        case unavailable(String)
    }

    enum MultimodalProjectorState {
        case missing
        case deferred(url: URL, declaredModalities: Set<LLMInputModality>)
        case ready(context: UnsafeMutableRawPointer, supportedModalities: Set<LLMInputModality>)
        case failed(detail: String)

        var context: UnsafeMutableRawPointer? {
            guard case .ready(let context, _) = self else { return nil }
            return context
        }
    }

    enum ModelLifecycleState: Equatable {
        case unloaded
        case ready
        case unloadPending
    }

    let configuration: LlamaEngineConfiguration

    var model: OpaquePointer?
    var context: OpaquePointer?
    var vocabulary: OpaquePointer?
    private var mtpContext: UnsafeMutableRawPointer?
    var multimodalProjectorState: MultimodalProjectorState = .missing
    var mtpCachedPromptTokens: [llama_token]?
    var promptCheckpoint: UnsafeMutableRawPointer?
    var promptCheckpointTokens: [llama_token]?
    var loadedDescriptor: LlamaModelDescriptor?
    var loadedInfo: LlamaLoadedModelInfo?
    var chatTemplate: String?
    var preparedChatTemplate: PreparedChatTemplate?
    var outputSanitizationProfile: OutputSanitizationProfile = .empty
    var cachedPromptTokens: [llama_token]?
    var promptFormattingCache: (key: PromptFormattingCacheKey, result: PromptFormattingResult)?
    var promptTokenizationCache: PromptTokenizationCache?
    var toolAwareGenerationSegmentOverride: ToolAwareGenerationSegmentOverride?
    private(set) var modelLifecycleState: ModelLifecycleState = .unloaded
    private var runtimeIdentity: LlamaRuntimeIdentity?
    private var newestManagedLifecycleEpoch: UInt64 = 0
    private var activeGenerationCount = 0

    public init(configuration: LlamaEngineConfiguration = LlamaEngineConfiguration()) {
        self.configuration = configuration
        LlamaBackend.ensureInitialized()
    }

    deinit {
        if case .ready(let context, _) = multimodalProjectorState {
            carbocation_mtmd_free_bridge(context)
        }
        if let promptCheckpoint { carbocation_llama_prompt_checkpoint_free_bridge(promptCheckpoint) }
        if let mtpContext { carbocation_llama_mtp_free_bridge(mtpContext) }
        if let context { llama_free(context) }
        if let model { llama_model_free(model) }
    }

    public func currentModelID() -> UUID? {
        loadedInfo?.modelID
    }

    public func currentContextSize() -> Int {
        loadedInfo?.contextSize ?? 0
    }

    public func currentTrainingContextSize() -> Int {
        loadedInfo?.trainingContextSize ?? 0
    }

    public func currentLoadedModelInfo() -> LlamaLoadedModelInfo? {
        loadedInfo
    }

    static func shouldUseMTPAcceleration(
        policy: LLMAccelerationPolicy,
        mtpContext: UnsafeMutableRawPointer?,
        grammarMode: GenerationGrammarMode,
        control: LLMGenerationControl?,
        hasIncompatibleControlPath: Bool
    ) -> Bool {
        _ = grammarMode
        _ = control
        return mtpAccelerationStatus(
            policy: policy,
            supportsMTPAcceleration: mtpContext != nil,
            mtpContext: mtpContext,
            hasIncompatibleControlPath: hasIncompatibleControlPath
        ) == .active
    }

    static func mtpAccelerationStatus(
        policy: LLMAccelerationPolicy,
        supportsMTPAcceleration: Bool,
        mtpContext: UnsafeMutableRawPointer?,
        hasIncompatibleControlPath: Bool
    ) -> LLMGenerationAccelerationStatus {
        guard supportsMTPAcceleration else {
            return .unsupported
        }
        guard policy == .automatic else {
            return .disabledByPolicy
        }
        guard !hasIncompatibleControlPath else {
            return .disabledByIncompatibleControlPath
        }
        guard mtpContext != nil else {
            return .runtimeUnavailable
        }
        return .active
    }

    static func mtpDiagnosticMessage(
        enabled: Bool,
        makeMessage: () -> String
    ) -> String? {
        guard enabled else { return nil }
        return makeMessage()
    }

    struct MTPAccelerationRuntime {
        var context: UnsafeMutableRawPointer?
        var stats: LLMGenerationAccelerationStats
    }

    func mtpAccelerationRuntime(
        hasIncompatibleControlPath: Bool
    ) -> MTPAccelerationRuntime {
        let maxDraftTokens = mtpMaxDraftTokens
        let status = Self.mtpAccelerationStatus(
            policy: configuration.accelerationPolicy,
            supportsMTPAcceleration: loadedInfo?.supportsMTPAcceleration ?? false,
            mtpContext: mtpContext,
            hasIncompatibleControlPath: hasIncompatibleControlPath
        )
        let activeContext = status == .active ? mtpContext : nil
        return MTPAccelerationRuntime(
            context: activeContext,
            stats: LLMGenerationAccelerationStats(
                status: status,
                accelerator: Self.mtpAcceleratorName,
                maxDraftTokens: activeContext == nil ? 0 : maxDraftTokens
            )
        )
    }

    var mtpMaxDraftTokens: Int {
        Self.effectiveMTPMaxDraftTokens(requested: configuration.mtpMaxDraftTokens)
    }

    static func clampedMTPMaxDraftTokens(_ requested: Int) -> Int {
        max(1, min(maximumRecurrentStateSnapshots, requested))
    }

    static func effectiveMTPMaxDraftTokens(
        requested: Int,
        metadata: GGUFMetadata.ModelMetadata
    ) -> Int {
        effectiveMTPMaxDraftTokens(requested: requested)
    }

    static func effectiveMTPMaxDraftTokens(requested: Int) -> Int {
        clampedMTPMaxDraftTokens(requested)
    }

    public func preflight(
        system: String,
        prompt: String,
        options: GenerationOptions
    ) async throws -> LLMGenerationPreflight {
        try requireReadyRuntime()
        guard context != nil,
              loadedInfo != nil else {
            throw LLMEngineError.noModelLoaded
        }

        let promptFormatting = try applyChatTemplate(system: system, user: prompt, options: options)
        return try preflight(promptFormatting: promptFormatting, options: options)
    }

    func preflight(
        promptFormatting: PromptFormattingResult,
        options: GenerationOptions
    ) throws -> LLMGenerationPreflight {
        try requireReadyRuntime()
        guard context != nil,
              let vocabulary,
              let loadedInfo else {
            throw LLMEngineError.noModelLoaded
        }
        try options.validateForLlamaBackend()

        let renderedPrompt = promptFormatting.text
        let promptTokens = try tokenizePrompt(renderedPrompt, vocab: vocabulary)
        guard !promptTokens.isEmpty else {
            throw LLMEngineError.tokenizationFailed
        }

        let activeOutputProfile = promptFormatting.outputProfile.merging(options.streamPhaseConfiguration)
        let continuingOpenThinkingPairs = Self.continuingOpenThinkingPairs(
            in: renderedPrompt,
            profile: activeOutputProfile
        )
        let grammarMode = Self.generationGrammarMode(
            for: options,
            profile: activeOutputProfile,
            continuingOpenThinkingPairs: continuingOpenThinkingPairs
        )
        let reasoningBudgetPlan = Self.reasoningBudgetPlan(
            for: options,
            profile: activeOutputProfile,
            continuingOpenThinkingPairs: continuingOpenThinkingPairs,
            startsInThinking: options.streamPhaseConfiguration.startsInThinking == true
        )
        if options.grammar != nil {
            let samplerRuntime = try buildSampler(
                grammarMode: grammarMode,
                options: options,
                vocab: vocabulary,
                reasoningBudgetPlan: reasoningBudgetPlan
            )
            llama_sampler_free(samplerRuntime.chain)
        }

        return LLMGenerationPreflight(
            loadedContextSize: loadedInfo.contextSize,
            modelTrainingContextSize: loadedInfo.trainingContextSize,
            promptTokens: promptTokens.count,
            reservedOutputTokens: configuration.promptReserveTokens,
            requestedMaxOutputTokens: options.maxOutputTokens,
            usesExactTokenCounts: true,
            templateMode: promptFormatting.mode
        )
    }

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
        params.configureLoadMode(useMemoryMap: true)

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

    @discardableResult
    public func load(
        model installed: InstalledModel,
        from root: URL,
        requestedContext: Int
    ) throws -> LlamaLoadedModelInfo {
        try load(descriptor: LlamaModelDescriptor(model: installed, root: root), requestedContext: requestedContext)
    }

    package func load(
        model installed: InstalledModel,
        from root: URL,
        requestedContext: Int,
        managedLifecycleEpoch: UInt64
    ) throws -> LlamaManagedRuntimeSnapshot {
        guard acceptManagedLifecycleEpoch(managedLifecycleEpoch) else {
            throw LlamaManagedRuntimeError.lifecycleOperationSuperseded
        }
        let info = try load(
            model: installed,
            from: root,
            requestedContext: requestedContext
        )
        guard let runtimeIdentity else {
            throw LLMEngineError.modelLoadFailed("Loaded runtime has no identity.")
        }
        return LlamaManagedRuntimeSnapshot(info: info, identity: runtimeIdentity)
    }

    @discardableResult
    public func load(
        modelAt url: URL,
        id: UUID? = nil,
        displayName: String? = nil,
        filename: String? = nil,
        hfRepo: String? = nil,
        hfFilename: String? = nil,
        requestedContext: Int
    ) throws -> LlamaLoadedModelInfo {
        try load(
            descriptor: LlamaModelDescriptor(
                id: id,
                url: url,
                displayName: displayName,
                filename: filename,
                hfRepo: hfRepo,
                hfFilename: hfFilename
            ),
            requestedContext: requestedContext
        )
    }

    @discardableResult
    public func load(
        descriptor: LlamaModelDescriptor,
        requestedContext: Int
    ) throws -> LlamaLoadedModelInfo {
        try Self.validateContextConfiguration(
            requestedContext: requestedContext,
            batchSizeLimit: configuration.batchSizeLimit,
            microBatchSizeLimit: configuration.microBatchSizeLimit
        )
        try requireModelLoadAvailable()

        let path = descriptor.url.path

        if modelLifecycleState == .ready,
           var loadedInfo,
           loadedInfo.modelPath == path,
           loadedDescriptor?.mmprojURL?.standardizedFileURL == descriptor.mmprojURL?.standardizedFileURL,
           context != nil {
            let desiredContext = Self.clampedContextSize(
                requestedContext: requestedContext,
                trainingContext: loadedInfo.trainingContextSize
            )
            if desiredContext == loadedInfo.contextSize {
                loadedInfo.modelID = descriptor.id
                loadedInfo.displayName = descriptor.displayName ?? loadedInfo.displayName
                loadedInfo.filename = descriptor.filename
                self.loadedDescriptor = descriptor
                self.loadedInfo = loadedInfo
                return loadedInfo
            }
        }

        performUnload()
        let ggufMetadata = GGUFMetadata.modelMetadata(at: descriptor.url)

        let supportsMTPAcceleration = ggufMetadata.supportsMTPAcceleration
        let shouldPrepareMTPAcceleration = supportsMTPAcceleration &&
            configuration.accelerationPolicy == .automatic
        var modelParams = llama_model_default_params()
        modelParams.n_gpu_layers = configuration.gpuLayerCount
        modelParams.configureLoadMode(useMemoryMap: configuration.useMemoryMap)
        modelParams.load_mtp = shouldPrepareMTPAcceleration
        if !Self.usesGPUOffload(gpuLayerCount: configuration.gpuLayerCount) {
            modelParams.configureForCPUOnly()
        }

        guard let loadedModel = path.withCString({ cPath in
            llama_model_load_from_file(cPath, modelParams)
        }) else {
            throw LLMEngineError.modelLoadFailed("llama_model_load_from_file returned null")
        }

        let trainingContext = Int(llama_model_n_ctx_train(loadedModel))
        let chosenContext = Self.clampedContextSize(
            requestedContext: requestedContext,
            trainingContext: trainingContext
        )

        let threads = configuration.threadCount
            ?? Int32(max(1, ProcessInfo.processInfo.activeProcessorCount / 2))

        let logicalBatchSize = min(
            max(1, chosenContext),
            max(1, configuration.batchSizeLimit)
        )
        let microBatchCandidates = Self.contextBatchCandidates(
            contextSize: chosenContext,
            batchSizeLimit: configuration.batchSizeLimit,
            microBatchSizeLimit: configuration.microBatchSizeLimit
        )
        let requestedMTPDraftTokens = Self.clampedMTPMaxDraftTokens(configuration.mtpMaxDraftTokens)
        let maxMTPDraftTokens = Self.effectiveMTPMaxDraftTokens(requested: requestedMTPDraftTokens)
        var attemptedMicroBatchSizes: [Int] = []
        var initializedContext: OpaquePointer?
        for microBatchSize in microBatchCandidates {
            attemptedMicroBatchSizes.append(microBatchSize)
            let contextParams = Self.contextParams(
                contextSize: chosenContext,
                batchSize: logicalBatchSize,
                microBatchSize: microBatchSize,
                threads: threads,
                recurrentStateSnapshots: shouldPrepareMTPAcceleration ? maxMTPDraftTokens : 0
            )
            if let context = llama_init_from_model(loadedModel, contextParams) {
                initializedContext = context
                break
            }
        }

        guard let loadedContext = initializedContext else {
            llama_model_free(loadedModel)
            let attempted = attemptedMicroBatchSizes.map(String.init).joined(separator: ", ")
            throw LLMEngineError.contextInitFailed(
                "llama_init_from_model returned null (context=\(chosenContext), logical batch=\(logicalBatchSize), attempted micro-batch sizes: \(attempted))"
            )
        }

        guard let loadedVocabulary = llama_model_get_vocab(loadedModel) else {
            llama_free(loadedContext)
            llama_model_free(loadedModel)
            throw LLMEngineError.modelLoadFailed("llama_model_get_vocab returned null")
        }
        let template = llama_model_chat_template(loadedModel, nil).map { String(cString: $0) }
        let preparedTemplate = Self.prepareChatTemplate(template)
        let outputProfile = OutputSanitizationProfile.derived(fromChatTemplate: template)
        let projectorPreparation = Self.deferredMultimodalProjectorState(
            mmprojURL: descriptor.mmprojURL
        )
        let info = LlamaLoadedModelInfo(
            modelID: descriptor.id,
            modelPath: path,
            displayName: descriptor.displayName,
            filename: descriptor.filename,
            contextSize: chosenContext,
            trainingContextSize: max(0, trainingContext),
            hasEmbeddedChatTemplate: template != nil,
            thinkingCapabilities: .derived(fromChatTemplate: template),
            supportsMTPAcceleration: supportsMTPAcceleration,
            supportedInputModalities: projectorPreparation.supportedInputModalities,
            architecture: ggufMetadata.architecture,
            nextNPredictLayers: ggufMetadata.nextNPredictLayers,
            logicalBatchSize: Int(llama_n_batch(loadedContext)),
            physicalMicroBatchSize: Int(llama_n_ubatch(loadedContext))
        )
        let loadedMTPContext: UnsafeMutableRawPointer?
        if shouldPrepareMTPAcceleration {
            loadedMTPContext = carbocation_llama_mtp_create_bridge(
                loadedModel,
                loadedContext,
                UInt32(chosenContext),
                UInt32(logicalBatchSize),
                threads,
                Int32(maxMTPDraftTokens),
                0
            )
            if loadedMTPContext == nil {
                llamaRuntimeLog.info(
                    "MTP acceleration unavailable after runtime initialization; generation will use standard decoding."
                )
            }
        } else {
            loadedMTPContext = nil
        }
        let loadedPromptCheckpoint = loadedMTPContext == nil
            ? carbocation_llama_prompt_checkpoint_create_bridge(loadedContext)
            : nil

        self.model = loadedModel
        self.context = loadedContext
        self.mtpContext = loadedMTPContext
        self.promptCheckpoint = loadedPromptCheckpoint
        self.multimodalProjectorState = projectorPreparation.state
        self.vocabulary = loadedVocabulary
        self.loadedDescriptor = descriptor
        self.loadedInfo = info
        self.modelLifecycleState = .ready
        self.runtimeIdentity = LlamaRuntimeIdentity()
        self.chatTemplate = template
        self.preparedChatTemplate = preparedTemplate
        self.outputSanitizationProfile = outputProfile
        clearPromptCaches()

        Self.logOutputSanitizationProfile(outputProfile, descriptor: descriptor, hasEmbeddedTemplate: template != nil)

        return info
    }

    @discardableResult
    func loadVocabularyOnlyForTesting(
        modelAt url: URL,
        displayName: String? = nil,
        filename: String? = nil,
        requestedContext: Int
    ) throws -> LlamaLoadedModelInfo {
        try requireModelLoadAvailable()

        let descriptor = LlamaModelDescriptor(
            url: url,
            displayName: displayName,
            filename: filename
        )
        let path = descriptor.url.path

        performUnload()

        var modelParams = llama_model_default_params()
        modelParams.vocab_only = true
        modelParams.configureLoadMode(useMemoryMap: true)
        modelParams.configureForCPUOnly()

        guard let loadedModel = path.withCString({ cPath in
            llama_model_load_from_file(cPath, modelParams)
        }) else {
            throw LLMEngineError.modelLoadFailed("llama_model_load_from_file returned null")
        }

        guard let loadedVocabulary = llama_model_get_vocab(loadedModel) else {
            llama_model_free(loadedModel)
            throw LLMEngineError.modelLoadFailed("llama_model_get_vocab returned null")
        }

        let trainingContext = Int(llama_model_n_ctx_train(loadedModel))
        let chosenContext = Self.clampedContextSize(
            requestedContext: requestedContext,
            trainingContext: trainingContext
        )
        let template = llama_model_chat_template(loadedModel, nil).map { String(cString: $0) }
        let preparedTemplate = Self.prepareChatTemplate(template)
        let outputProfile = OutputSanitizationProfile.derived(fromChatTemplate: template)
        let info = LlamaLoadedModelInfo(
            modelID: nil,
            modelPath: path,
            displayName: descriptor.displayName,
            filename: descriptor.filename,
            contextSize: chosenContext,
            trainingContextSize: max(0, trainingContext),
            hasEmbeddedChatTemplate: template != nil,
            thinkingCapabilities: .derived(fromChatTemplate: template),
            supportsMTPAcceleration: false
        )

        self.model = loadedModel
        self.context = nil
        self.mtpContext = nil
        self.promptCheckpoint = nil
        self.multimodalProjectorState = .missing
        self.vocabulary = loadedVocabulary
        self.loadedDescriptor = descriptor
        self.loadedInfo = info
        self.modelLifecycleState = .ready
        self.runtimeIdentity = LlamaRuntimeIdentity()
        self.chatTemplate = template
        self.preparedChatTemplate = preparedTemplate
        self.outputSanitizationProfile = outputProfile
        clearPromptCaches()

        Self.logOutputSanitizationProfile(outputProfile, descriptor: descriptor, hasEmbeddedTemplate: template != nil)

        return info
    }

    func setToolAwareGenerationSegmentOverrideForTesting(
        _ override: ToolAwareGenerationSegmentOverride?
    ) {
        toolAwareGenerationSegmentOverride = override
    }

    func setChatTemplateForTesting(_ template: String?) {
        chatTemplate = template
        preparedChatTemplate = Self.prepareChatTemplate(template)
        outputSanitizationProfile = OutputSanitizationProfile.derived(fromChatTemplate: template)
        clearPromptCaches()
        clearPromptPreparationCaches()
    }

    public func unload() {
        guard activeGenerationCount == 0 else {
            modelLifecycleState = .unloadPending
            return
        }
        performUnload()
    }

    @discardableResult
    package func unload(managedLifecycleEpoch: UInt64) -> Bool {
        guard acceptManagedLifecycleEpoch(managedLifecycleEpoch) else { return false }
        unload()
        return true
    }

    @discardableResult
    package func supersedeManagedLifecycleOperations(epoch: UInt64) -> Bool {
        acceptManagedLifecycleEpoch(epoch)
    }

    package func currentManagedRuntimeSnapshot() -> LlamaManagedRuntimeSnapshot? {
        guard modelLifecycleState == .ready,
              let loadedInfo,
              let runtimeIdentity else {
            return nil
        }
        return LlamaManagedRuntimeSnapshot(info: loadedInfo, identity: runtimeIdentity)
    }

    private func acceptManagedLifecycleEpoch(_ epoch: UInt64) -> Bool {
        guard epoch > newestManagedLifecycleEpoch else { return false }
        newestManagedLifecycleEpoch = epoch
        return true
    }

    func requireReadyRuntime() throws {
        if let expectedIdentity = LlamaManagedRuntimeContext.expectedIdentity {
            guard modelLifecycleState == .ready,
                  expectedIdentity == runtimeIdentity else {
                throw LlamaManagedRuntimeError.lifecycleOperationSuperseded
            }
            return
        }
        guard modelLifecycleState == .ready else {
            throw LLMEngineError.noModelLoaded
        }
    }

    func acquireGenerationLease() throws {
        try requireReadyRuntime()
        activeGenerationCount += 1
    }

    func endGenerationLease() {
        activeGenerationCount = max(0, activeGenerationCount - 1)
        if activeGenerationCount == 0, modelLifecycleState == .unloadPending {
            performUnload()
        }
    }

    func requireModelLoadAvailable() throws {
        guard activeGenerationCount == 0,
              modelLifecycleState != .unloadPending else {
            throw LLMEngineError.modelLoadFailed(
                "Cannot load a model while generation or a deferred unload is active."
            )
        }
    }

    private func performUnload() {
        clearPromptCaches()
        clearPromptPreparationCaches()
        if case .ready(let context, _) = multimodalProjectorState {
            carbocation_mtmd_free_bridge(context)
        }
        if let promptCheckpoint { carbocation_llama_prompt_checkpoint_free_bridge(promptCheckpoint) }
        if let mtpContext { carbocation_llama_mtp_free_bridge(mtpContext) }
        if let context { llama_free(context) }
        if let model { llama_model_free(model) }

        modelLifecycleState = .unloaded
        self.model = nil
        self.context = nil
        self.mtpContext = nil
        self.promptCheckpoint = nil
        self.multimodalProjectorState = .missing
        self.vocabulary = nil
        self.loadedDescriptor = nil
        self.loadedInfo = nil
        self.runtimeIdentity = nil
        self.chatTemplate = nil
        self.preparedChatTemplate = nil
        self.outputSanitizationProfile = .empty
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
        try await generate(
            system: system,
            prompt: prompt,
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
        guard context != nil, vocabulary != nil, loadedInfo != nil else {
            throw LLMEngineError.noModelLoaded
        }

        try acquireGenerationLease()
        defer { endGenerationLease() }
        let controlGenerationID = control?.beginGeneration()
        defer {
            if let controlGenerationID {
                control?.finishGeneration(controlGenerationID)
            }
        }

        let promptFormatting: PromptFormattingResult
        do {
            promptFormatting = try applyChatTemplate(system: system, user: prompt, options: options)
        } catch let error as LLMEngineError {
            if case .chatTemplateUnavailable = error {
                onPhaseAwareEvent(.requestSent(phase: .unknown))
                onPhaseAwareEvent(.generationStats(
                    promptTokens: 0,
                    generatedTokens: 0,
                    stopReason: "template-unavailable",
                    templateMode: .unavailable,
                    phase: .unknown
                ))
            }
            throw error
        }
        return try await generate(
            promptFormatting: promptFormatting,
            options: options,
            control: control,
            controlGenerationID: controlGenerationID,
            onPhaseAwareEvent: onPhaseAwareEvent
        )
    }

    public func generatePhased(
        system: String,
        prompt: String,
        options: GenerationOptions,
        control: LLMGenerationControl? = nil,
        onEvent: @escaping @Sendable (LLMGenerationStreamEvent) -> Void = { _ in }
    ) async throws -> LLMGenerationResult {
        guard context != nil, vocabulary != nil, loadedInfo != nil else {
            throw LLMEngineError.noModelLoaded
        }

        try acquireGenerationLease()
        defer { endGenerationLease() }
        let controlGenerationID = control?.beginGeneration()
        defer {
            if let controlGenerationID {
                control?.finishGeneration(controlGenerationID)
            }
        }

        let promptFormatting: PromptFormattingResult
        do {
            promptFormatting = try applyChatTemplate(system: system, user: prompt, options: options)
        } catch let error as LLMEngineError {
            if case .chatTemplateUnavailable = error {
                onEvent(.requestSent(phase: .unknown))
                onEvent(.generationStats(
                    promptTokens: 0,
                    generatedTokens: 0,
                    stopReason: "template-unavailable",
                    templateMode: .unavailable,
                    phase: .unknown
                ))
            }
            throw error
        }
        return try await generatePhased(
            promptFormatting: promptFormatting,
            options: options,
            control: control,
            controlGenerationID: controlGenerationID,
            onEvent: onEvent
        )
    }

    func generate(
        promptFormatting: PromptFormattingResult,
        options: GenerationOptions,
        control: LLMGenerationControl? = nil,
        controlGenerationID: UInt64? = nil,
        onPhaseAwareEvent: @Sendable (LLMPhaseAwareStreamEvent) -> Void
    ) async throws -> String {
        let result = try await generatePhased(
            promptFormatting: promptFormatting,
            options: options,
            control: control,
            controlGenerationID: controlGenerationID,
            onEvent: { event in
                if let phaseAwareEvent = event.phaseAwareEvent {
                    onPhaseAwareEvent(phaseAwareEvent)
                }
            }
        )
        return result.finalText
    }

    func generatePhased(
        promptFormatting: PromptFormattingResult,
        multimodalPrefill: LlamaPreparedMultimodalPrompt? = nil,
        options: GenerationOptions,
        control: LLMGenerationControl? = nil,
        controlGenerationID: UInt64? = nil,
        onEvent: @Sendable (LLMGenerationStreamEvent) -> Void
    ) async throws -> LLMGenerationResult {
        guard let context, let vocabulary else {
            throw LLMEngineError.noModelLoaded
        }
        try options.validateForLlamaBackend()

        func onPhaseAwareEvent(_ event: LLMPhaseAwareStreamEvent) {
            onEvent(LLMGenerationStreamEvent(adapting: event))
        }

        var currentPhase = LLMStreamContentPhase.unknown
        func updatePhase(_ nextPhase: LLMStreamContentPhase) {
            guard nextPhase != currentPhase else { return }
            let previousPhase = currentPhase
            currentPhase = nextPhase
            onPhaseAwareEvent(.phaseChanged(from: previousPhase, to: nextPhase))
        }

        onPhaseAwareEvent(.requestSent(phase: currentPhase))
        let startedAt = Date()

        var templateMode: LLMChatTemplateMode = .unavailable
        var promptTokenCount = 0
        var generatedTokenCount = 0
        var stopReason = "cancelled"
        var emittedStats = false
        var emittedAccelerationStats = false
        var accelerationStats: LLMGenerationAccelerationStats?
        var promptContextPrepared = false
        var promptCacheCommitted = false

        func emitAccelerationStatsIfNeeded() {
            guard !emittedAccelerationStats, let accelerationStats else {
                return
            }
            emittedAccelerationStats = true
            onPhaseAwareEvent(.accelerationStats(accelerationStats))
        }

        defer {
            if promptContextPrepared && !promptCacheCommitted {
                clearPromptCaches()
            }
            emitAccelerationStatsIfNeeded()
            if !emittedStats {
                onPhaseAwareEvent(.generationStats(
                    promptTokens: promptTokenCount,
                    generatedTokens: generatedTokenCount,
                    stopReason: stopReason,
                    templateMode: templateMode,
                    phase: currentPhase
                ))
            }
        }

        let renderedPrompt = promptFormatting.text
        templateMode = promptFormatting.mode

        let promptTokens: [llama_token]
        let promptPositionCount: Int
        if let multimodalPrefill {
            promptTokens = []
            promptTokenCount = multimodalPrefill.promptTokenCount
            promptPositionCount = multimodalPrefill.promptPositionCount
        } else {
            promptTokens = try tokenizePrompt(renderedPrompt, vocab: vocabulary)
            promptTokenCount = promptTokens.count
            promptPositionCount = promptTokens.count
        }
        guard promptTokenCount > 0 else {
            if multimodalPrefill == nil {
                throw LLMEngineError.tokenizationFailed
            }
            throw LLMEngineError.imageTokenizationFailed("mtmd produced no prompt tokens.")
        }
        let promptCheckpointTokenCount = multimodalPrefill == nil
            ? self.promptCheckpointTokenCount(
                promptFormatting: promptFormatting,
                promptTokens: promptTokens,
                vocab: vocabulary
            )
            : nil
        guard promptPositionCount < currentContextSize() else {
            if multimodalPrefill == nil {
                throw LLMEngineError.insufficientGenerationBudget(
                    contextSize: currentContextSize(),
                    promptTokens: promptTokenCount,
                    reserve: configuration.promptReserveTokens
                )
            }
            throw LLMEngineError.contextBudgetExceeded(
                contextSize: currentContextSize(),
                promptTokens: promptTokenCount,
                reserve: configuration.promptReserveTokens
            )
        }

        let activeOutputProfile = promptFormatting.outputProfile.merging(options.streamPhaseConfiguration)
        let continuingOpenThinkingPairs = Self.continuingOpenThinkingPairs(
            in: renderedPrompt,
            profile: activeOutputProfile
        )
        let streamPhasePlan = StreamPhasePlan(
            profile: activeOutputProfile,
            continuingOpenThinkingPairs: continuingOpenThinkingPairs,
            startsInThinking: options.streamPhaseConfiguration.startsInThinking
        )
        updatePhase(Self.streamContentPhase(in: "", plan: streamPhasePlan))
        let grammarMode = Self.generationGrammarMode(
            for: options,
            profile: activeOutputProfile,
            continuingOpenThinkingPairs: continuingOpenThinkingPairs
        )
        let structuredOutputPlan = grammarMode.usesLazyGrammar
            ? StructuredOutputPlan(
                profile: activeOutputProfile,
                continuingOpenThinkingPairs: continuingOpenThinkingPairs,
                grammarMode: grammarMode
            )
            : nil
        let reasoningBudgetPlan = Self.reasoningBudgetPlan(
            for: options,
            profile: activeOutputProfile,
            continuingOpenThinkingPairs: continuingOpenThinkingPairs,
            startsInThinking: streamPhasePlan.startsInThinking == true,
            requiresSampler: control != nil
        )
        llamaRuntimeLog.info(
            "Generation grammar mode selected: mode=\(grammarMode.logLabel, privacy: .public) enableThinking=\(options.enableThinking, privacy: .public) continuingOpenThinkingPairs=\(continuingOpenThinkingPairs.count, privacy: .public) thinkingBudgetActive=\(reasoningBudgetPlan != nil, privacy: .public)"
        )

        let samplerRuntime = try buildSampler(
            grammarMode: grammarMode,
            options: options,
            vocab: vocabulary,
            reasoningBudgetPlan: reasoningBudgetPlan
        )
        let sampler = samplerRuntime.chain
        defer { llama_sampler_free(sampler) }

        let hasIncompatibleMTPControlPath = multimodalPrefill != nil
        let maxMTPDraftTokens = mtpMaxDraftTokens
        let mtpStatus = Self.mtpAccelerationStatus(
            policy: configuration.accelerationPolicy,
            supportsMTPAcceleration: loadedInfo?.supportsMTPAcceleration ?? false,
            mtpContext: mtpContext,
            hasIncompatibleControlPath: hasIncompatibleMTPControlPath
        )
        let activeMTPContext = mtpStatus == .active ? mtpContext : nil
        accelerationStats = LLMGenerationAccelerationStats(
            status: mtpStatus,
            accelerator: Self.mtpAcceleratorName,
            maxDraftTokens: activeMTPContext == nil ? 0 : maxMTPDraftTokens
        )
        if let multimodalPrefill {
            clearPromptRuntimeState(context: context, mtpContext: mtpContext)
            try multimodalPrefill.evaluate(
                mtmdContext: multimodalProjectorState.context,
                llamaContext: context,
                batchSize: Int32(max(1, llama_n_batch(context)))
            )
        } else {
            try preparePromptContext(
                promptTokens,
                context: context,
                mtpContext: activeMTPContext,
                checkpointTokenCount: promptCheckpointTokenCount
            )
        }
        promptContextPrepared = true

        var accumulatedData = Data()
        var accumulatedText = ""
        var incrementalUTF8 = IncrementalUTF8Accumulator()
        var reasoningBudgetExhaustionLogged = false
        func logReasoningBudgetExhaustionIfNeeded(
            state: carbocation_llama_reasoning_budget_state,
            generatedTokens: Int
        ) {
            guard !reasoningBudgetExhaustionLogged,
                  Self.reasoningBudgetStateIsExhausted(state),
                  let budgetTokens = reasoningBudgetPlan?.budgetTokens else {
                return
            }

            reasoningBudgetExhaustionLogged = true
            llamaRuntimeLog.info(
                "Reasoning budget exhausted: budgetTokens=\(budgetTokens, privacy: .public) generatedTokens=\(generatedTokens, privacy: .public) rawBytes=\(accumulatedData.count, privacy: .public) state=\(Self.reasoningBudgetStateLogLabel(state), privacy: .public)"
            )
        }

        if let reasoningBudgetSampler = samplerRuntime.reasoningBudgetSampler {
            logReasoningBudgetExhaustionIfNeeded(
                state: carbocation_llama_reasoning_budget_sampler_state(reasoningBudgetSampler),
                generatedTokens: generatedTokenCount
            )
        }
        var sawFirstToken = false
        func emitFirstByteIfNeeded() {
            guard !sawFirstToken else { return }
            sawFirstToken = true
            onPhaseAwareEvent(.firstByteReceived(
                after: Date().timeIntervalSince(startedAt),
                phase: currentPhase
            ))
        }

        let effectiveStopSequences = Self.mergingStopSequences(
            options.stopSequences,
            activeOutputProfile.extraStopStrings
        )
        let streamingControlLiterals = Self.streamingControlLiterals(
            plan: streamPhasePlan,
            stopSequences: effectiveStopSequences
        )
        var lastHeartbeat = Date()
        var streamedThinkingText = ""
        var streamedFinalAnswer = ""
        var hasEmittedThinkingContent = false
        var hasEmittedFinalAnswerContent = false
        var structuredPhase = structuredOutputPlan.map {
            Self.structuredOutputPhase(in: accumulatedText, plan: $0)
        }
        func emitThinking(_ nextThinkingText: String, snapshotReason: LLMGenerationContentSnapshotReason) {
            guard nextThinkingText != streamedThinkingText else { return }

            if nextThinkingText.hasPrefix(streamedThinkingText) {
                let delta = String(nextThinkingText.dropFirst(streamedThinkingText.count))
                if !delta.isEmpty {
                    onEvent(.contentDelta(
                        phase: .thinking,
                        text: delta,
                        bytesSoFar: nextThinkingText.utf8.count
                    ))
                }
            } else {
                onEvent(.contentSnapshot(
                    phase: .thinking,
                    text: nextThinkingText,
                    bytesSoFar: nextThinkingText.utf8.count,
                    reason: snapshotReason
                ))
            }

            streamedThinkingText = nextThinkingText
            if Self.hasVisibleStreamingContent(nextThinkingText) {
                hasEmittedThinkingContent = true
            }
        }

        func emitFinalAnswer(_ nextFinalAnswer: String, snapshotReason: LLMFinalAnswerSnapshotReason) {
            guard nextFinalAnswer != streamedFinalAnswer else { return }

            if nextFinalAnswer.hasPrefix(streamedFinalAnswer) {
                let delta = String(nextFinalAnswer.dropFirst(streamedFinalAnswer.count))
                if !delta.isEmpty {
                    onPhaseAwareEvent(.finalAnswerDelta(
                        text: delta,
                        bytesSoFar: nextFinalAnswer.utf8.count
                    ))
                }
            } else {
                onPhaseAwareEvent(.finalAnswerSnapshot(
                    text: nextFinalAnswer,
                    bytesSoFar: nextFinalAnswer.utf8.count,
                    reason: snapshotReason
                ))
            }

            streamedFinalAnswer = nextFinalAnswer
            if Self.hasVisibleStreamingContent(nextFinalAnswer) {
                hasEmittedFinalAnswerContent = true
            }
        }

        func emitFinalAnswerProgressIfNeeded(raw: String? = nil) {
            guard Self.shouldEmitFinalAnswerProgress(
                currentPhase: currentPhase,
                structuredPhase: structuredPhase
            ),
                  let nextFinalAnswer = try? Self.sanitizedGeneratedText(
                    raw ?? accumulatedText,
                    profile: activeOutputProfile,
                    continuingOpenThinkingPairs: continuingOpenThinkingPairs,
                    requiresNonEmptyStructuredOutput: false
                  )
            else { return }

            emitFinalAnswer(nextFinalAnswer, snapshotReason: .streamCorrection)
        }

        func emitThinkingProgressIfNeeded(raw: String? = nil) {
            let snapshot = Self.phasedGeneratedText(raw ?? accumulatedText, plan: streamPhasePlan)
            emitThinking(snapshot.thinkingText, snapshotReason: .streamCorrection)
        }

        let contextMaxNew = Self.maxGenerationTokens(
            contextSize: currentContextSize(),
            promptTokenCount: promptPositionCount,
            reserve: configuration.promptReserveTokens
        )
        let maxNew: Int
        if let requestedMax = options.maxOutputTokens, requestedMax > 0 {
            maxNew = min(contextMaxNew, requestedMax)
        } else {
            maxNew = contextMaxNew
        }

        guard maxNew > 0 else {
            if multimodalPrefill == nil {
                throw LLMEngineError.insufficientGenerationBudget(
                    contextSize: currentContextSize(),
                    promptTokens: promptTokenCount,
                    reserve: configuration.promptReserveTokens
                )
            }
            throw LLMEngineError.contextBudgetExceeded(
                contextSize: currentContextSize(),
                promptTokens: promptTokenCount,
                reserve: configuration.promptReserveTokens
            )
        }

        var incrementalParser = IncrementalGenerationParser(
            streamPhasePlan: streamPhasePlan,
            structuredOutputPlan: structuredOutputPlan,
            stopSequences: effectiveStopSequences,
            stopAtBalancedJSON: options.stopAtBalancedJSON
        )

        let tokenDiagnosticsEnabled = configuration.emitsMTPDiagnostics
        func emitTokenDiagnostic(
            _ token: llama_token,
            generatedTokensIncludingThisToken: Int,
            source: String
        ) {
            guard tokenDiagnosticsEnabled else { return }
            let position = promptPositionCount + generatedTokensIncludingThisToken - 1
            onPhaseAwareEvent(.diagnostic(
                message: "token-diagnostic generated index=\(generatedTokensIncludingThisToken) pos=\(position) source=\(source) token=\(diagnosticTokenDescription(token, vocab: vocabulary))"
            ))
        }

        func processGeneratedToken(
            _ token: llama_token,
            generatedTokensIncludingThisToken: Int,
            reasoningBudgetState: carbocation_llama_reasoning_budget_state?,
            diagnosticSource: String
        ) -> Bool {
            emitTokenDiagnostic(
                token,
                generatedTokensIncludingThisToken: generatedTokensIncludingThisToken,
                source: diagnosticSource
            )

            if llama_vocab_is_eog(vocabulary, token) {
                if let reasoningBudgetState {
                    logReasoningBudgetExhaustionIfNeeded(
                        state: reasoningBudgetState,
                        generatedTokens: max(0, generatedTokensIncludingThisToken - 1)
                    )
                }
                stopReason = "eog"
                return false
            }

            let rawPiece = tokenToPiece(vocab: vocabulary, token: token)
            let piece = rawPiece.isEmpty
                ? tokenToPiece(vocab: vocabulary, token: token, special: true)
                : rawPiece
            var appendedText = ""
            if !piece.isEmpty {
                accumulatedData.append(piece)
                appendedText = incrementalUTF8.append(piece)
                if !appendedText.isEmpty {
                    accumulatedText.append(contentsOf: appendedText)
                }
            }

            if let reasoningBudgetState {
                logReasoningBudgetExhaustionIfNeeded(
                    state: reasoningBudgetState,
                    generatedTokens: generatedTokensIncludingThisToken
                )
            }

            if !piece.isEmpty {
                let snapshot = incrementalParser.append(
                    appendedText,
                    fullText: accumulatedText
                )
                if snapshot.structuredPhase != structuredPhase {
                    structuredPhase = snapshot.structuredPhase
                    if let nextPhase = snapshot.structuredPhase {
                        llamaRuntimeLog.info(
                            "Structured output phase changed: phase=\(nextPhase.rawValue, privacy: .public) rawBytes=\(accumulatedData.count, privacy: .public)"
                        )
                    }
                }

                let boundary = snapshot.boundary
                if let boundary {
                    accumulatedText = boundary.text
                    stopReason = boundary.reason
                }

                updatePhase(boundary == nil
                    ? snapshot.phase
                    : Self.streamContentPhase(in: accumulatedText, plan: streamPhasePlan))

                if stopReason == "json-complete" || stopReason == "stop-sequence" || stopReason == "tool-call-complete" {
                    emitFirstByteIfNeeded()
                    emitThinkingProgressIfNeeded()
                    emitFinalAnswerProgressIfNeeded()
                    if stopReason == "json-complete", structuredOutputPlan != nil {
                        incrementalParser.markStructuredOutputComplete()
                        structuredPhase = .complete
                    }
                    return false
                }
            }

            emitFirstByteIfNeeded()

            let stableRaw = Self.stableStreamingPrefix(
                accumulatedText,
                protecting: streamingControlLiterals
            )
            if currentPhase == .thinking, !hasEmittedThinkingContent {
                emitThinkingProgressIfNeeded(raw: stableRaw)
                if hasEmittedThinkingContent {
                    lastHeartbeat = Date()
                }
            } else if currentPhase != .thinking, !hasEmittedFinalAnswerContent {
                emitFinalAnswerProgressIfNeeded(raw: stableRaw)
                if hasEmittedFinalAnswerContent {
                    lastHeartbeat = Date()
                }
            }

            let now = Date()
            if now.timeIntervalSince(lastHeartbeat) >= configuration.heartbeatInterval {
                lastHeartbeat = now
                emitThinkingProgressIfNeeded(raw: stableRaw)
                emitFinalAnswerProgressIfNeeded(raw: stableRaw)
                onPhaseAwareEvent(.tokenChunk(
                    preview: String(accumulatedText.suffix(60)),
                    bytesSoFar: accumulatedData.count,
                    phase: currentPhase
                ))
            }

            return true
        }

        func decodeSingleToken(_ token: llama_token) throws {
            var oneToken: [llama_token] = [token]
            let decodeResult = oneToken.withUnsafeMutableBufferPointer { buffer -> Int32 in
                let batch = llama_batch_get_one(buffer.baseAddress, 1)
                return llama_decode(context, batch)
            }
            if decodeResult != 0 {
                clearPromptCaches()
                throw LLMEngineError.decodeFailed
            }
        }

        func decodeMTPVerificationTargetTokens(
            _ tokens: [llama_token],
            startPosition: Int,
            mtpContext: UnsafeMutableRawPointer
        ) throws {
            var tokens = tokens
            let decodeResult = tokens.withUnsafeMutableBufferPointer { buffer -> Int32 in
                carbocation_llama_mtp_decode_verification_target_tokens_bridge(
                    mtpContext,
                    buffer.baseAddress,
                    Int32(buffer.count),
                    Int32(startPosition)
                )
            }
            if decodeResult != 0 {
                clearPromptCaches()
                throw LLMEngineError.decodeFailed
            }
        }

        func processLastMTPVerificationBatch(
            mtpContext: UnsafeMutableRawPointer
        ) throws {
            let processResult = carbocation_llama_mtp_process_last_target_batch_bridge(mtpContext)
            if processResult != 0 {
                clearPromptCaches()
                throw LLMEngineError.decodeFailed
            }
        }

        struct PendingMTPToken {
            var token: llama_token
            var reasoningBudgetState: carbocation_llama_reasoning_budget_state?
            var alreadyProcessed: Bool
            var diagnosticSource: String
        }

        var pendingNextToken: PendingMTPToken?
        var generatedTokenHistory: [llama_token] = []

        func sampleTargetToken(at index: Int32) -> llama_token {
            return llama_sampler_sample(sampler, context, index)
        }

        let mtpDiagnosticsEnabled = configuration.emitsMTPDiagnostics
            && activeMTPContext != nil
            && maxMTPDraftTokens > 2
        func emitMTPDiagnostic(_ message: @autoclosure () -> String) {
            guard let message = Self.mtpDiagnosticMessage(
                enabled: mtpDiagnosticsEnabled,
                makeMessage: message
            ) else { return }
            onPhaseAwareEvent(.diagnostic(message: "mtp-diagnostic \(message)"))
        }

        while generatedTokenCount < maxNew {
            try Task.checkCancellation()
            if accumulatedData.isEmpty || incrementalUTF8.isAtUTF8Boundary {
                applyThinkingTerminationIfRequested(
                    control: control,
                    generationID: controlGenerationID,
                    currentPhase: currentPhase,
                    samplerRuntime: samplerRuntime,
                    reasoningBudgetPlan: reasoningBudgetPlan,
                    vocab: vocabulary
                )
            }

            let tokenPosition = promptPositionCount + generatedTokenHistory.count
            let next: llama_token
            let nextDiagnosticSource: String
            let reasoningBudgetState: carbocation_llama_reasoning_budget_state?
            let nextAlreadyProcessed: Bool
            if let pending = pendingNextToken {
                next = pending.token
                nextDiagnosticSource = pending.diagnosticSource
                reasoningBudgetState = pending.reasoningBudgetState
                nextAlreadyProcessed = pending.alreadyProcessed
                pendingNextToken = nil
            } else {
                next = sampleTargetToken(at: -1)
                nextDiagnosticSource = "sampled"
                reasoningBudgetState = samplerRuntime.reasoningBudgetSampler.map {
                    carbocation_llama_reasoning_budget_sampler_state($0)
                }
                nextAlreadyProcessed = false
            }
            var emittedNextTokenCount = 0
            if !nextAlreadyProcessed {
                guard processGeneratedToken(
                    next,
                    generatedTokensIncludingThisToken: generatedTokenCount + 1,
                    reasoningBudgetState: reasoningBudgetState,
                    diagnosticSource: nextDiagnosticSource
                ) else {
                    break
                }
                emittedNextTokenCount = 1
            }

            if let activeMTPContext {
                let remainingDraftCapacity = maxNew - generatedTokenCount - emittedNextTokenCount
                let draftLimit = min(maxMTPDraftTokens, remainingDraftCapacity)
                var draftCallIndex: Int?
                let draftTokens: [llama_token]
                if draftLimit > 0 {
                    if mtpDiagnosticsEnabled {
                        draftCallIndex = (accelerationStats?.draftCalls ?? 0) + 1
                    }
                    var draftBuffer = [llama_token](repeating: 0, count: draftLimit)
                    let draftCount = draftBuffer.withUnsafeMutableBufferPointer { buffer in
                        carbocation_llama_mtp_draft_bridge(
                            activeMTPContext,
                            next,
                            Int32(tokenPosition),
                            buffer.baseAddress,
                            Int32(buffer.count)
                        )
                    }
                    draftTokens = Array(draftBuffer.prefix(min(draftBuffer.count, max(0, Int(draftCount)))))
                    accelerationStats?.draftCalls += 1
                    accelerationStats?.draftTokensGenerated += draftTokens.count
                    emitMTPDiagnostic(
                        "draft call=\(draftCallIndex ?? 0) pos=\(tokenPosition) next=\(diagnosticTokenDescription(next, vocab: vocabulary)) draft=\(diagnosticTokenList(draftTokens, vocab: vocabulary))"
                    )
                } else {
                    draftTokens = []
                }

                var emittedAcceptedDraftCount = 0
                var emittedCorrectionToken: llama_token?
                var shouldContinue = true

                if draftTokens.isEmpty {
                    try decodeMTPVerificationTargetTokens(
                        [next],
                        startPosition: tokenPosition,
                        mtpContext: activeMTPContext
                    )
                    if generatedTokenCount + emittedNextTokenCount < maxNew {
                        let sampledPrefetch = sampleTargetToken(at: 0)
                        pendingNextToken = PendingMTPToken(
                            token: sampledPrefetch,
                            reasoningBudgetState: samplerRuntime.reasoningBudgetSampler.map {
                                carbocation_llama_reasoning_budget_sampler_state($0)
                            },
                            alreadyProcessed: false,
                            diagnosticSource: "prefetched"
                        )
                    }
                    try processLastMTPVerificationBatch(mtpContext: activeMTPContext)
                } else {
                    let verificationTokens = [next] + draftTokens
                    try decodeMTPVerificationTargetTokens(
                        verificationTokens,
                        startPosition: tokenPosition,
                        mtpContext: activeMTPContext
                    )

                    var correctionToken: llama_token?
                    var correctionReasoningBudgetState: carbocation_llama_reasoning_budget_state?
                    var verifierSamples: [llama_token]? = mtpDiagnosticsEnabled ? [] : nil
                    for (verifierIndex, draftToken) in draftTokens.enumerated() {
                        let sampled = sampleTargetToken(at: Int32(verifierIndex))
                        verifierSamples?.append(sampled)
                        let acceptedReasoningBudgetState = samplerRuntime.reasoningBudgetSampler.map {
                            carbocation_llama_reasoning_budget_sampler_state($0)
                        }

                        if sampled != draftToken {
                            correctionToken = sampled
                            correctionReasoningBudgetState = acceptedReasoningBudgetState
                            break
                        }

                        emittedAcceptedDraftCount += 1
                        shouldContinue = processGeneratedToken(
                            draftToken,
                            generatedTokensIncludingThisToken: generatedTokenCount + emittedNextTokenCount + emittedAcceptedDraftCount,
                            reasoningBudgetState: acceptedReasoningBudgetState,
                            diagnosticSource: "accepted-draft"
                        )
                        if !shouldContinue {
                            break
                        }
                    }

                    if emittedAcceptedDraftCount < draftTokens.count {
                        let retainedTokens = [next] + Array(draftTokens.prefix(emittedAcceptedDraftCount))
                        try processLastMTPVerificationBatch(mtpContext: activeMTPContext)
                        _ = carbocation_llama_mtp_accept_bridge(
                            activeMTPContext,
                            Int32(emittedAcceptedDraftCount)
                        )
                        let rollbackPosition = tokenPosition + retainedTokens.count
                        let rollbackResult = carbocation_llama_mtp_rollback_bridge(
                            activeMTPContext,
                            Int32(rollbackPosition)
                        )
                        if rollbackResult == 0 {
                            clearPromptCaches()
                            throw LLMEngineError.decodeFailed
                        }

                        if let correctionToken, shouldContinue {
                            emittedCorrectionToken = correctionToken
                            shouldContinue = processGeneratedToken(
                                correctionToken,
                                generatedTokensIncludingThisToken: generatedTokenCount + emittedNextTokenCount + emittedAcceptedDraftCount + 1,
                                reasoningBudgetState: correctionReasoningBudgetState,
                                diagnosticSource: "correction"
                            )
                            if shouldContinue {
                                pendingNextToken = PendingMTPToken(
                                    token: correctionToken,
                                    reasoningBudgetState: correctionReasoningBudgetState,
                                    alreadyProcessed: true,
                                    diagnosticSource: "correction"
                                )
                            }
                        }
                        if mtpDiagnosticsEnabled {
                            let verificationPrefixTokens = promptTokens + generatedTokenHistory
                            if let command = diagnosticBatchEquivalenceCommand(
                                modelPath: loadedInfo?.modelPath,
                                prefixTokens: verificationPrefixTokens,
                                windowTokens: verificationTokens,
                                recurrentStateSnapshots: maxMTPDraftTokens,
                                samplerPrefixSkip: promptTokens.count
                            ) {
                                emitMTPDiagnostic("repro-command \(command)")
                            }
                            emitMTPDiagnostic(
                                "repro call=\(draftCallIndex ?? 0) pos=\(tokenPosition) prefix-token-count=\(verificationPrefixTokens.count) prefix-token-ids=\(diagnosticTokenIDList(verificationPrefixTokens)) window-token-ids=\(diagnosticTokenIDList(verificationTokens))"
                            )
                            emitMTPDiagnostic(
                                "verify call=\(draftCallIndex ?? 0) pos=\(tokenPosition) accepted=\(emittedAcceptedDraftCount)/\(draftTokens.count) draft=\(diagnosticTokenList(draftTokens, vocab: vocabulary)) sampled=\(diagnosticTokenList(verifierSamples ?? [], vocab: vocabulary)) correction=\(correctionToken.map { diagnosticTokenDescription($0, vocab: vocabulary) } ?? "none")"
                            )
                        }
                    } else {
                        var prefetchedDiagnosticToken: llama_token?
                        if shouldContinue,
                           generatedTokenCount + emittedNextTokenCount + emittedAcceptedDraftCount < maxNew {
                            let sampledPrefetch = sampleTargetToken(at: Int32(draftTokens.count))
                            if mtpDiagnosticsEnabled {
                                prefetchedDiagnosticToken = sampledPrefetch
                            }
                            pendingNextToken = PendingMTPToken(
                                token: sampledPrefetch,
                                reasoningBudgetState: samplerRuntime.reasoningBudgetSampler.map {
                                    carbocation_llama_reasoning_budget_sampler_state($0)
                                },
                                alreadyProcessed: false,
                                diagnosticSource: "prefetched"
                            )
                        }

                        try processLastMTPVerificationBatch(mtpContext: activeMTPContext)

                        _ = carbocation_llama_mtp_accept_bridge(
                            activeMTPContext,
                            Int32(emittedAcceptedDraftCount)
                        )
                        emitMTPDiagnostic(
                            "verify call=\(draftCallIndex ?? 0) pos=\(tokenPosition) accepted=full/\(draftTokens.count) draft=\(diagnosticTokenList(draftTokens, vocab: vocabulary)) sampled=\(diagnosticTokenList(verifierSamples ?? [], vocab: vocabulary)) prefetch=\(prefetchedDiagnosticToken.map { diagnosticTokenDescription($0, vocab: vocabulary) } ?? "none")"
                        )
                    }
                }

                accelerationStats?.draftTokensAccepted += emittedAcceptedDraftCount
                let retainedTokenCount = emittedNextTokenCount + emittedAcceptedDraftCount + (emittedCorrectionToken == nil ? 0 : 1)
                generatedTokenHistory.append(next)
                if emittedAcceptedDraftCount > 0 {
                    generatedTokenHistory.append(contentsOf: draftTokens.prefix(emittedAcceptedDraftCount))
                }
                generatedTokenCount += retainedTokenCount
                if !shouldContinue {
                    break
                }
                continue
            }

            try decodeSingleToken(next)

            generatedTokenHistory.append(next)
            generatedTokenCount += 1
        }

        if stopReason == "cancelled" {
            stopReason = "max-tokens"
        }

        if accumulatedText.isEmpty, !accumulatedData.isEmpty {
            accumulatedText = String(decoding: accumulatedData, as: UTF8.self)
        }

        updatePhase(Self.streamContentPhase(in: accumulatedText, plan: streamPhasePlan))
        if stopReason == "max-tokens", currentPhase == .thinking {
            stopReason = "thinking-not-closed"
        }
        let shouldReturnIncompleteThinking = stopReason == "thinking-not-closed"

        if let plan = structuredOutputPlan {
            let finalPhase = structuredPhase ?? Self.structuredOutputPhase(in: accumulatedText, plan: plan)
            if finalPhase == .thinking || finalPhase == .awaitingFinal {
                stopReason = finalPhase == .thinking
                    ? "thinking-not-closed"
                    : "structured-output-not-started"
                llamaRuntimeLog.error(
                    "Structured output phase failed before sanitization: phase=\(finalPhase.rawValue, privacy: .public) rawBytes=\(accumulatedText.utf8.count, privacy: .public) stopReason=\(stopReason, privacy: .public)"
                )
                if !(shouldReturnIncompleteThinking && finalPhase == .thinking) {
                    throw LLMEngineError.structuredOutputPhaseFailed(
                        "Generation ended before final structured output began."
                    )
                }
            }
        }

        let phasedText = Self.phasedGeneratedText(accumulatedText, plan: streamPhasePlan)
        let returnedText: String
        if stopReason == "thinking-not-closed" {
            returnedText = phasedText.finalText
        } else {
            do {
                returnedText = try Self.sanitizedGeneratedText(
                    accumulatedText,
                    profile: activeOutputProfile,
                    continuingOpenThinkingPairs: continuingOpenThinkingPairs,
                    requiresNonEmptyStructuredOutput: options.grammar != nil && !activeOutputProfile.isEmpty
                )
            } catch let error as LLMEngineError {
                if case .structuredOutputPhaseFailed = error {
                    stopReason = "structured-sanitization-empty"
                    llamaRuntimeLog.error(
                        "Structured output sanitization failed: rawBytes=\(accumulatedText.utf8.count, privacy: .public) stopReason=\(stopReason, privacy: .public)"
                    )
                }
                throw error
            }
        }

        let resultPhases = phasedText.replacingPhaseText(.final, with: returnedText)
        emitThinking(resultPhases.thinkingText, snapshotReason: .completed)
        emitFinalAnswer(returnedText, snapshotReason: .completed)

        llamaRuntimeLog.info(
            "Generation sanitized output: grammarMode=\(grammarMode.logLabel, privacy: .public) rawBytes=\(accumulatedText.utf8.count, privacy: .public) sanitizedBytes=\(returnedText.utf8.count, privacy: .public) stopReason=\(stopReason, privacy: .public)"
        )

        if multimodalPrefill == nil {
            commitPromptCache(
                promptTokens,
                decodedGeneratedTokens: generatedTokenHistory,
                mtpSynchronized: activeMTPContext != nil
            )
            promptCacheCommitted = true
        }
        emitAccelerationStatsIfNeeded()
        emittedStats = true
        onPhaseAwareEvent(.generationStats(
            promptTokens: promptTokenCount,
            generatedTokens: generatedTokenCount,
            stopReason: stopReason,
            templateMode: templateMode,
            phase: currentPhase
        ))
        onPhaseAwareEvent(.done(
            totalBytes: returnedText.utf8.count,
            duration: Date().timeIntervalSince(startedAt),
            phase: currentPhase
        ))
        return resultPhases.generationResult(
            stopReason: stopReason,
            promptTokens: promptTokenCount,
            generatedTokens: generatedTokenCount,
            templateMode: templateMode,
            accelerationStats: accelerationStats
        )
    }

}
