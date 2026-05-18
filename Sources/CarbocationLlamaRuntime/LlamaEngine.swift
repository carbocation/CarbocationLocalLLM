import CarbocationLocalLLM
import CarbocationLlamaCommonBridge
import Foundation
import OSLog
import llama

let llamaRuntimeLog = Logger(
    subsystem: "com.carbocation.CarbocationLocalLLM",
    category: "LlamaRuntime"
)

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
    public static let defaultMTPMaxDraftTokens = 3

    public var gpuLayerCount: Int32
    public var useMemoryMap: Bool
    public var batchSizeLimit: Int
    public var threadCount: Int32?
    public var promptReserveTokens: Int
    public var heartbeatInterval: TimeInterval
    public var accelerationPolicy: LLMAccelerationPolicy
    public var mtpMaxDraftTokens: Int
    public var allowsUnsafeMTPDraftWidthsForDebugging: Bool

    public init(
        gpuLayerCount: Int32 = LlamaEngineConfiguration.defaultGPULayerCount,
        useMemoryMap: Bool = true,
        batchSizeLimit: Int = LlamaEngineConfiguration.defaultBatchSizeLimit,
        threadCount: Int32? = nil,
        promptReserveTokens: Int = LLMGenerationBudget.outputTokenReserve,
        heartbeatInterval: TimeInterval = 2,
        accelerationPolicy: LLMAccelerationPolicy = .automatic,
        mtpMaxDraftTokens: Int = LlamaEngineConfiguration.defaultMTPMaxDraftTokens,
        allowsUnsafeMTPDraftWidthsForDebugging: Bool = false
    ) {
        self.gpuLayerCount = gpuLayerCount
        self.useMemoryMap = useMemoryMap
        self.batchSizeLimit = batchSizeLimit
        self.threadCount = threadCount
        self.promptReserveTokens = promptReserveTokens
        self.heartbeatInterval = heartbeatInterval
        self.accelerationPolicy = accelerationPolicy
        self.mtpMaxDraftTokens = mtpMaxDraftTokens
        self.allowsUnsafeMTPDraftWidthsForDebugging = allowsUnsafeMTPDraftWidthsForDebugging
    }
}

public struct LlamaModelDescriptor: Hashable, Sendable {
    public var id: UUID?
    public var url: URL
    public var displayName: String?
    public var filename: String
    public var hfRepo: String?
    public var hfFilename: String?

    public init(
        id: UUID? = nil,
        url: URL,
        displayName: String? = nil,
        filename: String? = nil,
        hfRepo: String? = nil,
        hfFilename: String? = nil
    ) {
        self.id = id
        self.url = url
        self.displayName = displayName
        self.filename = filename ?? url.lastPathComponent
        self.hfRepo = hfRepo
        self.hfFilename = hfFilename
    }

    public init(model: InstalledModel, root: URL) {
        self.init(
            id: model.id,
            url: model.weightsURL(in: root),
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
    public var supportsMTPAcceleration: Bool
    public var architecture: String?
    public var nextNPredictLayers: Int?

    public init(
        modelID: UUID?,
        modelPath: String,
        displayName: String?,
        filename: String,
        contextSize: Int,
        trainingContextSize: Int,
        hasEmbeddedChatTemplate: Bool,
        supportsMTPAcceleration: Bool = false,
        architecture: String? = nil,
        nextNPredictLayers: Int? = nil
    ) {
        self.modelID = modelID
        self.modelPath = modelPath
        self.displayName = displayName
        self.filename = filename
        self.contextSize = contextSize
        self.trainingContextSize = trainingContextSize
        self.hasEmbeddedChatTemplate = hasEmbeddedChatTemplate
        self.supportsMTPAcceleration = supportsMTPAcceleration
        self.architecture = architecture
        self.nextNPredictLayers = nextNPredictLayers
    }
}

public actor LlamaEngine: LLMEngine {
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
    }

    struct ToolAwareGenerationSegmentOverrideInput: Sendable {
        var renderedPrompt: String
        var templateMode: LLMChatTemplateMode
        var isInternalContinuation: Bool
    }

    struct ToolAwareGenerationSegmentOverrideOutput: Sendable {
        var finalText: String?
        var toolCalls: [LLMToolCall]
        var stopReason: String
        var triggerPhase: LLMStreamContentPhase?
        var remainingThinkingBudgetTokens: Int?
        var generatedTokens: Int

        init(
            finalText: String? = nil,
            toolCalls: [LLMToolCall] = [],
            stopReason: String,
            triggerPhase: LLMStreamContentPhase? = nil,
            remainingThinkingBudgetTokens: Int? = nil,
            generatedTokens: Int = 0
        ) {
            self.finalText = finalText
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

    let configuration: LlamaEngineConfiguration

    private var model: OpaquePointer?
    var context: OpaquePointer?
    var vocabulary: OpaquePointer?
    private var mtpContext: UnsafeMutableRawPointer?
    var mtpCachedPromptTokens: [llama_token]?
    var loadedDescriptor: LlamaModelDescriptor?
    private var loadedInfo: LlamaLoadedModelInfo?
    var chatTemplate: String?
    var preparedChatTemplate: PreparedChatTemplate?
    var outputSanitizationProfile: OutputSanitizationProfile = .empty
    var cachedPromptTokens: [llama_token]?
    var toolAwareGenerationSegmentOverride: ToolAwareGenerationSegmentOverride?
    private var activeGenerationCount = 0
    private var unloadAfterActiveGeneration = false

    public init(configuration: LlamaEngineConfiguration = LlamaEngineConfiguration()) {
        self.configuration = configuration
        LlamaBackend.ensureInitialized()
    }

    deinit {
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
        Self.effectiveMTPMaxDraftTokens(
            requested: configuration.mtpMaxDraftTokens,
            architecture: loadedInfo?.architecture,
            nextNPredictLayers: loadedInfo?.nextNPredictLayers,
            allowsUnsafeDraftWidths: configuration.allowsUnsafeMTPDraftWidthsForDebugging
        )
    }

    static func clampedMTPMaxDraftTokens(_ requested: Int) -> Int {
        max(1, min(32, requested))
    }

    static func effectiveMTPMaxDraftTokens(
        requested: Int,
        metadata: GGUFMetadata.ModelMetadata
    ) -> Int {
        effectiveMTPMaxDraftTokens(
            requested: requested,
            architecture: metadata.architecture,
            nextNPredictLayers: metadata.nextNPredictLayers,
            allowsUnsafeDraftWidths: false
        )
    }

    static func effectiveMTPMaxDraftTokens(
        requested: Int,
        architecture: String?,
        nextNPredictLayers: Int?,
        allowsUnsafeDraftWidths: Bool = false
    ) -> Int {
        let clamped = clampedMTPMaxDraftTokens(requested)
        guard !allowsUnsafeDraftWidths else {
            return clamped
        }
        guard architecture == "qwen35", nextNPredictLayers == 1 else {
            return clamped
        }
        return min(clamped, 2)
    }

    public func preflight(
        system: String,
        prompt: String,
        options: GenerationOptions
    ) async throws -> LLMGenerationPreflight {
        guard context != nil, let vocabulary, let loadedInfo else {
            throw LLMEngineError.noModelLoaded
        }

        let promptFormatting = try applyChatTemplate(system: system, user: prompt, options: options)
        let renderedPrompt = promptFormatting.text
        let promptForTokenization = promptWithAutoAddedSpecialTokensStripped(
            renderedPrompt,
            vocab: vocabulary
        )
        let promptTokens = try tokenize(vocab: vocabulary, text: promptForTokenization, addSpecial: true)
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

    @discardableResult
    public func load(
        model installed: InstalledModel,
        from root: URL,
        requestedContext: Int
    ) throws -> LlamaLoadedModelInfo {
        try load(descriptor: LlamaModelDescriptor(model: installed, root: root), requestedContext: requestedContext)
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
        let path = descriptor.url.path
        let ggufMetadata = GGUFMetadata.modelMetadata(at: descriptor.url)

        if let loadedInfo,
           loadedInfo.modelPath == path,
           context != nil {
            let desiredContext = Self.clampedContextSize(
                requestedContext: requestedContext,
                trainingContext: loadedInfo.trainingContextSize
            )
            if desiredContext == loadedInfo.contextSize {
                return loadedInfo
            }
        }

        guard activeGenerationCount == 0 else {
            throw LLMEngineError.modelLoadFailed("Cannot load a new model while generation is active.")
        }

        performUnload()

        var modelParams = llama_model_default_params()
        modelParams.n_gpu_layers = configuration.gpuLayerCount
        modelParams.use_mmap = configuration.useMemoryMap
        if configuration.gpuLayerCount <= 0 {
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

        let batchCandidates = Self.contextBatchCandidates(
            contextSize: chosenContext,
            batchSizeLimit: configuration.batchSizeLimit
        )
        let supportsMTPAcceleration = ggufMetadata.supportsMTPAcceleration
        let requestedMTPDraftTokens = Self.clampedMTPMaxDraftTokens(configuration.mtpMaxDraftTokens)
        let maxMTPDraftTokens = Self.effectiveMTPMaxDraftTokens(
            requested: requestedMTPDraftTokens,
            architecture: ggufMetadata.architecture,
            nextNPredictLayers: ggufMetadata.nextNPredictLayers,
            allowsUnsafeDraftWidths: configuration.allowsUnsafeMTPDraftWidthsForDebugging
        )
        let shouldPrepareMTPAcceleration = supportsMTPAcceleration &&
            configuration.accelerationPolicy == .automatic
        if supportsMTPAcceleration, maxMTPDraftTokens != requestedMTPDraftTokens {
            llamaRuntimeLog.info(
                "MTP draft width capped: requested=\(requestedMTPDraftTokens, privacy: .public) effective=\(maxMTPDraftTokens, privacy: .public) architecture=\(ggufMetadata.architecture ?? "unknown", privacy: .public) nextNPredictLayers=\(ggufMetadata.nextNPredictLayers ?? 0, privacy: .public)"
            )
        }
        var attemptedBatchSizes: [Int] = []
        var initializedContext: OpaquePointer?
        var initializedBatchSize: Int?
        for batchSize in batchCandidates {
            attemptedBatchSizes.append(batchSize)
            let contextParams = Self.contextParams(
                contextSize: chosenContext,
                batchSize: batchSize,
                threads: threads,
                recurrentStateSnapshots: shouldPrepareMTPAcceleration ? maxMTPDraftTokens : 0
            )
            if let context = llama_init_from_model(loadedModel, contextParams) {
                initializedContext = context
                initializedBatchSize = batchSize
                break
            }
        }

        guard let loadedContext = initializedContext else {
            llama_model_free(loadedModel)
            let attempted = attemptedBatchSizes.map(String.init).joined(separator: ", ")
            throw LLMEngineError.contextInitFailed(
                "llama_init_from_model returned null (context=\(chosenContext), attempted batch sizes: \(attempted))"
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
        let info = LlamaLoadedModelInfo(
            modelID: descriptor.id,
            modelPath: path,
            displayName: descriptor.displayName,
            filename: descriptor.filename,
            contextSize: chosenContext,
            trainingContextSize: max(0, trainingContext),
            hasEmbeddedChatTemplate: template != nil,
            supportsMTPAcceleration: supportsMTPAcceleration,
            architecture: ggufMetadata.architecture,
            nextNPredictLayers: ggufMetadata.nextNPredictLayers
        )
        let loadedMTPContext: UnsafeMutableRawPointer?
        if shouldPrepareMTPAcceleration, let initializedBatchSize {
            loadedMTPContext = carbocation_llama_mtp_create_bridge(
                loadedModel,
                loadedContext,
                UInt32(chosenContext),
                UInt32(initializedBatchSize),
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

        self.model = loadedModel
        self.context = loadedContext
        self.mtpContext = loadedMTPContext
        self.vocabulary = loadedVocabulary
        self.loadedDescriptor = descriptor
        self.loadedInfo = info
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
        let descriptor = LlamaModelDescriptor(
            url: url,
            displayName: displayName,
            filename: filename
        )
        let path = descriptor.url.path

        guard activeGenerationCount == 0 else {
            throw LLMEngineError.modelLoadFailed("Cannot load a new model while generation is active.")
        }

        performUnload()

        var modelParams = llama_model_default_params()
        modelParams.vocab_only = true
        modelParams.use_mmap = true
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
            supportsMTPAcceleration: false
        )

        self.model = loadedModel
        self.context = nil
        self.mtpContext = nil
        self.vocabulary = loadedVocabulary
        self.loadedDescriptor = descriptor
        self.loadedInfo = info
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

    public func unload() {
        guard activeGenerationCount == 0 else {
            unloadAfterActiveGeneration = true
            return
        }
        performUnload()
    }

    func beginGenerationLease() {
        activeGenerationCount += 1
    }

    func endGenerationLease() {
        activeGenerationCount = max(0, activeGenerationCount - 1)
        if activeGenerationCount == 0, unloadAfterActiveGeneration {
            performUnload()
        }
    }

    private func performUnload() {
        if let mtpContext { carbocation_llama_mtp_free_bridge(mtpContext) }
        if let context { llama_free(context) }
        if let model { llama_model_free(model) }

        unloadAfterActiveGeneration = false
        self.model = nil
        self.context = nil
        self.mtpContext = nil
        self.vocabulary = nil
        self.loadedDescriptor = nil
        self.loadedInfo = nil
        self.chatTemplate = nil
        self.preparedChatTemplate = nil
        self.outputSanitizationProfile = .empty
        clearPromptCaches()
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

        beginGenerationLease()
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

    func generate(
        promptFormatting: PromptFormattingResult,
        options: GenerationOptions,
        control: LLMGenerationControl? = nil,
        controlGenerationID: UInt64? = nil,
        onPhaseAwareEvent: @Sendable (LLMPhaseAwareStreamEvent) -> Void
    ) async throws -> String {
        guard let context, let vocabulary else {
            throw LLMEngineError.noModelLoaded
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

        let promptForTokenization = promptWithAutoAddedSpecialTokensStripped(
            renderedPrompt,
            vocab: vocabulary
        )
        let promptTokens = try tokenize(vocab: vocabulary, text: promptForTokenization, addSpecial: true)
        promptTokenCount = promptTokens.count
        guard !promptTokens.isEmpty else {
            throw LLMEngineError.tokenizationFailed
        }
        guard promptTokens.count < currentContextSize() else {
            throw LLMEngineError.insufficientGenerationBudget(
                contextSize: currentContextSize(),
                promptTokens: promptTokens.count,
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

        let hasIncompatibleMTPControlPath = false
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

        try preparePromptContext(promptTokens, context: context, mtpContext: activeMTPContext)
        promptContextPrepared = true

        var accumulatedData = Data()
        var accumulatedText = ""
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

        var lastHeartbeat = Date()
        var streamedFinalAnswer = ""
        var structuredPhase = structuredOutputPlan.map {
            Self.structuredOutputPhase(in: accumulatedText, plan: $0)
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
        }

        func emitFinalAnswerProgressIfNeeded() {
            guard Self.shouldEmitFinalAnswerProgress(
                currentPhase: currentPhase,
                structuredPhase: structuredPhase
            ),
                  let nextFinalAnswer = try? Self.sanitizedGeneratedText(
                    accumulatedText,
                    profile: activeOutputProfile,
                    continuingOpenThinkingPairs: continuingOpenThinkingPairs,
                    requiresNonEmptyStructuredOutput: false
                  )
            else { return }

            emitFinalAnswer(nextFinalAnswer, snapshotReason: .streamCorrection)
        }

        let contextMaxNew = Self.maxGenerationTokens(
            contextSize: currentContextSize(),
            promptTokenCount: promptTokens.count,
            reserve: configuration.promptReserveTokens
        )
        let maxNew: Int
        if let requestedMax = options.maxOutputTokens, requestedMax > 0 {
            maxNew = min(contextMaxNew, requestedMax)
        } else {
            maxNew = contextMaxNew
        }

        guard maxNew > 0 else {
            throw LLMEngineError.insufficientGenerationBudget(
                contextSize: currentContextSize(),
                promptTokens: promptTokens.count,
                reserve: configuration.promptReserveTokens
            )
        }

        let effectiveStopSequences = Self.mergingStopSequences(
            options.stopSequences,
            activeOutputProfile.extraStopStrings
        )

        let tokenDiagnosticsEnabled = configuration.allowsUnsafeMTPDraftWidthsForDebugging
        func emitTokenDiagnostic(
            _ token: llama_token,
            generatedTokensIncludingThisToken: Int,
            source: String
        ) {
            guard tokenDiagnosticsEnabled else { return }
            let position = promptTokens.count + generatedTokensIncludingThisToken - 1
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
            if !piece.isEmpty {
                accumulatedData.append(piece)
                if let decoded = String(data: accumulatedData, encoding: .utf8) {
                    accumulatedText = decoded
                }
            }

            if let reasoningBudgetState {
                logReasoningBudgetExhaustionIfNeeded(
                    state: reasoningBudgetState,
                    generatedTokens: generatedTokensIncludingThisToken
                )
            }

            if !piece.isEmpty {
                if let plan = structuredOutputPlan {
                    let nextPhase = Self.structuredOutputPhase(in: accumulatedText, plan: plan)
                    if nextPhase != structuredPhase {
                        structuredPhase = nextPhase
                        llamaRuntimeLog.info(
                            "Structured output phase changed: phase=\(nextPhase.rawValue, privacy: .public) rawBytes=\(accumulatedData.count, privacy: .public)"
                        )
                    }
                }

                let boundary = if let plan = structuredOutputPlan {
                    Self.firstStructuredGenerationBoundary(
                        in: accumulatedText,
                        stopSequences: effectiveStopSequences,
                        stopAtBalancedJSON: options.stopAtBalancedJSON,
                        plan: plan
                    )
                } else {
                    Self.firstGenerationBoundary(
                        in: accumulatedText,
                        stopSequences: effectiveStopSequences,
                        stopAtBalancedJSON: options.stopAtBalancedJSON
                    )
                }

                if let boundary {
                    accumulatedText = boundary.text
                    stopReason = boundary.reason
                }

                updatePhase(Self.streamContentPhase(in: accumulatedText, plan: streamPhasePlan))

                if stopReason == "json-complete" || stopReason == "stop-sequence" || stopReason == "tool-call-complete" {
                    emitFirstByteIfNeeded()
                    emitFinalAnswerProgressIfNeeded()
                    if stopReason == "json-complete", structuredOutputPlan != nil {
                        structuredPhase = .complete
                    }
                    return false
                }
            }

            emitFirstByteIfNeeded()

            let now = Date()
            if now.timeIntervalSince(lastHeartbeat) >= configuration.heartbeatInterval {
                lastHeartbeat = now
                emitFinalAnswerProgressIfNeeded()
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

        func decodeMTPAcceptedVerificationTokens(
            _ tokens: [llama_token],
            startPosition: Int,
            mtpContext: UnsafeMutableRawPointer
        ) throws {
            var tokens = tokens
            let decodeResult = tokens.withUnsafeMutableBufferPointer { buffer -> Int32 in
                carbocation_llama_mtp_decode_target_tokens_bridge(
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

        func restoreMTPVerificationCheckpoint(
            mtpContext: UnsafeMutableRawPointer
        ) throws {
            let restoreResult = carbocation_llama_mtp_restore_verification_checkpoint_bridge(mtpContext)
            if restoreResult == 0 {
                clearPromptCaches()
                throw LLMEngineError.decodeFailed
            }
        }

        var prefetchedNextToken: llama_token?
        var prefetchedReasoningBudgetState: carbocation_llama_reasoning_budget_state?
        var generatedTokenHistory: [llama_token] = []
        // A partial rejection rolls target recurrent state through restore/replay.
        // Keep a short sequential runway before the next batched verify window.
        var mtpSequentialCooldown = 0

        func sampleTargetToken(at index: Int32) -> llama_token {
            llama_synchronize(context)
            return llama_sampler_sample(sampler, context, index)
        }

        let mtpDiagnosticsEnabled = configuration.allowsUnsafeMTPDraftWidthsForDebugging
            && activeMTPContext != nil
            && maxMTPDraftTokens > 2
        func emitMTPDiagnostic(_ message: String) {
            guard mtpDiagnosticsEnabled else { return }
            onPhaseAwareEvent(.diagnostic(message: "mtp-diagnostic \(message)"))
        }

        while generatedTokenCount < maxNew {
            try Task.checkCancellation()
            if accumulatedData.isEmpty || String(data: accumulatedData, encoding: .utf8) != nil {
                applyThinkingTerminationIfRequested(
                    control: control,
                    generationID: controlGenerationID,
                    currentPhase: currentPhase,
                    samplerRuntime: samplerRuntime,
                    reasoningBudgetPlan: reasoningBudgetPlan,
                    vocab: vocabulary
                )
            }

            let tokenPosition = promptTokens.count + generatedTokenCount
            let next: llama_token
            let nextDiagnosticSource: String
            let reasoningBudgetState: carbocation_llama_reasoning_budget_state?
            if let prefetched = prefetchedNextToken {
                next = prefetched
                nextDiagnosticSource = "prefetched"
                reasoningBudgetState = prefetchedReasoningBudgetState
                prefetchedNextToken = nil
                prefetchedReasoningBudgetState = nil
            } else {
                next = sampleTargetToken(at: -1)
                nextDiagnosticSource = "sampled"
                reasoningBudgetState = samplerRuntime.reasoningBudgetSampler.map {
                    carbocation_llama_reasoning_budget_sampler_state($0)
                }
            }
            guard processGeneratedToken(
                next,
                generatedTokensIncludingThisToken: generatedTokenCount + 1,
                reasoningBudgetState: reasoningBudgetState,
                diagnosticSource: nextDiagnosticSource
            ) else {
                break
            }

            if let activeMTPContext {
                if mtpSequentialCooldown > 0 {
                    try decodeMTPAcceptedVerificationTokens(
                        [next],
                        startPosition: tokenPosition,
                        mtpContext: activeMTPContext
                    )
                    generatedTokenHistory.append(next)
                    generatedTokenCount += 1
                    mtpSequentialCooldown -= 1
                    continue
                }

                let remainingDraftCapacity = maxNew - generatedTokenCount - 1
                let draftLimit = min(maxMTPDraftTokens, remainingDraftCapacity)
                var draftCallIndex: Int?
                let draftTokens: [llama_token]
                if draftLimit > 0 {
                    draftCallIndex = (accelerationStats?.draftCalls ?? 0) + 1
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
                    if generatedTokenCount + 1 < maxNew {
                        let sampledPrefetch = sampleTargetToken(at: 0)
                        prefetchedNextToken = sampledPrefetch
                        prefetchedReasoningBudgetState = samplerRuntime.reasoningBudgetSampler.map {
                            carbocation_llama_reasoning_budget_sampler_state($0)
                        }
                    }
                    try processLastMTPVerificationBatch(mtpContext: activeMTPContext)
                } else {
                    let verificationPrefixTokens = promptTokens + generatedTokenHistory
                    let verificationTokens = [next] + draftTokens
                    try decodeMTPVerificationTargetTokens(
                        verificationTokens,
                        startPosition: tokenPosition,
                        mtpContext: activeMTPContext
                    )

                    var correctionToken: llama_token?
                    var correctionReasoningBudgetState: carbocation_llama_reasoning_budget_state?
                    var verifierSamples: [llama_token] = []
                    for (verifierIndex, draftToken) in draftTokens.enumerated() {
                        let sampled = sampleTargetToken(at: Int32(verifierIndex))
                        verifierSamples.append(sampled)
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
                            generatedTokensIncludingThisToken: generatedTokenCount + 1 + emittedAcceptedDraftCount,
                            reasoningBudgetState: acceptedReasoningBudgetState,
                            diagnosticSource: "accepted-draft"
                        )
                        if !shouldContinue {
                            break
                        }
                    }

                    if emittedAcceptedDraftCount < draftTokens.count {
                        let retainedTokens = [next] + Array(draftTokens.prefix(emittedAcceptedDraftCount))
                        try restoreMTPVerificationCheckpoint(mtpContext: activeMTPContext)
                        try decodeMTPAcceptedVerificationTokens(
                            retainedTokens,
                            startPosition: tokenPosition,
                            mtpContext: activeMTPContext
                        )
                        _ = carbocation_llama_mtp_accept_bridge(
                            activeMTPContext,
                            Int32(emittedAcceptedDraftCount)
                        )

                        if let correctionToken, shouldContinue {
                            let correctionPosition = tokenPosition + retainedTokens.count
                            try decodeMTPAcceptedVerificationTokens(
                                [correctionToken],
                                startPosition: correctionPosition,
                                mtpContext: activeMTPContext
                            )
                            emittedCorrectionToken = correctionToken
                            shouldContinue = processGeneratedToken(
                                correctionToken,
                                generatedTokensIncludingThisToken: generatedTokenCount + retainedTokens.count + 1,
                                reasoningBudgetState: correctionReasoningBudgetState,
                                diagnosticSource: "correction"
                            )
                            mtpSequentialCooldown = max(mtpSequentialCooldown, maxMTPDraftTokens * 2)
                        }
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
                            "verify call=\(draftCallIndex ?? 0) pos=\(tokenPosition) accepted=\(emittedAcceptedDraftCount)/\(draftTokens.count) draft=\(diagnosticTokenList(draftTokens, vocab: vocabulary)) sampled=\(diagnosticTokenList(verifierSamples, vocab: vocabulary)) correction=\(correctionToken.map { diagnosticTokenDescription($0, vocab: vocabulary) } ?? "none")"
                        )
                    } else {
                        var prefetchedDiagnosticToken: llama_token?
                        if shouldContinue,
                           generatedTokenCount + 1 + emittedAcceptedDraftCount < maxNew {
                            let sampledPrefetch = sampleTargetToken(at: Int32(draftTokens.count))
                            prefetchedNextToken = sampledPrefetch
                            prefetchedDiagnosticToken = sampledPrefetch
                            prefetchedReasoningBudgetState = samplerRuntime.reasoningBudgetSampler.map {
                                carbocation_llama_reasoning_budget_sampler_state($0)
                            }
                        }

                        try processLastMTPVerificationBatch(mtpContext: activeMTPContext)

                        _ = carbocation_llama_mtp_accept_bridge(
                            activeMTPContext,
                            Int32(emittedAcceptedDraftCount)
                        )
                        emitMTPDiagnostic(
                            "verify call=\(draftCallIndex ?? 0) pos=\(tokenPosition) accepted=full/\(draftTokens.count) draft=\(diagnosticTokenList(draftTokens, vocab: vocabulary)) sampled=\(diagnosticTokenList(verifierSamples, vocab: vocabulary)) prefetch=\(prefetchedDiagnosticToken.map { diagnosticTokenDescription($0, vocab: vocabulary) } ?? "none")"
                        )
                    }
                }

                accelerationStats?.draftTokensAccepted += emittedAcceptedDraftCount
                let retainedTokenCount = 1 + emittedAcceptedDraftCount + (emittedCorrectionToken == nil ? 0 : 1)
                generatedTokenHistory.append(next)
                if emittedAcceptedDraftCount > 0 {
                    generatedTokenHistory.append(contentsOf: draftTokens.prefix(emittedAcceptedDraftCount))
                }
                if let emittedCorrectionToken {
                    generatedTokenHistory.append(emittedCorrectionToken)
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

        if let plan = structuredOutputPlan {
            let finalPhase = structuredPhase ?? Self.structuredOutputPhase(in: accumulatedText, plan: plan)
            if finalPhase == .thinking || finalPhase == .awaitingFinal {
                stopReason = finalPhase == .thinking
                    ? "thinking-not-closed"
                    : "structured-output-not-started"
                llamaRuntimeLog.error(
                    "Structured output phase failed before sanitization: phase=\(finalPhase.rawValue, privacy: .public) rawBytes=\(accumulatedText.utf8.count, privacy: .public) stopReason=\(stopReason, privacy: .public)"
                )
                throw LLMEngineError.structuredOutputPhaseFailed(
                    "Generation ended before final structured output began."
                )
            }
        }

        let returnedText: String
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

        emitFinalAnswer(returnedText, snapshotReason: .completed)

        llamaRuntimeLog.info(
            "Generation sanitized output: grammarMode=\(grammarMode.logLabel, privacy: .public) rawBytes=\(accumulatedText.utf8.count, privacy: .public) sanitizedBytes=\(returnedText.utf8.count, privacy: .public) stopReason=\(stopReason, privacy: .public)"
        )

        commitPromptCache(promptTokens, mtpSynchronized: activeMTPContext != nil)
        promptCacheCommitted = true
        emitAccelerationStatsIfNeeded()
        emittedStats = true
        onPhaseAwareEvent(.generationStats(
            promptTokens: promptTokens.count,
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
        return returnedText
    }

}
