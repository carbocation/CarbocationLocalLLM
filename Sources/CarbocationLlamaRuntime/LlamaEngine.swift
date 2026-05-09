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

    public var gpuLayerCount: Int32
    public var useMemoryMap: Bool
    public var batchSizeLimit: Int
    public var threadCount: Int32?
    public var promptReserveTokens: Int
    public var heartbeatInterval: TimeInterval

    public init(
        gpuLayerCount: Int32 = LlamaEngineConfiguration.defaultGPULayerCount,
        useMemoryMap: Bool = true,
        batchSizeLimit: Int = LlamaEngineConfiguration.defaultBatchSizeLimit,
        threadCount: Int32? = nil,
        promptReserveTokens: Int = LLMGenerationBudget.outputTokenReserve,
        heartbeatInterval: TimeInterval = 2
    ) {
        self.gpuLayerCount = gpuLayerCount
        self.useMemoryMap = useMemoryMap
        self.batchSizeLimit = batchSizeLimit
        self.threadCount = threadCount
        self.promptReserveTokens = promptReserveTokens
        self.heartbeatInterval = heartbeatInterval
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

    public init(
        modelID: UUID?,
        modelPath: String,
        displayName: String?,
        filename: String,
        contextSize: Int,
        trainingContextSize: Int,
        hasEmbeddedChatTemplate: Bool
    ) {
        self.modelID = modelID
        self.modelPath = modelPath
        self.displayName = displayName
        self.filename = filename
        self.contextSize = contextSize
        self.trainingContextSize = trainingContextSize
        self.hasEmbeddedChatTemplate = hasEmbeddedChatTemplate
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

    enum PreparedChatTemplate {
        case swiftJinja(ChatTemplatePromptFormatter)
        case unavailable(String)
    }

    let configuration: LlamaEngineConfiguration

    private var model: OpaquePointer?
    var context: OpaquePointer?
    var vocabulary: OpaquePointer?
    var loadedDescriptor: LlamaModelDescriptor?
    private var loadedInfo: LlamaLoadedModelInfo?
    var chatTemplate: String?
    var preparedChatTemplate: PreparedChatTemplate?
    var outputSanitizationProfile: OutputSanitizationProfile = .empty
    var cachedPromptTokens: [llama_token]?

    public init(configuration: LlamaEngineConfiguration = LlamaEngineConfiguration()) {
        self.configuration = configuration
        LlamaBackend.ensureInitialized()
    }

    deinit {
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

        unload()

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
        var attemptedBatchSizes: [Int] = []
        var initializedContext: OpaquePointer?
        for batchSize in batchCandidates {
            attemptedBatchSizes.append(batchSize)
            let contextParams = Self.contextParams(
                contextSize: chosenContext,
                batchSize: batchSize,
                threads: threads
            )
            if let context = llama_init_from_model(loadedModel, contextParams) {
                initializedContext = context
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
            hasEmbeddedChatTemplate: template != nil
        )

        self.model = loadedModel
        self.context = loadedContext
        self.vocabulary = loadedVocabulary
        self.loadedDescriptor = descriptor
        self.loadedInfo = info
        self.chatTemplate = template
        self.preparedChatTemplate = preparedTemplate
        self.outputSanitizationProfile = outputProfile
        self.cachedPromptTokens = nil

        Self.logOutputSanitizationProfile(outputProfile, descriptor: descriptor, hasEmbeddedTemplate: template != nil)

        return info
    }

    public func unload() {
        if let context { llama_free(context) }
        if let model { llama_model_free(model) }

        self.model = nil
        self.context = nil
        self.vocabulary = nil
        self.loadedDescriptor = nil
        self.loadedInfo = nil
        self.chatTemplate = nil
        self.preparedChatTemplate = nil
        self.outputSanitizationProfile = .empty
        self.cachedPromptTokens = nil
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
        var promptContextPrepared = false
        var promptCacheCommitted = false

        defer {
            if promptContextPrepared && !promptCacheCommitted {
                cachedPromptTokens = nil
            }
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

        let promptFormatting: PromptFormattingResult
        let renderedPrompt: String
        do {
            promptFormatting = try applyChatTemplate(system: system, user: prompt, options: options)
            renderedPrompt = promptFormatting.text
            templateMode = promptFormatting.mode
        } catch let error as LLMEngineError {
            if case .chatTemplateUnavailable = error {
                stopReason = "template-unavailable"
            }
            throw error
        }

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
            startsInThinking: streamPhasePlan.startsInThinking == true
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

        try preparePromptContext(promptTokens, context: context)
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
            guard currentPhase == .final,
                  let nextFinalAnswer = try? Self.sanitizedGeneratedText(
                    accumulatedText,
                    profile: activeOutputProfile,
                    continuingOpenThinkingPairs: continuingOpenThinkingPairs,
                    requiresNonEmptyStructuredOutput: false
                  )
            else { return }

            emitFinalAnswer(nextFinalAnswer, snapshotReason: .streamCorrection)
        }

        var structuredPhase = structuredOutputPlan.map {
            Self.structuredOutputPhase(in: accumulatedText, plan: $0)
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

        while generatedTokenCount < maxNew {
            try Task.checkCancellation()

            let next = llama_sampler_sample(sampler, context, -1)
            let reasoningBudgetState = samplerRuntime.reasoningBudgetSampler.map {
                carbocation_llama_reasoning_budget_sampler_state($0)
            }
            if llama_vocab_is_eog(vocabulary, next) {
                if let reasoningBudgetState {
                    logReasoningBudgetExhaustionIfNeeded(
                        state: reasoningBudgetState,
                        generatedTokens: generatedTokenCount
                    )
                }
                stopReason = "eog"
                break
            }

            let rawPiece = tokenToPiece(vocab: vocabulary, token: next)
            let piece = rawPiece.isEmpty
                ? tokenToPiece(vocab: vocabulary, token: next, special: true)
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
                    generatedTokens: generatedTokenCount + 1
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

                if stopReason == "json-complete" || stopReason == "stop-sequence" {
                    emitFirstByteIfNeeded()
                    emitFinalAnswerProgressIfNeeded()
                    if stopReason == "json-complete", structuredOutputPlan != nil {
                        structuredPhase = .complete
                    }
                    break
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

            var oneToken: [llama_token] = [next]
            let decodeResult = oneToken.withUnsafeMutableBufferPointer { buffer -> Int32 in
                let batch = llama_batch_get_one(buffer.baseAddress, 1)
                return llama_decode(context, batch)
            }
            if decodeResult != 0 {
                cachedPromptTokens = nil
                throw LLMEngineError.decodeFailed
            }

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

        cachedPromptTokens = promptTokens
        promptCacheCommitted = true
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
