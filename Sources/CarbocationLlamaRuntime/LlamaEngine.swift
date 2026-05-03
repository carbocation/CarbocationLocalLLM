import CarbocationLocalLLM
import Foundation
import OSLog
import llama

private let llamaRuntimeLog = Logger(
    subsystem: "com.carbocation.CarbocationLocalLLM",
    category: "LlamaRuntime"
)

#if os(iOS)
private let platformDefaultGPULayerCount: Int32 = 0
// Large mobile GGUFs can fit model + KV memory but fail llama.cpp's 512-token graph reservation.
private let platformDefaultBatchSizeLimit = 64
private let platformMaximumContextSize = Int.max
#else
private let platformDefaultGPULayerCount: Int32 = 999
private let platformDefaultBatchSizeLimit = 2_048
private let platformMaximumContextSize = Int.max
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

    private struct PromptFormattingResult {
        var text: String
        var mode: LLMChatTemplateMode
        var outputProfile: OutputSanitizationProfile
    }

    private enum PreparedChatTemplate {
        case swiftJinja(ChatTemplatePromptFormatter)
        case unavailable(String)
    }

    let configuration: LlamaEngineConfiguration

    private var model: OpaquePointer?
    private var context: OpaquePointer?
    private var vocabulary: OpaquePointer?
    private var loadedDescriptor: LlamaModelDescriptor?
    private var loadedInfo: LlamaLoadedModelInfo?
    private var chatTemplate: String?
    private var preparedChatTemplate: PreparedChatTemplate?
    private var outputSanitizationProfile: OutputSanitizationProfile = .empty
    private var cachedPromptTokens: [llama_token]?

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

        let continuingOpenThinkingPairs = Self.continuingOpenThinkingPairs(
            in: renderedPrompt,
            profile: promptFormatting.outputProfile
        )
        let grammarMode = Self.generationGrammarMode(
            for: options,
            profile: promptFormatting.outputProfile,
            continuingOpenThinkingPairs: continuingOpenThinkingPairs
        )
        if options.grammar != nil {
            let sampler = try buildSampler(grammarMode: grammarMode, options: options, vocab: vocabulary)
            llama_sampler_free(sampler)
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
        guard let context, let vocabulary else {
            throw LLMEngineError.noModelLoaded
        }

        onEvent(.requestSent)
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
                onEvent(.generationStats(
                    promptTokens: promptTokenCount,
                    generatedTokens: generatedTokenCount,
                    stopReason: stopReason,
                    templateMode: templateMode
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

        let activeOutputProfile = promptFormatting.outputProfile
        let continuingOpenThinkingPairs = Self.continuingOpenThinkingPairs(
            in: renderedPrompt,
            profile: activeOutputProfile
        )
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
        llamaRuntimeLog.info(
            "Generation grammar mode selected: mode=\(grammarMode.logLabel, privacy: .public) enableThinking=\(options.enableThinking, privacy: .public) continuingOpenThinkingPairs=\(continuingOpenThinkingPairs.count, privacy: .public)"
        )

        let sampler = try buildSampler(grammarMode: grammarMode, options: options, vocab: vocabulary)
        defer { llama_sampler_free(sampler) }

        try preparePromptContext(promptTokens, context: context)
        promptContextPrepared = true

        var accumulatedData = Data()
        var accumulatedText = ""
        var sawFirstToken = false
        var lastHeartbeat = Date()
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
            if llama_vocab_is_eog(vocabulary, next) {
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

                if stopReason == "json-complete" || stopReason == "stop-sequence" {
                    if stopReason == "json-complete", structuredOutputPlan != nil {
                        structuredPhase = .complete
                    }
                    break
                }
            }

            if !sawFirstToken {
                sawFirstToken = true
                onEvent(.firstByteReceived(after: Date().timeIntervalSince(startedAt)))
            }

            let now = Date()
            if now.timeIntervalSince(lastHeartbeat) >= configuration.heartbeatInterval {
                lastHeartbeat = now
                onEvent(.tokenChunk(
                    preview: String(accumulatedText.suffix(60)),
                    bytesSoFar: accumulatedData.count
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

        llamaRuntimeLog.info(
            "Generation sanitized output: grammarMode=\(grammarMode.logLabel, privacy: .public) rawBytes=\(accumulatedText.utf8.count, privacy: .public) sanitizedBytes=\(returnedText.utf8.count, privacy: .public) stopReason=\(stopReason, privacy: .public)"
        )

        cachedPromptTokens = promptTokens
        promptCacheCommitted = true
        emittedStats = true
        onEvent(.generationStats(
            promptTokens: promptTokens.count,
            generatedTokens: generatedTokenCount,
            stopReason: stopReason,
            templateMode: templateMode
        ))
        onEvent(.done(totalBytes: returnedText.utf8.count, duration: Date().timeIntervalSince(startedAt)))
        return returnedText
    }

    public func encodeCalgacus(
        _ request: CalgacusEncodeRequest,
        onEvent: @Sendable (CalgacusEvent) -> Void
    ) async throws -> CalgacusEncodeResult {
        guard let context, let vocabulary else {
            throw LLMEngineError.noModelLoaded
        }
        cachedPromptTokens = nil

        let startedAt = Date()
        onEvent(.started(operation: "encode"))

        let secretTokens = try tokenize(vocab: vocabulary, text: request.secretText, addSpecial: false)
        guard !secretTokens.isEmpty else {
            throw CalgacusError.emptySecretText
        }
        onEvent(.tokensPrepared(operation: "secret", count: secretTokens.count))

        let secretContextTokens = try calgacusInitialTokens(vocab: vocabulary)
        try Self.calgacusValidateBudget(
            operation: "Secret ranking",
            contextSize: currentContextSize(),
            contextTokenCount: secretContextTokens.count,
            payloadTokenCount: secretTokens.count
        )

        let secretTrace = try calgacusTrace(
            tokens: secretTokens,
            initialContextTokens: secretContextTokens,
            context: context,
            vocabulary: vocabulary,
            stage: .secretRanking,
            onEvent: onEvent
        )
        let ranks = secretTrace.map(\.rank)

        let coverPromptTokens = try calgacusPromptTokens(vocab: vocabulary, text: request.coverPrompt)
        try Self.calgacusValidateBudget(
            operation: "Cover generation",
            contextSize: currentContextSize(),
            contextTokenCount: coverPromptTokens.count,
            payloadTokenCount: ranks.count
        )

        let coverPayload = try calgacusSelectTokens(
            ranks: ranks,
            initialContextTokens: coverPromptTokens,
            context: context,
            vocabulary: vocabulary,
            stage: .coverGeneration,
            operation: "Cover generation",
            rejectsControlTokens: true,
            onEvent: onEvent
        )

        guard let coverText = String(data: coverPayload.data, encoding: .utf8) else {
            throw CalgacusError.textRenderingFailed(operation: "Cover generation")
        }

        let verification = try calgacusDecodePayload(
            coverText: coverText,
            coverPrompt: request.coverPrompt,
            context: context,
            vocabulary: vocabulary,
            emitsEvents: false,
            onEvent: onEvent
        )
        guard verification.recoveredTokens == secretTokens else {
            throw CalgacusError.verificationFailed(
                expectedTokenCount: secretTokens.count,
                recoveredTokenCount: verification.recoveredTokens.count
            )
        }

        onEvent(.completed(operation: "encode", duration: Date().timeIntervalSince(startedAt)))
        return CalgacusEncodeResult(
            coverText: coverText,
            secretTokenCount: secretTokens.count,
            coverTokenCount: coverPayload.tokens.count,
            stats: Self.calgacusStats(for: secretTrace),
            trace: secretTrace
        )
    }

    public func decodeCalgacus(
        _ request: CalgacusDecodeRequest,
        onEvent: @Sendable (CalgacusEvent) -> Void
    ) async throws -> CalgacusDecodeResult {
        guard let context, let vocabulary else {
            throw LLMEngineError.noModelLoaded
        }
        cachedPromptTokens = nil

        let startedAt = Date()
        onEvent(.started(operation: "decode"))
        let payload = try calgacusDecodePayload(
            coverText: request.coverText,
            coverPrompt: request.coverPrompt,
            context: context,
            vocabulary: vocabulary,
            emitsEvents: true,
            onEvent: onEvent
        )
        onEvent(.completed(operation: "decode", duration: Date().timeIntervalSince(startedAt)))
        return payload.result
    }

    private enum KnownTemplateFamily {
        case gemma
        case chatML

        var mode: LLMChatTemplateMode {
            switch self {
            case .gemma:
                return .gemmaFallback
            case .chatML:
                return .chatMLFallback
            }
        }
    }

    private func applyChatTemplate(
        system: String,
        user: String,
        options: GenerationOptions
    ) throws -> PromptFormattingResult {
        if let chatTemplate {
            switch preparedChatTemplate {
            case .swiftJinja(let formatter):
                do {
                    let formatted = try formatMessagesViaSwiftJinja(
                        formatter: formatter,
                        system: system,
                        user: user,
                        options: options
                    )
                    Self.logChatTemplateSelection(
                        mode: .embedded,
                        descriptor: loadedDescriptor,
                        hasEmbeddedTemplate: true,
                        formatter: "swift-jinja"
                    )
                    return PromptFormattingResult(
                        text: formatted,
                        mode: .embedded,
                        outputProfile: outputSanitizationProfile
                    )
                } catch {
                    llamaRuntimeLog.info(
                        "Swift Jinja chat template render failed: \(String(describing: error), privacy: .public)"
                    )
                }
            case .unavailable(let detail):
                llamaRuntimeLog.info(
                    "Swift Jinja chat template unavailable: \(detail, privacy: .public)"
                )
            case nil:
                break
            }

            if let formatted = Self.formatMessagesWithLegacyTemplate(
                template: chatTemplate,
                system: system,
                user: user
            ) {
                Self.logChatTemplateSelection(
                    mode: .embedded,
                    descriptor: loadedDescriptor,
                    hasEmbeddedTemplate: true,
                    formatter: "legacy-c-api"
                )
                return PromptFormattingResult(
                    text: formatted,
                    mode: .embedded,
                    outputProfile: outputSanitizationProfile
                )
            }

            throw LLMEngineError.chatTemplateUnavailable(Self.embeddedTemplateFailureDescription(
                descriptor: loadedDescriptor
            ))
        }

        let fallback = try Self.fallbackPrompt(
            system: system,
            user: user,
            embeddedTemplate: nil,
            descriptor: loadedDescriptor
        )
        Self.logChatTemplateSelection(
            mode: fallback.mode,
            descriptor: loadedDescriptor,
            hasEmbeddedTemplate: false,
            formatter: "fallback"
        )
        return PromptFormattingResult(
            text: fallback.text,
            mode: fallback.mode,
            outputProfile: fallback.outputProfile
        )
    }

    static func fallbackPrompt(
        system: String,
        user: String,
        embeddedTemplate: String?,
        descriptor: LlamaModelDescriptor?
    ) throws -> (text: String, mode: LLMChatTemplateMode, outputProfile: OutputSanitizationProfile) {
        guard let family = inferredTemplateFamily(
            embeddedTemplate: embeddedTemplate,
            descriptor: descriptor
        ) else {
            throw LLMEngineError.chatTemplateUnavailable(templateUnavailableDescription(
                embeddedTemplate: embeddedTemplate,
                descriptor: descriptor
            ))
        }

        return (
            renderFallbackPrompt(system: system, user: user, family: family),
            family.mode,
            fallbackOutputProfile(for: family)
        )
    }

    private static func inferredTemplateFamily(
        embeddedTemplate: String?,
        descriptor: LlamaModelDescriptor?
    ) -> KnownTemplateFamily? {
        if let embeddedTemplate {
            return templateFamily(from: embeddedTemplate)
        }

        let probes = [
            descriptor?.displayName,
            descriptor?.filename,
            descriptor?.hfRepo,
            descriptor?.hfFilename,
            descriptor?.url.path
        ]
        .compactMap { $0?.lowercased() }

        if probes.contains(where: { $0.contains("gemma") }) {
            return .gemma
        }
        if probes.contains(where: { $0.contains("qwen") || $0.contains("chatml") }) {
            return .chatML
        }
        return nil
    }

    private static func templateFamily(from template: String) -> KnownTemplateFamily? {
        let lowered = template.lowercased()
        if lowered.contains("start_of_turn") {
            return .gemma
        }
        if lowered.contains("im_start") {
            return .chatML
        }
        return nil
    }

    private static func prepareChatTemplate(_ template: String?) -> PreparedChatTemplate? {
        guard let template else { return nil }
        do {
            return .swiftJinja(try ChatTemplatePromptFormatter(template: template))
        } catch {
            return .unavailable(String(describing: error))
        }
    }

    private func formatMessagesViaSwiftJinja(
        formatter: ChatTemplatePromptFormatter,
        system: String,
        user: String,
        options: GenerationOptions
    ) throws -> String {
        guard let vocabulary else {
            throw LLMEngineError.noModelLoaded
        }
        return try formatter.format(
            system: system,
            user: user,
            bosToken: specialTokenString(vocab: vocabulary, token: llama_vocab_bos(vocabulary)) ?? "",
            eosToken: specialTokenString(vocab: vocabulary, token: llama_vocab_eos(vocabulary)) ?? "",
            enableThinking: options.enableThinking
        )
    }

    private static func templateUnavailableDescription(
        embeddedTemplate: String?,
        descriptor: LlamaModelDescriptor?
    ) -> String {
        if embeddedTemplate != nil {
            if let descriptor {
                return "Embedded template exists but its family is not supported. Model: \(descriptor.displayName ?? descriptor.filename) (\(descriptor.filename))."
            }
            return "Embedded template exists but its family is not supported."
        }
        if let descriptor {
            return "Model: \(descriptor.displayName ?? descriptor.filename) (\(descriptor.filename))."
        }
        return "The GGUF metadata did not expose a known template family."
    }

    private static func embeddedTemplateFailureDescription(
        descriptor: LlamaModelDescriptor?
    ) -> String {
        if let descriptor {
            return "Embedded template exists but could not be applied. Model: \(descriptor.displayName ?? descriptor.filename) (\(descriptor.filename))."
        }
        return "Embedded template exists but could not be applied."
    }

    private static func logChatTemplateSelection(
        mode: LLMChatTemplateMode,
        descriptor: LlamaModelDescriptor?,
        hasEmbeddedTemplate: Bool,
        formatter: String
    ) {
        let source = mode == .embedded ? "embedded" : "fallback"
        let modelName = descriptor?.displayName ?? descriptor?.filename ?? "unknown"
        llamaRuntimeLog.info(
            "Chat template selected: source=\(source, privacy: .public) mode=\(mode.rawValue, privacy: .public) formatter=\(formatter, privacy: .public) embeddedTemplatePresent=\(hasEmbeddedTemplate, privacy: .public) model=\(modelName, privacy: .public)"
        )
    }

    private static func logOutputSanitizationProfile(
        _ profile: OutputSanitizationProfile,
        descriptor: LlamaModelDescriptor?,
        hasEmbeddedTemplate: Bool
    ) {
        let modelName = descriptor?.displayName ?? descriptor?.filename ?? "unknown"
        let stops = profile.extraStopStrings.joined(separator: ",")
        let scrubTokens = profile.scrubTokens.joined(separator: ",")
        let sliceAfter = profile.sliceAfterMarker ?? "none"
        llamaRuntimeLog.info(
            "Output sanitization profile selected: embeddedTemplatePresent=\(hasEmbeddedTemplate, privacy: .public) thinkingPairs=\(profile.thinkingPairs.count, privacy: .public) stopStrings=\(stops, privacy: .public) sliceAfter=\(sliceAfter, privacy: .public) scrubTokens=\(scrubTokens, privacy: .public) model=\(modelName, privacy: .public)"
        )
    }

    private static func renderFallbackPrompt(
        system: String,
        user: String,
        family: KnownTemplateFamily
    ) -> String {
        switch family {
        case .gemma:
            return "<start_of_turn>user\n\(system)\n\n\(user)<end_of_turn>\n<start_of_turn>think\n<end_of_turn>\n<start_of_turn>model\n"
        case .chatML:
            return "<|im_start|>system\n\(system)<|im_end|>\n<|im_start|>user\n\(user)<|im_end|>\n<|im_start|>assistant\n"
        }
    }

    private static func fallbackOutputProfile(for family: KnownTemplateFamily) -> OutputSanitizationProfile {
        switch family {
        case .gemma:
            return OutputSanitizationProfile.derived(fromChatTemplate: "<start_of_turn><end_of_turn>")
        case .chatML:
            return OutputSanitizationProfile.derived(fromChatTemplate: "<|im_start|><|im_end|>")
        }
    }

    static func formatMessagesWithLegacyTemplate(template: String, system: String, user: String) -> String? {
        guard let roleSystem = strdup("system"),
              let roleUser = strdup("user"),
              let systemContent = strdup(system),
              let userContent = strdup(user)
        else {
            return nil
        }
        defer {
            free(roleSystem)
            free(roleUser)
            free(systemContent)
            free(userContent)
        }

        var messages = [
            llama_chat_message(role: UnsafePointer(roleSystem), content: UnsafePointer(systemContent)),
            llama_chat_message(role: UnsafePointer(roleUser), content: UnsafePointer(userContent))
        ]

        var capacity = max(2_048, (system.utf8.count + user.utf8.count) * 2 + 1_024)
        for _ in 0..<3 {
            var buffer = [CChar](repeating: 0, count: capacity)
            let result: Int32 = template.withCString { templatePointer in
                messages.withUnsafeMutableBufferPointer { messagePointer in
                    buffer.withUnsafeMutableBufferPointer { bufferPointer in
                        llama_chat_apply_template(
                            templatePointer,
                            messagePointer.baseAddress,
                            messagePointer.count,
                            true,
                            bufferPointer.baseAddress,
                            Int32(bufferPointer.count)
                        )
                    }
                }
            }

            if result > 0 && Int(result) <= capacity {
                return String(cString: buffer)
            }
            if result > 0 {
                capacity = Int(result) + 64
                continue
            }
            break
        }
        return nil
    }

    private func preparePromptContext(_ promptTokens: [llama_token], context: OpaquePointer) throws {
        let memory = llama_get_memory(context)
        let plan = Self.promptPrefillPlan(
            cachedPromptTokens: cachedPromptTokens,
            newPromptTokens: promptTokens
        )
        cachedPromptTokens = nil

        if plan.shouldClearMemory {
            llama_memory_clear(memory, false)
            try decodePromptTokens(promptTokens, startingAt: 0, context: context)
            return
        }

        if let removeStartPosition = plan.removeStartPosition {
            let removed = llama_memory_seq_rm(memory, 0, Int32(removeStartPosition), -1)
            if !removed {
                llamaRuntimeLog.info(
                    "Prompt prefix cache removal failed; falling back to full prompt prefill."
                )
                llama_memory_clear(memory, false)
                try decodePromptTokens(promptTokens, startingAt: 0, context: context)
                return
            }
        }

        do {
            try decodePromptTokens(promptTokens, startingAt: plan.decodeStartIndex, context: context)
        } catch {
            llamaRuntimeLog.info(
                "Prompt prefix cache decode failed; falling back to full prompt prefill."
            )
            llama_memory_clear(memory, false)
            try decodePromptTokens(promptTokens, startingAt: 0, context: context)
        }
    }

    private func decodePromptTokens(
        _ tokens: [llama_token],
        startingAt startIndex: Int,
        context: OpaquePointer
    ) throws {
        guard startIndex < tokens.count else { return }

        let maxBatchSize = max(1, Int(llama_n_batch(context)))
        for range in Self.prefillRanges(tokenCount: tokens.count - startIndex, maxBatchSize: maxBatchSize) {
            let lower = startIndex + range.lowerBound
            let upper = startIndex + range.upperBound
            var chunk = Array(tokens[lower..<upper])
            try chunk.withUnsafeMutableBufferPointer { buffer in
                let batch = llama_batch_get_one(buffer.baseAddress, Int32(buffer.count))
                if llama_decode(context, batch) != 0 {
                    throw LLMEngineError.decodeFailed
                }
            }
        }
    }

    private func tokenize(vocab: OpaquePointer, text: String, addSpecial: Bool) throws -> [llama_token] {
        let utf8 = Array(text.utf8CString)
        let characterCount = Int32(utf8.count - 1)

        var probe = [llama_token](repeating: 0, count: max(8, Int(characterCount)))
        let probeCount = utf8.withUnsafeBufferPointer { cBuffer in
            probe.withUnsafeMutableBufferPointer { tokenBuffer in
                llama_tokenize(
                    vocab,
                    cBuffer.baseAddress,
                    characterCount,
                    tokenBuffer.baseAddress,
                    Int32(tokenBuffer.count),
                    addSpecial,
                    true
                )
            }
        }

        if probeCount >= 0 {
            return Array(probe.prefix(Int(probeCount)))
        }

        let neededCount = Int(-probeCount)
        var tokens = [llama_token](repeating: 0, count: neededCount)
        let tokenCount = utf8.withUnsafeBufferPointer { cBuffer in
            tokens.withUnsafeMutableBufferPointer { tokenBuffer in
                llama_tokenize(
                    vocab,
                    cBuffer.baseAddress,
                    characterCount,
                    tokenBuffer.baseAddress,
                    Int32(tokenBuffer.count),
                    addSpecial,
                    true
                )
            }
        }

        guard tokenCount > 0 else {
            throw LLMEngineError.tokenizationFailed
        }
        return Array(tokens.prefix(Int(tokenCount)))
    }

    private func promptWithAutoAddedSpecialTokensStripped(
        _ prompt: String,
        vocab: OpaquePointer
    ) -> String {
        var output = prompt
        if llama_vocab_get_add_bos(vocab),
           let bosToken = specialTokenString(vocab: vocab, token: llama_vocab_bos(vocab)),
           !bosToken.isEmpty,
           output.hasPrefix(bosToken) {
            output = String(output.dropFirst(bosToken.count))
        }
        if llama_vocab_get_add_eos(vocab),
           let eosToken = specialTokenString(vocab: vocab, token: llama_vocab_eos(vocab)),
           !eosToken.isEmpty,
           output.hasSuffix(eosToken) {
            output = String(output.dropLast(eosToken.count))
        }
        return output
    }

    private func specialTokenString(vocab: OpaquePointer, token: llama_token) -> String? {
        guard token >= 0 else { return nil }
        let data = tokenToPiece(vocab: vocab, token: token, special: true)
        guard !data.isEmpty else { return nil }
        return String(data: data, encoding: .utf8)
    }

    private func tokenToPiece(vocab: OpaquePointer, token: llama_token, special: Bool = false) -> Data {
        var probe = [CChar](repeating: 0, count: 32)
        let probeCount = probe.withUnsafeMutableBufferPointer { buffer in
            llama_token_to_piece(vocab, token, buffer.baseAddress, Int32(buffer.count), 0, special)
        }

        if probeCount >= 0 {
            return probe.withUnsafeBytes { rawBuffer in
                Data(rawBuffer.prefix(Int(probeCount)))
            }
        }

        let neededCount = Int(-probeCount)
        var bytes = [CChar](repeating: 0, count: neededCount)
        let byteCount = bytes.withUnsafeMutableBufferPointer { buffer in
            llama_token_to_piece(vocab, token, buffer.baseAddress, Int32(buffer.count), 0, special)
        }

        guard byteCount > 0 else {
            return Data()
        }
        return bytes.withUnsafeBytes { rawBuffer in
            Data(rawBuffer.prefix(Int(byteCount)))
        }
    }

    private static let penaltyLastN: Int32 = 16
    private static let penaltyRepeat: Float = 1.3

    private func buildSampler(
        grammarMode: GenerationGrammarMode,
        options: GenerationOptions,
        vocab: OpaquePointer
    ) throws -> UnsafeMutablePointer<llama_sampler> {
        let params = llama_sampler_chain_default_params()
        guard let chain = llama_sampler_chain_init(params) else {
            throw LLMEngineError.samplerInitFailed
        }

        llama_sampler_chain_add(
            chain,
            llama_sampler_init_penalties(Self.penaltyLastN, Self.penaltyRepeat, 0.0, 0.0)
        )

        switch grammarMode {
        case .none:
            break
        case .eager(let grammar):
            let grammarSampler = Self.makeEagerGrammarSampler(
                grammar: grammar,
                vocab: vocab
            )
            guard let grammarSampler else {
                llama_sampler_free(chain)
                throw LLMEngineError.grammarParseFailed
            }
            llama_sampler_chain_add(chain, grammarSampler)
        case .lazy(let grammar, let triggerPatterns):
            let grammarSampler = Self.makeLazyGrammarSampler(
                grammar: grammar,
                triggerPatterns: triggerPatterns,
                vocab: vocab
            )
            guard let grammarSampler else {
                llama_sampler_free(chain)
                throw LLMEngineError.grammarParseFailed
            }
            llamaRuntimeLog.info(
                "Lazy grammar sampler initialized: triggerPatterns=\(triggerPatterns.count, privacy: .public)"
            )
            llama_sampler_chain_add(chain, grammarSampler)
        }

        let temperature = Float(options.temperature ?? 0)
        if temperature <= 0 {
            llama_sampler_chain_add(chain, llama_sampler_init_greedy())
        } else {
            if let topK = options.topK, topK > 0 {
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(Int32(topK)))
            }
            if let topP = options.topP {
                llama_sampler_chain_add(chain, llama_sampler_init_top_p(Float(topP), 1))
            }
            llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature))
            let seed = options.seed ?? UInt32.random(in: 1...UInt32.max)
            llama_sampler_chain_add(chain, llama_sampler_init_dist(seed))
        }

        return chain
    }

    private static func makeEagerGrammarSampler(
        grammar: String,
        vocab: OpaquePointer
    ) -> UnsafeMutablePointer<llama_sampler>? {
        grammar.withCString { grammarPointer in
            "root".withCString { rootPointer in
                llama_sampler_init_grammar(vocab, grammarPointer, rootPointer)
            }
        }
    }

    private static func makeLazyGrammarSampler(
        grammar: String,
        triggerPatterns: [String],
        vocab: OpaquePointer
    ) -> UnsafeMutablePointer<llama_sampler>? {
        let allocatedPatternPointers: [UnsafeMutablePointer<CChar>?] = triggerPatterns.map { strdup($0) }
        defer {
            for pointer in allocatedPatternPointers {
                free(pointer)
            }
        }
        guard allocatedPatternPointers.allSatisfy({ $0 != nil }) else {
            return nil
        }
        var patternPointers: [UnsafePointer<CChar>?] = allocatedPatternPointers.map { pointer in
            guard let pointer else { return nil }
            return UnsafePointer(pointer)
        }

        return grammar.withCString { grammarPointer in
            "root".withCString { rootPointer in
                patternPointers.withUnsafeMutableBufferPointer { buffer in
                    llama_sampler_init_grammar_lazy_patterns(
                        vocab,
                        grammarPointer,
                        rootPointer,
                        buffer.baseAddress,
                        buffer.count,
                        nil,
                        0
                    )
                }
            }
        }
    }

    static func generationGrammarMode(
        for options: GenerationOptions,
        profile: OutputSanitizationProfile,
        continuingOpenThinkingPairs: [OutputDelimiterPair]
    ) -> GenerationGrammarMode {
        guard let grammar = options.grammar else { return .none }

        let canStageStructuredOutput = !profile.thinkingPairs.isEmpty
            || profile.sliceAfterMarker != nil
            || !continuingOpenThinkingPairs.isEmpty
        guard options.enableThinking, canStageStructuredOutput else {
            return .eager(grammar: grammar)
        }

        let triggerPatterns = lazyGrammarTriggerPatterns(
            profile: profile,
            continuingOpenThinkingPairs: continuingOpenThinkingPairs
        )
        guard !triggerPatterns.isEmpty else {
            return .eager(grammar: grammar)
        }
        return .lazy(grammar: grammar, triggerPatterns: triggerPatterns)
    }

    static func lazyGrammarTriggerPatterns(
        profile: OutputSanitizationProfile,
        continuingOpenThinkingPairs: [OutputDelimiterPair]
    ) -> [String] {
        var patterns: [String] = []
        let jsonStartCapture = #"\s*(\{|\[)"#

        for pair in continuingOpenThinkingPairs {
            patterns.append(regexEscaped(pair.close) + jsonStartCapture)
        }

        for pair in profile.thinkingPairs {
            patterns.append(regexEscaped(pair.close) + jsonStartCapture)
        }

        if let marker = profile.sliceAfterMarker {
            patterns.append(regexEscaped(marker) + jsonStartCapture)
        }

        patterns.append(#"^\s*(\{|\[)"#)

        var deduplicated: [String] = []
        for pattern in patterns where !deduplicated.contains(pattern) {
            deduplicated.append(pattern)
        }
        return deduplicated
    }

    static func regexEscaped(_ literal: String) -> String {
        var escaped = ""
        for character in literal {
            if #"\\.^$|?*+()[]{}"#.contains(character) {
                escaped.append("\\")
            }
            escaped.append(character)
        }
        return escaped
    }

    static func mergingStopSequences(_ callerStops: [String], _ templateStops: [String]) -> [String] {
        var merged: [String] = []
        for stop in callerStops + templateStops where !stop.isEmpty && !merged.contains(stop) {
            merged.append(stop)
        }
        return merged
    }

    static func continuingOpenThinkingPairs(
        in renderedPrompt: String,
        profile: OutputSanitizationProfile
    ) -> [OutputDelimiterPair] {
        profile.thinkingPairs.filter { pair in
            guard let openRange = renderedPrompt.range(of: pair.open, options: .backwards) else {
                return false
            }
            guard let closeRange = renderedPrompt.range(of: pair.close, options: .backwards) else {
                return renderedPrompt[openRange.upperBound...]
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .isEmpty
            }
            return closeRange.lowerBound < openRange.lowerBound
                && renderedPrompt[openRange.upperBound...]
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .isEmpty
        }
    }

    static func structuredOutputPhase(
        in text: String,
        plan: StructuredOutputPlan
    ) -> StructuredOutputPhase {
        guard structuredFinalOutputSearchRange(in: text, plan: plan) == nil else {
            return .final
        }
        return structuredOutputIsInsideThinking(in: text, plan: plan)
            ? .thinking
            : .awaitingFinal
    }

    private static func structuredOutputIsInsideThinking(
        in text: String,
        plan: StructuredOutputPlan
    ) -> Bool {
        var lowerBound = text.startIndex
        for pair in plan.continuingOpenThinkingPairs {
            guard let closeRange = text.range(of: pair.close, range: lowerBound..<text.endIndex) else {
                return true
            }
            lowerBound = closeRange.upperBound
        }

        let trimmedLowerBound = indexAfterWhitespace(in: text, from: lowerBound)
        for pair in plan.profile.thinkingPairs where text[trimmedLowerBound...].hasPrefix(pair.open) {
            return text.range(of: pair.close, range: trimmedLowerBound..<text.endIndex) == nil
        }
        return false
    }

    static func firstStructuredGenerationBoundary(
        in text: String,
        stopSequences: [String],
        stopAtBalancedJSON: Bool,
        plan: StructuredOutputPlan
    ) -> GenerationBoundary? {
        let phase = structuredOutputPhase(in: text, plan: plan)
        let activeStopSequences = stopSequencesForStructuredPhase(
            stopSequences,
            phase: phase,
            plan: plan
        )
        var boundaryIndex: String.Index?
        var boundaryText: String?
        var reason: String?

        if let stopRange = firstStopSequenceRange(in: text, stopSequences: activeStopSequences) {
            boundaryIndex = stopRange.lowerBound
            boundaryText = String(text[..<stopRange.lowerBound])
            reason = "stop-sequence"
        }

        if stopAtBalancedJSON,
           let finalSearchRange = structuredFinalOutputSearchRange(in: text, plan: plan),
           let jsonRange = balancedJSONValueRange(in: text, searchRange: finalSearchRange),
           boundaryIndex.map({ jsonRange.upperBound < $0 }) ?? true {
            boundaryIndex = jsonRange.upperBound
            boundaryText = String(text[..<jsonRange.upperBound])
            reason = "json-complete"
        }

        guard boundaryIndex != nil, let boundaryText, let reason else {
            return nil
        }
        return GenerationBoundary(text: boundaryText, reason: reason)
    }

    private static func stopSequencesForStructuredPhase(
        _ stopSequences: [String],
        phase: StructuredOutputPhase,
        plan: StructuredOutputPlan
    ) -> [String] {
        guard phase == .thinking || phase == .awaitingFinal else {
            return stopSequences
        }

        let structuralStops = Set(
            plan.profile.thinkingPairs.map(\.close)
                + plan.continuingOpenThinkingPairs.map(\.close)
                + [plan.profile.sliceAfterMarker].compactMap { $0 }
        )
        return stopSequences.filter { !structuralStops.contains($0) }
    }

    static func structuredFinalOutputSearchRange(
        in text: String,
        plan: StructuredOutputPlan
    ) -> Range<String.Index>? {
        guard !text.isEmpty else { return nil }

        var lowerBound = text.startIndex
        for pair in plan.continuingOpenThinkingPairs {
            guard let closeRange = text.range(of: pair.close, range: lowerBound..<text.endIndex) else {
                return nil
            }
            lowerBound = closeRange.upperBound
        }

        guard let afterThinkingBlocks = indexAfterGeneratedThinkingPrefix(
            in: text,
            from: lowerBound,
            pairs: plan.profile.thinkingPairs
        ) else {
            return nil
        }
        lowerBound = afterThinkingBlocks

        if let jsonStart = immediateJSONStartIndex(in: text, from: lowerBound) {
            return jsonStart..<text.endIndex
        }

        if let marker = plan.profile.sliceAfterMarker,
           let markerRange = text.range(of: marker, range: lowerBound..<text.endIndex) {
            lowerBound = markerRange.upperBound
            guard let afterMarkerThinkingBlocks = indexAfterGeneratedThinkingPrefix(
                in: text,
                from: lowerBound,
                pairs: plan.profile.thinkingPairs
            ) else {
                return nil
            }
            lowerBound = afterMarkerThinkingBlocks

            if let jsonStart = immediateJSONStartIndex(in: text, from: lowerBound) {
                return jsonStart..<text.endIndex
            }
        }

        return nil
    }

    private static func indexAfterGeneratedThinkingPrefix(
        in text: String,
        from start: String.Index,
        pairs: [OutputDelimiterPair]
    ) -> String.Index? {
        var lowerBound = indexAfterWhitespace(in: text, from: start)
        var strippedBlock = true

        while strippedBlock {
            strippedBlock = false
            for pair in pairs where text[lowerBound...].hasPrefix(pair.open) {
                guard let closeRange = text.range(of: pair.close, range: lowerBound..<text.endIndex) else {
                    return nil
                }
                lowerBound = indexAfterWhitespace(in: text, from: closeRange.upperBound)
                strippedBlock = true
                break
            }
        }

        return lowerBound
    }

    private static func immediateJSONStartIndex(
        in text: String,
        from start: String.Index
    ) -> String.Index? {
        let lowerBound = indexAfterWhitespace(in: text, from: start)
        guard lowerBound < text.endIndex else { return nil }
        return text[lowerBound] == "{" || text[lowerBound] == "["
            ? lowerBound
            : nil
    }

    private static func indexAfterWhitespace(
        in text: String,
        from start: String.Index
    ) -> String.Index {
        var index = start
        while index < text.endIndex, text[index].isWhitespace {
            index = text.index(after: index)
        }
        return index
    }

    static func sanitizedGeneratedText(
        _ accumulatedText: String,
        profile: OutputSanitizationProfile,
        continuingOpenThinkingPairs: [OutputDelimiterPair],
        requiresNonEmptyStructuredOutput: Bool
    ) throws -> String {
        let returnedText = profile.isEmpty
            ? accumulatedText
            : LLMResponseSanitizer.unwrapStructuredOutput(
                accumulatedText,
                using: profile,
                continuingOpenThinkingPairs: continuingOpenThinkingPairs
            )

        if requiresNonEmptyStructuredOutput,
           !accumulatedText.isEmpty,
           returnedText.isEmpty {
            throw LLMEngineError.structuredOutputPhaseFailed(
                "Sanitization removed all generated structured output."
            )
        }

        return returnedText
    }

    static func trimmingAtFirstStopSequence(
        _ text: String,
        stopSequences: [String]
    ) -> String? {
        guard let earliest = firstStopSequenceRange(in: text, stopSequences: stopSequences) else {
            return nil
        }
        return String(text[..<earliest.lowerBound])
    }

    static func firstStopSequenceRange(
        in text: String,
        stopSequences: [String]
    ) -> Range<String.Index>? {
        stopSequences
            .filter { !$0.isEmpty }
            .compactMap { sequence -> Range<String.Index>? in
                text.range(of: sequence)
            }
            .min(by: { $0.lowerBound < $1.lowerBound })
    }

    struct GenerationBoundary: Equatable {
        var text: String
        var reason: String
    }

    static func firstGenerationBoundary(
        in text: String,
        stopSequences: [String],
        stopAtBalancedJSON: Bool
    ) -> GenerationBoundary? {
        var boundaryIndex: String.Index?
        var boundaryText: String?
        var reason: String?

        if let stopRange = firstStopSequenceRange(in: text, stopSequences: stopSequences) {
            boundaryIndex = stopRange.lowerBound
            boundaryText = String(text[..<stopRange.lowerBound])
            reason = "stop-sequence"
        }

        if stopAtBalancedJSON,
           let jsonRange = balancedJSONValueRange(in: text),
           boundaryIndex.map({ jsonRange.upperBound < $0 }) ?? true {
            boundaryIndex = jsonRange.upperBound
            boundaryText = String(text[jsonRange])
            reason = "json-complete"
        }

        guard boundaryIndex != nil, let boundaryText, let reason else {
            return nil
        }
        return GenerationBoundary(text: boundaryText, reason: reason)
    }

    static func balancedJSONValueRange(in text: String) -> Range<String.Index>? {
        balancedJSONValueRange(in: text, searchRange: text.startIndex..<text.endIndex)
    }

    static func balancedJSONValueRange(
        in text: String,
        searchRange: Range<String.Index>
    ) -> Range<String.Index>? {
        var startIndex: String.Index?
        var expectedClosers: [Character] = []
        var inString = false
        var escaped = false

        var index = searchRange.lowerBound
        while index < searchRange.upperBound {
            let character = text[index]

            if startIndex == nil {
                if character == "{" {
                    startIndex = index
                    expectedClosers = ["}"]
                } else if character == "[" {
                    startIndex = index
                    expectedClosers = ["]"]
                }
                index = text.index(after: index)
                continue
            }

            if escaped {
                escaped = false
                index = text.index(after: index)
                continue
            }

            if character == "\\" && inString {
                escaped = true
                index = text.index(after: index)
                continue
            }

            if character == "\"" {
                inString.toggle()
                index = text.index(after: index)
                continue
            }

            if !inString {
                if character == "{" {
                    expectedClosers.append("}")
                } else if character == "[" {
                    expectedClosers.append("]")
                } else if character == "}" || character == "]" {
                    guard expectedClosers.last == character else {
                        return nil
                    }
                    expectedClosers.removeLast()
                    if expectedClosers.isEmpty, let startIndex {
                        return startIndex..<text.index(after: index)
                    }
                }
            }

            index = text.index(after: index)
        }

        return nil
    }

    static func promptPrefillPlan(
        cachedPromptTokens: [llama_token]?,
        newPromptTokens: [llama_token]
    ) -> PromptPrefillPlan {
        guard let cachedPromptTokens,
              !cachedPromptTokens.isEmpty,
              !newPromptTokens.isEmpty
        else {
            return PromptPrefillPlan(
                commonPrefixCount: 0,
                retainedPrefixCount: 0,
                shouldClearMemory: true,
                removeStartPosition: nil,
                decodeStartIndex: 0
            )
        }

        let common = commonTokenPrefixCount(cachedPromptTokens, newPromptTokens)
        guard common > 0 else {
            return PromptPrefillPlan(
                commonPrefixCount: 0,
                retainedPrefixCount: 0,
                shouldClearMemory: true,
                removeStartPosition: nil,
                decodeStartIndex: 0
            )
        }

        let decodeStart = common == newPromptTokens.count ? max(0, common - 1) : common
        guard decodeStart > 0 else {
            return PromptPrefillPlan(
                commonPrefixCount: common,
                retainedPrefixCount: 0,
                shouldClearMemory: true,
                removeStartPosition: nil,
                decodeStartIndex: 0
            )
        }

        return PromptPrefillPlan(
            commonPrefixCount: common,
            retainedPrefixCount: decodeStart,
            shouldClearMemory: false,
            removeStartPosition: decodeStart,
            decodeStartIndex: decodeStart
        )
    }

    static func commonTokenPrefixCount(_ lhs: [llama_token], _ rhs: [llama_token]) -> Int {
        let count = min(lhs.count, rhs.count)
        var index = 0
        while index < count, lhs[index] == rhs[index] {
            index += 1
        }
        return index
    }

    static func prefillRanges(tokenCount: Int, maxBatchSize: Int) -> [Range<Int>] {
        guard tokenCount > 0, maxBatchSize > 0 else { return [] }

        var ranges: [Range<Int>] = []
        var start = 0
        while start < tokenCount {
            let end = min(start + maxBatchSize, tokenCount)
            ranges.append(start..<end)
            start = end
        }
        return ranges
    }

    static func maxGenerationTokens(contextSize: Int, promptTokenCount: Int, reserve: Int) -> Int {
        max(0, contextSize - promptTokenCount - reserve)
    }

    static func calgacusRank(of tokenID: Int32, in logits: [Float]) throws -> Int {
        guard tokenID >= 0, Int(tokenID) < logits.count else {
            throw CalgacusError.invalidTokenID(tokenID, vocabularySize: logits.count)
        }

        let targetIndex = Int(tokenID)
        let targetLogit = calgacusComparableLogit(logits[targetIndex])
        var rank = 1
        for index in logits.indices where index != targetIndex {
            let logit = calgacusComparableLogit(logits[index])
            if logit > targetLogit || (logit == targetLogit && index < targetIndex) {
                rank += 1
            }
        }
        return rank
    }

    static func calgacusToken(atRank rank: Int, in logits: [Float]) throws -> Int32 {
        guard rank >= 1, rank <= logits.count else {
            throw CalgacusError.invalidRank(rank, vocabularySize: logits.count)
        }

        let sorted = logits.indices.sorted { lhs, rhs in
            let lhsLogit = calgacusComparableLogit(logits[lhs])
            let rhsLogit = calgacusComparableLogit(logits[rhs])
            if lhsLogit == rhsLogit {
                return lhs < rhs
            }
            return lhsLogit > rhsLogit
        }
        return Int32(sorted[rank - 1])
    }

    static func calgacusNegativeLogProbability(of tokenID: Int32, in logits: [Float]) throws -> Double {
        guard tokenID >= 0, Int(tokenID) < logits.count else {
            throw CalgacusError.invalidTokenID(tokenID, vocabularySize: logits.count)
        }

        let finiteLogits = logits
            .map { Double($0) }
            .filter { $0.isFinite }
        guard let maxLogit = finiteLogits.max() else {
            return .infinity
        }

        let normalizer = finiteLogits.reduce(0.0) { partial, logit in
            partial + exp(logit - maxLogit)
        }
        let targetLogit = Double(calgacusComparableLogit(logits[Int(tokenID)]))
        guard targetLogit.isFinite, normalizer > 0 else {
            return .infinity
        }

        return maxLogit + log(normalizer) - targetLogit
    }

    static func calgacusStats(for trace: [CalgacusTraceEntry]) -> CalgacusRankStats {
        guard !trace.isEmpty else {
            return CalgacusRankStats(
                tokenCount: 0,
                maxRank: 0,
                meanRank: 0,
                medianRank: 0,
                cumulativeNegativeLogProbability: 0,
                averageNegativeLogProbability: 0
            )
        }

        let ranks = trace.map(\.rank).sorted()
        let rankSum = ranks.reduce(0, +)
        let nll = trace.reduce(0.0) { $0 + $1.negativeLogProbability }
        let median: Double
        if ranks.count.isMultiple(of: 2) {
            median = Double(ranks[(ranks.count / 2) - 1] + ranks[ranks.count / 2]) / 2.0
        } else {
            median = Double(ranks[ranks.count / 2])
        }

        return CalgacusRankStats(
            tokenCount: trace.count,
            maxRank: ranks.last ?? 0,
            meanRank: Double(rankSum) / Double(ranks.count),
            medianRank: median,
            cumulativeNegativeLogProbability: nll,
            averageNegativeLogProbability: nll / Double(trace.count)
        )
    }

    static func calgacusValidateBudget(
        operation: String,
        contextSize: Int,
        contextTokenCount: Int,
        payloadTokenCount: Int
    ) throws {
        let requiredTokens = contextTokenCount + payloadTokenCount
        guard requiredTokens <= contextSize else {
            throw CalgacusError.contextBudgetExceeded(
                operation: operation,
                contextSize: contextSize,
                requiredTokens: requiredTokens
            )
        }
    }

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
        threads: Int32
    ) -> llama_context_params {
        var params = llama_context_default_params()
        let clampedContext = max(1, contextSize)
        let clampedBatch = min(clampedContext, max(1, batchSize))
        params.n_ctx = UInt32(clampedContext)
        params.n_batch = UInt32(clampedBatch)
        params.n_ubatch = UInt32(clampedBatch)
        params.n_threads = threads
        params.n_threads_batch = threads
        return params
    }

    private struct CalgacusSelectedPayload {
        var tokens: [llama_token]
        var data: Data
    }

    private struct CalgacusDecodedPayload {
        var result: CalgacusDecodeResult
        var recoveredTokens: [llama_token]
    }

    private static func calgacusComparableLogit(_ logit: Float) -> Float {
        logit.isFinite ? logit : -.infinity
    }

    private func calgacusDecodePayload(
        coverText: String,
        coverPrompt: String,
        context: OpaquePointer,
        vocabulary: OpaquePointer,
        emitsEvents: Bool,
        onEvent: @Sendable (CalgacusEvent) -> Void
    ) throws -> CalgacusDecodedPayload {
        let coverTokens = try tokenize(vocab: vocabulary, text: coverText, addSpecial: false)
        guard !coverTokens.isEmpty else {
            throw CalgacusError.emptyCoverText
        }
        if emitsEvents {
            onEvent(.tokensPrepared(operation: "cover", count: coverTokens.count))
        }

        let coverPromptTokens = try calgacusPromptTokens(vocab: vocabulary, text: coverPrompt)
        try Self.calgacusValidateBudget(
            operation: "Cover ranking",
            contextSize: currentContextSize(),
            contextTokenCount: coverPromptTokens.count,
            payloadTokenCount: coverTokens.count
        )
        let coverTrace: [CalgacusTraceEntry]
        if emitsEvents {
            coverTrace = try calgacusTrace(
                tokens: coverTokens,
                initialContextTokens: coverPromptTokens,
                context: context,
                vocabulary: vocabulary,
                stage: .coverRanking,
                onEvent: onEvent
            )
        } else {
            coverTrace = try calgacusTrace(
                tokens: coverTokens,
                initialContextTokens: coverPromptTokens,
                context: context,
                vocabulary: vocabulary,
                stage: .verification,
                onEvent: { (_: CalgacusEvent) in }
            )
        }
        let ranks = coverTrace.map(\.rank)

        let secretContextTokens = try calgacusInitialTokens(vocab: vocabulary)
        try Self.calgacusValidateBudget(
            operation: "Secret recovery",
            contextSize: currentContextSize(),
            contextTokenCount: secretContextTokens.count,
            payloadTokenCount: ranks.count
        )
        let recovered: CalgacusSelectedPayload
        if emitsEvents {
            recovered = try calgacusSelectTokens(
                ranks: ranks,
                initialContextTokens: secretContextTokens,
                context: context,
                vocabulary: vocabulary,
                stage: .secretRecovery,
                operation: "Secret recovery",
                rejectsControlTokens: false,
                onEvent: onEvent
            )
        } else {
            recovered = try calgacusSelectTokens(
                ranks: ranks,
                initialContextTokens: secretContextTokens,
                context: context,
                vocabulary: vocabulary,
                stage: .verification,
                operation: "Secret recovery",
                rejectsControlTokens: false,
                onEvent: { (_: CalgacusEvent) in }
            )
        }
        guard let secretText = String(data: recovered.data, encoding: .utf8) else {
            throw CalgacusError.textRenderingFailed(operation: "Secret recovery")
        }

        return CalgacusDecodedPayload(
            result: CalgacusDecodeResult(
                secretText: secretText,
                coverTokenCount: coverTokens.count,
                recoveredTokenCount: recovered.tokens.count,
                stats: Self.calgacusStats(for: coverTrace),
                trace: coverTrace
            ),
            recoveredTokens: recovered.tokens
        )
    }

    private func calgacusTrace(
        tokens: [llama_token],
        initialContextTokens: [llama_token],
        context: OpaquePointer,
        vocabulary: OpaquePointer,
        stage: CalgacusStage,
        onEvent: @Sendable (CalgacusEvent) -> Void
    ) throws -> [CalgacusTraceEntry] {
        try prefillCalgacusContext(initialContextTokens, context: context)

        var trace: [CalgacusTraceEntry] = []
        trace.reserveCapacity(tokens.count)

        for (index, token) in tokens.enumerated() {
            try Task.checkCancellation()
            let logits = try calgacusLogits(context: context, vocabulary: vocabulary)
            let rank = try Self.calgacusRank(of: token, in: logits)
            let nll = try Self.calgacusNegativeLogProbability(of: token, in: logits)
            trace.append(CalgacusTraceEntry(
                index: index,
                tokenID: token,
                tokenText: calgacusDisplayText(for: token, vocab: vocabulary),
                rank: rank,
                negativeLogProbability: nll
            ))
            onEvent(.tokenProcessed(stage: stage, index: index + 1, total: tokens.count, rank: rank))
            try decodeCalgacusToken(token, context: context)
        }

        return trace
    }

    private func calgacusSelectTokens(
        ranks: [Int],
        initialContextTokens: [llama_token],
        context: OpaquePointer,
        vocabulary: OpaquePointer,
        stage: CalgacusStage,
        operation: String,
        rejectsControlTokens: Bool,
        onEvent: @Sendable (CalgacusEvent) -> Void
    ) throws -> CalgacusSelectedPayload {
        try prefillCalgacusContext(initialContextTokens, context: context)

        var tokens: [llama_token] = []
        var data = Data()
        tokens.reserveCapacity(ranks.count)

        for (index, rank) in ranks.enumerated() {
            try Task.checkCancellation()
            let logits = try calgacusLogits(context: context, vocabulary: vocabulary)
            let token = try Self.calgacusToken(atRank: rank, in: logits)
            if rejectsControlTokens {
                if llama_vocab_is_eog(vocabulary, token) {
                    throw CalgacusError.invalidGeneratedCoverToken(rank: rank, tokenID: token, reason: "end-of-generation token")
                }
                if llama_vocab_is_control(vocabulary, token) {
                    throw CalgacusError.invalidGeneratedCoverToken(rank: rank, tokenID: token, reason: "control token")
                }
            }

            let piece = tokenToPiece(vocab: vocabulary, token: token)
            guard !piece.isEmpty else {
                if rejectsControlTokens {
                    throw CalgacusError.invalidGeneratedCoverToken(rank: rank, tokenID: token, reason: "empty rendered token")
                }
                throw CalgacusError.textRenderingFailed(operation: operation)
            }

            tokens.append(token)
            data.append(piece)
            onEvent(.tokenProcessed(stage: stage, index: index + 1, total: ranks.count, rank: rank))
            try decodeCalgacusToken(token, context: context)
        }

        return CalgacusSelectedPayload(tokens: tokens, data: data)
    }

    private func calgacusInitialTokens(vocab: OpaquePointer) throws -> [llama_token] {
        let tokens = try tokenize(vocab: vocab, text: "", addSpecial: true)
        if !tokens.isEmpty {
            return tokens
        }

        let bos = llama_vocab_bos(vocab)
        guard bos >= 0 else {
            throw CalgacusError.noInitialContext
        }
        return [bos]
    }

    private func calgacusPromptTokens(vocab: OpaquePointer, text: String) throws -> [llama_token] {
        let tokens = try tokenize(vocab: vocab, text: text, addSpecial: true)
        if !tokens.isEmpty {
            return tokens
        }
        return try calgacusInitialTokens(vocab: vocab)
    }

    private func prefillCalgacusContext(_ tokens: [llama_token], context: OpaquePointer) throws {
        llama_memory_clear(llama_get_memory(context), false)

        let maxBatchSize = max(1, Int(llama_n_batch(context)))
        for range in Self.prefillRanges(tokenCount: tokens.count, maxBatchSize: maxBatchSize) {
            var chunk = Array(tokens[range])
            let decodeResult = chunk.withUnsafeMutableBufferPointer { buffer -> Int32 in
                let batch = llama_batch_get_one(buffer.baseAddress, Int32(buffer.count))
                return llama_decode(context, batch)
            }
            if decodeResult != 0 {
                throw LLMEngineError.decodeFailed
            }
        }
    }

    private func decodeCalgacusToken(_ token: llama_token, context: OpaquePointer) throws {
        var oneToken: [llama_token] = [token]
        let decodeResult = oneToken.withUnsafeMutableBufferPointer { buffer -> Int32 in
            let batch = llama_batch_get_one(buffer.baseAddress, 1)
            return llama_decode(context, batch)
        }
        if decodeResult != 0 {
            throw LLMEngineError.decodeFailed
        }
    }

    private func calgacusLogits(context: OpaquePointer, vocabulary: OpaquePointer) throws -> [Float] {
        let vocabularySize = Int(llama_vocab_n_tokens(vocabulary))
        guard vocabularySize > 0,
              let logits = llama_get_logits_ith(context, -1)
        else {
            throw CalgacusError.logitsUnavailable
        }

        let buffer = UnsafeBufferPointer(start: logits, count: vocabularySize)
        return Array(buffer)
    }

    private func calgacusDisplayText(for token: llama_token, vocab: OpaquePointer) -> String {
        let data = tokenToPiece(vocab: vocab, token: token)
        if let text = String(data: data, encoding: .utf8), !text.isEmpty {
            return text
        }
        return "<token:\(token)>"
    }
}

extension llama_model_params {
    mutating func configureForCPUOnly() {
        n_gpu_layers = 0
        split_mode = LLAMA_SPLIT_MODE_NONE
        main_gpu = -1
    }
}
