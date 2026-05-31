import CarbocationLocalLLM
import CarbocationLlamaMTMDBridge
import Foundation
import llama

final class LlamaPreparedMultimodalPrompt: @unchecked Sendable {
    let chunks: UnsafeMutableRawPointer
    let promptTokenCount: Int
    let promptPositionCount: Int

    init(chunks: UnsafeMutableRawPointer, promptTokenCount: Int, promptPositionCount: Int) {
        self.chunks = chunks
        self.promptTokenCount = promptTokenCount
        self.promptPositionCount = promptPositionCount
    }

    deinit {
        carbocation_mtmd_input_chunks_free_bridge(chunks)
    }

    func evaluate(
        mtmdContext: UnsafeMutableRawPointer?,
        llamaContext: OpaquePointer,
        batchSize: Int32
    ) throws {
        guard let mtmdContext else {
            throw LLMEngineError.visionProjectorMissing
        }
        let result = carbocation_mtmd_helper_eval_chunks_bridge(
            mtmdContext,
            llamaContext,
            chunks,
            batchSize
        )
        guard result == 0 else {
            throw LLMEngineError.decodeFailed
        }
    }
}

struct LlamaVisionTokenAccounting {
    static func contextCost(tokenCount: Int, positionCount: Int) -> Int {
        max(tokenCount, positionCount)
    }
}

extension LlamaEngine {
    struct VisionProjectorLoadResult {
        var context: UnsafeMutableRawPointer?
        var unsupportedDetail: String?
    }

    struct PublicMessageFormatting {
        var messages: [ChatTemplateMessage]
        var images: [(location: LLMContentLocation, image: LLMRGBImage)]
    }

    public nonisolated static func projectorSupportsVision(at url: URL) -> Bool {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return false
        }
        return url.path.withCString { cPath in
            carbocation_mtmd_get_cap_from_file_bridge(cPath).inp_vision
        }
    }

    static func loadVisionProjectorIfAvailable(
        mmprojURL: URL?,
        model: OpaquePointer,
        threads: Int32,
        useGPU: Bool
    ) -> VisionProjectorLoadResult {
        guard let mmprojURL else {
            return VisionProjectorLoadResult(context: nil, unsupportedDetail: nil)
        }
        llamaRuntimeLog.info("Loading vision projector at \(mmprojURL.path, privacy: .public)")
        guard FileManager.default.fileExists(atPath: mmprojURL.path) else {
            let detail = "The mmproj artifact file was not found."
            llamaRuntimeLog.info("Vision projector unavailable: \(detail, privacy: .public)")
            return VisionProjectorLoadResult(
                context: nil,
                unsupportedDetail: detail
            )
        }
        let caps = mmprojURL.path.withCString { cPath in
            carbocation_mtmd_get_cap_from_file_bridge(cPath)
        }
        guard caps.inp_vision else {
            let detail = "mtmd_get_cap_from_file did not report image input support."
            llamaRuntimeLog.info("Vision projector rejected: \(detail, privacy: .public)")
            return VisionProjectorLoadResult(
                context: nil,
                unsupportedDetail: detail
            )
        }

        guard let context = mmprojURL.path.withCString({ cPath in
            carbocation_mtmd_init_from_file_bridge(
                cPath,
                model,
                useGPU,
                Int32(max(1, threads))
            )
        }) else {
            let detail = "mtmd_init_from_file returned null."
            llamaRuntimeLog.info("Vision projector rejected: \(detail, privacy: .public)")
            return VisionProjectorLoadResult(
                context: nil,
                unsupportedDetail: detail
            )
        }

        guard carbocation_mtmd_support_vision_bridge(context) else {
            carbocation_mtmd_free_bridge(context)
            let detail = "The initialized mtmd context does not support image input."
            llamaRuntimeLog.info("Vision projector rejected: \(detail, privacy: .public)")
            return VisionProjectorLoadResult(
                context: nil,
                unsupportedDetail: detail
            )
        }

        return VisionProjectorLoadResult(context: context, unsupportedDetail: nil)
    }

    public func preflight(
        messages: [LLMChatMessage],
        options: GenerationOptions
    ) async throws -> LLMGenerationPreflight {
        guard context != nil, let loadedInfo else {
            throw LLMEngineError.noModelLoaded
        }

        if LLMChatMessage.inputModalities(in: messages).contains(.image) {
            let prepared = try prepareMultimodalPrompt(messages: messages, options: options)
            try validateSamplerForPreflight(promptFormatting: prepared.promptFormatting, options: options)
            let promptCost = LlamaVisionTokenAccounting.contextCost(
                tokenCount: prepared.multimodalPrefill.promptTokenCount,
                positionCount: prepared.multimodalPrefill.promptPositionCount
            )
            guard promptCost < loadedInfo.contextSize else {
                throw LLMEngineError.contextBudgetExceeded(
                    contextSize: loadedInfo.contextSize,
                    promptTokens: promptCost,
                    reserve: configuration.promptReserveTokens
                )
            }
            return LLMGenerationPreflight(
                loadedContextSize: loadedInfo.contextSize,
                modelTrainingContextSize: loadedInfo.trainingContextSize,
                promptTokens: promptCost,
                reservedOutputTokens: configuration.promptReserveTokens,
                requestedMaxOutputTokens: options.maxOutputTokens,
                usesExactTokenCounts: true,
                templateMode: prepared.promptFormatting.mode
            )
        }

        if chatTemplate == nil {
            let textOnly = try LLMChatTextRenderer.textOnlySystemAndPrompt(from: messages)
            return try await preflight(system: textOnly.system, prompt: textOnly.prompt, options: options)
        }

        let promptFormatting = try applyChatTemplate(
            messages: publicChatTemplateMessages(from: messages, mediaMarker: nil).messages,
            tools: [],
            options: options
        )
        return try preflight(promptFormatting: promptFormatting, options: options)
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
        control: LLMGenerationControl? = nil,
        onPhaseAwareEvent: @escaping @Sendable (LLMPhaseAwareStreamEvent) -> Void
    ) async throws -> String {
        let result = try await generatePhased(
            messages: messages,
            options: options,
            control: control,
            onEvent: { event in
                if let phaseAwareEvent = event.phaseAwareEvent {
                    onPhaseAwareEvent(phaseAwareEvent)
                }
            }
        )
        return result.finalText
    }

    public func generatePhased(
        messages: [LLMChatMessage],
        options: GenerationOptions,
        control: LLMGenerationControl? = nil,
        onEvent: @escaping @Sendable (LLMGenerationStreamEvent) -> Void = { _ in }
    ) async throws -> LLMGenerationResult {
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

        if LLMChatMessage.inputModalities(in: messages).contains(.image) {
            let prepared = try prepareMultimodalPrompt(messages: messages, options: options)
            return try await generatePhased(
                promptFormatting: prepared.promptFormatting,
                multimodalPrefill: prepared.multimodalPrefill,
                options: options,
                control: control,
                controlGenerationID: controlGenerationID,
                onEvent: onEvent
            )
        }

        if chatTemplate == nil {
            let textOnly = try LLMChatTextRenderer.textOnlySystemAndPrompt(from: messages)
            let promptFormatting = try applyChatTemplate(
                system: textOnly.system,
                user: textOnly.prompt,
                options: options
            )
            return try await generatePhased(
                promptFormatting: promptFormatting,
                options: options,
                control: control,
                controlGenerationID: controlGenerationID,
                onEvent: onEvent
            )
        }

        let promptFormatting = try applyChatTemplate(
            messages: publicChatTemplateMessages(from: messages, mediaMarker: nil).messages,
            tools: [],
            options: options
        )
        return try await generatePhased(
            promptFormatting: promptFormatting,
            options: options,
            control: control,
            controlGenerationID: controlGenerationID,
            onEvent: onEvent
        )
    }

    func prepareMultimodalPrompt(
        messages: [LLMChatMessage],
        options: GenerationOptions
    ) throws -> (promptFormatting: PromptFormattingResult, multimodalPrefill: LlamaPreparedMultimodalPrompt) {
        let mtmdContext = try requireVisionProjector()
        let marker = String(cString: carbocation_mtmd_default_marker_bridge())
        let formattedMessages = try publicChatTemplateMessages(from: messages, mediaMarker: marker)
        let promptFormatting = try applyChatTemplate(
            messages: formattedMessages.messages,
            tools: [],
            options: options
        )
        let prefill = try makeMultimodalPrefill(
            prompt: promptFormatting.text,
            images: formattedMessages.images,
            mtmdContext: mtmdContext
        )
        return (promptFormatting, prefill)
    }

    func requireVisionProjector() throws -> UnsafeMutableRawPointer {
        if let mtmdContext, loadedInfo?.supportedInputModalities.contains(.image) == true {
            return mtmdContext
        }
        guard let mmprojURL = loadedDescriptor?.mmprojURL,
              FileManager.default.fileExists(atPath: mmprojURL.path)
        else {
            throw LLMEngineError.visionProjectorMissing
        }
        throw LLMEngineError.visionProjectorUnsupported(
            visionProjectorUnsupportedDetail ?? "The loaded mmproj artifact does not support image input."
        )
    }

    func publicChatTemplateMessages(
        from messages: [LLMChatMessage],
        mediaMarker: String?
    ) throws -> PublicMessageFormatting {
        var chatMessages: [ChatTemplateMessage] = []
        var images: [(location: LLMContentLocation, image: LLMRGBImage)] = []

        for (messageIndex, message) in messages.enumerated() {
            var renderedContent = ""
            for (partIndex, part) in message.content.enumerated() {
                switch part {
                case .text(let text):
                    renderedContent += text
                case .image(let imageInput):
                    let location = LLMContentLocation(messageIndex: messageIndex, partIndex: partIndex)
                    guard message.role == .user else {
                        throw LLMEngineError.unsupportedImagePlacement(location: location)
                    }
                    guard let mediaMarker else {
                        throw LLMEngineError.unsupportedInputModality(.image, location: location)
                    }
                    let image = try imageInput.normalizedRGB8(location: location)
                    images.append((location, image))
                    renderedContent += mediaMarker
                }
            }
            chatMessages.append(ChatTemplateMessage(role: message.role.rawValue, content: renderedContent))
        }

        return PublicMessageFormatting(messages: chatMessages, images: images)
    }

    func makeMultimodalPrefill(
        prompt: String,
        images: [(location: LLMContentLocation, image: LLMRGBImage)],
        mtmdContext: UnsafeMutableRawPointer
    ) throws -> LlamaPreparedMultimodalPrompt {
        guard !images.isEmpty else {
            throw LLMEngineError.imageTokenizationFailed("No image payloads were supplied.")
        }

        var bitmaps: [UnsafeMutableRawPointer] = []
        bitmaps.reserveCapacity(images.count)
        defer {
            for bitmap in bitmaps {
                carbocation_mtmd_bitmap_free_bridge(bitmap)
            }
        }

        for item in images {
            guard item.image.width <= Int(UInt32.max),
                  item.image.height <= Int(UInt32.max) else {
                throw LLMEngineError.invalidImageData(
                    "Image dimensions exceed mtmd bitmap limits.",
                    location: item.location
                )
            }
            let bitmap = item.image.data.withUnsafeBytes { rawBuffer -> UnsafeMutableRawPointer? in
                guard let baseAddress = rawBuffer.bindMemory(to: UInt8.self).baseAddress else {
                    return nil
                }
                return carbocation_mtmd_bitmap_init_bridge(
                    UInt32(item.image.width),
                    UInt32(item.image.height),
                    baseAddress
                )
            }
            guard let bitmap else {
                throw LLMEngineError.imageTokenizationFailed(
                    "mtmd_bitmap_init returned null.",
                    location: item.location
                )
            }
            bitmaps.append(bitmap)
        }

        guard let chunks = carbocation_mtmd_input_chunks_init_bridge() else {
            throw LLMEngineError.imageTokenizationFailed("mtmd_input_chunks_init returned null.")
        }

        let tokenizationResult = prompt.withCString { cPrompt in
            var bitmapPointers = bitmaps.map { Optional($0) }
            return bitmapPointers.withUnsafeMutableBufferPointer { buffer in
                carbocation_mtmd_tokenize_bridge(
                    mtmdContext,
                    chunks,
                    cPrompt,
                    buffer.baseAddress,
                    buffer.count
                )
            }
        }

        guard tokenizationResult == 0 else {
            carbocation_mtmd_input_chunks_free_bridge(chunks)
            let detail: String
            switch tokenizationResult {
            case 1:
                detail = "The rendered prompt media-marker count did not match the image count."
            case 2:
                detail = "Image preprocessing failed."
            default:
                detail = "mtmd_tokenize returned \(tokenizationResult)."
            }
            throw LLMEngineError.imageTokenizationFailed(detail, location: images.first?.location)
        }

        let tokenCount = Int(carbocation_mtmd_helper_get_n_tokens_bridge(chunks))
        let positionCount = Int(carbocation_mtmd_helper_get_n_pos_bridge(chunks))
        guard tokenCount > 0, positionCount > 0 else {
            carbocation_mtmd_input_chunks_free_bridge(chunks)
            throw LLMEngineError.imageTokenizationFailed(
                "mtmd produced empty prompt chunks.",
                location: images.first?.location
            )
        }

        return LlamaPreparedMultimodalPrompt(
            chunks: chunks,
            promptTokenCount: tokenCount,
            promptPositionCount: positionCount
        )
    }

    func validateSamplerForPreflight(
        promptFormatting: PromptFormattingResult,
        options: GenerationOptions
    ) throws {
        guard options.grammar != nil, let vocabulary else { return }

        let activeOutputProfile = promptFormatting.outputProfile.merging(options.streamPhaseConfiguration)
        let continuingOpenThinkingPairs = Self.continuingOpenThinkingPairs(
            in: promptFormatting.text,
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
        let samplerRuntime = try buildSampler(
            grammarMode: grammarMode,
            options: options,
            vocab: vocabulary,
            reasoningBudgetPlan: reasoningBudgetPlan
        )
        llama_sampler_free(samplerRuntime.chain)
    }
}
