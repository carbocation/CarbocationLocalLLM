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
            throw LLMEngineError.multimodalProjectorMissing
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
    struct MultimodalProjectorLoadResult {
        var context: UnsafeMutableRawPointer?
        var unsupportedDetail: String?
        var supportedInputModalities: Set<LLMInputModality>
    }

    enum LlamaMediaPayload {
        case image(LLMRGBImage)
        case audio(LLMAudioInput)

        var inputModality: LLMInputModality {
            switch self {
            case .image:
                return .image
            case .audio:
                return .audio
            }
        }
    }

    struct PublicMessageFormatting {
        var messages: [ChatTemplateMessage]
        var media: [(location: LLMContentLocation, payload: LlamaMediaPayload)]

        var images: [(location: LLMContentLocation, image: LLMRGBImage)] {
            media.compactMap { item in
                if case .image(let image) = item.payload {
                    return (item.location, image)
                }
                return nil
            }
        }

        var audio: [(location: LLMContentLocation, audio: LLMAudioInput)] {
            media.compactMap { item in
                if case .audio(let audio) = item.payload {
                    return (item.location, audio)
                }
                return nil
            }
        }
    }

    public nonisolated static func projectorSupportsVision(at url: URL) -> Bool {
        projectorSupportedInputModalities(at: url).contains(.image)
    }

    public nonisolated static func projectorSupportsAudio(at url: URL) -> Bool {
        projectorSupportedInputModalities(at: url).contains(.audio)
    }

    public nonisolated static func projectorSupportedInputModalities(at url: URL) -> Set<LLMInputModality> {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return []
        }
        let caps = url.path.withCString { cPath in
            carbocation_mtmd_get_cap_from_file_bridge(cPath)
        }
        var modalities: Set<LLMInputModality> = []
        if caps.inp_vision { modalities.insert(.image) }
        if caps.inp_audio { modalities.insert(.audio) }
        return modalities
    }

    static func loadMultimodalProjectorIfAvailable(
        mmprojURL: URL?,
        model: OpaquePointer,
        threads: Int32,
        useGPU: Bool
    ) -> MultimodalProjectorLoadResult {
        guard let mmprojURL else {
            return MultimodalProjectorLoadResult(
                context: nil,
                unsupportedDetail: nil,
                supportedInputModalities: [.text]
            )
        }
        llamaRuntimeLog.info("Loading multimodal projector at \(mmprojURL.path, privacy: .public)")
        guard FileManager.default.fileExists(atPath: mmprojURL.path) else {
            let detail = "The mmproj artifact file was not found."
            llamaRuntimeLog.info("Multimodal projector unavailable: \(detail, privacy: .public)")
            return MultimodalProjectorLoadResult(
                context: nil,
                unsupportedDetail: detail,
                supportedInputModalities: [.text]
            )
        }
        let caps = mmprojURL.path.withCString { cPath in
            carbocation_mtmd_get_cap_from_file_bridge(cPath)
        }
        guard caps.inp_vision || caps.inp_audio else {
            let detail = "mtmd_get_cap_from_file did not report image or audio input support."
            llamaRuntimeLog.info("Multimodal projector rejected: \(detail, privacy: .public)")
            return MultimodalProjectorLoadResult(
                context: nil,
                unsupportedDetail: detail,
                supportedInputModalities: [.text]
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
            llamaRuntimeLog.info("Multimodal projector rejected: \(detail, privacy: .public)")
            return MultimodalProjectorLoadResult(
                context: nil,
                unsupportedDetail: detail,
                supportedInputModalities: [.text]
            )
        }

        var supportedInputModalities: Set<LLMInputModality> = [.text]
        if carbocation_mtmd_support_vision_bridge(context) {
            supportedInputModalities.insert(.image)
        }
        if carbocation_mtmd_support_audio_bridge(context) {
            supportedInputModalities.insert(.audio)
        }

        guard supportedInputModalities.count > 1 else {
            carbocation_mtmd_free_bridge(context)
            let detail = "The initialized mtmd context does not support image or audio input."
            llamaRuntimeLog.info("Multimodal projector rejected: \(detail, privacy: .public)")
            return MultimodalProjectorLoadResult(
                context: nil,
                unsupportedDetail: detail,
                supportedInputModalities: [.text]
            )
        }

        return MultimodalProjectorLoadResult(
            context: context,
            unsupportedDetail: nil,
            supportedInputModalities: supportedInputModalities
        )
    }

    public func preflight(
        messages: [LLMChatMessage],
        options: GenerationOptions
    ) async throws -> LLMGenerationPreflight {
        guard context != nil, let loadedInfo else {
            throw LLMEngineError.noModelLoaded
        }

        if LLMChatMessage.containsMultimodalInput(in: messages) {
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

        if LLMChatMessage.containsMultimodalInput(in: messages) {
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
        let mtmdContext = try requireMultimodalProjector()
        let marker = String(cString: carbocation_mtmd_default_marker_bridge())
        let formattedMessages = try publicChatTemplateMessages(from: messages, mediaMarker: marker)
        let supportedInputModalities = loadedInfo?.supportedInputModalities ?? [.text]
        if let unsupported = formattedMessages.media.first(where: {
            !supportedInputModalities.contains($0.payload.inputModality)
        }) {
            throw LLMEngineError.unsupportedInputModality(
                unsupported.payload.inputModality,
                location: unsupported.location
            )
        }
        let promptFormatting = try applyChatTemplate(
            messages: formattedMessages.messages,
            tools: [],
            options: options
        )
        let prefill = try makeMultimodalPrefill(
            prompt: promptFormatting.text,
            media: formattedMessages.media,
            mtmdContext: mtmdContext
        )
        return (promptFormatting, prefill)
    }

    func requireMultimodalProjector() throws -> UnsafeMutableRawPointer {
        if let mtmdContext,
           loadedInfo?.supportedInputModalities.contains(where: { $0 != .text }) == true {
            return mtmdContext
        }
        guard let mmprojURL = loadedDescriptor?.mmprojURL,
              FileManager.default.fileExists(atPath: mmprojURL.path)
        else {
            throw LLMEngineError.multimodalProjectorMissing
        }
        throw LLMEngineError.multimodalProjectorUnsupported(
            visionProjectorUnsupportedDetail ?? "The loaded mmproj artifact does not support image or audio input."
        )
    }

    func publicChatTemplateMessages(
        from messages: [LLMChatMessage],
        mediaMarker: String?
    ) throws -> PublicMessageFormatting {
        var chatMessages: [ChatTemplateMessage] = []
        var media: [(location: LLMContentLocation, payload: LlamaMediaPayload)] = []

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
                    media.append((location, .image(image)))
                    renderedContent += mediaMarker
                case .audio(let audioInput):
                    let location = LLMContentLocation(messageIndex: messageIndex, partIndex: partIndex)
                    guard message.role == .user else {
                        throw LLMEngineError.unsupportedAudioPlacement(location: location)
                    }
                    guard let mediaMarker else {
                        throw LLMEngineError.unsupportedInputModality(.audio, location: location)
                    }
                    if case .encoded = audioInput {
                        _ = try audioInput.encodedFormat(location: location)
                    }
                    media.append((location, .audio(audioInput)))
                    renderedContent += mediaMarker
                }
            }
            chatMessages.append(ChatTemplateMessage(role: message.role.rawValue, content: renderedContent))
        }

        return PublicMessageFormatting(messages: chatMessages, media: media)
    }

    func makeMultimodalPrefill(
        prompt: String,
        media: [(location: LLMContentLocation, payload: LlamaMediaPayload)],
        mtmdContext: UnsafeMutableRawPointer
    ) throws -> LlamaPreparedMultimodalPrompt {
        guard !media.isEmpty else {
            throw LLMEngineError.imageTokenizationFailed("No media payloads were supplied.")
        }

        var bitmaps: [UnsafeMutableRawPointer] = []
        bitmaps.reserveCapacity(media.count)
        defer {
            for bitmap in bitmaps {
                carbocation_mtmd_bitmap_free_bridge(bitmap)
            }
        }

        for item in media {
            let bitmap: UnsafeMutableRawPointer?
            switch item.payload {
            case .image(let image):
                guard image.width <= Int(UInt32.max),
                      image.height <= Int(UInt32.max) else {
                    throw LLMEngineError.invalidImageData(
                        "Image dimensions exceed mtmd bitmap limits.",
                        location: item.location
                    )
                }
                bitmap = image.data.withUnsafeBytes { rawBuffer -> UnsafeMutableRawPointer? in
                    guard let baseAddress = rawBuffer.bindMemory(to: UInt8.self).baseAddress else {
                        return nil
                    }
                    return carbocation_mtmd_bitmap_init_bridge(
                        UInt32(image.width),
                        UInt32(image.height),
                        baseAddress
                    )
                }
                guard bitmap != nil else {
                    throw LLMEngineError.imageTokenizationFailed(
                        "mtmd_bitmap_init returned null.",
                        location: item.location
                    )
                }
            case .audio(let audio):
                bitmap = try makeAudioBitmap(
                    audio,
                    mtmdContext: mtmdContext,
                    location: item.location
                )
            }
            guard let bitmap else {
                throw LLMEngineError.invalidImageData(
                    "Failed to create media bitmap.",
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
                detail = "The rendered prompt media-marker count did not match the media count."
            case 2:
                detail = "Media preprocessing failed."
            default:
                detail = "mtmd_tokenize returned \(tokenizationResult)."
            }
            throw mediaTokenizationFailed(detail, media: media)
        }

        let tokenCount = Int(carbocation_mtmd_helper_get_n_tokens_bridge(chunks))
        let positionCount = Int(carbocation_mtmd_helper_get_n_pos_bridge(chunks))
        guard tokenCount > 0, positionCount > 0 else {
            carbocation_mtmd_input_chunks_free_bridge(chunks)
            throw mediaTokenizationFailed("mtmd produced empty prompt chunks.", media: media)
        }

        return LlamaPreparedMultimodalPrompt(
            chunks: chunks,
            promptTokenCount: tokenCount,
            promptPositionCount: positionCount
        )
    }

    func mediaTokenizationFailed(
        _ detail: String,
        media: [(location: LLMContentLocation, payload: LlamaMediaPayload)]
    ) -> LLMEngineError {
        let hasAudio = media.contains { $0.payload.inputModality == .audio }
        let hasImage = media.contains { $0.payload.inputModality == .image }
        if hasAudio, !hasImage {
            return .audioTokenizationFailed(detail, location: media.first?.location)
        }
        return .imageTokenizationFailed(detail, location: media.first?.location)
    }

    func makeAudioBitmap(
        _ audio: LLMAudioInput,
        mtmdContext: UnsafeMutableRawPointer,
        location: LLMContentLocation
    ) throws -> UnsafeMutableRawPointer {
        let sampleRate = Int(carbocation_mtmd_get_audio_sample_rate_bridge(mtmdContext))
        guard sampleRate > 0 else {
            throw LLMEngineError.unsupportedInputModality(.audio, location: location)
        }

        switch audio {
        case .encoded(let data, let mimeType):
            _ = try LLMAudioInput.encodedFormat(data: data, mimeType: mimeType, location: location)
            guard let bitmap = data.withUnsafeBytes({ rawBuffer -> UnsafeMutableRawPointer? in
                guard let baseAddress = rawBuffer.bindMemory(to: UInt8.self).baseAddress else {
                    return nil
                }
                return carbocation_mtmd_helper_bitmap_init_from_buf_bridge(
                    mtmdContext,
                    baseAddress,
                    rawBuffer.count
                )
            }) else {
                throw LLMEngineError.audioTokenizationFailed(
                    "mtmd_helper_bitmap_init_from_buf returned null.",
                    location: location
                )
            }
            guard carbocation_mtmd_bitmap_is_audio_bridge(bitmap) else {
                carbocation_mtmd_bitmap_free_bridge(bitmap)
                throw LLMEngineError.unsupportedAudioFormat(
                    "encoded bytes did not decode as audio",
                    location: location
                )
            }
            let sampleCount = Int(carbocation_mtmd_bitmap_get_nx_bridge(bitmap))
            let duration = TimeInterval(sampleCount) / TimeInterval(sampleRate)
            guard duration <= LLMAudioInput.maximumDuration else {
                carbocation_mtmd_bitmap_free_bridge(bitmap)
                throw LLMEngineError.audioDurationExceeded(
                    LLMAudioDurationLimit(
                        maxSeconds: LLMAudioInput.maximumDuration,
                        actualSeconds: duration
                    ),
                    location: location
                )
            }
            return bitmap

        case .pcmFloat32Mono:
            let samples = try audio.validatedPCMFloat32Mono(
                expectedSampleRate: sampleRate,
                location: location
            )
            guard samples.count <= Int(UInt32.max) else {
                throw LLMEngineError.invalidAudioData(
                    "PCM sample count exceeds mtmd audio limits.",
                    location: location
                )
            }
            guard let bitmap = samples.withUnsafeBufferPointer({ buffer -> UnsafeMutableRawPointer? in
                guard let baseAddress = buffer.baseAddress else {
                    return nil
                }
                return carbocation_mtmd_bitmap_init_from_audio_bridge(
                    buffer.count,
                    baseAddress
                )
            }) else {
                throw LLMEngineError.audioTokenizationFailed(
                    "mtmd_bitmap_init_from_audio returned null.",
                    location: location
                )
            }
            return bitmap
        }
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
