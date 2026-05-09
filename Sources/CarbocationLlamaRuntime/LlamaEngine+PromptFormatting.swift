import CarbocationLocalLLM
import Foundation
import llama

extension LlamaEngine {
    private enum KnownTemplateFamily {
        case gemmaLegacy
        case gemma4
        case chatML

        var mode: LLMChatTemplateMode {
            switch self {
            case .gemmaLegacy, .gemma4:
                return .gemmaFallback
            case .chatML:
                return .chatMLFallback
            }
        }
    }

    func applyChatTemplate(
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

            if let fallback = try? Self.fallbackPrompt(
                system: system,
                user: user,
                embeddedTemplate: chatTemplate,
                descriptor: loadedDescriptor
            ) {
                Self.logChatTemplateSelection(
                    mode: fallback.mode,
                    descriptor: loadedDescriptor,
                    hasEmbeddedTemplate: true,
                    formatter: "family-fallback"
                )
                return PromptFormattingResult(
                    text: fallback.text,
                    mode: fallback.mode,
                    outputProfile: outputSanitizationProfile.isEmpty
                        ? fallback.outputProfile
                        : outputSanitizationProfile
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

        if probes.contains(where: { $0.contains("gemma-4") || $0.contains("gemma4") }) {
            return .gemma4
        }
        if probes.contains(where: { $0.contains("gemma") }) {
            return .gemmaLegacy
        }
        if probes.contains(where: { $0.contains("qwen") || $0.contains("chatml") }) {
            return .chatML
        }
        return nil
    }

    private static func templateFamily(from template: String) -> KnownTemplateFamily? {
        let lowered = template.lowercased()
        if lowered.contains("<|turn>") && lowered.contains("<turn|>") {
            return .gemma4
        }
        if lowered.contains("start_of_turn") {
            return .gemmaLegacy
        }
        if lowered.contains("im_start") {
            return .chatML
        }
        return nil
    }

    static func prepareChatTemplate(_ template: String?) -> PreparedChatTemplate? {
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

    static func logOutputSanitizationProfile(
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
        case .gemmaLegacy:
            return "<start_of_turn>user\n\(system)\n\n\(user)<end_of_turn>\n<start_of_turn>think\n<end_of_turn>\n<start_of_turn>model\n"
        case .gemma4:
            return "<|turn>system\n\(system.trimmingCharacters(in: .whitespacesAndNewlines))<turn|>\n<|turn>user\n\(user.trimmingCharacters(in: .whitespacesAndNewlines))<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
        case .chatML:
            return "<|im_start|>system\n\(system)<|im_end|>\n<|im_start|>user\n\(user)<|im_end|>\n<|im_start|>assistant\n"
        }
    }

    private static func fallbackOutputProfile(for family: KnownTemplateFamily) -> OutputSanitizationProfile {
        switch family {
        case .gemmaLegacy:
            return OutputSanitizationProfile.derived(fromChatTemplate: "<start_of_turn><end_of_turn>")
        case .gemma4:
            return OutputSanitizationProfile.derived(fromChatTemplate: "<|turn><turn|><|channel>thought<channel|>")
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

}
