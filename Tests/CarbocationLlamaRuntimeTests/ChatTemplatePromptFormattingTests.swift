import CarbocationLocalLLM
import XCTest
@testable import CarbocationLlamaRuntime

final class ChatTemplatePromptFormattingTests: XCTestCase {
    func testQwenTemplateAcceptsWhitespaceThatItTrimsFromUserContent() throws {
        let template = try String(contentsOf: Self.qwen35TemplateURL, encoding: .utf8)

        let prompt = try ChatTemplatePromptFormatter.format(
            template: template,
            system: "System",
            user: "  padded user  ",
            bosToken: "<bos>",
            eosToken: "<eos>",
            enableThinking: false
        )

        XCTAssertTrue(prompt.contains("<|im_start|>user\npadded user<|im_end|>"))
        XCTAssertFalse(prompt.contains("  padded user  "))
    }

    func testFormatterRejectsStructurallyEmptyPrompt() throws {
        let formatter = try ChatTemplatePromptFormatter(
            template: "{% for message in messages %}{% set ignored = message.content %}{% endfor %}"
        )

        XCTAssertThrowsError(try formatter.format(
            system: "System",
            user: "User",
            bosToken: "",
            eosToken: ""
        )) { error in
            XCTAssertEqual(error as? ChatTemplatePromptFormatter.Error, .emptyPrompt)
        }
    }

    func testCompiledTemplateRenderFailureDoesNotFallBackToLegacyRenderer() async throws {
        let engine = try await makeVocabularyOnlyEngine()
        await engine.setChatTemplateForTesting("""
        {%- if enable_thinking -%}
        {{- raise_exception('semantic failure') -}}
        {%- endif -%}
        <|im_start|>system
        {{ messages[0].content }}<|im_end|>
        <|im_start|>user
        {{ messages[1].content }}<|im_end|>
        <|im_start|>assistant
        """)

        do {
            _ = try await engine.applyChatTemplate(
                system: "System",
                user: "User",
                options: GenerationOptions(enableThinking: true)
            )
            XCTFail("Expected the Swift Jinja semantic error to be reported.")
        } catch LLMEngineError.chatTemplateUnavailable(let detail) {
            XCTAssertTrue(detail.contains("Swift Jinja template failed while rendering"))
        } catch {
            XCTFail("Expected chatTemplateUnavailable, got \(error).")
        }
    }

    func testUnpreparedThinkingTemplateDoesNotUseSemanticallyLossyFallback() async throws {
        let engine = try await makeVocabularyOnlyEngine()
        await engine.setChatTemplateForTesting("""
        enable_thinking
        {% unsupported_tag %}
        <|im_start|>user
        """)

        do {
            _ = try await engine.applyChatTemplate(
                system: "System",
                user: "User",
                options: GenerationOptions()
            )
            XCTFail("Expected reasoning-control semantic loss to be rejected.")
        } catch LLMEngineError.chatTemplateUnavailable(let detail) {
            XCTAssertTrue(detail.contains("legacy template renderer cannot preserve"))
        } catch {
            XCTFail("Expected chatTemplateUnavailable, got \(error).")
        }
    }

    func testUnpreparedTemplateAliasCanStillUseLegacyRenderer() async throws {
        let engine = try await makeVocabularyOnlyEngine()
        await engine.setChatTemplateForTesting("chatml")

        let result = try await engine.applyChatTemplate(
            system: "System",
            user: "User",
            options: GenerationOptions()
        )

        XCTAssertEqual(result.mode, .embedded)
        XCTAssertTrue(result.text.contains("<|im_start|>system"))
        XCTAssertTrue(result.text.contains("<|im_start|>assistant"))
    }

    private func makeVocabularyOnlyEngine() async throws -> LlamaEngine {
        let engine = LlamaEngine()
        try await engine.loadVocabularyOnlyForTesting(
            modelAt: Self.qwen35VocabModelURL,
            displayName: "Prompt Formatting Test Vocab",
            requestedContext: 2_048
        )
        return engine
    }

    private static var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private static var qwen35TemplateURL: URL {
        packageRoot
            .appendingPathComponent("Vendor/llama.cpp/models/templates/Qwen3.5-4B.jinja")
    }

    private static var qwen35VocabModelURL: URL {
        packageRoot
            .appendingPathComponent("Vendor/llama.cpp/models/ggml-vocab-qwen35.gguf")
    }
}
