import CarbocationLocalLLM
import Foundation
import XCTest
@testable import CarbocationLlamaRuntime

final class NativeToolProtocolCapabilityTests: XCTestCase {
    func testCapabilityDetectionUsesTemplateGrammarWithoutModelIdentity() async throws {
        let arbitraryTemplate = """
        {%- if tools -%}
        <tools>{{ tools | tojson }}</tools>
        {%- endif -%}
        {%- for message in messages -%}
        {%- if message.tool_calls -%}
        {%- for tool_call in message.tool_calls -%}
        <tool_call><function={{ tool_call.function.name }}>
        {%- for argument_name, argument_value in tool_call.function.arguments | items -%}
        <parameter={{ argument_name }}>{{ argument_value }}</parameter>
        {%- endfor -%}
        </function></tool_call>
        {%- endfor -%}
        {%- elif message.role == "tool" -%}
        <tool_response>{{ message.content }}</tool_response>
        {%- else -%}
        {{ message.content }}
        {%- endif -%}
        {%- endfor -%}
        """
        _ = try ChatTemplatePromptFormatter(template: arbitraryTemplate)
        let engine = LlamaEngine()

        await engine.setChatTemplateForTesting(arbitraryTemplate)

        let protocolKind = await engine.nativeToolTemplateProtocol
        XCTAssertEqual(protocolKind, .functionParameterXML)
    }

    func testTemplateWithOnlyGenericToolWordsDoesNotClaimNativeProtocol() async {
        let engine = LlamaEngine()
        await engine.setChatTemplateForTesting("""
        {%- for message in messages -%}{{ message.content }}{%- endfor -%}
        Tools may be returned as ordinary JSON.
        """)

        let protocolKind = await engine.nativeToolTemplateProtocol
        XCTAssertNil(protocolKind)
    }

    func testFunctionParameterProtocolUsesStructuredTemplateAndHistory() async throws {
        let engine = LlamaEngine()
        try await engine.loadVocabularyOnlyForTesting(
            modelAt: Self.vocabularyModelURL,
            displayName: "Native protocol test vocabulary",
            requestedContext: 2_048
        )
        let template = try String(contentsOf: Self.functionParameterTemplateURL, encoding: .utf8)
        await engine.setChatTemplateForTesting(template)

        let script = NativeProtocolSegmentScript()
        await engine.setToolAwareGenerationSegmentOverrideForTesting { input in
            script.next(input)
        }
        let lookup = LLMTool(
            definition: LLMToolDefinition(
                name: "lookup",
                description: "Look up current information.",
                parameters: [
                    "type": "object",
                    "properties": [
                        "query": ["type": "string"]
                    ],
                    "required": ["query"]
                ]
            )
        ) { arguments in
            .object([
                "answer": .string("result for \(arguments.string(forKey: "query") ?? "")")
            ])
        }

        let result = try await engine.generateWithTools(LLMToolGenerationRequest(
            system: "Follow the system policy.",
            prompt: "Find current Swift information.",
            tools: [lookup],
            maxToolRounds: 1
        ))
        let prompts = script.prompts

        XCTAssertEqual(result.toolCalls.map(\.name), ["lookup"])
        XCTAssertEqual(result.finalText, "Final answer.")
        XCTAssertEqual(prompts.count, 2)
        XCTAssertTrue(prompts[0].contains("# Tools"))
        XCTAssertTrue(prompts[0].contains("\"name\":\"lookup\""))
        XCTAssertFalse(prompts[0].contains("Available tools:"))
        XCTAssertTrue(prompts[1].contains("<tool_call>\n<function=lookup>"))
        XCTAssertTrue(prompts[1].contains("<parameter=query>\nSwift"))
        XCTAssertTrue(prompts[1].contains("<tool_response>"))
        XCTAssertTrue(prompts[1].contains(#"{"answer":"result for Swift"}"#))
        XCTAssertTrue(prompts[1].contains("<think>\nA lookup is needed.\n</think>"))
        XCTAssertFalse(prompts[1].contains("Use the tool outputs above to continue"))
    }

    func testFunctionParameterStreamingInterceptionMatchesWholeOutputAtEverySplit() {
        let text = "🙂 I will check. " + Self.parallelFunctionParameterCalls + " trailing"
        let expected = LlamaToolStreamInterpreter.completedToolCall(in: text)
        XCTAssertEqual(expected?.calls.map(\.name), ["lookup", "calculate"])

        for split in Array(text.indices) + [text.endIndex] {
            var parser = IncrementalToolStreamParser()
            let first = String(text[..<split])
            let second = String(text[split...])
            var fullText = first
            _ = parser.append(first, fullText: fullText)
            fullText += second
            let snapshot = parser.append(second, fullText: fullText)

            XCTAssertEqual(snapshot.completed?.calls.map(\.name), expected?.calls.map(\.name))
            XCTAssertEqual(snapshot.completed?.utf8Range.lowerBound, "🙂 I will check. ".utf8.count)
        }
    }

    func testFunctionParameterEnvelopeRemainsPendingUntilSiblingDecision() {
        var parser = IncrementalToolStreamParser()
        let envelope = Self.singleFunctionParameterCall

        let pending = parser.append(envelope, fullText: envelope)
        XCTAssertNil(pending.completed)
        XCTAssertEqual(pending.pendingNative?.calls.map(\.name), ["lookup"])

        let completeText = envelope + " done"
        let completed = parser.append(" done", fullText: completeText)
        XCTAssertEqual(completed.completed?.calls.map(\.name), ["lookup"])
    }

    func testOtherProtocolEndMarkerInsideParameterDoesNotCompleteXMLCall() throws {
        let prefix = "I will check. "
        let call = """
        <tool_call>
        <function=lookup>
        <parameter=query>literal <tool_call|> marker</parameter>
        </function>
        </tool_call>
        """
        let falseEnd = try XCTUnwrap(call.range(of: "<tool_call|>")?.upperBound)
        var partialParser = IncrementalToolStreamParser()
        let partial = prefix + String(call[..<falseEnd])
        let partialSnapshot = partialParser.append(partial, fullText: partial)
        XCTAssertNil(partialSnapshot.completed)
        XCTAssertEqual(partialSnapshot.visiblePrefixUTF8Count, prefix.utf8.count)

        let text = prefix + call + " trailing"
        for split in Array(text.indices) + [text.endIndex] {
            var parser = IncrementalToolStreamParser()
            let first = String(text[..<split])
            let second = String(text[split...])
            _ = parser.append(first, fullText: first)
            let snapshot = parser.append(second, fullText: text)
            XCTAssertEqual(snapshot.completed?.calls.map(\.name), ["lookup"])
            XCTAssertEqual(snapshot.completed?.utf8Range.lowerBound, prefix.utf8.count)
        }
    }

    func testDirectToolCallArrayIsCapturedAndInterceptedIncrementally() {
        let prefix = "I will check. "
        let callArray = #"[{"parameters":{"query":"Swift"},"name":"lookup"}]"#
        let partial = prefix + #"[{"parameters":{"query":"Swift"},"name":"lookup""#
        XCTAssertEqual(LlamaToolStreamInterpreter.visibleRawPrefix(in: partial), prefix)

        let text = prefix + callArray + " trailing"
        for split in Array(text.indices) + [text.endIndex] {
            var parser = IncrementalToolStreamParser()
            let first = String(text[..<split])
            let second = String(text[split...])
            _ = parser.append(first, fullText: first)
            let snapshot = parser.append(second, fullText: text)

            XCTAssertEqual(snapshot.completed?.calls.map(\.name), ["lookup"])
            XCTAssertEqual(snapshot.completed?.utf8Range.lowerBound, prefix.utf8.count)
        }
    }

    func testBalancedOutputBoundaryRecognizesFunctionParameterEnvelope() {
        let text = "reasoning " + Self.singleFunctionParameterCall + " suffix"

        let boundary = LlamaEngine.firstGenerationBoundary(
            in: text,
            stopSequences: [],
            stopAtBalancedJSON: true
        )

        XCTAssertEqual(boundary?.reason, "tool-call-complete")
        XCTAssertEqual(boundary?.text, "reasoning " + Self.singleFunctionParameterCall)
    }

    func testToolEntryPointsValidateSamplingOptionsBeforeSegmentOverride() async throws {
        let engine = LlamaEngine()
        try await engine.loadVocabularyOnlyForTesting(
            modelAt: Self.vocabularyModelURL,
            displayName: "Tool option validation vocabulary",
            requestedContext: 1_024
        )
        let script = NativeProtocolSegmentScript()
        await engine.setToolAwareGenerationSegmentOverrideForTesting { input in
            script.next(input)
        }
        let tool = LLMTool(
            definition: LLMToolDefinition(name: "lookup", description: "Lookup data.")
        ) { _ in
            .object([:])
        }
        let request = LLMToolGenerationRequest(
            prompt: "Use lookup.",
            options: GenerationOptions(temperature: .nan),
            tools: [tool]
        )

        do {
            _ = try await engine.generateWithTools(request)
            XCTFail("Expected standard tool generation to reject NaN temperature.")
        } catch let error as GenerationOptionsValidationError {
            XCTAssertEqual(error, .mustBeFinite(.temperature))
        }

        do {
            _ = try await engine.generateWithToolsPhased(request)
            XCTFail("Expected phased tool generation to reject NaN temperature.")
        } catch let error as GenerationOptionsValidationError {
            XCTAssertEqual(error, .mustBeFinite(.temperature))
        }
        XCTAssertTrue(script.prompts.isEmpty)
    }
}

private extension NativeToolProtocolCapabilityTests {
    static let singleFunctionParameterCall = """
    <tool_call>
    <function=lookup>
    <parameter=query>
    Swift
    </parameter>
    </function>
    </tool_call>
    """

    static let parallelFunctionParameterCalls = singleFunctionParameterCall + "\n" + """
    <tool_call>
    <function=calculate>
    <parameter=operands>
    [2,3]
    </parameter>
    </function>
    </tool_call>
    """

    static var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    static var functionParameterTemplateURL: URL {
        packageRoot.appendingPathComponent(
            "Vendor/llama.cpp/models/templates/Qwen3.5-4B.jinja"
        )
    }

    static var vocabularyModelURL: URL {
        packageRoot.appendingPathComponent(
            "Vendor/llama.cpp/models/ggml-vocab-gemma-4.gguf"
        )
    }
}

private final class NativeProtocolSegmentScript: @unchecked Sendable {
    private let lock = NSLock()
    private var callCount = 0
    private var promptStorage: [String] = []

    var prompts: [String] {
        lock.lock()
        defer { lock.unlock() }
        return promptStorage
    }

    func next(
        _ input: LlamaEngine.ToolAwareGenerationSegmentOverrideInput
    ) -> LlamaEngine.ToolAwareGenerationSegmentOverrideOutput {
        lock.lock()
        defer {
            callCount += 1
            lock.unlock()
        }
        promptStorage.append(input.renderedPrompt)

        if callCount == 0 {
            return LlamaEngine.ToolAwareGenerationSegmentOverrideOutput(
                reasoningContent: "A lookup is needed.",
                toolCalls: [
                    LLMToolCall(
                        executionID: "call_1",
                        name: "lookup",
                        arguments: ["query": "Swift"]
                    )
                ],
                stopReason: "tool-call-complete",
                triggerPhase: .thinking
            )
        }
        return LlamaEngine.ToolAwareGenerationSegmentOverrideOutput(
            finalText: "Final answer.",
            stopReason: "eog",
            triggerPhase: .final
        )
    }
}
