import CarbocationLocalLLM
import XCTest

final class ToolCallingTests: XCTestCase {
    func testJSONValueRoundTripsObjectsAndArrays() throws {
        let value: LLMJSONValue = [
            "name": "calculate",
            "arguments": [
                "operation": "add",
                "operands": [1, 2.5]
            ]
        ]

        let encoded = try value.jsonString()
        let decoded = try LLMJSONValue(jsonString: encoded)

        XCTAssertEqual(decoded.value(forKey: "name")?.stringValue, "calculate")
        XCTAssertEqual(decoded.value(forKey: "arguments")?.array(forKey: "operands")?.count, 2)
    }

    func testToolCallParserReadsOpenAIStyleFunctionCalls() {
        let text = """
        ```json
        {"tool_calls":[{"id":"call_7","function":{"name":"calculate","arguments":"{\\"operation\\":\\"add\\",\\"operands\\":[1,2]}"}}]}
        ```
        """

        let calls = LLMToolCallParser.parseToolCalls(in: text)

        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls[0].id, "call_7")
        XCTAssertEqual(calls[0].name, "calculate")
        XCTAssertEqual(calls[0].arguments.string(forKey: "operation"), "add")
        XCTAssertEqual(calls[0].arguments.array(forKey: "operands")?.count, 2)
    }

    func testToolCallParserReadsGemmaStyleNativeCalls() {
        let text = #"<|tool_call>call:calculate{operands:[17.5,23],operation:"multiply"}<tool_call|>"#

        let calls = LLMToolCallParser.parseToolCalls(in: text)

        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls[0].id, "call_1")
        XCTAssertEqual(calls[0].name, "calculate")
        XCTAssertEqual(calls[0].arguments.string(forKey: "operation"), "multiply")
        XCTAssertEqual(calls[0].arguments.array(forKey: "operands") ?? [], [.number(17.5), .number(23)])
    }

    func testToolCallParserReadsGemmaStyleNestedAndParallelCalls() {
        let text = #"<|tool_call>call:set_config{config:{theme:<|"|>dark<|"|>,count:3},enabled:true,items:[null,false,1.5e10]}<tool_call|><|tool_call>call:empty_args{}<tool_call|>"#

        let calls = LLMToolCallParser.parseToolCalls(in: text)

        XCTAssertEqual(calls.count, 2)
        XCTAssertEqual(calls[0].name, "set_config")
        XCTAssertEqual(calls[0].arguments.value(forKey: "config")?.string(forKey: "theme"), "dark")
        XCTAssertEqual(calls[0].arguments.value(forKey: "config")?.double(forKey: "count"), 3)
        XCTAssertEqual(calls[0].arguments.value(forKey: "enabled"), .bool(true))
        XCTAssertEqual(calls[0].arguments.array(forKey: "items") ?? [], [.null, .bool(false), .number(1.5e10)])
        XCTAssertEqual(calls[1].id, "call_2")
        XCTAssertEqual(calls[1].name, "empty_args")
        XCTAssertEqual(calls[1].arguments, .object([:]))
    }

    func testToolOutputSerializesStructuredContent() throws {
        let output = LLMToolOutput(
            callID: "call_1",
            name: "calculate",
            content: ["ok": true, "result": "3"]
        )

        let data = try JSONEncoder().encode(output)
        let decoded = try JSONDecoder().decode(LLMToolOutput.self, from: data)

        XCTAssertEqual(decoded, output)
        XCTAssertEqual(decoded.content.string(forKey: "result"), "3")
    }

    func testToolChoiceUsesStableCodableShape() throws {
        let encoded = try JSONEncoder().encode(LLMToolChoice.named("calculate"))
        let value = try LLMJSONValue(jsonString: String(decoding: encoded, as: UTF8.self))

        XCTAssertEqual(value.string(forKey: "type"), "named")
        XCTAssertEqual(value.string(forKey: "name"), "calculate")
        XCTAssertEqual(try JSONDecoder().decode(LLMToolChoice.self, from: encoded), .named("calculate"))
    }

    func testToolGenerationRejectsDuplicateToolNames() async throws {
        let engine = ScriptedEngine(responses: [])
        let tool = LLMTool(
            definition: LLMToolDefinition(name: "duplicate", description: "Duplicate test tool.")
        ) { _ in
            .null
        }

        do {
            _ = try await engine.generateWithTools(LLMToolGenerationRequest(
                prompt: "test",
                tools: [tool, tool]
            ))
            XCTFail("Expected duplicate tool validation to fail.")
        } catch let error as LLMToolError {
            XCTAssertEqual(error, .duplicateToolName("duplicate"))
        }
    }

    func testToolGenerationStopsAtMaxToolRounds() async throws {
        let toolCall = #"{"tool_calls":[{"id":"call_1","name":"noop","arguments":{"value":"x"}}]}"#
        let engine = ScriptedEngine(responses: [toolCall, toolCall])
        let tool = LLMTool(
            definition: LLMToolDefinition(name: "noop", description: "No-op test tool.")
        ) { arguments in
            ["ok": true, "echo": arguments]
        }

        let result = try await engine.generateWithTools(LLMToolGenerationRequest(
            prompt: "Use a tool.",
            tools: [tool],
            maxToolRounds: 1
        ))

        XCTAssertEqual(result.stopReason, "max-tool-rounds")
        XCTAssertEqual(result.roundsCompleted, 1)
        XCTAssertEqual(result.toolCalls.count, 2)
        XCTAssertEqual(result.toolOutputs.count, 1)
        XCTAssertEqual(result.toolOutputs[0].content.value(forKey: "echo")?.string(forKey: "value"), "x")
    }

    func testToolGenerationExecutesGemmaStyleToolCall() async throws {
        let toolCall = #"<|tool_call>call:calculate{operands:[17.5,23],operation:"multiply"}<tool_call|>"#
        let engine = ScriptedEngine(responses: [toolCall, "17.5 times 23 is 402.5."])
        let tool = LLMTool(
            definition: LLMToolDefinition(name: "calculate", description: "Math test tool.")
        ) { arguments in
            [
                "ok": true,
                "operation": arguments.value(forKey: "operation") ?? .null,
                "operands": arguments.value(forKey: "operands") ?? .null,
                "result": "402.5"
            ]
        }

        let result = try await engine.generateWithTools(LLMToolGenerationRequest(
            prompt: "Use calculate.",
            tools: [tool]
        ))

        XCTAssertEqual(result.finalText, "17.5 times 23 is 402.5.")
        XCTAssertEqual(result.roundsCompleted, 1)
        XCTAssertEqual(result.toolCalls.count, 1)
        XCTAssertEqual(result.toolCalls[0].name, "calculate")
        XCTAssertEqual(result.toolOutputs.count, 1)
        XCTAssertEqual(result.toolOutputs[0].content.string(forKey: "operation"), "multiply")
        XCTAssertEqual(result.toolOutputs[0].content.array(forKey: "operands") ?? [], [.number(17.5), .number(23)])
        XCTAssertFalse(result.finalText.contains("<|tool_call>"))
    }
}

private actor ScriptedEngine: LLMEngine {
    private var responses: [String]

    init(responses: [String]) {
        self.responses = responses
    }

    func currentModelID() async -> UUID? {
        nil
    }

    func currentContextSize() async -> Int {
        4_096
    }

    func generate(
        system: String,
        prompt: String,
        options: GenerationOptions,
        onEvent: @Sendable (LLMStreamEvent) -> Void
    ) async throws -> String {
        onEvent(.requestSent)
        guard !responses.isEmpty else { return "" }
        return responses.removeFirst()
    }
}
