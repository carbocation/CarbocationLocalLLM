import CarbocationLocalLLM
import Foundation
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
        XCTAssertEqual(calls[0].executionID, "call_7")
        XCTAssertEqual(calls[0].rawID, "call_7")
        XCTAssertEqual(calls[0].name, "calculate")
        XCTAssertEqual(calls[0].arguments.string(forKey: "operation"), "add")
        XCTAssertEqual(calls[0].arguments.array(forKey: "operands")?.count, 2)
    }

    func testToolCallParserTreatsEmptyIDsAsMissing() {
        let text = """
        {"tool_calls":[
            {"id":"","name":"lookup","arguments":{"query":"first"}},
            {"id":"   ","name":"lookup","arguments":{"query":"second"}}
        ]}
        """

        let calls = LLMToolCallParser.parseToolCalls(in: text)

        XCTAssertEqual(calls.map(\.id), ["call_1", "call_2"])
        XCTAssertEqual(calls.map(\.executionID), ["call_1", "call_2"])
        XCTAssertEqual(calls.map(\.rawID), [String?](repeating: nil, count: 2))
    }

    func testToolCallParserReadsGemmaStyleNativeCalls() {
        let text = #"<|tool_call>call:calculate{operands:[17.5,23],operation:"multiply"}<tool_call|>"#

        let calls = LLMToolCallParser.parseToolCalls(in: text)

        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls[0].id, "call_1")
        XCTAssertEqual(calls[0].executionID, "call_1")
        XCTAssertNil(calls[0].rawID)
        XCTAssertEqual(calls[0].name, "calculate")
        XCTAssertEqual(calls[0].arguments.string(forKey: "operation"), "multiply")
        XCTAssertEqual(calls[0].arguments.array(forKey: "operands") ?? [], [.number(17.5), .number(23)])
    }

    func testToolCallParserReadsObservedGemma4SearchCall() {
        let text = #"<|tool_call>call:bing_search{queries:["is Ted Turner still alive"]}<tool_call|>"#

        let calls = LLMToolCallParser.parseToolCalls(in: text)

        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls[0].name, "bing_search")
        XCTAssertEqual(calls[0].arguments.array(forKey: "queries"), ["is Ted Turner still alive"])
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

    func testToolCallSerializesExecutionAndRawIDs() throws {
        let call = LLMToolCall(
            executionID: "call_2",
            rawID: "call_1",
            name: "lookup",
            arguments: ["query": "swift"],
            triggerPhase: .thinking
        )

        let data = try JSONEncoder().encode(call)
        let value = try LLMJSONValue(jsonString: String(decoding: data, as: UTF8.self))
        let decoded = try JSONDecoder().decode(LLMToolCall.self, from: data)

        XCTAssertEqual(value.string(forKey: "executionID"), "call_2")
        XCTAssertEqual(value.string(forKey: "rawID"), "call_1")
        XCTAssertEqual(value.string(forKey: "triggerPhase"), "thinking")
        XCTAssertNil(value.string(forKey: "id"))
        XCTAssertEqual(decoded.id, "call_2")
        XCTAssertEqual(decoded.executionID, "call_2")
        XCTAssertEqual(decoded.rawID, "call_1")
        XCTAssertEqual(decoded.name, "lookup")
        XCTAssertEqual(decoded.arguments.string(forKey: "query"), "swift")
        XCTAssertEqual(decoded.triggerPhase, .thinking)
    }

    func testToolCallDecodesLegacyIDAsExecutionID() throws {
        let data = #"{"id":"call_legacy","name":"lookup","arguments":{"query":"swift"}}"#.data(using: .utf8)!

        let decoded = try JSONDecoder().decode(LLMToolCall.self, from: data)

        XCTAssertEqual(decoded.id, "call_legacy")
        XCTAssertEqual(decoded.executionID, "call_legacy")
        XCTAssertNil(decoded.rawID)
        XCTAssertNil(decoded.triggerPhase)
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

    func testToolGenerationRequestUsesSingleOptionsSurface() {
        let options = GenerationOptions(maxOutputTokens: 128, enableThinking: true)
        let request = LLMToolGenerationRequest(prompt: "test", options: options)

        XCTAssertEqual(request.options, options)
        XCTAssertEqual(request.maxToolRounds, 4)
    }

    func testEnabledToolsAreUnsupportedByDefaultEngineImplementation() async throws {
        let engine = ScriptedEngine(responses: [])
        let tool = LLMTool(
            definition: LLMToolDefinition(name: "noop", description: "No-op test tool.")
        ) { _ in
            ["ok": true]
        }

        do {
            _ = try await engine.generateWithTools(LLMToolGenerationRequest(
                prompt: "Use a tool.",
                tools: [tool]
            ))
            XCTFail("Expected default tool generation to be unsupported.")
        } catch let error as LLMToolError {
            guard case .unsupportedToolMode = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
    }

    func testToolCallIDAllocatorAssignsUniqueFallbackIDsAcrossRounds() {
        var allocator = LLMToolCallIDAllocator()
        let first = allocator.materialize(LLMToolCallParser.parseToolCalls(
            in: #"{"tool_calls":[{"name":"lookup","arguments":{"query":"first"}}]}"#
        ))
        let second = allocator.materialize(LLMToolCallParser.parseToolCalls(
            in: #"{"tool_calls":[{"name":"lookup","arguments":{"query":"second"}}]}"#
        ))

        XCTAssertEqual((first + second).map(\.executionID), ["call_1", "call_2"])
        XCTAssertEqual((first + second).map(\.rawID), [String?](repeating: nil, count: 2))
    }

    func testToolCallIDAllocatorAvoidsRawIDCollisions() {
        var allocator = LLMToolCallIDAllocator()
        let mixed = allocator.materialize(LLMToolCallParser.parseToolCalls(
            in: #"{"tool_calls":[{"name":"lookup","arguments":{"query":"fallback"}},{"id":"call_1","name":"lookup","arguments":{"query":"raw"}}]}"#
        ))
        let repeated = allocator.materialize(LLMToolCallParser.parseToolCalls(
            in: #"{"tool_calls":[{"id":"call_1","name":"lookup","arguments":{"query":"again"}}]}"#
        ))

        XCTAssertEqual(mixed.map(\.executionID), ["call_2", "call_1"])
        XCTAssertEqual(mixed.map(\.rawID), [nil, "call_1"])
        XCTAssertEqual(repeated.map(\.executionID), ["call_3"])
        XCTAssertEqual(repeated.map(\.rawID), ["call_1"])
    }

    func testToolCallIDAllocatorPreservesTriggerPhase() {
        var allocator = LLMToolCallIDAllocator()
        let materialized = allocator.materialize([
            LLMToolCall(
                executionID: "call_1",
                name: "lookup",
                arguments: ["query": "swift"],
                triggerPhase: .final
            )
        ])

        XCTAssertEqual(materialized.first?.executionID, "call_1")
        XCTAssertEqual(materialized.first?.triggerPhase, .final)
    }

    func testToolRuntimeExecutesToolsAndEmitsLifecycleEvents() async throws {
        let recorder = ToolEventRecorder()
        let call = LLMToolCall(
            executionID: "call_1",
            name: "lookup",
            arguments: ["query": "swift"]
        )
        let tool = LLMTool(
            definition: LLMToolDefinition(name: "lookup", description: "Lookup test tool.")
        ) { arguments in
            ["ok": true, "query": arguments.value(forKey: "query") ?? .null]
        }

        let outputs = try await LLMToolRuntime.execute(
            calls: [call],
            toolsByName: ["lookup": tool],
            onPhaseAwareEvent: { recorder.append($0) }
        )

        XCTAssertEqual(outputs.map(\.callID), ["call_1"])
        XCTAssertEqual(outputs[0].content.string(forKey: "query"), "swift")
        XCTAssertEqual(recorder.events.startedToolCallIDs, ["call_1"])
        XCTAssertEqual(recorder.events.completedToolCallIDs, ["call_1"])
    }

    func testToolRuntimeTurnsUnknownToolsIntoErrorOutputs() async throws {
        let recorder = ToolEventRecorder()
        let call = LLMToolCall(executionID: "call_1", name: "missing", arguments: [:])

        let outputs = try await LLMToolRuntime.execute(
            calls: [call],
            toolsByName: [:],
            onPhaseAwareEvent: { recorder.append($0) }
        )

        XCTAssertEqual(outputs.count, 1)
        XCTAssertTrue(outputs[0].isError)
        XCTAssertEqual(outputs[0].content.value(forKey: "error")?.string(forKey: "code"), "unknown_tool")
        XCTAssertEqual(recorder.events.failedToolCallIDs, ["call_1"])
    }

    func testToolGenerationWithoutEnabledToolsUsesFinalAnswerEvents() async throws {
        let engine = ScriptedEngine(responses: ["Plain answer."])
        let recorder = ToolEventRecorder()

        let result = try await engine.generateWithTools(
            LLMToolGenerationRequest(
                prompt: "Answer plainly.",
                tools: []
            ),
            onPhaseAwareEvent: { event in recorder.append(event) }
        )

        XCTAssertEqual(result.finalText, "Plain answer.")
        XCTAssertEqual(recorder.events.finalAnswerDeltaText, "Plain answer.")
        XCTAssertEqual(recorder.events.aggregateAccelerationStats?.status, .active)
        XCTAssertEqual(recorder.events.aggregateAccelerationStats?.draftTokensGenerated, 3)
        XCTAssertEqual(recorder.events.aggregateAccelerationStats?.draftTokensAccepted, 2)
    }

    func testToolGenerationWithToolChoiceNoneUsesFinalAnswerEvents() async throws {
        let engine = ScriptedEngine(responses: ["No tools answer."])
        let recorder = ToolEventRecorder()
        let tool = LLMTool(
            definition: LLMToolDefinition(name: "noop", description: "No-op test tool.")
        ) { _ in
            ["ok": true]
        }

        let result = try await engine.generateWithTools(
            LLMToolGenerationRequest(
                prompt: "Answer without tools.",
                tools: [tool],
                toolChoice: .none
            ),
            onPhaseAwareEvent: { event in recorder.append(event) }
        )

        XCTAssertEqual(result.finalText, "No tools answer.")
        XCTAssertEqual(result.toolCalls.count, 0)
        XCTAssertEqual(recorder.events.finalAnswerDeltaText, "No tools answer.")
    }
}

private struct GenerationInvocation: Sendable {
    var options: GenerationOptions
    var phaseAware: Bool
}

private actor ScriptedEngine: LLMEngine {
    private var responses: [String]
    private var recordedInvocations: [GenerationInvocation] = []

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
        recordedInvocations.append(GenerationInvocation(options: options, phaseAware: false))
        onEvent(.requestSent)
        let response = nextResponse()
        if !response.isEmpty {
            onEvent(.tokenChunk(preview: response, bytesSoFar: response.utf8.count))
        }
        onEvent(.generationStats(
            promptTokens: 2,
            generatedTokens: TokenEstimator.estimate(text: response),
            stopReason: "complete",
            templateMode: .unavailable
        ))
        onEvent(.done(totalBytes: response.utf8.count, duration: 0))
        return response
    }

    func generate(
        system: String,
        prompt: String,
        options: GenerationOptions,
        onPhaseAwareEvent: @Sendable (LLMPhaseAwareStreamEvent) -> Void,
        _ phaseAwareOverload: Void = ()
    ) async throws -> String {
        recordedInvocations.append(GenerationInvocation(options: options, phaseAware: true))
        let initialPhase: LLMStreamContentPhase = options.enableThinking ? .thinking : .final
        onPhaseAwareEvent(.requestSent(phase: initialPhase))
        if options.enableThinking {
            onPhaseAwareEvent(.tokenChunk(preview: "thinking", bytesSoFar: "thinking".utf8.count, phase: .thinking))
            onPhaseAwareEvent(.phaseChanged(from: .thinking, to: .final))
        }
        let response = nextResponse()
        if !response.isEmpty {
            onPhaseAwareEvent(.finalAnswerDelta(text: response, bytesSoFar: response.utf8.count))
            onPhaseAwareEvent(.tokenChunk(preview: response, bytesSoFar: response.utf8.count, phase: .final))
        }
        onPhaseAwareEvent(.accelerationStats(LLMGenerationAccelerationStats(
            status: .active,
            accelerator: "mtp",
            maxDraftTokens: 3,
            draftCalls: 1,
            draftTokensGenerated: 3,
            draftTokensAccepted: 2
        )))
        onPhaseAwareEvent(.generationStats(
            promptTokens: 3,
            generatedTokens: TokenEstimator.estimate(text: response),
            stopReason: "complete",
            templateMode: .unavailable,
            phase: .final
        ))
        onPhaseAwareEvent(.done(totalBytes: response.utf8.count, duration: 0, phase: .final))
        return response
    }

    func invocations() -> [GenerationInvocation] {
        recordedInvocations
    }

    private func nextResponse() -> String {
        guard !responses.isEmpty else { return "" }
        return responses.removeFirst()
    }
}

private final class ToolEventRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [LLMToolPhaseAwareStreamEvent] = []

    func append(_ event: LLMToolPhaseAwareStreamEvent) {
        lock.lock()
        storage.append(event)
        lock.unlock()
    }

    var events: [LLMToolPhaseAwareStreamEvent] {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }
}

private extension Array where Element == LLMToolPhaseAwareStreamEvent {
    var finalAnswerDeltaText: String {
        reduce(into: "") { result, event in
            if case .finalAnswerEvent(.finalAnswerDelta(let text, _)) = event {
                result += text
            }
        }
    }

    func finalAnswerTextContains(_ text: String) -> Bool {
        contains { event in
            switch event {
            case .finalAnswerEvent(.finalAnswerDelta(let delta, _)):
                return delta.contains(text)
            case .finalAnswerEvent(.finalAnswerSnapshot(let snapshot, _, _)):
                return snapshot.contains(text)
            default:
                return false
            }
        }
    }

    func containsAggregateStats(stopReason: String) -> Bool {
        contains { event in
            guard case .aggregateGenerationStats(let promptTokens, let generatedTokens, let actualStopReason) = event else {
                return false
            }
            return promptTokens > 0 && generatedTokens > 0 && actualStopReason == stopReason
        }
    }

    var aggregateAccelerationStats: LLMGenerationAccelerationStats? {
        compactMap { event in
            if case .aggregateAccelerationStats(let stats) = event {
                return stats
            }
            return nil
        }.last
    }

    var startedToolCallIDs: [String] {
        compactMap { event in
            if case .toolCallStarted(let call) = event {
                return call.id
            }
            return nil
        }
    }

    var failedToolCallIDs: [String] {
        compactMap { event in
            if case .toolCallFailed(let output) = event {
                return output.callID
            }
            return nil
        }
    }

    var completedToolCallIDs: [String] {
        compactMap { event in
            if case .toolCallCompleted(let output) = event {
                return output.callID
            }
            return nil
        }
    }
}
