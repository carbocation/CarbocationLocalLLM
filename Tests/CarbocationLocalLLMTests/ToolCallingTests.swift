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
            arguments: ["query": "swift"]
        )

        let data = try JSONEncoder().encode(call)
        let value = try LLMJSONValue(jsonString: String(decoding: data, as: UTF8.self))
        let decoded = try JSONDecoder().decode(LLMToolCall.self, from: data)

        XCTAssertEqual(value.string(forKey: "executionID"), "call_2")
        XCTAssertEqual(value.string(forKey: "rawID"), "call_1")
        XCTAssertNil(value.string(forKey: "id"))
        XCTAssertEqual(decoded.id, "call_2")
        XCTAssertEqual(decoded.executionID, "call_2")
        XCTAssertEqual(decoded.rawID, "call_1")
        XCTAssertEqual(decoded.name, "lookup")
        XCTAssertEqual(decoded.arguments.string(forKey: "query"), "swift")
    }

    func testToolCallDecodesLegacyIDAsExecutionID() throws {
        let data = #"{"id":"call_legacy","name":"lookup","arguments":{"query":"swift"}}"#.data(using: .utf8)!

        let decoded = try JSONDecoder().decode(LLMToolCall.self, from: data)

        XCTAssertEqual(decoded.id, "call_legacy")
        XCTAssertEqual(decoded.executionID, "call_legacy")
        XCTAssertNil(decoded.rawID)
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

    func testToolGenerationRequestDefaultsToFastCandidateOptions() {
        let request = LLMToolGenerationRequest(prompt: "test")

        XCTAssertEqual(request.toolCandidateOptions, .toolCandidateDefault)
        XCTAssertFalse(request.toolCandidateOptions.enableThinking)
        XCTAssertNil(request.toolCandidateOptions.thinkingBudgetTokens)
        XCTAssertEqual(request.toolCandidateOptions.maxOutputTokens, 256)
        XCTAssertTrue(request.toolCandidateOptions.stopAtBalancedJSON)
    }

    func testToolGenerationUsesFastCandidateOptionsByDefault() async throws {
        let finalOptions = GenerationOptions(
            maxOutputTokens: 1_024,
            grammar: "root ::= object",
            enableThinking: true,
            thinkingBudgetTokens: 128,
            thinkingBudgetMessage: "Final thinking budget reached."
        )
        let engine = ScriptedEngine(responses: [#"{"tool_calls":[]}"#, "Final answer."])
        let tool = LLMTool(
            definition: LLMToolDefinition(name: "noop", description: "No-op test tool.")
        ) { _ in
            ["ok": true]
        }

        _ = try await engine.generateWithTools(LLMToolGenerationRequest(
            prompt: "Usually answer directly.",
            options: finalOptions,
            tools: [tool]
        ))

        let invocations = await engine.invocations()
        XCTAssertEqual(invocations.count, 2)
        XCTAssertTrue(invocations[0].phaseAware)
        XCTAssertEqual(invocations[0].options, .toolCandidateDefault)
        XCTAssertEqual(invocations[1].options, finalOptions)
    }

    func testToolGenerationRespectsExplicitCandidateOptions() async throws {
        let finalOptions = GenerationOptions(maxOutputTokens: 1_024, enableThinking: true)
        let candidateOptions = GenerationOptions(
            temperature: 0.2,
            maxOutputTokens: 42,
            stopAtBalancedJSON: false,
            grammar: "root ::= object",
            enableThinking: true,
            thinkingBudgetTokens: 4
        )
        let engine = ScriptedEngine(responses: [#"{"tool_calls":[]}"#, "Final answer."])
        let tool = LLMTool(
            definition: LLMToolDefinition(name: "noop", description: "No-op test tool.")
        ) { _ in
            ["ok": true]
        }

        _ = try await engine.generateWithTools(LLMToolGenerationRequest(
            prompt: "Use explicit candidate options.",
            options: finalOptions,
            toolCandidateOptions: candidateOptions,
            tools: [tool]
        ))

        let invocations = await engine.invocations()
        XCTAssertEqual(invocations.count, 2)
        XCTAssertEqual(invocations[0].options, candidateOptions)
        XCTAssertEqual(invocations[0].options.grammar, "root ::= object")
        XCTAssertEqual(invocations[1].options, finalOptions)
    }

    func testToolGenerationStopsAtMaxToolRounds() async throws {
        let toolCall = #"{"tool_calls":[{"id":"call_1","name":"noop","arguments":{"value":"x"}}]}"#
        let engine = ScriptedEngine(responses: [toolCall, toolCall, "I cannot call more tools, so I will answer with the available result."])
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
        XCTAssertEqual(result.finalText, "I cannot call more tools, so I will answer with the available result.")
        XCTAssertEqual(result.roundsCompleted, 1)
        XCTAssertEqual(result.toolCalls.count, 2)
        XCTAssertEqual(result.toolOutputs.count, 1)
        XCTAssertEqual(result.toolOutputs[0].content.value(forKey: "echo")?.string(forKey: "value"), "x")
    }

    func testToolGenerationExecutesGemmaStyleToolCall() async throws {
        let toolCall = #"<|tool_call>call:calculate{operands:[17.5,23],operation:"multiply"}<tool_call|>"#
        let engine = ScriptedEngine(responses: [toolCall, #"{"tool_calls":[]}"#, "17.5 times 23 is 402.5."])
        let recorder = ToolEventRecorder()
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
        ), onPhaseAwareEvent: { event in recorder.append(event) })

        XCTAssertEqual(result.finalText, "17.5 times 23 is 402.5.")
        XCTAssertEqual(result.roundsCompleted, 1)
        XCTAssertEqual(result.toolCalls.count, 1)
        XCTAssertEqual(result.toolCalls[0].name, "calculate")
        XCTAssertEqual(result.toolOutputs.count, 1)
        XCTAssertEqual(result.toolOutputs[0].content.string(forKey: "operation"), "multiply")
        XCTAssertEqual(result.toolOutputs[0].content.array(forKey: "operands") ?? [], [.number(17.5), .number(23)])
        XCTAssertFalse(result.finalText.contains("<|tool_call>"))

        let events = recorder.events
        XCTAssertTrue(events.containsToolCandidateText("<|tool_call>"))
        XCTAssertEqual(events.finalAnswerDeltaText, "17.5 times 23 is 402.5.")
        XCTAssertFalse(events.finalAnswerTextContains("<|tool_call>"))
        XCTAssertTrue(events.containsAggregateStats(stopReason: "complete"))
    }

    func testToolGenerationAssignsUniqueFallbackCallIDsAcrossRounds() async throws {
        let firstToolCall = #"{"tool_calls":[{"name":"lookup","arguments":{"query":"first"}}]}"#
        let secondToolCall = #"{"tool_calls":[{"name":"lookup","arguments":{"query":"second"}}]}"#
        let engine = ScriptedEngine(responses: [firstToolCall, secondToolCall, #"{"tool_calls":[]}"#, "Done."])
        let recorder = ToolEventRecorder()
        let tool = LLMTool(
            definition: LLMToolDefinition(name: "lookup", description: "Lookup test tool.")
        ) { arguments in
            ["ok": true, "query": arguments.value(forKey: "query") ?? .null]
        }

        let result = try await engine.generateWithTools(
            LLMToolGenerationRequest(
                prompt: "Use lookup twice.",
                tools: [tool]
            ),
            onPhaseAwareEvent: { event in recorder.append(event) }
        )

        XCTAssertEqual(result.finalText, "Done.")
        XCTAssertEqual(result.toolCalls.map(\.id), ["call_1", "call_2"])
        XCTAssertEqual(result.toolCalls.map(\.executionID), ["call_1", "call_2"])
        XCTAssertEqual(result.toolCalls.map(\.rawID), [String?](repeating: nil, count: 2))
        XCTAssertEqual(result.toolOutputs.map(\.callID), ["call_1", "call_2"])
        XCTAssertEqual(result.toolOutputs.map { $0.content.string(forKey: "query") }, ["first", "second"])
        XCTAssertEqual(recorder.events.startedToolCallIDs, ["call_1", "call_2"])
        XCTAssertEqual(recorder.events.completedToolCallIDs, ["call_1", "call_2"])
    }

    func testToolGenerationFallbackCallIDsDoNotCollideWithRawIDs() async throws {
        let mixedToolCalls = #"{"tool_calls":[{"name":"lookup","arguments":{"query":"fallback"}},{"id":"call_1","name":"lookup","arguments":{"query":"raw"}}]}"#
        let engine = ScriptedEngine(responses: [mixedToolCalls, #"{"tool_calls":[]}"#, "Done."])
        let tool = LLMTool(
            definition: LLMToolDefinition(name: "lookup", description: "Lookup test tool.")
        ) { arguments in
            ["ok": true, "query": arguments.value(forKey: "query") ?? .null]
        }

        let result = try await engine.generateWithTools(
            LLMToolGenerationRequest(
                prompt: "Use lookup.",
                tools: [tool]
            )
        )

        XCTAssertEqual(result.toolCalls.map(\.id), ["call_2", "call_1"])
        XCTAssertEqual(result.toolCalls.map(\.executionID), ["call_2", "call_1"])
        XCTAssertEqual(result.toolCalls.map(\.rawID), [nil, "call_1"])
        XCTAssertEqual(result.toolOutputs.map(\.callID), ["call_2", "call_1"])
    }

    func testToolGenerationPreservesCollidingRawIDsSeparatelyFromExecutionIDs() async throws {
        let repeatedRawIDCalls = #"{"tool_calls":[{"id":"call_1","name":"lookup","arguments":{"query":"first"}},{"id":"call_1","name":"lookup","arguments":{"query":"second"}}]}"#
        let engine = ScriptedEngine(responses: [repeatedRawIDCalls, #"{"tool_calls":[]}"#, "Done."])
        let tool = LLMTool(
            definition: LLMToolDefinition(name: "lookup", description: "Lookup test tool.")
        ) { arguments in
            ["ok": true, "query": arguments.value(forKey: "query") ?? .null]
        }

        let result = try await engine.generateWithTools(
            LLMToolGenerationRequest(
                prompt: "Use lookup.",
                tools: [tool]
            )
        )

        XCTAssertEqual(result.toolCalls.map(\.executionID), ["call_1", "call_2"])
        XCTAssertEqual(result.toolCalls.map(\.rawID), ["call_1", "call_1"])
        XCTAssertEqual(result.toolOutputs.map(\.callID), ["call_1", "call_2"])
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
        XCTAssertFalse(recorder.events.containsToolCandidateEvent)
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
        XCTAssertFalse(recorder.events.containsToolCandidateEvent)
    }

    func testToolCallJSONIsDiagnosticOnlyAndFinalAnswerStreamsAfterToolRound() async throws {
        let toolCall = #"{"tool_calls":[{"id":"call_1","name":"lookup","arguments":{"query":"swift"}}]}"#
        let engine = ScriptedEngine(responses: [toolCall, #"{"tool_calls":[]}"#, "Swift is a programming language."])
        let recorder = ToolEventRecorder()
        let tool = LLMTool(
            definition: LLMToolDefinition(name: "lookup", description: "Lookup test tool.")
        ) { arguments in
            ["ok": true, "query": arguments.value(forKey: "query") ?? .null]
        }

        let result = try await engine.generateWithTools(
            LLMToolGenerationRequest(
                prompt: "Use lookup.",
                tools: [tool]
            ),
            onPhaseAwareEvent: { event in recorder.append(event) }
        )

        XCTAssertEqual(result.finalText, "Swift is a programming language.")
        XCTAssertTrue(recorder.events.containsToolCandidateText(#""tool_calls""#))
        XCTAssertFalse(recorder.events.finalAnswerTextContains("tool_calls"))
        XCTAssertEqual(recorder.events.finalAnswerDeltaText, "Swift is a programming language.")
    }

    func testToolCandidateEventsArePhaseAwareHiddenTelemetry() async throws {
        let candidateOptions = GenerationOptions(enableThinking: true, thinkingBudgetTokens: 4)
        let engine = ScriptedEngine(responses: [#"{"tool_calls":[]}"#, "Visible answer."])
        let recorder = ToolEventRecorder()
        let tool = LLMTool(
            definition: LLMToolDefinition(name: "noop", description: "No-op test tool.")
        ) { _ in
            ["ok": true]
        }

        let result = try await engine.generateWithTools(
            LLMToolGenerationRequest(
                prompt: "Answer if no tool is needed.",
                toolCandidateOptions: candidateOptions,
                tools: [tool]
            ),
            onPhaseAwareEvent: { event in recorder.append(event) }
        )

        XCTAssertEqual(result.finalText, "Visible answer.")
        XCTAssertTrue(recorder.events.containsToolCandidateThinking)
        XCTAssertTrue(recorder.events.containsToolCandidateText(#""tool_calls""#))
        XCTAssertFalse(recorder.events.finalAnswerTextContains("tool_calls"))
        XCTAssertEqual(recorder.events.finalAnswerDeltaText, "Visible answer.")
    }

    func testMaxToolRoundsDoesNotStreamToolJSONAsFinalAnswer() async throws {
        let toolCall = #"{"tool_calls":[{"id":"call_1","name":"noop","arguments":{}}]}"#
        let engine = ScriptedEngine(responses: [toolCall, "Final answer after max rounds."])
        let recorder = ToolEventRecorder()
        let tool = LLMTool(
            definition: LLMToolDefinition(name: "noop", description: "No-op test tool.")
        ) { _ in
            ["ok": true]
        }

        let result = try await engine.generateWithTools(
            LLMToolGenerationRequest(
                prompt: "Use a tool.",
                tools: [tool],
                maxToolRounds: 0
            ),
            onPhaseAwareEvent: { event in recorder.append(event) }
        )

        XCTAssertEqual(result.stopReason, "max-tool-rounds")
        XCTAssertEqual(result.finalText, "Final answer after max rounds.")
        XCTAssertEqual(result.toolCalls.count, 1)
        XCTAssertEqual(result.toolOutputs.count, 0)
        XCTAssertTrue(recorder.events.containsToolCandidateText("tool_calls"))
        XCTAssertFalse(recorder.events.finalAnswerTextContains("tool_calls"))
        XCTAssertEqual(recorder.events.finalAnswerDeltaText, "Final answer after max rounds.")
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
    var containsToolCandidateEvent: Bool {
        contains {
            if case .toolCandidateEvent = $0 { return true }
            return false
        }
    }

    var finalAnswerDeltaText: String {
        reduce(into: "") { result, event in
            if case .finalAnswerEvent(.finalAnswerDelta(let text, _)) = event {
                result += text
            }
        }
    }

    func containsToolCandidateText(_ text: String) -> Bool {
        contains { event in
            guard case .toolCandidateEvent(_, let streamEvent) = event else {
                return false
            }
            switch streamEvent {
            case .tokenChunk(let preview, _, _):
                return preview.contains(text)
            case .finalAnswerDelta(let preview, _):
                return preview.contains(text)
            case .finalAnswerSnapshot(let preview, _, _):
                return preview.contains(text)
            default:
                return false
            }
        }
    }

    var containsToolCandidateThinking: Bool {
        contains { event in
            guard case .toolCandidateEvent(_, let streamEvent) = event else {
                return false
            }
            switch streamEvent {
            case .requestSent(.thinking),
                 .firstByteReceived(_, .thinking),
                 .tokenChunk(_, _, .thinking),
                 .generationStats(_, _, _, _, .thinking),
                 .done(_, _, .thinking):
                return true
            case .phaseChanged(_, .thinking):
                return true
            default:
                return false
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

    var startedToolCallIDs: [String] {
        compactMap { event in
            if case .toolCallStarted(let call) = event {
                return call.id
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
