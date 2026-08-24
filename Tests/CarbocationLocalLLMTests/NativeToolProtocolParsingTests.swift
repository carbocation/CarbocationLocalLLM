import CarbocationLocalLLM
import XCTest

final class NativeToolProtocolParsingTests: XCTestCase {
    func testParsesFunctionParameterEnvelopeWithTypedAndMultilineValues() {
        let text = """
        I should look that up.
        <tool_call>
        <function=lookup>
        <parameter=query>
        Swift concurrency migration
        </parameter>
        <parameter=limit>
        3
        </parameter>
        <parameter=filters>
        {"name":"recent","enabled":true}
        </parameter>
        </function>
        </tool_call>
        """

        let calls = LLMToolCallParser.parseToolCalls(in: text)

        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls[0].name, "lookup")
        XCTAssertEqual(calls[0].arguments.string(forKey: "query"), "Swift concurrency migration")
        XCTAssertEqual(calls[0].arguments.double(forKey: "limit"), 3)
        XCTAssertEqual(calls[0].arguments.value(forKey: "filters")?.string(forKey: "name"), "recent")
        XCTAssertEqual(calls[0].arguments.value(forKey: "filters")?.value(forKey: "enabled"), .bool(true))
    }

    func testParsesParallelCallsAndNormalizesFunctionsPrefix() {
        let text = """
        <tool_call>
        <function=functions.lookup>
        <parameter=query>first</parameter>
        </function>
        </tool_call>
        <tool_call>
        <function=calculate>
        <parameter=operands>[2,3]</parameter>
        </function>
        </tool_call>
        """

        let calls = LLMToolCallParser.parseToolCalls(in: text)

        XCTAssertEqual(calls.map(\.id), ["call_1", "call_2"])
        XCTAssertEqual(calls.map(\.name), ["lookup", "calculate"])
        XCTAssertEqual(calls[0].arguments.string(forKey: "query"), "first")
        XCTAssertEqual(calls[1].arguments.array(forKey: "operands"), [2, 3])
    }

    func testRejectsIncompleteFunctionParameterEnvelope() {
        let text = """
        visible prefix
        <tool_call>
        <function=lookup>
        <parameter=query>unfinished
        """

        XCTAssertTrue(LLMToolCallParser.parseToolCalls(in: text).isEmpty)
    }

    func testMalformedFunctionParameterEnvelopeDoesNotExecuteNestedJSON() {
        let text = """
        <tool_call>
        <function=lookup>
        <parameter=query>{"name":"dangerous","arguments":{"value":1}}</parameter>
        </function>
        """

        XCTAssertTrue(LLMToolCallParser.parseToolCalls(in: text).isEmpty)
    }

    func testGenericJSONAliasesPreserveParametersAndToolCallID() {
        let text = """
        {"tool_call_id":"provider-call-7","name":"lookup","parameters":{"query":"Swift"}}
        """

        let calls = LLMToolCallParser.parseToolCalls(in: text)

        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls[0].executionID, "provider-call-7")
        XCTAssertEqual(calls[0].rawID, "provider-call-7")
        XCTAssertEqual(calls[0].name, "lookup")
        XCTAssertEqual(calls[0].arguments.string(forKey: "query"), "Swift")
    }

    func testNestedFunctionParametersAliasIsAccepted() {
        let text = """
        {"id":"call-8","function":{"name":"calculate","parameters":{"value":42}}}
        """

        let calls = LLMToolCallParser.parseToolCalls(in: text)

        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls[0].name, "calculate")
        XCTAssertEqual(calls[0].arguments.double(forKey: "value"), 42)
    }

    func testDirectArrayCallsPreserveGenericAliases() {
        let text = """
        [
          {"tool_call_id":"call-9","name":"lookup","parameters":{"query":"Swift"}},
          {"name":"calculate","arguments":{"value":42}}
        ]
        """

        let calls = LLMToolCallParser.parseToolCalls(in: text)

        XCTAssertEqual(calls.map(\.name), ["lookup", "calculate"])
        XCTAssertEqual(calls[0].rawID, "call-9")
        XCTAssertEqual(calls[0].arguments.string(forKey: "query"), "Swift")
        XCTAssertEqual(calls[1].arguments.double(forKey: "value"), 42)
    }

    func testTaggedJSONEnvelopeKeepsGenericParserCompatibility() {
        let text = """
        <tool_call>{"name":"lookup","parameters":{"query":"Swift"}}</tool_call>
        """

        let calls = LLMToolCallParser.parseToolCalls(in: text)

        XCTAssertEqual(calls.map(\.name), ["lookup"])
        XCTAssertEqual(calls[0].arguments.string(forKey: "query"), "Swift")
    }
}
