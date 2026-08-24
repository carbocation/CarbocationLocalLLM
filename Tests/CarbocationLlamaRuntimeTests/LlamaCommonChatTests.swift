@testable import CarbocationLlamaRuntime
import CarbocationLocalLLM
import XCTest

final class LlamaCommonChatTests: XCTestCase {
    func testRepresentativeUpstreamTemplatesRenderWithoutModelNameBranches() throws {
        let fixtures = [
            "Qwen3.5-4B.jinja",
            "google-gemma-4-31B-it.jinja",
            "meta-llama-Llama-3.1-8B-Instruct.jinja",
            "Mistral-Small-3.2-24B-Instruct-2506.jinja",
            "deepseek-ai-DeepSeek-V3.2.jinja",
            "ibm-granite-granite-4.1.jinja",
            "Kimi-K2-Thinking.jinja",
        ]

        for fixture in fixtures {
            let source = try String(contentsOf: Self.templatesURL.appendingPathComponent(fixture))
            let templates = try LlamaCommonChatTemplates(
                model: nil,
                templateOverride: source,
                bosTokenOverride: "<s>",
                eosTokenOverride: "</s>"
            )
            let plan = try templates.render(
                messages: [ChatTemplateMessage(role: "user", content: "bridge-fixture-user")],
                tools: [],
                options: GenerationOptions(enableThinking: true)
            )

            XCTAssertTrue(templates.hasExplicitTemplate, fixture)
            XCTAssertFalse(plan.metadata.prompt.isEmpty, fixture)
            XCTAssertTrue(plan.metadata.hasParser, fixture)
        }
    }

    func testNamedChoiceFiltersToolsBeforeQwenRendering() throws {
        let templates = try Self.templates(named: "Qwen3-Coder.jinja")
        let selected = LLMToolDefinition(
            name: "special_function",
            description: "Selected tool",
            parameters: .object([
                "type": .string("object"),
                "properties": .object([
                    "arg1": .object(["type": .string("integer")])
                ]),
                "required": .array([.string("arg1")])
            ])
        )
        let omitted = LLMToolDefinition(name: "must_not_appear", description: "Filtered tool")
        let plan = try templates.render(
            messages: [ChatTemplateMessage(role: "user", content: "Use the selected tool")],
            tools: [selected, omitted],
            toolChoice: .named(selected.name),
            options: GenerationOptions()
        )

        XCTAssertTrue(plan.metadata.prompt.contains(selected.name))
        XCTAssertFalse(plan.metadata.prompt.contains(omitted.name))
        XCTAssertFalse(plan.metadata.grammar.isEmpty)
    }

    func testQwenParserBuffersPartialToolCallAndReturnsStructuredCallAtEOF() throws {
        let templates = try Self.templates(named: "Qwen3-Coder.jinja")
        let tool = LLMToolDefinition(
            name: "special_function",
            description: "A test function",
            parameters: .object([
                "type": .string("object"),
                "properties": .object([
                    "arg1": .object(["type": .string("integer")])
                ]),
                "required": .array([.string("arg1")])
            ])
        )
        let plan = try templates.render(
            messages: [ChatTemplateMessage(role: "user", content: "Call it")],
            tools: [tool],
            options: GenerationOptions()
        )
        let parser = try plan.makeParser()

        let partial = try parser.append(
            "<tool_call>\n<function=special_function>\n<parameter=arg1>\n",
            isPartial: true
        )
        XCTAssertTrue(partial.isPartial)

        let complete = try parser.append(
            "1\n</parameter>\n</function>\n</tool_call>",
            isPartial: false
        )
        XCTAssertEqual(complete.message.toolCalls.count, 1)
        XCTAssertEqual(complete.message.toolCalls.first?.name, tool.name)
        XCTAssertEqual(
            complete.message.toolCalls.first?.materialized()?.arguments,
            .object(["arg1": .number(1)])
        )
    }

    func testMalformedQwenToolCallIsNeverMaterialized() throws {
        let templates = try Self.templates(named: "Qwen3-Coder.jinja")
        let tool = LLMToolDefinition(name: "special_function", description: "A test function")
        let plan = try templates.render(
            messages: [ChatTemplateMessage(role: "user", content: "Call it")],
            tools: [tool],
            options: GenerationOptions()
        )
        let parser = try plan.makeParser()
        let result = try parser.append(
            "<tool_call>\n<function=special_function>\n<parameter=arg1>",
            isPartial: false
        )

        XCTAssertTrue(result.message.toolCalls.compactMap { $0.materialized() }.isEmpty)
    }

    func testReasoningControlsAndHistoryReachUpstreamRenderer() throws {
        let source = """
        {{ reasoning_effort|default('default') }}|{{ preserve_thinking|default(true) }}|
        {%- for message in messages -%}
        {{ message.role }}={{ message.content }}:{{ message.reasoning_content|default('none') }};
        {%- endfor -%}
        {%- if add_generation_prompt -%}assistant={%- endif -%}
        """
        let templates = try LlamaCommonChatTemplates(model: nil, templateOverride: source)
        let plan = try templates.render(
            messages: [
                ChatTemplateMessage(role: "user", content: "Question"),
                ChatTemplateMessage(
                    role: "assistant",
                    content: "Visible answer",
                    reasoningContent: "Historical reasoning"
                ),
            ],
            tools: [],
            options: GenerationOptions(
                enableThinking: true,
                reasoningEffort: .medium,
                preserveThinking: false
            )
        )

        XCTAssertTrue(plan.metadata.prompt.lowercased().contains("medium|false|"), plan.metadata.prompt)
        XCTAssertTrue(plan.metadata.prompt.contains("Historical reasoning"), plan.metadata.prompt)
    }

    private static func templates(named name: String) throws -> LlamaCommonChatTemplates {
        try LlamaCommonChatTemplates(
            model: nil,
            templateOverride: String(contentsOf: templatesURL.appendingPathComponent(name)),
            bosTokenOverride: "<s>",
            eosTokenOverride: "</s>"
        )
    }

    private static var templatesURL: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Vendor/llama.cpp/models/templates")
    }
}
