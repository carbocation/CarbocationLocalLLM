import CarbocationLocalLLM
import Foundation
import XCTest
@testable import CarbocationLlamaRuntime

final class LlamaCommonChatLiveTests: XCTestCase {
    func testLiveEmbeddedTemplateThinkingAndNamedToolChoice() async throws {
        let path = try LiveModelTestPolicy.requiredReadablePath(
            "CLLM_COMMON_CHAT_LIVE_MODEL_PATH",
            for: .commonChatThinkingAndTools
        )

        let engine = LlamaEngine(configuration: LlamaEngineConfiguration(
            gpuLayerCount: -1,
            accelerationPolicy: .disabled
        ))
        _ = try await engine.load(
            modelAt: URL(fileURLWithPath: path),
            displayName: "Common-chat live smoke",
            requestedContext: 2_048
        )
        do {
            let thinking = try await engine.generatePhased(
                system: "Answer accurately and concisely.",
                prompt: "Think briefly, then answer: what is 2 + 2?",
                options: GenerationOptions(
                    temperature: 0,
                    maxOutputTokens: 64,
                    enableThinking: true,
                    reasoningEffort: .low,
                    thinkingBudgetTokens: 16
                )
            )
            XCTAssertEqual(thinking.templateMode, .embedded)
            XCTAssertFalse(thinking.thinkingText.isEmpty)
            XCTAssertFalse(thinking.finalText.isEmpty)

            let recordValue = LLMTool(
                definition: LLMToolDefinition(
                    name: "record_value",
                    description: "Record the integer value supplied by the user.",
                    parameters: [
                        "type": "object",
                        "properties": [
                            "value": ["type": "integer"]
                        ],
                        "required": ["value"]
                    ]
                )
            ) { arguments in
                let recorded: Bool
                if case .object(let values) = arguments {
                    recorded = values["value"] != nil
                } else {
                    recorded = false
                }
                return .object([
                    "recorded": .bool(recorded)
                ])
            }

            let tools = try await engine.generateWithToolsPhased(LLMToolGenerationRequest(
                system: "Use the selected tool exactly once, then answer DONE.",
                prompt: "Record the value 7.",
                options: GenerationOptions(
                    temperature: 0,
                    maxOutputTokens: 96,
                    enableThinking: false
                ),
                tools: [recordValue],
                toolChoice: .named("record_value"),
                maxToolRounds: 1
            ))
            XCTAssertEqual(tools.toolCalls.map { $0.name }, ["record_value"])
            XCTAssertEqual(tools.toolOutputs.count, 1)
            XCTAssertFalse(tools.finalText.isEmpty)
            await engine.unload()
        } catch {
            await engine.unload()
            throw error
        }
    }
}
