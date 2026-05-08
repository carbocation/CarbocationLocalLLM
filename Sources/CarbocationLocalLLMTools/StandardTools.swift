import CarbocationLocalLLM
import Foundation

public enum LLMStandardTools {
    public static func calculate() -> LLMTool {
        LLMCalculateTool.tool()
    }

    public static func convertUnits() -> LLMTool {
        LLMConvertUnitsTool.tool()
    }

    public static func loadWebpage(
        configuration: LLMLoadWebpageTool.Configuration = LLMLoadWebpageTool.Configuration(),
        fetcher: any LLMWebpageFetching = URLSessionWebpageFetcher()
    ) -> LLMTool {
        LLMLoadWebpageTool.tool(configuration: configuration, fetcher: fetcher)
    }

    public static func initialTools(
        webpageConfiguration: LLMLoadWebpageTool.Configuration = LLMLoadWebpageTool.Configuration(),
        webpageFetcher: any LLMWebpageFetching = URLSessionWebpageFetcher()
    ) -> [LLMTool] {
        [
            loadWebpage(configuration: webpageConfiguration, fetcher: webpageFetcher),
            calculate(),
            convertUnits()
        ]
    }
}

enum LLMStandardToolResult {
    static func success(_ fields: [String: LLMJSONValue]) -> LLMJSONValue {
        var output = fields
        output["ok"] = .bool(true)
        return .object(output)
    }

    static func failure(message: String, code: String = "invalid_arguments") -> LLMJSONValue {
        .object([
            "ok": .bool(false),
            "error": .object([
                "code": .string(code),
                "message": .string(message)
            ])
        ])
    }
}
