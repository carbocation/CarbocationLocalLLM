import CarbocationLocalLLM
import Foundation
import Jinja

enum ChatTemplatePromptFormatter {
    enum Error: Swift.Error, LocalizedError {
        case notJinjaTemplate
        case missingUserContent

        var errorDescription: String? {
            switch self {
            case .notJinjaTemplate:
                return "Embedded chat template is not a Jinja template."
            case .missingUserContent:
                return "Applied chat template did not include the user message."
            }
        }
    }

    static func format(
        template source: String,
        system: String,
        user: String,
        bosToken: String,
        eosToken: String
    ) throws -> String {
        guard source.contains("{%") || source.contains("{{") || source.contains("{#") else {
            throw Error.notJinjaTemplate
        }

        let template = try Template(
            source,
            with: Template.Options(lstripBlocks: true, trimBlocks: true)
        )
        let context: [String: Value] = [
            "messages": [
                [
                    "role": "system",
                    "content": .string(system)
                ],
                [
                    "role": "user",
                    "content": .string(user)
                ]
            ],
            "bos_token": .string(bosToken),
            "eos_token": .string(eosToken),
            "add_generation_prompt": true,
            "enable_thinking": false,
            "tools": []
        ]

        let formatted = try template.render(context)
        guard user.isEmpty || formatted.contains(user) else {
            throw Error.missingUserContent
        }
        return formatted
    }
}
