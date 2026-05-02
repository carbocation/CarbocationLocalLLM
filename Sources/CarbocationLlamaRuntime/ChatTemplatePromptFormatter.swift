import CarbocationLocalLLM
import Foundation
import Jinja

struct ChatTemplatePromptFormatter {
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

    private let template: Template

    init(template source: String) throws {
        guard source.contains("{%") || source.contains("{{") || source.contains("{#") else {
            throw Error.notJinjaTemplate
        }

        self.template = try Template(
            source,
            with: Template.Options(lstripBlocks: true, trimBlocks: true)
        )
    }

    static func format(
        template source: String,
        system: String,
        user: String,
        bosToken: String,
        eosToken: String,
        enableThinking: Bool = false
    ) throws -> String {
        try Self(template: source).format(
            system: system,
            user: user,
            bosToken: bosToken,
            eosToken: eosToken,
            enableThinking: enableThinking
        )
    }

    func format(
        system: String,
        user: String,
        bosToken: String,
        eosToken: String,
        enableThinking: Bool = false
    ) throws -> String {
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
            "enable_thinking": .boolean(enableThinking),
            "tools": []
        ]

        let formatted = try template.render(context)
        guard user.isEmpty || formatted.contains(user) else {
            throw Error.missingUserContent
        }
        return formatted
    }
}
