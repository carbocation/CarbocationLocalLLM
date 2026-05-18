import Foundation

public struct LLMToolDefinition: Codable, Hashable, Sendable {
    public var name: String
    public var description: String
    public var parameters: LLMJSONValue

    public init(
        name: String,
        description: String,
        parameters: LLMJSONValue = .object([
            "type": .string("object"),
            "properties": .object([:]),
            "additionalProperties": .bool(false)
        ])
    ) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }

    public func validate() throws {
        guard Self.isValidName(name) else {
            throw LLMToolError.invalidDefinition("Tool name must start with a letter or underscore and contain only letters, digits, and underscores: \(name)")
        }
        guard !description.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw LLMToolError.invalidDefinition("Tool \(name) must have a description.")
        }
        guard case .object = parameters else {
            throw LLMToolError.invalidDefinition("Tool \(name) parameters must be a JSON schema object.")
        }
    }

    public static func isValidName(_ name: String) -> Bool {
        guard let first = name.unicodeScalars.first,
              first == "_" || CharacterSet.letters.contains(first),
              name.unicodeScalars.count <= 64 else {
            return false
        }
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "_"))
        return name.unicodeScalars.allSatisfy { allowed.contains($0) }
    }
}

public struct LLMTool: Sendable {
    public typealias Executor = @Sendable (LLMJSONValue) async throws -> LLMJSONValue

    public var definition: LLMToolDefinition
    private let executor: Executor

    public init(
        definition: LLMToolDefinition,
        execute: @escaping Executor
    ) {
        self.definition = definition
        self.executor = execute
    }

    public func call(arguments: LLMJSONValue) async throws -> LLMJSONValue {
        try await executor(arguments)
    }
}

public struct LLMToolCall: Codable, Hashable, Sendable, Identifiable {
    /// Stable identity for this tool execution. Use this for UI diffing, persistence, and matching tool outputs.
    public var id: String { executionID }
    /// Library-generated execution identity, unique within a `generateWithTools` result.
    public var executionID: String
    /// Raw model-provided call ID, when the model supplied a non-empty ID.
    public var rawID: String?
    public var name: String
    public var arguments: LLMJSONValue
    /// Phase of the raw assistant stream where this tool call began, when the provider exposes that signal.
    public var triggerPhase: LLMStreamContentPhase?

    public init(
        executionID: String,
        rawID: String? = nil,
        name: String,
        arguments: LLMJSONValue,
        triggerPhase: LLMStreamContentPhase? = nil
    ) {
        self.executionID = executionID
        self.rawID = Self.normalizedRawID(rawID)
        self.name = name
        self.arguments = arguments
        self.triggerPhase = triggerPhase
    }

    private enum CodingKeys: String, CodingKey {
        case executionID
        case rawID
        case legacyID = "id"
        case name
        case arguments
        case triggerPhase
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let executionID = try container.decodeIfPresent(String.self, forKey: .executionID)
            ?? container.decode(String.self, forKey: .legacyID)
        self.init(
            executionID: executionID,
            rawID: try container.decodeIfPresent(String.self, forKey: .rawID),
            name: try container.decode(String.self, forKey: .name),
            arguments: try container.decode(LLMJSONValue.self, forKey: .arguments),
            triggerPhase: try container.decodeIfPresent(LLMStreamContentPhase.self, forKey: .triggerPhase)
        )
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(executionID, forKey: .executionID)
        try container.encodeIfPresent(rawID, forKey: .rawID)
        try container.encode(name, forKey: .name)
        try container.encode(arguments, forKey: .arguments)
        try container.encodeIfPresent(triggerPhase, forKey: .triggerPhase)
    }

    private static func normalizedRawID(_ rawID: String?) -> String? {
        guard let rawID,
              !rawID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return nil
        }
        return rawID
    }
}

public struct LLMToolOutput: Codable, Hashable, Sendable, Identifiable {
    /// Same value as `callID`, used for `Identifiable` conformance.
    public var id: String { callID }
    /// Execution ID of the `LLMToolCall` that produced this output.
    public var callID: String
    public var name: String
    public var content: LLMJSONValue
    public var isError: Bool

    public init(
        callID: String,
        name: String,
        content: LLMJSONValue,
        isError: Bool = false
    ) {
        self.callID = callID
        self.name = name
        self.content = content
        self.isError = isError
    }
}

public struct LLMToolInteractionRound: Codable, Hashable, Sendable {
    public var calls: [LLMToolCall]
    public var outputs: [LLMToolOutput]

    public init(calls: [LLMToolCall], outputs: [LLMToolOutput]) {
        self.calls = calls
        self.outputs = outputs
    }
}

public enum LLMToolChoice: Hashable, Sendable {
    case auto
    case none
    case required
    case named(String)
}

extension LLMToolChoice: Codable {
    private enum CodingKeys: String, CodingKey {
        case type
        case name
    }

    public init(from decoder: Decoder) throws {
        if let container = try? decoder.singleValueContainer(),
           let rawValue = try? container.decode(String.self) {
            switch rawValue {
            case "auto":
                self = .auto
            case "none":
                self = .none
            case "required":
                self = .required
            default:
                self = .named(rawValue)
            }
            return
        }

        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        switch type {
        case "auto":
            self = .auto
        case "none":
            self = .none
        case "required":
            self = .required
        case "named":
            self = .named(try container.decode(String.self, forKey: .name))
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .type,
                in: container,
                debugDescription: "Unknown tool choice: \(type)"
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .auto:
            try container.encode("auto", forKey: .type)
        case .none:
            try container.encode("none", forKey: .type)
        case .required:
            try container.encode("required", forKey: .type)
        case .named(let name):
            try container.encode("named", forKey: .type)
            try container.encode(name, forKey: .name)
        }
    }
}

public struct LLMToolGenerationRequest: Sendable {
    public var system: String
    public var prompt: String
    /// Options for the single tool-aware generation flow.
    public var options: GenerationOptions
    public var tools: [LLMTool]
    public var toolChoice: LLMToolChoice
    public var maxToolRounds: Int

    public init(
        system: String = "",
        prompt: String,
        options: GenerationOptions = GenerationOptions(),
        tools: [LLMTool] = [],
        toolChoice: LLMToolChoice = .auto,
        maxToolRounds: Int = 4
    ) {
        self.system = system
        self.prompt = prompt
        self.options = options
        self.tools = tools
        self.toolChoice = toolChoice
        self.maxToolRounds = maxToolRounds
    }
}

public struct LLMToolGenerationResult: Codable, Hashable, Sendable {
    public var finalText: String
    public var toolCalls: [LLMToolCall]
    public var toolOutputs: [LLMToolOutput]
    public var roundsCompleted: Int
    public var stopReason: String

    public init(
        finalText: String,
        toolCalls: [LLMToolCall] = [],
        toolOutputs: [LLMToolOutput] = [],
        roundsCompleted: Int = 0,
        stopReason: String = "complete"
    ) {
        self.finalText = finalText
        self.toolCalls = toolCalls
        self.toolOutputs = toolOutputs
        self.roundsCompleted = roundsCompleted
        self.stopReason = stopReason
    }
}

public enum LLMToolPhaseAwareStreamEvent: Sendable {
    /// Phase-aware events for user-visible assistant output. Apps can render final-answer deltas and snapshots from this event.
    case finalAnswerEvent(LLMPhaseAwareStreamEvent)
    case toolRoundStarted(round: Int)
    case toolCallStarted(LLMToolCall)
    case toolCallCompleted(LLMToolOutput)
    case toolCallFailed(LLMToolOutput)
    case aggregateGenerationStats(promptTokens: Int, generatedTokens: Int, stopReason: String)
    case aggregateAccelerationStats(LLMGenerationAccelerationStats)
}

public enum LLMToolError: Error, LocalizedError, Sendable, Equatable {
    case duplicateToolName(String)
    case invalidDefinition(String)
    case invalidRequest(String)
    case unsupportedToolMode(String)

    public var errorDescription: String? {
        switch self {
        case .duplicateToolName(let name):
            return "Duplicate tool name: \(name)."
        case .invalidDefinition(let detail):
            return "Invalid tool definition: \(detail)"
        case .invalidRequest(let detail):
            return "Invalid tool request: \(detail)"
        case .unsupportedToolMode(let detail):
            return "Tool generation is unsupported: \(detail)"
        }
    }
}

public struct LLMToolCallIDAllocator: Sendable {
    private var usedToolCallIDs = Set<String>()
    private var nextFallbackToolCallNumber = 1

    public init() {}

    public mutating func materialize(_ parsedCalls: [LLMToolCall]) -> [LLMToolCall] {
        let reservedRawIDs = Set(parsedCalls.compactMap(\.rawID).filter { !usedToolCallIDs.contains($0) })
        return parsedCalls.map { parsedCall in
            if let rawID = parsedCall.rawID,
               !usedToolCallIDs.contains(rawID) {
                usedToolCallIDs.insert(rawID)
                return LLMToolCall(
                    executionID: rawID,
                    rawID: rawID,
                    name: parsedCall.name,
                    arguments: parsedCall.arguments,
                    triggerPhase: parsedCall.triggerPhase
                )
            }

            return LLMToolCall(
                executionID: nextUniqueFallbackToolCallID(reserving: reservedRawIDs),
                rawID: parsedCall.rawID,
                name: parsedCall.name,
                arguments: parsedCall.arguments,
                triggerPhase: parsedCall.triggerPhase
            )
        }
    }

    private mutating func nextUniqueFallbackToolCallID(reserving reservedIDs: Set<String>) -> String {
        while true {
            let candidate = "call_\(nextFallbackToolCallNumber)"
            nextFallbackToolCallNumber += 1
            guard !usedToolCallIDs.contains(candidate),
                  !reservedIDs.contains(candidate) else {
                continue
            }
            usedToolCallIDs.insert(candidate)
            return candidate
        }
    }
}

public enum LLMToolRuntime {
    public static func validate(_ request: LLMToolGenerationRequest) throws {
        guard request.maxToolRounds >= 0 else {
            throw LLMToolError.invalidRequest("maxToolRounds must be nonnegative.")
        }
        if case .required = request.toolChoice, request.tools.isEmpty {
            throw LLMToolError.invalidRequest("toolChoice required cannot be used without tools.")
        }
        if case .named(let name) = request.toolChoice,
           !request.tools.contains(where: { $0.definition.name == name }) {
            throw LLMToolError.invalidRequest("toolChoice named an unavailable tool: \(name).")
        }

        var seen = Set<String>()
        for tool in request.tools {
            try tool.definition.validate()
            guard seen.insert(tool.definition.name).inserted else {
                throw LLMToolError.duplicateToolName(tool.definition.name)
            }
        }
    }

    public static func index(_ tools: [LLMTool]) -> [String: LLMTool] {
        Dictionary(uniqueKeysWithValues: tools.map { ($0.definition.name, $0) })
    }

    public static func execute(
        calls: [LLMToolCall],
        toolsByName: [String: LLMTool],
        onPhaseAwareEvent: @Sendable (LLMToolPhaseAwareStreamEvent) -> Void
    ) async throws -> [LLMToolOutput] {
        var outputs: [LLMToolOutput] = []
        for call in calls {
            try Task.checkCancellation()
            onPhaseAwareEvent(.toolCallStarted(call))
            let output: LLMToolOutput
            if let tool = toolsByName[call.name] {
                do {
                    let content = try await tool.call(arguments: call.arguments)
                    output = LLMToolOutput(
                        callID: call.executionID,
                        name: call.name,
                        content: content,
                        isError: false
                    )
                    onPhaseAwareEvent(.toolCallCompleted(output))
                } catch is CancellationError {
                    throw CancellationError()
                } catch {
                    output = LLMToolOutput(
                        callID: call.executionID,
                        name: call.name,
                        content: toolErrorContent(
                            message: error.localizedDescription,
                            code: "tool_execution_failed"
                        ),
                        isError: true
                    )
                    onPhaseAwareEvent(.toolCallFailed(output))
                }
            } else {
                output = LLMToolOutput(
                    callID: call.executionID,
                    name: call.name,
                    content: toolErrorContent(
                        message: "Unknown tool: \(call.name)",
                        code: "unknown_tool"
                    ),
                    isError: true
                )
                onPhaseAwareEvent(.toolCallFailed(output))
            }
            outputs.append(output)
        }
        return outputs
    }

    public static func toolErrorContent(message: String, code: String) -> LLMJSONValue {
        .object([
            "ok": .bool(false),
            "error": .object([
                "code": .string(code),
                "message": .string(message)
            ])
        ])
    }
}

public enum LLMToolCallParser {
    struct ParsedToolCall: Hashable, Sendable {
        var rawID: String?
        var fallbackIndex: Int
        var name: String
        var arguments: LLMJSONValue

        var defaultID: String {
            rawID ?? "call_\(fallbackIndex + 1)"
        }

        func toolCall(executionID: String? = nil) -> LLMToolCall {
            LLMToolCall(
                executionID: executionID ?? defaultID,
                rawID: rawID,
                name: name,
                arguments: arguments
            )
        }
    }

    public static func parseToolCalls(in text: String) -> [LLMToolCall] {
        parsedToolCalls(in: text).map { $0.toolCall() }
    }

    static func parsedToolCalls(in text: String) -> [ParsedToolCall] {
        for candidate in jsonCandidates(in: text) {
            guard let value = try? LLMJSONValue(jsonString: candidate),
                  let calls = parsedToolCalls(from: value),
                  !calls.isEmpty else {
                continue
            }
            return calls
        }
        let nativeCalls = gemmaToolCalls(in: text)
        if !nativeCalls.isEmpty {
            return nativeCalls
        }
        return []
    }

    private static func parsedToolCalls(from value: LLMJSONValue) -> [ParsedToolCall]? {
        switch value {
        case .object(let object):
            if let toolCalls = object["tool_calls"]?.arrayValue {
                return toolCalls.enumerated().compactMap { index, value in
                    parsedToolCall(from: value, fallbackIndex: index)
                }
            }
            if let toolCall = parsedToolCall(from: value, fallbackIndex: 0) {
                return [toolCall]
            }
            return nil
        case .array(let values):
            return values.enumerated().compactMap { index, value in
                parsedToolCall(from: value, fallbackIndex: index)
            }
        case .null, .bool, .number, .string:
            return nil
        }
    }

    private static func parsedToolCall(from value: LLMJSONValue, fallbackIndex: Int) -> ParsedToolCall? {
        guard case .object(let object) = value else { return nil }

        let rawID = nonEmptyRawID(from: object)
        if let function = object["function"]?.objectValue {
            guard let name = function["name"]?.stringValue else { return nil }
            return ParsedToolCall(
                rawID: rawID,
                fallbackIndex: fallbackIndex,
                name: name,
                arguments: normalizedArguments(function["arguments"])
            )
        }

        guard let name = object["name"]?.stringValue ?? object["tool_name"]?.stringValue else {
            return nil
        }
        return ParsedToolCall(
            rawID: rawID,
            fallbackIndex: fallbackIndex,
            name: name,
            arguments: normalizedArguments(object["arguments"] ?? object["args"])
        )
    }

    private static func nonEmptyRawID(from object: [String: LLMJSONValue]) -> String? {
        guard let id = object["id"]?.stringValue,
              !id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return nil
        }
        return id
    }

    private static func normalizedArguments(_ value: LLMJSONValue?) -> LLMJSONValue {
        guard let value else { return .object([:]) }
        switch value {
        case .object:
            return value
        case .string(let text):
            if let decoded = try? LLMJSONValue(jsonString: text),
               case .object = decoded {
                return decoded
            }
            return .object(["value": .string(text)])
        case .null:
            return .object([:])
        case .bool, .number, .array:
            return .object(["value": value])
        }
    }

    private static func jsonCandidates(in text: String) -> [String] {
        let stripped = text.trimmingCharacters(in: .whitespacesAndNewlines)
        var candidates = fencedJSONBlocks(in: stripped)
        if let object = firstBalancedJSON(in: stripped, opening: "{", closing: "}") {
            candidates.append(object)
        }
        if let array = firstBalancedJSON(in: stripped, opening: "[", closing: "]") {
            candidates.append(array)
        }
        if candidates.isEmpty {
            candidates.append(stripped)
        }
        return candidates
    }

    private static func fencedJSONBlocks(in text: String) -> [String] {
        guard let regex = try? NSRegularExpression(
            pattern: "(?is)```(?:json)?\\s*([\\s\\S]*?)```",
            options: []
        ) else { return [] }

        let nsString = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: nsString.length))
        return matches.compactMap { match in
            guard match.numberOfRanges > 1 else { return nil }
            return nsString.substring(with: match.range(at: 1))
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }

    private static func firstBalancedJSON(
        in text: String,
        opening: Character,
        closing: Character
    ) -> String? {
        guard let start = text.firstIndex(of: opening) else { return nil }
        var depth = 0
        var inString = false
        var escaped = false
        var index = start
        while index < text.endIndex {
            let character = text[index]
            if escaped {
                escaped = false
            } else if character == "\\" && inString {
                escaped = true
            } else if character == "\"" {
                inString.toggle()
            } else if !inString {
                if character == opening {
                    depth += 1
                } else if character == closing {
                    depth -= 1
                    if depth == 0 {
                        return String(text[start...index])
                    }
                }
            }
            index = text.index(after: index)
        }
        return nil
    }

    private static func gemmaToolCalls(in text: String) -> [ParsedToolCall] {
        let startMarker = "<|tool_call>call:"
        var searchStart = text.startIndex
        var calls: [ParsedToolCall] = []

        while let markerRange = text.range(of: startMarker, range: searchStart..<text.endIndex) {
            var scanner = GemmaToolCallScanner(text: text, index: markerRange.upperBound)
            if let call = scanner.parseCall(fallbackIndex: calls.count) {
                calls.append(call)
                searchStart = scanner.index
            } else {
                searchStart = markerRange.upperBound
            }
        }

        return calls
    }

    private struct GemmaToolCallScanner {
        private static let endMarker = "<tool_call|>"
        private static let stringMarker = #"<|"|>"#

        let text: String
        var index: String.Index

        mutating func parseCall(fallbackIndex: Int) -> ParsedToolCall? {
            skipWhitespace()
            guard let openBrace = text[index..<text.endIndex].firstIndex(of: "{") else {
                return nil
            }

            let rawName = String(text[index..<openBrace])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !rawName.isEmpty else { return nil }

            index = openBrace
            guard let arguments = parseObject() else { return nil }
            skipWhitespace()
            guard consume(Self.endMarker) else { return nil }

            let name = rawName.hasPrefix("functions.")
                ? String(rawName.dropFirst("functions.".count))
                : rawName
            return ParsedToolCall(
                rawID: nil,
                fallbackIndex: fallbackIndex,
                name: name,
                arguments: arguments
            )
        }

        private mutating func parseValue() -> LLMJSONValue? {
            skipWhitespace()
            guard index < text.endIndex else { return nil }

            if text[index] == "{" {
                return parseObject()
            }
            if text[index] == "[" {
                return parseArray()
            }
            if let string = parseString() {
                return .string(string)
            }
            if consumeKeyword("true") {
                return .bool(true)
            }
            if consumeKeyword("false") {
                return .bool(false)
            }
            if consumeKeyword("null") {
                return .null
            }
            if let number = parseNumber() {
                return .number(number)
            }
            if let identifier = parseBareIdentifier() {
                return .string(identifier)
            }
            return nil
        }

        private mutating func parseObject() -> LLMJSONValue? {
            guard consume("{") else { return nil }
            skipWhitespace()

            var object: [String: LLMJSONValue] = [:]
            if consume("}") {
                return .object(object)
            }

            while index < text.endIndex {
                guard let key = parseKey() else { return nil }
                skipWhitespace()
                guard consume(":") else { return nil }
                guard let value = parseValue() else { return nil }
                object[key] = value
                skipWhitespace()

                if consume("}") {
                    return .object(object)
                }
                guard consume(",") else { return nil }
                skipWhitespace()
            }

            return nil
        }

        private mutating func parseArray() -> LLMJSONValue? {
            guard consume("[") else { return nil }
            skipWhitespace()

            var values: [LLMJSONValue] = []
            if consume("]") {
                return .array(values)
            }

            while index < text.endIndex {
                guard let value = parseValue() else { return nil }
                values.append(value)
                skipWhitespace()

                if consume("]") {
                    return .array(values)
                }
                guard consume(",") else { return nil }
                skipWhitespace()
            }

            return nil
        }

        private mutating func parseKey() -> String? {
            skipWhitespace()
            if let string = parseString() {
                return string
            }

            let start = index
            while index < text.endIndex, text[index] != ":", text[index] != "}" {
                index = text.index(after: index)
            }

            let key = String(text[start..<index])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return key.isEmpty ? nil : key
        }

        private mutating func parseString() -> String? {
            if consume(Self.stringMarker) {
                let start = index
                guard let endRange = text.range(of: Self.stringMarker, range: index..<text.endIndex) else {
                    return nil
                }
                let value = String(text[start..<endRange.lowerBound])
                index = endRange.upperBound
                return value
            }

            guard index < text.endIndex, text[index] == "\"" else { return nil }

            let start = index
            index = text.index(after: index)
            var escaped = false

            while index < text.endIndex {
                let character = text[index]
                if escaped {
                    escaped = false
                } else if character == "\\" {
                    escaped = true
                } else if character == "\"" {
                    index = text.index(after: index)
                    let rawString = String(text[start..<index])
                    if let data = rawString.data(using: .utf8),
                       let decoded = try? JSONDecoder().decode(String.self, from: data) {
                        return decoded
                    }
                    return String(rawString.dropFirst().dropLast())
                }
                index = text.index(after: index)
            }

            return nil
        }

        private mutating func parseNumber() -> Double? {
            guard index < text.endIndex,
                  text[index] == "-" || text[index].isNumber else {
                return nil
            }

            let start = index
            while index < text.endIndex, isNumberCharacter(text[index]) {
                index = text.index(after: index)
            }

            guard let number = Double(String(text[start..<index])) else {
                index = start
                return nil
            }
            return number
        }

        private mutating func parseBareIdentifier() -> String? {
            let start = index
            while index < text.endIndex,
                  text[index] != ",",
                  text[index] != "}",
                  text[index] != "]" {
                index = text.index(after: index)
            }

            let identifier = String(text[start..<index])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return identifier.isEmpty ? nil : identifier
        }

        private mutating func consumeKeyword(_ keyword: String) -> Bool {
            guard text[index..<text.endIndex].hasPrefix(keyword) else {
                return false
            }

            let end = text.index(index, offsetBy: keyword.count)
            if end < text.endIndex, isIdentifierCharacter(text[end]) {
                return false
            }

            index = end
            return true
        }

        private mutating func consume(_ literal: String) -> Bool {
            guard text[index..<text.endIndex].hasPrefix(literal) else {
                return false
            }
            index = text.index(index, offsetBy: literal.count)
            return true
        }

        private mutating func skipWhitespace() {
            while index < text.endIndex, text[index].isWhitespace {
                index = text.index(after: index)
            }
        }

        private func isNumberCharacter(_ character: Character) -> Bool {
            character.isNumber || character == "-" || character == "+" || character == "." || character == "e" || character == "E"
        }

        private func isIdentifierCharacter(_ character: Character) -> Bool {
            character.isLetter || character.isNumber || character == "_"
        }
    }
}
