import Foundation

public enum LocalLLMJSON {
    public static func makeDecoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }

    public static func makePrettyEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }

    public static func prettyPrintedString<T: Encodable>(from value: T) throws -> String {
        let data = try makePrettyEncoder().encode(value)
        return String(decoding: data, as: UTF8.self)
    }

    public static func prettyPrintedJSONString(from object: Any) -> String {
        guard JSONSerialization.isValidJSONObject(object),
              let data = try? JSONSerialization.data(
                withJSONObject: object,
                options: [.prettyPrinted, .sortedKeys]
              ),
              let string = String(data: data, encoding: .utf8)
        else { return "{}" }
        return string
    }
}

public enum LLMResponsePreview {
    public static func describe(_ text: String, limit: Int = 200) -> String {
        let snippet = String(text.prefix(limit))
        guard !snippet.isEmpty else { return "<empty response>" }

        if snippet.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "<whitespace-only response: \(escape(snippet, visualizeSpaces: true))>"
        }

        return escape(snippet, visualizeSpaces: false)
    }

    private static func escape(_ text: String, visualizeSpaces: Bool) -> String {
        var output = ""
        for scalar in text.unicodeScalars {
            switch scalar {
            case "\n":
                output += "\\n"
            case "\r":
                output += "\\r"
            case "\t":
                output += "\\t"
            case " " where visualizeSpaces:
                output += "."
            default:
                if scalar.value < 0x20 {
                    output += String(format: "\\u{%02X}", scalar.value)
                } else {
                    output.unicodeScalars.append(scalar)
                }
            }
        }
        return output
    }
}

public enum LLMResponseSanitizer {
    public static func unwrapStructuredOutput(_ text: String) -> String {
        var output = stripReasoningBlocks(from: text)

        if let range = output.range(of: "<|channel|>final<|message|>", options: .backwards) {
            output = String(output[range.upperBound...])
        } else if let range = output.range(of: "<|message|>", options: .backwards) {
            let candidate = String(output[range.upperBound...])
            if candidate.contains("{") || candidate.contains("```") {
                output = candidate
            }
        }

        output = output.replacingOccurrences(of: "<|return|>", with: "")
        output = output.replacingOccurrences(of: "<|end|>", with: "")
        return output.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func stripReasoningBlocks(from text: String) -> String {
        var output = text
        for pattern in [
            #"(?is)<think>\s*.*?\s*</think>\s*"#,
            #"(?is)<\|channel(?:\|)?>thought\b[\s\S]*?(?:<channel\|>|<\|channel\|>)\s*"#
        ] {
            output = replacingMatches(in: output, pattern: pattern, with: "")
        }
        return output.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func replacingMatches(in text: String, pattern: String, with template: String) -> String {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else {
            return text
        }
        let range = NSRange(text.startIndex..., in: text)
        return regex.stringByReplacingMatches(in: text, options: [], range: range, withTemplate: template)
    }
}

public enum JSONSalvage {
    public enum Error: Swift.Error, LocalizedError {
        case decodingFailed(String)

        public var errorDescription: String? {
            switch self {
            case .decodingFailed(let detail):
                return "JSON salvage failed: \(detail)"
            }
        }
    }

    public static func decode<T: Decodable>(_ type: T.Type, from text: String) throws -> T {
        let stripped = LLMResponseSanitizer.unwrapStructuredOutput(text)
        let blocks = extractFencedBlocksReversed(from: stripped)
        let candidates = blocks.isEmpty ? [stripped] : blocks

        var lastError: Swift.Error?
        var sawNonEmptyCandidate = false

        for rawCandidate in candidates {
            var candidate = stripFences(rawCandidate).trimmingCharacters(in: .whitespacesAndNewlines)
            if let trimmed = trimToObject(candidate) {
                candidate = trimmed
            }
            candidate = normalizeControlCharsInStringLiterals(candidate)
            guard !candidate.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                continue
            }
            sawNonEmptyCandidate = true

            if let data = candidate.data(using: .utf8),
               let decoded = try? JSONDecoder().decode(type, from: data) {
                return decoded
            }

            if let salvaged = regexSalvage(candidate, for: type),
               let data = salvaged.data(using: .utf8),
               let decoded = try? JSONDecoder().decode(type, from: data) {
                return decoded
            }

            if let data = candidate.data(using: .utf8) {
                do {
                    return try JSONDecoder().decode(type, from: data)
                } catch {
                    lastError = error
                }
            }
        }

        if !sawNonEmptyCandidate {
            throw Error.decodingFailed("Response was empty after trimming wrappers and whitespace")
        }
        if let lastError {
            throw lastError
        }
        throw Error.decodingFailed("No fenced block decoded successfully")
    }

    public static func unwrapResponse(_ text: String) -> String {
        let stripped = LLMResponseSanitizer.unwrapStructuredOutput(text)
        let blocks = extractFencedBlocksReversed(from: stripped)
        let inner = blocks.first ?? stripped
        return stripFences(inner).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public static func normalizeStringControlChars(_ text: String) -> String {
        normalizeControlCharsInStringLiterals(text)
    }

    private static func stripFences(_ text: String) -> String {
        var output = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if output.hasPrefix("```json") {
            output = String(output.dropFirst(7))
        } else if output.hasPrefix("```") {
            output = String(output.dropFirst(3))
        }
        if output.hasSuffix("```") {
            output = String(output.dropLast(3))
        }
        return output.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func extractFencedBlocksReversed(from text: String) -> [String] {
        guard let regex = try? NSRegularExpression(
            pattern: "(?is)```(?:json)?\\s*([\\s\\S]*?)```",
            options: []
        ) else { return [] }

        let nsString = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: nsString.length))
        var results: [String] = []
        for match in matches {
            guard match.numberOfRanges > 1 else { continue }
            let candidate = nsString.substring(with: match.range(at: 1))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if candidate.contains("{") {
                results.append(candidate)
            }
        }
        return results.reversed()
    }

    private static func trimToObject(_ text: String) -> String? {
        guard let start = text.firstIndex(of: "{") else { return nil }

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
                if character == "{" {
                    depth += 1
                } else if character == "}" {
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

    private static func normalizeControlCharsInStringLiterals(_ text: String) -> String {
        var output = ""
        var inString = false
        var escaped = false

        for character in text {
            if escaped {
                output.append(character)
                escaped = false
                continue
            }

            if character == "\\" {
                output.append(character)
                if inString {
                    escaped = true
                }
                continue
            }

            if character == "\"" {
                output.append(character)
                inString.toggle()
                continue
            }

            if inString && character.isNewline {
                output += "\\n"
                continue
            }

            if inString,
               let value = character.asciiValue,
               value < 0x20,
               character != "\t" {
                continue
            }

            output.append(character)
        }
        return output
    }

    private static func regexSalvage<T: Decodable>(_ text: String, for type: T.Type) -> String? {
        guard let regex = try? NSRegularExpression(pattern: "\\{[\\s\\S]*?\\}", options: []) else {
            return nil
        }
        let nsString = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: nsString.length))
        for match in matches.reversed() {
            let candidate = nsString.substring(with: match.range)
            if let data = candidate.data(using: .utf8),
               (try? JSONDecoder().decode(type, from: data)) != nil {
                return candidate
            }
        }
        return nil
    }
}
