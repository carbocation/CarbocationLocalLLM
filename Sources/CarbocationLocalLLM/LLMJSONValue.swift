import Foundation

public enum LLMJSONValue: Codable, Hashable, Sendable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([LLMJSONValue])
    case object([String: LLMJSONValue])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([LLMJSONValue].self) {
            self = .array(value)
        } else if let value = try? container.decode([String: LLMJSONValue].self) {
            self = .object(value)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Unsupported JSON value."
            )
        }
    }

    public init(jsonString: String) throws {
        let data = Data(jsonString.utf8)
        self = try JSONDecoder().decode(LLMJSONValue.self, from: data)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null:
            try container.encodeNil()
        case .bool(let value):
            try container.encode(value)
        case .number(let value):
            try container.encode(value)
        case .string(let value):
            try container.encode(value)
        case .array(let value):
            try container.encode(value)
        case .object(let value):
            try container.encode(value)
        }
    }

    public var objectValue: [String: LLMJSONValue]? {
        if case .object(let value) = self { return value }
        return nil
    }

    public var arrayValue: [LLMJSONValue]? {
        if case .array(let value) = self { return value }
        return nil
    }

    public var stringValue: String? {
        switch self {
        case .string(let value):
            return value
        case .number(let value):
            return String(value)
        case .bool(let value):
            return String(value)
        case .null, .array, .object:
            return nil
        }
    }

    public var doubleValue: Double? {
        switch self {
        case .number(let value):
            return value
        case .string(let value):
            return Double(value)
        case .null, .bool, .array, .object:
            return nil
        }
    }

    public func value(forKey key: String) -> LLMJSONValue? {
        objectValue?[key]
    }

    public func string(forKey key: String) -> String? {
        value(forKey: key)?.stringValue
    }

    public func double(forKey key: String) -> Double? {
        value(forKey: key)?.doubleValue
    }

    public func array(forKey key: String) -> [LLMJSONValue]? {
        value(forKey: key)?.arrayValue
    }

    public func jsonString(prettyPrinted: Bool = false) throws -> String {
        let encoder = JSONEncoder()
        if prettyPrinted {
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        } else {
            encoder.outputFormatting = [.sortedKeys]
        }
        let data = try encoder.encode(self)
        return String(decoding: data, as: UTF8.self)
    }
}

extension LLMJSONValue: ExpressibleByNilLiteral,
    ExpressibleByBooleanLiteral,
    ExpressibleByFloatLiteral,
    ExpressibleByIntegerLiteral,
    ExpressibleByStringLiteral,
    ExpressibleByArrayLiteral,
    ExpressibleByDictionaryLiteral
{
    public init(nilLiteral: ()) {
        self = .null
    }

    public init(booleanLiteral value: Bool) {
        self = .bool(value)
    }

    public init(floatLiteral value: Double) {
        self = .number(value)
    }

    public init(integerLiteral value: Int) {
        self = .number(Double(value))
    }

    public init(stringLiteral value: String) {
        self = .string(value)
    }

    public init(arrayLiteral elements: LLMJSONValue...) {
        self = .array(elements)
    }

    public init(dictionaryLiteral elements: (String, LLMJSONValue)...) {
        self = .object(Dictionary(uniqueKeysWithValues: elements))
    }
}
