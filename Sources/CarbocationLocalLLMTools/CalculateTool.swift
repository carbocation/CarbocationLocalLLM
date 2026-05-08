import CarbocationLocalLLM
import Foundation

public enum LLMCalculateTool {
    public static let definition = LLMToolDefinition(
        name: "calculate",
        description: "Perform deterministic arithmetic with structured numeric operands.",
        parameters: .object([
            "type": .string("object"),
            "additionalProperties": .bool(false),
            "required": .array([.string("operation"), .string("operands")]),
            "properties": .object([
                "operation": .object([
                    "type": .string("string"),
                    "enum": .array([
                        .string("add"),
                        .string("subtract"),
                        .string("multiply"),
                        .string("divide"),
                        .string("power")
                    ])
                ]),
                "operands": .object([
                    "type": .string("array"),
                    "items": .object(["type": .string("number")]),
                    "minItems": .number(1)
                ])
            ])
        ])
    )

    public static func tool() -> LLMTool {
        LLMTool(definition: definition) { arguments in
            do {
                return try evaluate(arguments: arguments)
            } catch let error as CalculateError {
                return LLMStandardToolResult.failure(message: error.errorDescription ?? "\(error)")
            } catch {
                return LLMStandardToolResult.failure(message: error.localizedDescription)
            }
        }
    }

    public static func evaluate(arguments: LLMJSONValue) throws -> LLMJSONValue {
        guard let operation = arguments.string(forKey: "operation") else {
            throw CalculateError.missingField("operation")
        }
        let operands = try decimalOperands(from: arguments)
        let result: Decimal

        switch operation {
        case "add":
            guard !operands.isEmpty else { throw CalculateError.invalidArity(operation: operation) }
            result = operands.dropFirst().reduce(operands[0], +)
        case "subtract":
            guard operands.count >= 2 else { throw CalculateError.invalidArity(operation: operation) }
            result = operands.dropFirst().reduce(operands[0], -)
        case "multiply":
            guard !operands.isEmpty else { throw CalculateError.invalidArity(operation: operation) }
            result = operands.dropFirst().reduce(operands[0], *)
        case "divide":
            guard operands.count >= 2 else { throw CalculateError.invalidArity(operation: operation) }
            result = try operands.dropFirst().reduce(operands[0]) { partial, operand in
                guard operand != 0 else { throw CalculateError.divideByZero }
                return partial / operand
            }
        case "power":
            guard operands.count == 2 else { throw CalculateError.invalidArity(operation: operation) }
            result = try power(base: operands[0], exponent: operands[1])
        default:
            throw CalculateError.unsupportedOperation(operation)
        }

        return LLMStandardToolResult.success([
            "operation": .string(operation),
            "result": .string(format(decimal: result))
        ])
    }

    private static func decimalOperands(from arguments: LLMJSONValue) throws -> [Decimal] {
        guard let values = arguments.array(forKey: "operands") else {
            throw CalculateError.missingField("operands")
        }
        return try values.map { value in
            guard let decimal = decimal(from: value) else {
                throw CalculateError.invalidOperand
            }
            return decimal
        }
    }

    private static func decimal(from value: LLMJSONValue) -> Decimal? {
        switch value {
        case .number(let number):
            return Decimal(string: String(number), locale: Locale(identifier: "en_US_POSIX"))
        case .string(let string):
            return Decimal(string: string, locale: Locale(identifier: "en_US_POSIX"))
        case .null, .bool, .array, .object:
            return nil
        }
    }

    private static func power(base: Decimal, exponent: Decimal) throws -> Decimal {
        guard let integerExponent = integer(from: exponent) else {
            throw CalculateError.nonIntegerExponent
        }
        if integerExponent == 0 {
            return 1
        }
        if base == 0, integerExponent < 0 {
            throw CalculateError.divideByZero
        }

        var result: Decimal = 1
        for _ in 0..<abs(integerExponent) {
            result *= base
        }
        if integerExponent < 0 {
            result = 1 / result
        }
        return result
    }

    private static func integer(from decimal: Decimal) -> Int? {
        let double = NSDecimalNumber(decimal: decimal).doubleValue
        guard double.isFinite,
              double.rounded(.towardZero) == double,
              double >= Double(Int.min),
              double <= Double(Int.max) else {
            return nil
        }
        return Int(double)
    }

    private static func format(decimal: Decimal) -> String {
        NSDecimalNumber(decimal: decimal).stringValue
    }

    enum CalculateError: Error, LocalizedError, Equatable {
        case missingField(String)
        case invalidOperand
        case invalidArity(operation: String)
        case unsupportedOperation(String)
        case divideByZero
        case nonIntegerExponent

        var errorDescription: String? {
            switch self {
            case .missingField(let field):
                return "Missing required field: \(field)."
            case .invalidOperand:
                return "Operands must be numeric values."
            case .invalidArity(let operation):
                return "Invalid operand count for operation: \(operation)."
            case .unsupportedOperation(let operation):
                return "Unsupported operation: \(operation)."
            case .divideByZero:
                return "Division by zero is not allowed."
            case .nonIntegerExponent:
                return "Power requires an integer exponent."
            }
        }
    }
}
