import CarbocationLocalLLM
import Foundation

public enum LLMConvertUnitsTool {
    public static let definition = LLMToolDefinition(
        name: "convert_units",
        description: "Convert between length, mass, volume, temperature, and speed units. Accepts common names and abbreviations such as mile, miles, mi, kilometer, kilometers, and km. Currency is not supported.",
        parameters: .object([
            "type": .string("object"),
            "additionalProperties": .bool(false),
            "required": .array([
                .string("category"),
                .string("value"),
                .string("from_unit"),
                .string("to_unit")
            ]),
            "properties": .object([
                "category": .object([
                    "type": .string("string"),
                    "enum": .array([
                        .string("length"),
                        .string("mass"),
                        .string("volume"),
                        .string("temperature"),
                        .string("speed")
                    ])
                ]),
                "value": .object(["type": .string("number")]),
                "from_unit": .object([
                    "type": .string("string"),
                    "description": .string("Source unit name or abbreviation, for example miles, mile, mi, kilometers, kilometer, km, fahrenheit, or celsius.")
                ]),
                "to_unit": .object([
                    "type": .string("string"),
                    "description": .string("Target unit name or abbreviation, for example miles, mile, mi, kilometers, kilometer, km, fahrenheit, or celsius.")
                ])
            ])
        ])
    )

    public static func tool() -> LLMTool {
        LLMTool(definition: definition) { arguments in
            do {
                return try convert(arguments: arguments)
            } catch let error as ConvertUnitsError {
                return LLMStandardToolResult.failure(message: error.errorDescription ?? "\(error)")
            } catch {
                return LLMStandardToolResult.failure(message: error.localizedDescription)
            }
        }
    }

    public static func convert(arguments: LLMJSONValue) throws -> LLMJSONValue {
        guard let category = arguments.string(forKey: "category") else {
            throw ConvertUnitsError.missingField("category")
        }
        let normalizedCategory = normalizedIdentifier(category)
        guard normalizedCategory != "currency" else {
            throw ConvertUnitsError.unsupportedCategory(category)
        }
        guard let value = arguments.double(forKey: "value"), value.isFinite else {
            throw ConvertUnitsError.missingField("value")
        }
        guard let fromUnitID = arguments.string(forKey: "from_unit") else {
            throw ConvertUnitsError.missingField("from_unit")
        }
        guard let toUnitID = arguments.string(forKey: "to_unit") else {
            throw ConvertUnitsError.missingField("to_unit")
        }

        let convertedValue: Double
        let sourceCanonicalID: String
        let targetCanonicalID: String
        switch normalizedCategory {
        case "length":
            let source = try lengthUnit(fromUnitID)
            let target = try lengthUnit(toUnitID)
            convertedValue = try convert(value, from: source.unit, to: target.unit)
            sourceCanonicalID = source.id
            targetCanonicalID = target.id
        case "mass":
            let source = try massUnit(fromUnitID)
            let target = try massUnit(toUnitID)
            convertedValue = try convert(value, from: source.unit, to: target.unit)
            sourceCanonicalID = source.id
            targetCanonicalID = target.id
        case "volume":
            let source = try volumeUnit(fromUnitID)
            let target = try volumeUnit(toUnitID)
            convertedValue = try convert(value, from: source.unit, to: target.unit)
            sourceCanonicalID = source.id
            targetCanonicalID = target.id
        case "temperature":
            let source = try temperatureUnit(fromUnitID)
            let target = try temperatureUnit(toUnitID)
            convertedValue = try convert(value, from: source.unit, to: target.unit)
            sourceCanonicalID = source.id
            targetCanonicalID = target.id
        case "speed":
            let source = try speedUnit(fromUnitID)
            let target = try speedUnit(toUnitID)
            convertedValue = try convert(value, from: source.unit, to: target.unit)
            sourceCanonicalID = source.id
            targetCanonicalID = target.id
        default:
            throw ConvertUnitsError.unsupportedCategory(category)
        }

        return LLMStandardToolResult.success([
            "category": .string(normalizedCategory),
            "value": .number(value),
            "from_unit": .string(fromUnitID),
            "to_unit": .string(toUnitID),
            "from_unit_canonical": .string(sourceCanonicalID),
            "to_unit_canonical": .string(targetCanonicalID),
            "converted_value": .number(convertedValue),
            "converted_value_text": .string(format(convertedValue))
        ])
    }

    private static func convert<UnitType: Dimension>(
        _ value: Double,
        from sourceUnit: UnitType,
        to targetUnit: UnitType
    ) throws -> Double {
        let measurement = Measurement(value: value, unit: sourceUnit)
        let converted = measurement.converted(to: targetUnit).value
        guard converted.isFinite else {
            throw ConvertUnitsError.nonFiniteResult
        }
        return converted
    }

    private static func lengthUnit(_ id: String) throws -> (unit: UnitLength, id: String) {
        switch normalizedIdentifier(id) {
        case "meter", "meters", "metre", "metres", "m": return (.meters, "meters")
        case "kilometer", "kilometers", "kilometre", "kilometres", "km": return (.kilometers, "kilometers")
        case "centimeter", "centimeters", "centimetre", "centimetres", "cm": return (.centimeters, "centimeters")
        case "millimeter", "millimeters", "millimetre", "millimetres", "mm": return (.millimeters, "millimeters")
        case "inch", "inches", "in": return (.inches, "inches")
        case "foot", "feet", "ft": return (.feet, "feet")
        case "yard", "yards", "yd": return (.yards, "yards")
        case "mile", "miles", "mi": return (.miles, "miles")
        default: throw ConvertUnitsError.unsupportedUnit(id)
        }
    }

    private static func massUnit(_ id: String) throws -> (unit: UnitMass, id: String) {
        switch normalizedIdentifier(id) {
        case "gram", "grams", "g": return (.grams, "grams")
        case "kilogram", "kilograms", "kg": return (.kilograms, "kilograms")
        case "milligram", "milligrams", "mg": return (.milligrams, "milligrams")
        case "ounce", "ounces", "oz": return (.ounces, "ounces")
        case "pound", "pounds", "lb", "lbs": return (.pounds, "pounds")
        default: throw ConvertUnitsError.unsupportedUnit(id)
        }
    }

    private static func volumeUnit(_ id: String) throws -> (unit: UnitVolume, id: String) {
        switch normalizedIdentifier(id) {
        case "liter", "liters", "litre", "litres", "l": return (.liters, "liters")
        case "milliliter", "milliliters", "millilitre", "millilitres", "ml": return (.milliliters, "milliliters")
        case "cup", "cups": return (.cups, "cups")
        case "pint", "pints", "pt": return (.pints, "pints")
        case "quart", "quarts", "qt": return (.quarts, "quarts")
        case "gallon", "gallons", "gal": return (.gallons, "gallons")
        case "fluid_ounce", "fluid_ounces", "fl_oz": return (.fluidOunces, "fluid_ounces")
        default: throw ConvertUnitsError.unsupportedUnit(id)
        }
    }

    private static func temperatureUnit(_ id: String) throws -> (unit: UnitTemperature, id: String) {
        switch normalizedIdentifier(id) {
        case "celsius", "centigrade", "c": return (.celsius, "celsius")
        case "fahrenheit", "f": return (.fahrenheit, "fahrenheit")
        case "kelvin", "k": return (.kelvin, "kelvin")
        default: throw ConvertUnitsError.unsupportedUnit(id)
        }
    }

    private static func speedUnit(_ id: String) throws -> (unit: UnitSpeed, id: String) {
        switch normalizedIdentifier(id) {
        case "meters_per_second", "meter_per_second", "mps", "m_s": return (.metersPerSecond, "meters_per_second")
        case "kilometers_per_hour", "kilometer_per_hour", "kph", "kmh", "km_h": return (.kilometersPerHour, "kilometers_per_hour")
        case "miles_per_hour", "mile_per_hour", "mph": return (.milesPerHour, "miles_per_hour")
        case "knot", "knots", "kt", "kts": return (.knots, "knots")
        default: throw ConvertUnitsError.unsupportedUnit(id)
        }
    }

    private static func normalizedIdentifier(_ id: String) -> String {
        var normalized = id
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "-", with: "_")
            .replacingOccurrences(of: " ", with: "_")
        while normalized.contains("__") {
            normalized = normalized.replacingOccurrences(of: "__", with: "_")
        }
        return normalized
    }

    private static func format(_ value: Double) -> String {
        if value.rounded() == value,
           value >= Double(Int64.min),
           value <= Double(Int64.max) {
            return String(Int64(value))
        }
        return String(format: "%.12g", value)
    }

    enum ConvertUnitsError: Error, LocalizedError, Equatable {
        case missingField(String)
        case unsupportedCategory(String)
        case unsupportedUnit(String)
        case nonFiniteResult

        var errorDescription: String? {
            switch self {
            case .missingField(let field):
                return "Missing or invalid required field: \(field)."
            case .unsupportedCategory(let category):
                return "Unsupported conversion category: \(category)."
            case .unsupportedUnit(let unit):
                return "Unsupported unit identifier: \(unit)."
            case .nonFiniteResult:
                return "Conversion produced a non-finite result."
            }
        }
    }
}
