import Foundation

public enum GGUFMetadata {
    public static func trainingContextLength(at url: URL) -> Int? {
        (try? GGUFMetadataReader(url: url).trainingContextLength()).flatMap { $0 }
    }

    public static func trainingContextLength(atPath path: String) -> Int? {
        trainingContextLength(at: URL(fileURLWithPath: path))
    }
}

private enum GGUFMetadataReaderError: Error {
    case invalidFormat
    case unsupportedType
}

private enum GGUFValueType: Int32 {
    case uint8 = 0
    case int8 = 1
    case uint16 = 2
    case int16 = 3
    case uint32 = 4
    case int32 = 5
    case float32 = 6
    case bool = 7
    case string = 8
    case array = 9
    case uint64 = 10
    case int64 = 11
    case float64 = 12

    var fixedWidth: UInt64? {
        switch self {
        case .uint8, .int8, .bool:
            return 1
        case .uint16, .int16:
            return 2
        case .uint32, .int32, .float32:
            return 4
        case .uint64, .int64, .float64:
            return 8
        case .string, .array:
            return nil
        }
    }

    var isInteger: Bool {
        switch self {
        case .uint8, .int8, .uint16, .int16, .uint32, .int32, .uint64, .int64:
            return true
        case .float32, .bool, .string, .array, .float64:
            return false
        }
    }
}

private final class GGUFMetadataReader {
    private static let magic = Data([0x47, 0x47, 0x55, 0x46])
    private static let maxKeyLength: UInt64 = 16 * 1_024
    private static let maxVersion: UInt32 = 3

    private let handle: FileHandle
    private let fileSize: UInt64

    init(url: URL) throws {
        handle = try FileHandle(forReadingFrom: url)
        let sizeValue = try FileManager.default.attributesOfItem(atPath: url.path)[.size]
        fileSize = Self.int64Size(from: sizeValue)
    }

    deinit {
        try? handle.close()
    }

    func trainingContextLength() throws -> Int? {
        guard try readData(count: 4) == Self.magic else {
            return nil
        }

        let version = try readUInt32()
        guard version >= 2, version <= Self.maxVersion else {
            return nil
        }

        _ = try readNonNegativeInt64()
        let keyValueCount = try readNonNegativeInt64()

        for _ in 0..<keyValueCount {
            let key = try readString(maxLength: Self.maxKeyLength)
            let rawValueType = try readInt32()

            guard let valueType = GGUFValueType(rawValue: rawValueType) else {
                throw GGUFMetadataReaderError.unsupportedType
            }

            if valueType == .array {
                try skipArray()
                continue
            }

            if key.hasSuffix(".context_length") {
                if let value = try readIntegerValue(type: valueType),
                   value > 0,
                   value <= Int64(Int.max) {
                    return Int(value)
                }
                if !valueType.isInteger {
                    try skipScalarValue(type: valueType)
                }
                continue
            }

            try skipScalarValue(type: valueType)
        }

        return nil
    }

    private static func int64Size(from value: Any?) -> UInt64 {
        if let number = value as? NSNumber {
            return max(0, number.int64Value).magnitude
        }
        if let value = value as? Int64 {
            return max(0, value).magnitude
        }
        if let value = value as? UInt64 {
            return value
        }
        if let value = value as? Int {
            return UInt64(max(0, value))
        }
        return 0
    }

    private func readIntegerValue(type: GGUFValueType) throws -> Int64? {
        switch type {
        case .uint8:
            return Int64(try readUInt8())
        case .int8:
            return Int64(try readInt8())
        case .uint16:
            return Int64(try readUInt16())
        case .int16:
            return Int64(try readInt16())
        case .uint32:
            return Int64(try readUInt32())
        case .int32:
            return Int64(try readInt32())
        case .uint64:
            let value = try readUInt64()
            guard value <= UInt64(Int64.max) else { return nil }
            return Int64(value)
        case .int64:
            return try readInt64()
        case .float32, .float64, .bool, .string, .array:
            return nil
        }
    }

    private func skipScalarValue(type: GGUFValueType) throws {
        if type == .string {
            try skipString()
            return
        }

        guard let width = type.fixedWidth else {
            throw GGUFMetadataReaderError.unsupportedType
        }
        try skip(byteCount: width)
    }

    private func skipArray() throws {
        let rawElementType = try readInt32()
        let elementCount = try readUInt64()
        guard let elementType = GGUFValueType(rawValue: rawElementType),
              elementType != .array
        else {
            throw GGUFMetadataReaderError.unsupportedType
        }

        if elementType == .string {
            for _ in 0..<elementCount {
                try skipString()
            }
            return
        }

        guard let width = elementType.fixedWidth,
              elementCount <= UInt64.max / width
        else {
            throw GGUFMetadataReaderError.invalidFormat
        }
        try skip(byteCount: elementCount * width)
    }

    private func readString(maxLength: UInt64) throws -> String {
        let length = try readUInt64()
        guard length <= maxLength,
              length <= UInt64(Int.max)
        else {
            throw GGUFMetadataReaderError.invalidFormat
        }

        let data = try readData(count: Int(length))
        guard let string = String(data: data, encoding: .utf8) else {
            throw GGUFMetadataReaderError.invalidFormat
        }
        return string
    }

    private func skipString() throws {
        try skip(byteCount: readUInt64())
    }

    private func readData(count: Int) throws -> Data {
        guard count >= 0,
              let data = try handle.read(upToCount: count),
              data.count == count
        else {
            throw GGUFMetadataReaderError.invalidFormat
        }
        return data
    }

    private func skip(byteCount: UInt64) throws {
        let offset = try handle.offset()
        guard offset <= fileSize,
              byteCount <= fileSize - offset
        else {
            throw GGUFMetadataReaderError.invalidFormat
        }
        try handle.seek(toOffset: offset + byteCount)
    }

    private func readUInt8() throws -> UInt8 {
        try readBytes(count: 1)[0]
    }

    private func readInt8() throws -> Int8 {
        Int8(bitPattern: try readUInt8())
    }

    private func readUInt16() throws -> UInt16 {
        let bytes = try readBytes(count: 2)
        return UInt16(bytes[0]) | UInt16(bytes[1]) << 8
    }

    private func readInt16() throws -> Int16 {
        Int16(bitPattern: try readUInt16())
    }

    private func readUInt32() throws -> UInt32 {
        let bytes = try readBytes(count: 4)
        return UInt32(bytes[0])
            | UInt32(bytes[1]) << 8
            | UInt32(bytes[2]) << 16
            | UInt32(bytes[3]) << 24
    }

    private func readInt32() throws -> Int32 {
        Int32(bitPattern: try readUInt32())
    }

    private func readUInt64() throws -> UInt64 {
        let bytes = try readBytes(count: 8)
        return UInt64(bytes[0])
            | UInt64(bytes[1]) << 8
            | UInt64(bytes[2]) << 16
            | UInt64(bytes[3]) << 24
            | UInt64(bytes[4]) << 32
            | UInt64(bytes[5]) << 40
            | UInt64(bytes[6]) << 48
            | UInt64(bytes[7]) << 56
    }

    private func readInt64() throws -> Int64 {
        Int64(bitPattern: try readUInt64())
    }

    private func readNonNegativeInt64() throws -> Int64 {
        let value = try readInt64()
        guard value >= 0 else {
            throw GGUFMetadataReaderError.invalidFormat
        }
        return value
    }

    private func readBytes(count: Int) throws -> [UInt8] {
        Array(try readData(count: count))
    }
}
