import Foundation

public enum ModelSource: String, Codable, Sendable {
    case curated
    case customHF
    case imported
}

/// Persistent metadata for a locally-installed GGUF model.
/// Stored alongside the weights file as `metadata.json`.
public struct InstalledModel: Codable, Identifiable, Hashable, Sendable {
    public var id: UUID
    public var displayName: String
    public var filename: String
    public var sizeBytes: Int64
    /// Training context (`n_ctx_train`) when known, otherwise `0`.
    public var contextLength: Int
    public var quantization: String?
    public var source: ModelSource
    public var hfRepo: String?
    public var hfFilename: String?
    public var sha256: String?
    public var installedAt: Date

    public init(
        id: UUID = UUID(),
        displayName: String,
        filename: String,
        sizeBytes: Int64,
        contextLength: Int,
        quantization: String?,
        source: ModelSource,
        hfRepo: String? = nil,
        hfFilename: String? = nil,
        sha256: String? = nil,
        installedAt: Date = Date()
    ) {
        self.id = id
        self.displayName = displayName
        self.filename = filename
        self.sizeBytes = sizeBytes
        self.contextLength = contextLength
        self.quantization = quantization
        self.source = source
        self.hfRepo = hfRepo
        self.hfFilename = hfFilename
        self.sha256 = sha256
        self.installedAt = installedAt
    }

    public func directory(in root: URL) -> URL {
        root.appendingPathComponent(id.uuidString, isDirectory: true)
    }

    public func weightsURL(in root: URL) -> URL {
        directory(in: root).appendingPathComponent(filename)
    }

    public func metadataURL(in root: URL) -> URL {
        directory(in: root).appendingPathComponent("metadata.json")
    }

    public static func inferQuantization(from filename: String) -> String? {
        let tokens = filename
            .lowercased()
            .replacingOccurrences(of: ".gguf", with: "")
            .split { !$0.isLetter && !$0.isNumber && $0 != "_" }
            .map(String.init)

        for token in tokens.reversed() {
            guard token.hasPrefix("q"),
                  token.count >= 2,
                  token.dropFirst().first?.isNumber == true
            else { continue }
            return token.uppercased()
        }
        return nil
    }
}

