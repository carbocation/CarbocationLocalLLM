import Foundation

public enum ModelSource: String, Codable, Sendable {
    case curated
    case customHF
    case imported
}

public enum InstalledModelArtifactRole: String, Codable, Sendable {
    case primaryModel
    case splitModel
    case mmproj
}

public struct InstalledModelArtifact: Codable, Hashable, Sendable {
    public var role: InstalledModelArtifactRole
    public var relativePath: String
    public var sizeBytes: Int64
    public var sha256: String?

    public init(
        role: InstalledModelArtifactRole,
        relativePath: String,
        sizeBytes: Int64,
        sha256: String? = nil
    ) {
        self.role = role
        self.relativePath = relativePath
        self.sizeBytes = sizeBytes
        self.sha256 = sha256
    }

    public var filename: String {
        URL(fileURLWithPath: relativePath).lastPathComponent
    }
}

public struct ModelLibraryInstallArtifact: Hashable, Sendable {
    public var sourceURL: URL
    public var role: InstalledModelArtifactRole
    public var relativePath: String
    public var sizeBytes: Int64
    public var sha256: String?
    public var copySource: Bool

    public init(
        sourceURL: URL,
        role: InstalledModelArtifactRole,
        relativePath: String,
        sizeBytes: Int64,
        sha256: String? = nil,
        copySource: Bool = false
    ) {
        self.sourceURL = sourceURL
        self.role = role
        self.relativePath = relativePath
        self.sizeBytes = sizeBytes
        self.sha256 = sha256
        self.copySource = copySource
    }
}

/// Persistent metadata for a locally-installed GGUF model.
/// Stored alongside the primary weights file as `metadata.json`.
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
    public var artifacts: [InstalledModelArtifact]
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
        artifacts: [InstalledModelArtifact],
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
        if artifacts.isEmpty {
            self.artifacts = [
                InstalledModelArtifact(
                    role: .primaryModel,
                    relativePath: filename,
                    sizeBytes: sizeBytes,
                    sha256: sha256
                )
            ]
        } else {
            self.artifacts = artifacts
        }
        self.installedAt = installedAt
    }

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
        self.init(
            id: id,
            displayName: displayName,
            filename: filename,
            sizeBytes: sizeBytes,
            contextLength: contextLength,
            quantization: quantization,
            source: source,
            hfRepo: hfRepo,
            hfFilename: hfFilename,
            sha256: sha256,
            artifacts: [],
            installedAt: installedAt
        )
    }

    private enum CodingKeys: String, CodingKey {
        case id
        case displayName
        case filename
        case sizeBytes
        case contextLength
        case quantization
        case source
        case hfRepo
        case hfFilename
        case sha256
        case artifacts
        case installedAt
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        displayName = try container.decode(String.self, forKey: .displayName)
        filename = try container.decode(String.self, forKey: .filename)
        sizeBytes = try container.decode(Int64.self, forKey: .sizeBytes)
        contextLength = try container.decode(Int.self, forKey: .contextLength)
        quantization = try container.decodeIfPresent(String.self, forKey: .quantization)
        source = try container.decode(ModelSource.self, forKey: .source)
        hfRepo = try container.decodeIfPresent(String.self, forKey: .hfRepo)
        hfFilename = try container.decodeIfPresent(String.self, forKey: .hfFilename)
        sha256 = try container.decodeIfPresent(String.self, forKey: .sha256)
        installedAt = try container.decode(Date.self, forKey: .installedAt)
        artifacts = try container.decodeIfPresent([InstalledModelArtifact].self, forKey: .artifacts) ?? [
            InstalledModelArtifact(
                role: .primaryModel,
                relativePath: filename,
                sizeBytes: sizeBytes,
                sha256: sha256
            )
        ]
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(displayName, forKey: .displayName)
        try container.encode(filename, forKey: .filename)
        try container.encode(sizeBytes, forKey: .sizeBytes)
        try container.encode(contextLength, forKey: .contextLength)
        try container.encodeIfPresent(quantization, forKey: .quantization)
        try container.encode(source, forKey: .source)
        try container.encodeIfPresent(hfRepo, forKey: .hfRepo)
        try container.encodeIfPresent(hfFilename, forKey: .hfFilename)
        try container.encodeIfPresent(sha256, forKey: .sha256)
        try container.encode(artifacts, forKey: .artifacts)
        try container.encode(installedAt, forKey: .installedAt)
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
