import CryptoKit
import Darwin
import Foundation
import OSLog

let modelDownloaderLog = Logger(
    subsystem: "com.carbocation.CarbocationLocalLLM",
    category: "ModelDownloader"
)

public enum ModelDownloaderError: Error, LocalizedError, Sendable {
    case httpStatus(Int)
    case badURL(String)
    case hashMismatch(expected: String, actual: String)
    case noContentLength
    case incompleteResponse
    case cancelled

    public var errorDescription: String? {
        switch self {
        case .httpStatus(let code):
            return "Download failed with HTTP \(code)."
        case .badURL(let value):
            return "Invalid download URL: \(value)"
        case .hashMismatch(let expected, let actual):
            return "SHA256 mismatch; expected \(expected.prefix(12)), got \(actual.prefix(12))."
        case .noContentLength:
            return "Server did not report a content length."
        case .incompleteResponse:
            return "Server closed the connection before the model download completed."
        case .cancelled:
            return "Download cancelled."
        }
    }
}

public struct DownloadProgress: Sendable, Hashable {
    public let bytesReceived: Int64
    public let totalBytes: Int64
    public let bytesPerSecond: Double

    public init(bytesReceived: Int64, totalBytes: Int64, bytesPerSecond: Double) {
        self.bytesReceived = bytesReceived
        self.totalBytes = totalBytes
        self.bytesPerSecond = bytesPerSecond
    }

    public var fractionComplete: Double {
        totalBytes > 0 ? Double(bytesReceived) / Double(totalBytes) : 0
    }
}

public struct PartialDownload: Identifiable, Sendable, Hashable {
    public var id: String
    public var partialURL: URL
    public var sidecarURL: URL
    public var sourceURL: URL
    public var displayName: String
    public var hfRepo: String?
    public var hfFilename: String?
    public var totalBytes: Int64
    public var bytesOnDisk: Int64

    public init(
        id: String,
        partialURL: URL,
        sidecarURL: URL,
        sourceURL: URL,
        displayName: String,
        hfRepo: String?,
        hfFilename: String?,
        totalBytes: Int64,
        bytesOnDisk: Int64
    ) {
        self.id = id
        self.partialURL = partialURL
        self.sidecarURL = sidecarURL
        self.sourceURL = sourceURL
        self.displayName = displayName
        self.hfRepo = hfRepo
        self.hfFilename = hfFilename
        self.totalBytes = totalBytes
        self.bytesOnDisk = bytesOnDisk
    }

    public var fractionComplete: Double {
        totalBytes > 0 ? Double(bytesOnDisk) / Double(totalBytes) : 0
    }
}

public struct ModelDownloadResult: Sendable, Hashable {
    public let tempURL: URL
    public let sizeBytes: Int64
    public let sha256: String?

    public init(tempURL: URL, sizeBytes: Int64, sha256: String?) {
        self.tempURL = tempURL
        self.sizeBytes = sizeBytes
        self.sha256 = sha256
    }
}

public struct HuggingFaceModelDownloadResult: Sendable, Hashable {
    public let resolution: HuggingFaceResolution
    public let artifacts: [ModelLibraryInstallArtifact]
    public let sizeBytes: Int64
    public let primarySHA256: String?

    public init(
        resolution: HuggingFaceResolution,
        artifacts: [ModelLibraryInstallArtifact],
        sizeBytes: Int64,
        primarySHA256: String?
    ) {
        self.resolution = resolution
        self.artifacts = artifacts
        self.sizeBytes = sizeBytes
        self.primarySHA256 = primarySHA256
    }
}

public struct ModelDownloadConfiguration: Sendable, Hashable {
    public static let defaultChunkSize: Int64 = 16 * 1_024 * 1_024
    public static let defaultParallelConnections = 12
    public static let maximumParallelConnections = 32

    public var parallelConnections: Int
    public var chunkSize: Int64
    public var requestTimeout: TimeInterval

    public init(
        parallelConnections: Int = Self.defaultParallelConnections,
        chunkSize: Int64 = Self.defaultChunkSize,
        requestTimeout: TimeInterval = 3_600
    ) {
        self.parallelConnections = min(
            max(1, parallelConnections),
            Self.maximumParallelConnections
        )
        self.chunkSize = max(1_024 * 1_024, chunkSize)
        self.requestTimeout = max(30, requestTimeout)
    }

    public static let `default` = ModelDownloadConfiguration()
}

public enum ModelDownloader {
    static let currentPartialPrefix = "cllm-partial-"
    static let userAgent = "CarbocationLocalLLM/1.0"
    static let sha256ChunkSize = 1 << 20

    public static func huggingFaceResolveURL(hfRepo: String, hfFilename: String) throws -> URL {
        let urlString = "https://huggingface.co/\(hfRepo)/resolve/main/\(hfFilename)?download=true"
        guard let url = URL(string: urlString) else {
            throw ModelDownloaderError.badURL(urlString)
        }
        return url
    }

    public static func download(
        hfRepo: String,
        hfFilename: String,
        modelsRoot: URL,
        displayName: String? = nil,
        expectedSHA256: String? = nil,
        bearerToken: String? = nil,
        configuration: ModelDownloadConfiguration = .default,
        onProgress: @escaping @Sendable (DownloadProgress) -> Void = { _ in }
    ) async throws -> ModelDownloadResult {
        try await download(
            from: huggingFaceResolveURL(hfRepo: hfRepo, hfFilename: hfFilename),
            modelsRoot: modelsRoot,
            displayName: displayName,
            expectedSHA256: expectedSHA256,
            bearerToken: bearerToken,
            configuration: configuration,
            onProgress: onProgress
        )
    }

    public static func download(
        resolution: HuggingFaceResolution,
        modelsRoot: URL,
        expectedPrimarySHA256: String? = nil,
        bearerToken: String? = nil,
        configuration: ModelDownloadConfiguration = .default,
        onProgress: @escaping @Sendable (DownloadProgress) -> Void = { _ in }
    ) async throws -> HuggingFaceModelDownloadResult {
        let aggregateProgress = AggregateDownloadProgressTracker(artifacts: resolution.artifacts)
        var installedArtifacts: [ModelLibraryInstallArtifact] = []
        installedArtifacts.reserveCapacity(resolution.artifacts.count)

        do {
            for artifact in resolution.artifacts {
                let expectedSHA256 = artifact.role == .primaryModel ? expectedPrimarySHA256 : nil
                let result = try await download(
                    from: artifact.url,
                    modelsRoot: modelsRoot,
                    displayName: resolution.displayName,
                    expectedSHA256: expectedSHA256,
                    bearerToken: bearerToken,
                    configuration: configuration
                ) { progress in
                    onProgress(aggregateProgress.update(path: artifact.path, progress: progress))
                }
                installedArtifacts.append(ModelLibraryInstallArtifact(
                    sourceURL: result.tempURL,
                    role: artifact.role.installedModelRole,
                    relativePath: artifact.path,
                    sizeBytes: result.sizeBytes,
                    sha256: result.sha256
                ))
            }
        } catch {
            for artifact in installedArtifacts {
                try? FileManager.default.removeItem(at: artifact.sourceURL)
            }
            throw error
        }

        let totalSize = installedArtifacts.reduce(Int64(0)) { $0 + $1.sizeBytes }
        onProgress(DownloadProgress(
            bytesReceived: totalSize,
            totalBytes: max(totalSize, resolution.totalSizeBytes),
            bytesPerSecond: 0
        ))
        return HuggingFaceModelDownloadResult(
            resolution: resolution,
            artifacts: installedArtifacts,
            sizeBytes: totalSize,
            primarySHA256: installedArtifacts.first(where: { $0.role == .primaryModel })?.sha256
        )
    }

    /// Downloads a GGUF file to the model cache's `.partials` directory.
    /// Range-capable servers use a chunked v2 sidecar and parallel workers.
    /// Servers without Range support fall back to single-stream resume.
    public static func download(
        from url: URL,
        modelsRoot: URL,
        displayName: String? = nil,
        expectedSHA256: String? = nil,
        bearerToken: String? = nil,
        configuration: ModelDownloadConfiguration = .default,
        onProgress: @escaping @Sendable (DownloadProgress) -> Void = { _ in }
    ) async throws -> ModelDownloadResult {
        let (partialURL, sidecarURL) = try partialPaths(for: url, modelsRoot: modelsRoot)

        if let (plan, sidecar) = loadChunkedState(
            partialURL: partialURL,
            sidecarURL: sidecarURL,
            sourceURL: url
        ) {
            var updatedSidecar = sidecar
            if updatedSidecar.displayName == nil, let displayName {
                updatedSidecar.displayName = displayName
                writeSidecarValue(updatedSidecar, to: sidecarURL)
            }

            modelDownloaderLog.info(
                "Resuming chunked \(url.lastPathComponent, privacy: .public) (\(plan.doneChunks.count)/\(plan.chunkCount) chunks done)"
            )

            return try await downloadParallel(
                url: url,
                partialURL: partialURL,
                sidecarURL: sidecarURL,
                plan: plan,
                sidecar: updatedSidecar,
                configuration: configuration,
                expectedSHA256: expectedSHA256,
                bearerToken: bearerToken,
                onProgress: onProgress
            )
        }

        let legacy = loadSingleStreamState(
            partialURL: partialURL,
            sidecarURL: sidecarURL,
            sourceURL: url
        )

        guard let probe = try await probeServer(url: url, bearerToken: bearerToken) else {
            modelDownloaderLog.info(
                "Server does not support Range for \(url.lastPathComponent, privacy: .public); using single stream"
            )
            return try await downloadSingleStream(
                url: url,
                partialURL: partialURL,
                sidecarURL: sidecarURL,
                prior: legacy,
                displayName: displayName,
                expectedSHA256: expectedSHA256,
                bearerToken: bearerToken,
                onProgress: onProgress
            )
        }

        let totalBytes = probe.totalBytes
        var doneChunks = Set<Int>()
        if let legacy {
            if legacy.totalBytes == totalBytes {
                let fullChunks = legacy.existingSize / configuration.chunkSize
                for index in 0..<Int(fullChunks) {
                    doneChunks.insert(index)
                }
                modelDownloaderLog.info("Credited \(fullChunks) legacy chunks from single-stream partial")
            } else {
                discardPartial(partialURL: partialURL, sidecarURL: sidecarURL)
            }
        }

        let plan = ChunkPlan(
            totalBytes: totalBytes,
            chunkSize: configuration.chunkSize,
            doneChunks: doneChunks
        )
        let sidecar = PartialSidecar(
            url: url.absoluteString,
            etag: probe.etag,
            lastModified: probe.lastModified,
            totalBytes: totalBytes,
            displayName: displayName,
            schemaVersion: 2,
            chunkSize: configuration.chunkSize,
            doneChunks: Array(doneChunks).sorted()
        )

        if !FileManager.default.fileExists(atPath: partialURL.path) {
            FileManager.default.createFile(atPath: partialURL.path, contents: nil)
        }
        let allocationHandle = try FileHandle(forWritingTo: partialURL)
        try allocationHandle.truncate(atOffset: UInt64(totalBytes))
        try allocationHandle.close()

        writeSidecarValue(sidecar, to: sidecarURL)

        modelDownloaderLog.info(
            "Starting parallel download \(url.lastPathComponent, privacy: .public) (\(totalBytes) bytes, \(plan.chunkCount) chunks, \(configuration.parallelConnections) connections)"
        )

        return try await downloadParallel(
            url: url,
            partialURL: partialURL,
            sidecarURL: sidecarURL,
            plan: plan,
            sidecar: sidecar,
            configuration: configuration,
            expectedSHA256: expectedSHA256,
            bearerToken: bearerToken,
            onProgress: onProgress
        )
    }

}

private extension HuggingFaceArtifactRole {
    var installedModelRole: InstalledModelArtifactRole {
        switch self {
        case .primaryModel:
            return .primaryModel
        case .splitModel:
            return .splitModel
        case .mmproj:
            return .mmproj
        }
    }
}
