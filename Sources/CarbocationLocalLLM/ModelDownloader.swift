import CryptoKit
import Foundation

public enum ModelDownloaderError: Error, LocalizedError, Sendable {
    case httpStatus(Int)
    case badURL(String)
    case hashMismatch(expected: String, actual: String)
    case noContentLength
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
    public let sha256: String

    public init(tempURL: URL, sizeBytes: Int64, sha256: String) {
        self.tempURL = tempURL
        self.sizeBytes = sizeBytes
        self.sha256 = sha256
    }
}

private struct PartialSidecar: Codable {
    var url: String
    var etag: String?
    var lastModified: String?
    var totalBytes: Int64
    var displayName: String?
    var schemaVersion: Int?
}

private struct PartialDownloadState {
    let existingSize: Int64
    let totalBytes: Int64
    let etag: String?
    let lastModified: String?
}

public enum ModelDownloader {
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
        onProgress: @escaping @Sendable (DownloadProgress) -> Void = { _ in }
    ) async throws -> ModelDownloadResult {
        try await download(
            from: huggingFaceResolveURL(hfRepo: hfRepo, hfFilename: hfFilename),
            modelsRoot: modelsRoot,
            displayName: displayName,
            expectedSHA256: expectedSHA256,
            onProgress: onProgress
        )
    }

    /// Downloads a GGUF file to the model cache's `.partials` directory.
    /// The returned file should be registered with `ModelLibrary.add(...)`,
    /// which moves it into its final per-model directory.
    public static func download(
        from url: URL,
        modelsRoot: URL,
        displayName: String? = nil,
        expectedSHA256: String? = nil,
        onProgress: @escaping @Sendable (DownloadProgress) -> Void = { _ in }
    ) async throws -> ModelDownloadResult {
        let (partialURL, sidecarURL) = try partialPaths(for: url, modelsRoot: modelsRoot)
        let prior = loadPartialState(partialURL: partialURL, sidecarURL: sidecarURL, sourceURL: url)
        var request = URLRequest(url: url)
        request.timeoutInterval = 3_600
        request.setValue("CarbocationLocalLLM/1.0", forHTTPHeaderField: "User-Agent")

        if let prior {
            request.setValue("bytes=\(prior.existingSize)-", forHTTPHeaderField: "Range")
            if let validator = prior.etag ?? prior.lastModified {
                request.setValue(validator, forHTTPHeaderField: "If-Range")
            }
        }

        let (stream, response) = try await URLSession.shared.bytes(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw ModelDownloaderError.httpStatus(-1)
        }
        guard (200...299).contains(http.statusCode) else {
            throw ModelDownloaderError.httpStatus(http.statusCode)
        }

        let isResume = http.statusCode == 206
        let existingSize: Int64
        let totalBytes: Int64
        if isResume, let prior {
            existingSize = prior.existingSize
            totalBytes = parseContentRangeTotal(http) ?? prior.totalBytes
        } else {
            existingSize = 0
            totalBytes = http.expectedContentLength
            discardPartial(partialURL: partialURL, sidecarURL: sidecarURL)
            try FileManager.default.createDirectory(
                at: partialURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            FileManager.default.createFile(atPath: partialURL.path, contents: nil)
        }

        if totalBytes > 0 {
            writeSidecar(
                to: sidecarURL,
                sourceURL: url,
                totalBytes: totalBytes,
                etag: http.value(forHTTPHeaderField: "ETag"),
                lastModified: http.value(forHTTPHeaderField: "Last-Modified"),
                displayName: displayName
            )
        }

        var hasher = SHA256()
        if existingSize > 0 {
            try updateHasher(&hasher, withPrefixOf: partialURL, byteCount: existingSize)
        }

        let handle = try FileHandle(forWritingTo: partialURL)
        defer { try? handle.close() }
        if isResume {
            try handle.seekToEnd()
        }

        let started = Date()
        var lastEmit = Date(timeIntervalSince1970: 0)
        var received = existingSize
        var buffer: [UInt8] = []
        buffer.reserveCapacity(1 << 20)

        do {
            for try await byte in stream {
                try Task.checkCancellation()
                buffer.append(byte)
                if buffer.count >= (1 << 20) {
                    let data = Data(buffer)
                    try handle.write(contentsOf: data)
                    hasher.update(data: data)
                    received += Int64(data.count)
                    emitProgressIfNeeded(
                        received: received,
                        total: totalBytes,
                        started: started,
                        lastEmit: &lastEmit,
                        onProgress: onProgress
                    )
                    buffer.removeAll(keepingCapacity: true)
                }
            }

            if !buffer.isEmpty {
                let data = Data(buffer)
                try handle.write(contentsOf: data)
                hasher.update(data: data)
                received += Int64(data.count)
            }
        } catch is CancellationError {
            throw ModelDownloaderError.cancelled
        }

        let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        if let expected = expectedSHA256?.lowercased(),
           !expected.isEmpty,
           expected != digest.lowercased() {
            discardPartial(partialURL: partialURL, sidecarURL: sidecarURL)
            throw ModelDownloaderError.hashMismatch(expected: expected, actual: digest)
        }

        try? FileManager.default.removeItem(at: sidecarURL)
        onProgress(DownloadProgress(bytesReceived: received, totalBytes: max(totalBytes, received), bytesPerSecond: 0))
        return ModelDownloadResult(tempURL: partialURL, sizeBytes: received, sha256: digest)
    }

    public static func partialsDirectory(in modelsRoot: URL) throws -> URL {
        let directory = modelsRoot.appendingPathComponent(".partials", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    public static func listPartials(in modelsRoot: URL) -> [PartialDownload] {
        guard let partialsRoot = try? partialsDirectory(in: modelsRoot),
              let entries = try? FileManager.default.contentsOfDirectory(
                at: partialsRoot,
                includingPropertiesForKeys: [.fileSizeKey],
                options: []
              )
        else { return [] }

        var result: [PartialDownload] = []
        for sidecarURL in entries where sidecarURL.pathExtension == "json" && sidecarURL.lastPathComponent.hasPrefix("cllm-partial-") {
            let stem = sidecarURL.deletingPathExtension().lastPathComponent
            let partialURL = sidecarURL.deletingLastPathComponent().appendingPathComponent("\(stem).gguf")
            guard FileManager.default.fileExists(atPath: partialURL.path) else {
                try? FileManager.default.removeItem(at: sidecarURL)
                continue
            }
            guard let data = try? Data(contentsOf: sidecarURL),
                  let sidecar = try? JSONDecoder().decode(PartialSidecar.self, from: data),
                  let sourceURL = URL(string: sidecar.url),
                  sidecar.totalBytes > 0
            else { continue }

            let key = String(stem.dropFirst("cllm-partial-".count))
            let bytesOnDisk = (try? FileManager.default.attributesOfItem(atPath: partialURL.path)[.size] as? Int64) ?? 0
            let (hfRepo, hfFilename) = parseHFCoords(from: sourceURL) ?? (nil, nil)
            let displayName = sidecar.displayName
                ?? hfFilename?.replacingOccurrences(of: ".gguf", with: "")
                ?? sourceURL.lastPathComponent

            result.append(PartialDownload(
                id: key,
                partialURL: partialURL,
                sidecarURL: sidecarURL,
                sourceURL: sourceURL,
                displayName: displayName,
                hfRepo: hfRepo,
                hfFilename: hfFilename,
                totalBytes: sidecar.totalBytes,
                bytesOnDisk: min(bytesOnDisk, sidecar.totalBytes)
            ))
        }

        return result.sorted {
            $0.displayName.localizedCaseInsensitiveCompare($1.displayName) == .orderedAscending
        }
    }

    public static func deletePartial(_ partial: PartialDownload) {
        discardPartial(partialURL: partial.partialURL, sidecarURL: partial.sidecarURL)
    }

    private static func partialPaths(for url: URL, modelsRoot: URL) throws -> (partial: URL, sidecar: URL) {
        let directory = try partialsDirectory(in: modelsRoot)
        let key = partialKey(for: url)
        return (
            directory.appendingPathComponent("cllm-partial-\(key).gguf"),
            directory.appendingPathComponent("cllm-partial-\(key).json")
        )
    }

    private static func partialKey(for url: URL) -> String {
        SHA256.hash(data: Data(url.absoluteString.utf8))
            .prefix(6)
            .map { String(format: "%02x", $0) }
            .joined()
    }

    private static func loadPartialState(
        partialURL: URL,
        sidecarURL: URL,
        sourceURL: URL
    ) -> PartialDownloadState? {
        guard FileManager.default.fileExists(atPath: partialURL.path),
              FileManager.default.fileExists(atPath: sidecarURL.path),
              let data = try? Data(contentsOf: sidecarURL),
              let sidecar = try? JSONDecoder().decode(PartialSidecar.self, from: data),
              sidecar.url == sourceURL.absoluteString,
              sidecar.totalBytes > 0
        else { return nil }

        let size = (try? FileManager.default.attributesOfItem(atPath: partialURL.path)[.size] as? Int64) ?? 0
        guard size > 0, size < sidecar.totalBytes else {
            discardPartial(partialURL: partialURL, sidecarURL: sidecarURL)
            return nil
        }

        return PartialDownloadState(
            existingSize: size,
            totalBytes: sidecar.totalBytes,
            etag: sidecar.etag,
            lastModified: sidecar.lastModified
        )
    }

    private static func writeSidecar(
        to url: URL,
        sourceURL: URL,
        totalBytes: Int64,
        etag: String?,
        lastModified: String?,
        displayName: String?
    ) {
        let sidecar = PartialSidecar(
            url: sourceURL.absoluteString,
            etag: etag,
            lastModified: lastModified,
            totalBytes: totalBytes,
            displayName: displayName,
            schemaVersion: 1
        )
        if let data = try? JSONEncoder().encode(sidecar) {
            try? data.write(to: url, options: .atomic)
        }
    }

    private static func discardPartial(partialURL: URL, sidecarURL: URL) {
        try? FileManager.default.removeItem(at: partialURL)
        try? FileManager.default.removeItem(at: sidecarURL)
    }

    private static func updateHasher(_ hasher: inout SHA256, withPrefixOf url: URL, byteCount: Int64) throws {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        var remaining = byteCount
        while remaining > 0 {
            let count = Int(min(Int64(1 << 20), remaining))
            guard let chunk = try handle.read(upToCount: count), !chunk.isEmpty else { break }
            hasher.update(data: chunk)
            remaining -= Int64(chunk.count)
        }
    }

    private static func emitProgressIfNeeded(
        received: Int64,
        total: Int64,
        started: Date,
        lastEmit: inout Date,
        onProgress: @escaping @Sendable (DownloadProgress) -> Void
    ) {
        let now = Date()
        guard now.timeIntervalSince(lastEmit) >= 0.25 else { return }
        lastEmit = now
        let elapsed = now.timeIntervalSince(started)
        let bytesPerSecond = elapsed > 0 ? Double(received) / elapsed : 0
        onProgress(DownloadProgress(bytesReceived: received, totalBytes: total, bytesPerSecond: bytesPerSecond))
    }

    private static func parseContentRangeTotal(_ response: HTTPURLResponse) -> Int64? {
        guard let header = response.value(forHTTPHeaderField: "Content-Range"),
              let slash = header.lastIndex(of: "/")
        else { return nil }
        return Int64(header[header.index(after: slash)...].trimmingCharacters(in: .whitespaces))
    }

    private static func parseHFCoords(from url: URL) -> (repo: String, filename: String)? {
        guard url.host?.contains("huggingface.co") == true else { return nil }
        let parts = url.pathComponents.filter { $0 != "/" }
        guard parts.count >= 5, parts[2] == "resolve" || parts[2] == "blob" else { return nil }
        return ("\(parts[0])/\(parts[1])", parts.last ?? "")
    }
}

public enum HuggingFaceURL {
    public static func parse(_ input: String) -> (repo: String, filename: String)? {
        let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        if let url = URL(string: trimmed),
           let host = url.host,
           host.contains("huggingface.co") {
            let parts = url.pathComponents.filter { $0 != "/" }
            if parts.count >= 5, parts[2] == "resolve" || parts[2] == "blob" {
                let repo = "\(parts[0])/\(parts[1])"
                let filename = parts.last ?? ""
                if filename.lowercased().hasSuffix(".gguf") {
                    return (repo, filename)
                }
            }
            return nil
        }

        let parts = trimmed.split(separator: "/", omittingEmptySubsequences: true).map(String.init)
        if parts.count >= 3, parts.last?.lowercased().hasSuffix(".gguf") == true {
            return ("\(parts[0])/\(parts[1])", parts.suffix(from: 2).joined(separator: "/"))
        }
        return nil
    }
}

