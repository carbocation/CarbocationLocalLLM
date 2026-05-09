import CryptoKit
import Foundation

extension ModelDownloader {
    static func updateHasher(_ hasher: inout SHA256?, withPrefixOf url: URL, byteCount: Int64) throws {
        guard hasher != nil else { return }

        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        var remaining = byteCount
        while remaining > 0 {
            let count = Int(min(Int64(sha256ChunkSize), remaining))
            let bytesRead = try autoreleasepool {
                guard let chunk = try handle.read(upToCount: count), !chunk.isEmpty else { return 0 }
                updateHasher(&hasher, data: chunk)
                return chunk.count
            }
            guard bytesRead > 0 else { break }
            remaining -= Int64(bytesRead)
        }
    }

    static func updateHasher(_ hasher: inout SHA256?, data: Data) {
        guard var activeHasher = hasher else { return }
        activeHasher.update(data: data)
        hasher = activeHasher
    }

    static func emitProgressIfNeeded(
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

    static func verifyFinalHash(at url: URL, expected: String?) throws -> String? {
        guard let expected = expected.normalizedNonEmptySHA256 else { return nil }
        let actual = try computeSHA256(at: url)
        if expected != actual.lowercased() {
            throw ModelDownloaderError.hashMismatch(expected: expected, actual: actual)
        }
        return actual
    }

    static func finalizeHash(_ hasher: inout SHA256?, expected: String?) throws -> String? {
        guard let activeHasher = hasher else { return nil }
        let actual = activeHasher.finalize().map { String(format: "%02x", $0) }.joined()
        hasher = nil
        if let expected = expected.normalizedNonEmptySHA256,
           expected != actual.lowercased() {
            throw ModelDownloaderError.hashMismatch(expected: expected, actual: actual)
        }
        return actual
    }

    private static func computeSHA256(at url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        var hasher = SHA256()
        while true {
            let bytesRead = try autoreleasepool {
                guard let chunk = try handle.read(upToCount: sha256ChunkSize), !chunk.isEmpty else { return 0 }
                hasher.update(data: chunk)
                return chunk.count
            }
            guard bytesRead > 0 else { break }
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    static func parseContentRangeTotal(_ response: HTTPURLResponse) -> Int64? {
        guard let header = response.value(forHTTPHeaderField: "Content-Range"),
              let slash = header.lastIndex(of: "/")
        else { return nil }
        return Int64(header[header.index(after: slash)...].trimmingCharacters(in: .whitespaces))
    }

    static func parseHFCoords(from url: URL) -> (repo: String, filename: String)? {
        guard url.host?.contains("huggingface.co") == true else { return nil }
        let parts = url.pathComponents.filter { $0 != "/" }
        guard parts.count >= 5, parts[2] == "resolve" || parts[2] == "blob" else { return nil }
        return ("\(parts[0])/\(parts[1])", parts.suffix(from: 4).joined(separator: "/"))
    }

    static func applyStandardHeaders(to request: inout URLRequest, bearerToken: String?) {
        request.setValue(userAgent, forHTTPHeaderField: "User-Agent")
        if let token = bearerToken?.trimmingCharacters(in: .whitespacesAndNewlines),
           !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
    }
}

extension Optional where Wrapped == String {
    var hasNonEmptyValue: Bool {
        normalizedNonEmptySHA256 != nil
    }

    var normalizedNonEmptySHA256: String? {
        guard let value = self?.trimmingCharacters(in: .whitespacesAndNewlines),
              !value.isEmpty
        else { return nil }
        return value.lowercased()
    }
}
