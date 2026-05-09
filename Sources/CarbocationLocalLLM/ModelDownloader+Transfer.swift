import CryptoKit
import Foundation

extension ModelDownloader {
    static func downloadParallel(
        url: URL,
        partialURL: URL,
        sidecarURL: URL,
        plan: ChunkPlan,
        sidecar: PartialSidecar,
        configuration: ModelDownloadConfiguration,
        expectedSHA256: String?,
        bearerToken: String?,
        onProgress: @escaping @Sendable (DownloadProgress) -> Void
    ) async throws -> ModelDownloadResult {
        let alreadyHad = plan.completedBytes()
        let queue = ChunkWorkQueue(
            pending: plan.pendingRanges(),
            done: plan.doneChunks,
            sidecarURL: sidecarURL,
            sidecar: sidecar
        )
        let tracker = ProgressTracker(alreadyHad: alreadyHad, totalBytes: plan.totalBytes)
        let writer = try RandomAccessFileWriter(url: partialURL)
        let validator = sidecar.etag ?? sidecar.lastModified
        let delegate = RangedDownloadDelegate()
        let session = makeURLSession(configuration: configuration, delegate: delegate)

        do {
            try await withThrowingTaskGroup(of: Void.self) { group in
                for _ in 0..<configuration.parallelConnections {
                    group.addTask {
                        while let chunk = await queue.nextChunk() {
                            try Task.checkCancellation()
                            try await downloadChunk(
                                chunk: chunk,
                                from: url,
                                validator: validator,
                                session: session,
                                delegate: delegate,
                                requestTimeout: configuration.requestTimeout,
                                bearerToken: bearerToken,
                                writer: writer,
                                queue: queue,
                                tracker: tracker,
                                onProgress: onProgress
                            )
                        }
                    }
                }
                try await group.waitForAll()
            }
            try writer.close()
            session.finishTasksAndInvalidate()
        } catch is CancellationError {
            try? writer.close()
            session.invalidateAndCancel()
            await queue.flush()
            throw ModelDownloaderError.cancelled
        } catch let error as URLError where error.code == .cancelled {
            try? writer.close()
            session.invalidateAndCancel()
            await queue.flush()
            throw ModelDownloaderError.cancelled
        } catch {
            try? writer.close()
            session.invalidateAndCancel()
            await queue.flush()
            throw error
        }

        await queue.flush()
        let completedCount = await queue.completedCount
        guard completedCount == plan.chunkCount else {
            throw ModelDownloaderError.httpStatus(-2)
        }

        let digest: String?
        do {
            digest = try verifyFinalHash(at: partialURL, expected: expectedSHA256)
        } catch {
            discardPartial(partialURL: partialURL, sidecarURL: sidecarURL)
            throw error
        }

        try? FileManager.default.removeItem(at: sidecarURL)
        onProgress(DownloadProgress(
            bytesReceived: plan.totalBytes,
            totalBytes: plan.totalBytes,
            bytesPerSecond: 0
        ))

        return ModelDownloadResult(tempURL: partialURL, sizeBytes: plan.totalBytes, sha256: digest)
    }

    private static func downloadChunk(
        chunk: ChunkRange,
        from url: URL,
        validator: String?,
        session: URLSession,
        delegate: RangedDownloadDelegate,
        requestTimeout: TimeInterval,
        bearerToken: String?,
        writer: RandomAccessFileWriter,
        queue: ChunkWorkQueue,
        tracker: ProgressTracker,
        onProgress: @escaping @Sendable (DownloadProgress) -> Void
    ) async throws {
        var request = URLRequest(url: url)
        applyStandardHeaders(to: &request, bearerToken: bearerToken)
        request.setValue("bytes=\(chunk.start)-\(chunk.end)", forHTTPHeaderField: "Range")
        if let validator {
            request.setValue(validator, forHTTPHeaderField: "If-Range")
        }
        request.timeoutInterval = requestTimeout

        try Task.checkCancellation()
        try await delegate.download(
            request: request,
            chunk: chunk,
            session: session,
            writer: writer,
            tracker: tracker,
            onProgress: onProgress
        )
        await queue.markDone(chunk.index)
        tracker.maybeEmit(onProgress)
    }

    private static func makeURLSession(
        configuration: ModelDownloadConfiguration,
        delegate: URLSessionDataDelegate? = nil
    ) -> URLSession {
        let urlSessionConfiguration = URLSessionConfiguration.default
        urlSessionConfiguration.httpMaximumConnectionsPerHost = configuration.parallelConnections
        urlSessionConfiguration.requestCachePolicy = .reloadIgnoringLocalCacheData
        urlSessionConfiguration.timeoutIntervalForRequest = configuration.requestTimeout
        urlSessionConfiguration.timeoutIntervalForResource = configuration.requestTimeout
        urlSessionConfiguration.urlCache = nil
        urlSessionConfiguration.waitsForConnectivity = true

        let delegateQueue = OperationQueue()
        delegateQueue.name = "com.carbocation.CarbocationLocalLLM.ModelDownloader"
        delegateQueue.maxConcurrentOperationCount = 1

        return URLSession(
            configuration: urlSessionConfiguration,
            delegate: delegate,
            delegateQueue: delegate == nil ? nil : delegateQueue
        )
    }

    struct ProbeResult: Sendable {
        let totalBytes: Int64
        let etag: String?
        let lastModified: String?
    }

    static func probeServer(url: URL, bearerToken: String?) async throws -> ProbeResult? {
        var request = URLRequest(url: url)
        applyStandardHeaders(to: &request, bearerToken: bearerToken)
        request.setValue("bytes=0-0", forHTTPHeaderField: "Range")
        request.timeoutInterval = 30

        let (_, response) = try await URLSession.shared.bytes(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 206 else {
            return nil
        }
        guard let totalBytes = parseContentRangeTotal(http), totalBytes > 0 else {
            return nil
        }

        return ProbeResult(
            totalBytes: totalBytes,
            etag: http.value(forHTTPHeaderField: "ETag"),
            lastModified: http.value(forHTTPHeaderField: "Last-Modified")
        )
    }

    static func downloadSingleStream(
        url: URL,
        partialURL: URL,
        sidecarURL: URL,
        prior: PartialDownloadState?,
        displayName: String?,
        expectedSHA256: String?,
        bearerToken: String?,
        onProgress: @escaping @Sendable (DownloadProgress) -> Void
    ) async throws -> ModelDownloadResult {
        var request = URLRequest(url: url)
        applyStandardHeaders(to: &request, bearerToken: bearerToken)
        request.timeoutInterval = 3_600

        if let prior {
            request.setValue("bytes=\(prior.existingSize)-", forHTTPHeaderField: "Range")
            if let validator = prior.etag ?? prior.lastModified {
                request.setValue(validator, forHTTPHeaderField: "If-Range")
            }
            modelDownloaderLog.info(
                "Resuming single-stream \(url.lastPathComponent, privacy: .public) from byte \(prior.existingSize)"
            )
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
            if let rangeTotal = parseContentRangeTotal(http), rangeTotal != prior.totalBytes {
                discardPartial(partialURL: partialURL, sidecarURL: sidecarURL)
                throw ModelDownloaderError.httpStatus(200)
            }
            existingSize = prior.existingSize
            totalBytes = prior.totalBytes
        } else {
            existingSize = 0
            totalBytes = http.expectedContentLength
            discardPartial(partialURL: partialURL, sidecarURL: sidecarURL)
            try FileManager.default.createDirectory(
                at: partialURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            FileManager.default.createFile(atPath: partialURL.path, contents: nil)
            if totalBytes > 0 {
                writeSingleStreamSidecar(
                    to: sidecarURL,
                    sourceURL: url,
                    totalBytes: totalBytes,
                    etag: http.value(forHTTPHeaderField: "ETag"),
                    lastModified: http.value(forHTTPHeaderField: "Last-Modified"),
                    displayName: displayName
                )
            }
        }

        var hasher: SHA256?
        if expectedSHA256.hasNonEmptyValue {
            hasher = SHA256()
        }
        if existingSize > 0, hasher != nil {
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
                    updateHasher(&hasher, data: data)
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
        } catch is CancellationError {
            throw ModelDownloaderError.cancelled
        } catch let error as URLError where error.code == .cancelled {
            throw ModelDownloaderError.cancelled
        }

        if !buffer.isEmpty {
            let data = Data(buffer)
            try handle.write(contentsOf: data)
            updateHasher(&hasher, data: data)
            received += Int64(data.count)
        }

        guard totalBytes <= 0 || received == totalBytes else {
            throw ModelDownloaderError.incompleteResponse
        }

        let digest: String?
        do {
            digest = try finalizeHash(&hasher, expected: expectedSHA256)
        } catch {
            discardPartial(partialURL: partialURL, sidecarURL: sidecarURL)
            throw error
        }

        try? FileManager.default.removeItem(at: sidecarURL)
        onProgress(DownloadProgress(
            bytesReceived: received,
            totalBytes: max(totalBytes, received),
            bytesPerSecond: 0
        ))
        return ModelDownloadResult(tempURL: partialURL, sizeBytes: received, sha256: digest)
    }

}
