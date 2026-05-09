import Foundation

public struct HuggingFaceModelReference: Hashable, Sendable {
    public static let defaultEndpoint = URL(string: "https://huggingface.co")!

    public var repo: String
    public var quantization: String?
    public var file: String?
    public var revision: String
    public var endpoint: URL

    public init(
        repo: String,
        quantization: String? = nil,
        file: String? = nil,
        revision: String = "main",
        endpoint: URL = Self.defaultEndpoint
    ) {
        self.repo = repo
        self.quantization = quantization?.nilIfBlank?.uppercased()
        self.file = file?.nilIfBlank
        self.revision = revision.nilIfBlank ?? "main"
        self.endpoint = endpoint
    }

    public static func parse(
        _ input: String,
        endpoint: URL = Self.defaultEndpoint
    ) -> HuggingFaceModelReference? {
        let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        if let url = URL(string: trimmed),
           let host = url.host,
           host.caseInsensitiveCompare(endpoint.host ?? "huggingface.co") == .orderedSame
                || host.contains("huggingface.co") {
            let parts = url.pathComponents.filter { $0 != "/" }
            guard parts.count >= 2 else { return nil }
            if parts.count >= 5, parts[2] == "resolve" || parts[2] == "blob" {
                let file = parts.suffix(from: 4).joined(separator: "/")
                guard file.lowercased().hasSuffix(".gguf") else { return nil }
                return HuggingFaceModelReference(
                    repo: "\(parts[0])/\(parts[1])",
                    file: file,
                    revision: parts[3],
                    endpoint: endpoint
                )
            }
            if parts.count == 2 {
                return parseRepoAndQuant("\(parts[0])/\(parts[1])", endpoint: endpoint)
            }
            return nil
        }

        let parts = trimmed.split(separator: "/", omittingEmptySubsequences: true).map(String.init)
        guard parts.count >= 2 else { return nil }

        if parts.count >= 3, parts.last?.lowercased().hasSuffix(".gguf") == true {
            return HuggingFaceModelReference(
                repo: "\(parts[0])/\(parts[1])",
                file: parts.suffix(from: 2).joined(separator: "/"),
                endpoint: endpoint
            )
        }

        if parts.count == 2 {
            return parseRepoAndQuant(trimmed, endpoint: endpoint)
        }

        return nil
    }

    private static func parseRepoAndQuant(
        _ value: String,
        endpoint: URL
    ) -> HuggingFaceModelReference? {
        let parts = value.split(separator: "/", omittingEmptySubsequences: true).map(String.init)
        guard parts.count == 2 else { return nil }

        let modelAndQuant = parts[1].split(separator: ":", maxSplits: 1, omittingEmptySubsequences: false)
        guard let model = modelAndQuant.first, !model.isEmpty else { return nil }
        let quant = modelAndQuant.count == 2 ? String(modelAndQuant[1]).nilIfBlank : nil
        return HuggingFaceModelReference(
            repo: "\(parts[0])/\(String(model))",
            quantization: quant,
            endpoint: endpoint
        )
    }
}

public enum HuggingFaceArtifactRole: String, Codable, Hashable, Sendable {
    case primaryModel
    case splitModel
    case mmproj
}

public struct HuggingFaceResolvedArtifact: Hashable, Sendable {
    public var role: HuggingFaceArtifactRole
    public var path: String
    public var url: URL
    public var sizeBytes: Int64
    public var oid: String?

    public init(
        role: HuggingFaceArtifactRole,
        path: String,
        url: URL,
        sizeBytes: Int64,
        oid: String? = nil
    ) {
        self.role = role
        self.path = path
        self.url = url
        self.sizeBytes = sizeBytes
        self.oid = oid
    }
}

public struct HuggingFaceResolution: Hashable, Sendable {
    public var reference: HuggingFaceModelReference
    public var commit: String
    public var primaryArtifact: HuggingFaceResolvedArtifact
    public var artifacts: [HuggingFaceResolvedArtifact]
    public var quantization: String?
    public var displayName: String
    public var totalSizeBytes: Int64

    public init(
        reference: HuggingFaceModelReference,
        commit: String,
        primaryArtifact: HuggingFaceResolvedArtifact,
        artifacts: [HuggingFaceResolvedArtifact],
        quantization: String?,
        displayName: String,
        totalSizeBytes: Int64
    ) {
        self.reference = reference
        self.commit = commit
        self.primaryArtifact = primaryArtifact
        self.artifacts = artifacts
        self.quantization = quantization
        self.displayName = displayName
        self.totalSizeBytes = totalSizeBytes
    }

    public var splitCount: Int {
        artifacts.filter { $0.role == .primaryModel || $0.role == .splitModel }.count
    }

    public var mmprojArtifact: HuggingFaceResolvedArtifact? {
        artifacts.first { $0.role == .mmproj }
    }
}

public enum HuggingFaceModelResolverError: Error, LocalizedError, Sendable, Equatable {
    case invalidRepository(String)
    case invalidURL(String)
    case httpStatus(Int)
    case malformedResponse(String)
    case noCommit(String)
    case noGGUFFiles(String)
    case fileNotFound(String)
    case missingSplitPart(String)
    case quantizationNotFound(String)

    public var errorDescription: String? {
        switch self {
        case .invalidRepository(let repo):
            return "Invalid Hugging Face repository: \(repo)."
        case .invalidURL(let value):
            return "Invalid Hugging Face URL: \(value)."
        case .httpStatus(let code):
            return "Hugging Face request failed with HTTP \(code)."
        case .malformedResponse(let detail):
            return "Hugging Face returned an unexpected response: \(detail)."
        case .noCommit(let repo):
            return "Could not resolve a Hugging Face revision for \(repo)."
        case .noGGUFFiles(let repo):
            return "No GGUF model files were found in \(repo)."
        case .fileNotFound(let file):
            return "Hugging Face file not found: \(file)."
        case .missingSplitPart(let file):
            return "Split GGUF set is incomplete near \(file)."
        case .quantizationNotFound(let quantization):
            return "No GGUF file matched quantization \(quantization)."
        }
    }
}

public protocol HuggingFaceModelResolverHTTPClient: Sendable {
    func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse)
}

public struct HuggingFaceURLSessionHTTPClient: HuggingFaceModelResolverHTTPClient, @unchecked Sendable {
    public var session: URLSession

    public init(session: URLSession = .shared) {
        self.session = session
    }

    public func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw HuggingFaceModelResolverError.httpStatus(-1)
        }
        return (data, http)
    }
}

public struct HuggingFaceModelResolver: Sendable {
    private struct BranchRef: Decodable {
        var name: String
        var targetCommit: String
    }

    private struct RefsResponse: Decodable {
        var branches: [BranchRef]?
        var tags: [BranchRef]?
    }

    private struct TreeItem: Decodable {
        struct LFS: Decodable {
            var oid: String?
            var size: Int64?
        }

        var type: String
        var path: String
        var oid: String?
        var size: Int64?
        var lfs: LFS?
    }

    private struct RepoFile: Hashable {
        var path: String
        var oid: String?
        var sizeBytes: Int64
        var url: URL
    }

    private struct SplitInfo: Hashable {
        var prefix: String
        var tag: String
        var index: Int
        var count: Int
    }

    public var endpoint: URL
    private let httpClient: any HuggingFaceModelResolverHTTPClient

    public init(
        endpoint: URL = HuggingFaceModelReference.defaultEndpoint,
        httpClient: any HuggingFaceModelResolverHTTPClient = HuggingFaceURLSessionHTTPClient()
    ) {
        self.endpoint = endpoint
        self.httpClient = httpClient
    }

    public func resolve(
        _ reference: HuggingFaceModelReference,
        token: String? = nil
    ) async throws -> HuggingFaceResolution {
        guard Self.isValidRepo(reference.repo) else {
            throw HuggingFaceModelResolverError.invalidRepository(reference.repo)
        }

        let commit = try await resolveCommit(for: reference, token: token)
        let files = try await fetchFiles(repo: reference.repo, commit: commit, token: token)
        let primary = try selectPrimaryFile(from: files, reference: reference)
        let modelFiles = try splitFiles(for: primary, in: files)
        let mmproj = selectMMProj(for: primary, in: files)

        var artifacts: [HuggingFaceResolvedArtifact] = []
        for file in modelFiles {
            artifacts.append(HuggingFaceResolvedArtifact(
                role: file.path == primary.path ? .primaryModel : .splitModel,
                path: file.path,
                url: file.url,
                sizeBytes: file.sizeBytes,
                oid: file.oid
            ))
        }
        if let mmproj {
            artifacts.append(HuggingFaceResolvedArtifact(
                role: .mmproj,
                path: mmproj.path,
                url: mmproj.url,
                sizeBytes: mmproj.sizeBytes,
                oid: mmproj.oid
            ))
        }

        let primaryArtifact = artifacts.first { $0.role == .primaryModel }!
        let totalSize = artifacts.reduce(Int64(0)) { $0 + max(0, $1.sizeBytes) }
        let split = splitInfo(for: primary.path)
        return HuggingFaceResolution(
            reference: reference,
            commit: commit,
            primaryArtifact: primaryArtifact,
            artifacts: artifacts,
            quantization: reference.quantization?.nilIfBlank?.uppercased() ?? split.tag.nilIfBlank,
            displayName: displayName(for: primary.path),
            totalSizeBytes: totalSize
        )
    }

    private func resolveCommit(
        for reference: HuggingFaceModelReference,
        token: String?
    ) async throws -> String {
        if Self.isCommitHash(reference.revision) {
            return reference.revision
        }

        let url = try apiURL(path: "api/models/\(reference.repo)/refs")
        let request = Self.authorizedRequest(url: url, token: token)
        let (data, response) = try await httpClient.data(for: request)
        guard (200...299).contains(response.statusCode) else {
            throw HuggingFaceModelResolverError.httpStatus(response.statusCode)
        }

        let refs = try JSONDecoder().decode(RefsResponse.self, from: data)
        let candidates = (refs.branches ?? []) + (refs.tags ?? [])
        if let exact = candidates.first(where: { $0.name == reference.revision }),
           Self.isCommitHash(exact.targetCommit) {
            return exact.targetCommit
        }
        if let main = candidates.first(where: { $0.name == "main" }),
           Self.isCommitHash(main.targetCommit) {
            return main.targetCommit
        }
        if let first = candidates.first(where: { Self.isCommitHash($0.targetCommit) }) {
            return first.targetCommit
        }
        throw HuggingFaceModelResolverError.noCommit(reference.repo)
    }

    private func fetchFiles(
        repo: String,
        commit: String,
        token: String?
    ) async throws -> [RepoFile] {
        let url = try apiURL(path: "api/models/\(repo)/tree/\(commit)?recursive=true")
        let request = Self.authorizedRequest(url: url, token: token)
        let (data, response) = try await httpClient.data(for: request)
        guard (200...299).contains(response.statusCode) else {
            throw HuggingFaceModelResolverError.httpStatus(response.statusCode)
        }

        let tree = try JSONDecoder().decode([TreeItem].self, from: data)
        return try tree.compactMap { item in
            guard item.type == "file" else { return nil }
            guard Self.isSafeSubpath(item.path) else { return nil }
            let oid = item.lfs?.oid ?? item.oid
            let size = item.lfs?.size ?? item.size ?? 0
            return RepoFile(
                path: item.path,
                oid: oid,
                sizeBytes: size,
                url: try resolveURL(endpoint: endpoint, repo: repo, commit: commit, path: item.path)
            )
        }
    }

    private func selectPrimaryFile(
        from files: [RepoFile],
        reference: HuggingFaceModelReference
    ) throws -> RepoFile {
        if let explicitFile = reference.file {
            guard let match = files.first(where: { $0.path == explicitFile }) else {
                throw HuggingFaceModelResolverError.fileNotFound(explicitFile)
            }
            guard Self.isModelGGUF(match.path) else {
                throw HuggingFaceModelResolverError.noGGUFFiles(reference.repo)
            }
            return match
        }

        let quantizations: [String]
        if let quantization = reference.quantization?.nilIfBlank {
            quantizations = [quantization.uppercased()]
        } else {
            quantizations = ["Q4_K_M", "Q8_0"]
        }

        for quantization in quantizations {
            if let match = files.first(where: {
                Self.isModelGGUF($0.path)
                    && splitInfo(for: $0.path).index == 1
                    && Self.path($0.path, matchesQuantization: quantization)
            }) {
                return match
            }
        }

        if reference.quantization?.nilIfBlank != nil {
            throw HuggingFaceModelResolverError.quantizationNotFound(reference.quantization!.uppercased())
        }

        if let fallback = files.first(where: {
            Self.isModelGGUF($0.path) && splitInfo(for: $0.path).index == 1
        }) {
            return fallback
        }

        throw HuggingFaceModelResolverError.noGGUFFiles(reference.repo)
    }

    private func splitFiles(for primary: RepoFile, in files: [RepoFile]) throws -> [RepoFile] {
        let primarySplit = splitInfo(for: primary.path)
        guard primarySplit.count > 1 else { return [primary] }

        let matches = files
            .filter {
                let split = splitInfo(for: $0.path)
                return split.prefix == primarySplit.prefix && split.count == primarySplit.count
            }
            .sorted { splitInfo(for: $0.path).index < splitInfo(for: $1.path).index }

        guard matches.count == primarySplit.count else {
            throw HuggingFaceModelResolverError.missingSplitPart(primary.path)
        }
        return matches
    }

    private func selectMMProj(for primary: RepoFile, in files: [RepoFile]) -> RepoFile? {
        let primaryDirectory = directoryPath(primary.path)
        let primaryBits = quantBits(for: primary.path)
        var best: (file: RepoFile, depth: Int, diff: Int)?

        for file in files where Self.isMMProj(file.path) {
            let directory = directoryPath(file.path)
            guard primaryDirectory.hasPrefix(directory) || directory == primaryDirectory else { continue }
            let depth = directory.split(separator: "/").count
            let diff = abs(quantBits(for: file.path) - primaryBits)
            if best == nil || depth > best!.depth || (depth == best!.depth && diff < best!.diff) {
                best = (file, depth, diff)
            }
        }

        return best?.file
    }

    private func apiURL(path: String) throws -> URL {
        let base = endpoint.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard let url = URL(string: "\(base)/\(path)") else {
            throw HuggingFaceModelResolverError.invalidURL("\(base)/\(path)")
        }
        return url
    }

    private func resolveURL(
        endpoint: URL,
        repo: String,
        commit: String,
        path: String
    ) throws -> URL {
        let base = endpoint.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let escapedPath = path
            .split(separator: "/", omittingEmptySubsequences: false)
            .map { String($0).addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0) }
            .joined(separator: "/")
        let urlString = "\(base)/\(repo)/resolve/\(commit)/\(escapedPath)"
        guard let url = URL(string: urlString) else {
            throw HuggingFaceModelResolverError.invalidURL(urlString)
        }
        return url
    }

    static func authorizedRequest(url: URL, token: String?) -> URLRequest {
        var request = URLRequest(url: url)
        request.setValue("CarbocationLocalLLM/1.0", forHTTPHeaderField: "User-Agent")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        if let token = token?.nilIfBlank {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        return request
    }

    private static func isValidRepo(_ repo: String) -> Bool {
        let parts = repo.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2 else { return false }
        return parts.allSatisfy { part in
            !part.isEmpty && part.allSatisfy { character in
                character.isLetter || character.isNumber || character == "_" || character == "-" || character == "."
            }
        }
    }

    private static func isCommitHash(_ value: String) -> Bool {
        value.count == 40 && value.allSatisfy { $0.isHexDigit }
    }

    private static func isSafeSubpath(_ path: String) -> Bool {
        !path.isEmpty
            && !path.hasPrefix("/")
            && !path.split(separator: "/").contains("..")
    }

    private static func isModelGGUF(_ path: String) -> Bool {
        let filename = URL(fileURLWithPath: path).lastPathComponent.lowercased()
        return filename.hasSuffix(".gguf")
            && !filename.contains("mmproj")
            && !filename.contains("imatrix")
    }

    private static func isMMProj(_ path: String) -> Bool {
        let filename = URL(fileURLWithPath: path).lastPathComponent.lowercased()
        return filename.hasSuffix(".gguf") && filename.contains("mmproj")
    }

    private static func path(_ path: String, matchesQuantization quantization: String) -> Bool {
        let escaped = NSRegularExpression.escapedPattern(for: quantization)
        return path.range(
            of: "\(escaped)[.-]",
            options: [.regularExpression, .caseInsensitive]
        ) != nil
    }

    private func splitInfo(for path: String) -> SplitInfo {
        var prefix = path
        guard prefix.lowercased().hasSuffix(".gguf") else {
            return SplitInfo(prefix: path, tag: "", index: 1, count: 1)
        }
        prefix.removeLast(5)

        var index = 1
        var count = 1
        if let range = prefix.range(
            of: "-[0-9]{5}-of-[0-9]{5}$",
            options: [.regularExpression, .caseInsensitive]
        ) {
            let suffix = String(prefix[range]).dropFirst()
            let parts = suffix.split(separator: "-")
            if parts.count == 3,
               let parsedIndex = Int(parts[0]),
               let parsedCount = Int(parts[2]) {
                index = parsedIndex
                count = parsedCount
                prefix.removeSubrange(range)
            }
        }

        var tag = ""
        if let range = prefix.range(
            of: "[-.][A-Za-z0-9_]+$",
            options: [.regularExpression]
        ) {
            tag = String(prefix[range].dropFirst()).uppercased()
        }
        return SplitInfo(prefix: prefix, tag: tag, index: index, count: count)
    }

    private func quantBits(for path: String) -> Int {
        let tag = splitInfo(for: path).tag
        guard let digit = tag.firstIndex(where: { $0.isNumber }) else { return 0 }
        return Int(tag[digit...].prefix { $0.isNumber }) ?? 0
    }

    private func directoryPath(_ path: String) -> String {
        let url = URL(fileURLWithPath: path)
        let directory = url.deletingLastPathComponent().relativePath
        return directory == "." ? "" : directory
    }

    private func displayName(for path: String) -> String {
        var name = URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent
        if let range = name.range(
            of: "-[0-9]{5}-of-[0-9]{5}$",
            options: [.regularExpression, .caseInsensitive]
        ) {
            name.removeSubrange(range)
        }
        return name
    }
}

public enum HuggingFaceURL {
    public static func parse(_ input: String) -> (repo: String, filename: String)? {
        guard let reference = HuggingFaceModelReference.parse(input),
              let file = reference.file
        else { return nil }
        return (reference.repo, file)
    }
}

private extension String {
    var nilIfBlank: String? {
        let trimmed = trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}
