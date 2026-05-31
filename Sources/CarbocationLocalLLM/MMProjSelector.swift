import Foundation

enum MMProjSelector {
    static func selectBest<Candidate>(
        primaryPath: String,
        candidates: [Candidate],
        path: (Candidate) -> String
    ) -> Candidate? {
        let primaryDirectory = directoryComponents(for: primaryPath)
        let primaryBits = quantBits(for: primaryPath)
        var best: (candidate: Candidate, depth: Int, diff: Int)?

        for candidate in candidates {
            let candidatePath = path(candidate)
            guard isMMProj(candidatePath) else { continue }

            let candidateDirectory = directoryComponents(for: candidatePath)
            guard candidateDirectory.isSameOrAncestor(of: primaryDirectory) else { continue }

            let depth = candidateDirectory.count
            let diff = abs(quantBits(for: candidatePath) - primaryBits)
            if best == nil || depth > best!.depth || (depth == best!.depth && diff < best!.diff) {
                best = (candidate, depth, diff)
            }
        }

        return best?.candidate
    }

    static func selectBest(primaryPath: String, candidatePaths: [String]) -> String? {
        selectBest(primaryPath: primaryPath, candidates: candidatePaths) { $0 }
    }

    static func isMMProj(_ path: String) -> Bool {
        let lowercased = path.lowercased()
        return lowercased.hasSuffix(".gguf") && lowercased.contains("mmproj")
    }

    static func quantBits(for path: String) -> Int {
        let tag = splitInfo(for: path).tag
        guard let digit = tag.firstIndex(where: { $0.isNumber }) else { return 0 }
        return Int(tag[digit...].prefix { $0.isNumber }) ?? 0
    }

    private struct SplitInfo {
        var tag: String
    }

    private static func splitInfo(for path: String) -> SplitInfo {
        var prefix = path
        if prefix.lowercased().hasSuffix(".gguf") {
            prefix.removeLast(5)
        } else {
            return SplitInfo(tag: "")
        }
        if let range = prefix.range(
            of: "-[0-9]{5}-of-[0-9]{5}$",
            options: [.regularExpression, .caseInsensitive]
        ) {
            prefix.removeSubrange(range)
        }

        var tag = ""
        if let range = prefix.range(
            of: "[-.][A-Z0-9_]+$",
            options: [.regularExpression, .caseInsensitive]
        ) {
            tag = String(prefix[range].dropFirst()).uppercased()
        }
        return SplitInfo(tag: tag)
    }

    private static func directoryComponents(for path: String) -> [String] {
        let components = path.split(separator: "/", omittingEmptySubsequences: true).map(String.init)
        guard !components.isEmpty else { return [] }
        return Array(components.dropLast())
    }
}

private extension Array where Element == String {
    func isSameOrAncestor(of other: [String]) -> Bool {
        guard count <= other.count else { return false }
        return zip(self, other).allSatisfy { $0 == $1 }
    }
}
