import Foundation
import Security

public enum HuggingFaceTokenStoreError: Error, LocalizedError, Sendable, Equatable {
    case unexpectedStatus(OSStatus)
    case invalidTokenData

    public var errorDescription: String? {
        switch self {
        case .unexpectedStatus(let status):
            return "Keychain operation failed with status \(status)."
        case .invalidTokenData:
            return "Stored Hugging Face token data is invalid."
        }
    }
}

public struct HuggingFaceTokenStore: Sendable {
    public static let shared = HuggingFaceTokenStore()

    private let service: String

    public init(service: String = "com.carbocation.CarbocationLocalLLM.HuggingFaceToken") {
        self.service = service
    }

    public func token(for endpoint: URL = HuggingFaceModelReference.defaultEndpoint) throws -> String? {
        var query = baseQuery(endpoint: endpoint)
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne

        var result: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        if status == errSecItemNotFound {
            return nil
        }
        guard status == errSecSuccess else {
            throw HuggingFaceTokenStoreError.unexpectedStatus(status)
        }
        guard let data = result as? Data,
              let token = String(data: data, encoding: .utf8)
        else {
            throw HuggingFaceTokenStoreError.invalidTokenData
        }
        return token.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : token
    }

    public func save(
        _ token: String,
        for endpoint: URL = HuggingFaceModelReference.defaultEndpoint
    ) throws {
        let trimmed = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            try deleteToken(for: endpoint)
            return
        }
        let data = Data(trimmed.utf8)
        let query = baseQuery(endpoint: endpoint)
        let update = [kSecValueData as String: data]
        let updateStatus = SecItemUpdate(query as CFDictionary, update as CFDictionary)
        if updateStatus == errSecSuccess {
            return
        }
        guard updateStatus == errSecItemNotFound else {
            throw HuggingFaceTokenStoreError.unexpectedStatus(updateStatus)
        }

        var insert = query
        insert[kSecValueData as String] = data
        #if os(macOS)
        insert[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        #else
        insert[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        #endif
        let addStatus = SecItemAdd(insert as CFDictionary, nil)
        guard addStatus == errSecSuccess else {
            throw HuggingFaceTokenStoreError.unexpectedStatus(addStatus)
        }
    }

    public func deleteToken(for endpoint: URL = HuggingFaceModelReference.defaultEndpoint) throws {
        let status = SecItemDelete(baseQuery(endpoint: endpoint) as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw HuggingFaceTokenStoreError.unexpectedStatus(status)
        }
    }

    private func baseQuery(endpoint: URL) -> [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: endpoint.host ?? endpoint.absoluteString
        ]
    }
}
