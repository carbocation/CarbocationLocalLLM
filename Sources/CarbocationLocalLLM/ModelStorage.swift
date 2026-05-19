import Foundation

public enum ModelStorage {
    public static let carbocationSharedGroupID = "group.com.carbocation.shared"
    public static let defaultSharedGroupID = carbocationSharedGroupID

    typealias SharedGroupRootResolver = (String, FileManager) -> URL?

    public static func appSupportDirectory(
        appSupportFolderName: String,
        fileManager: FileManager = .default
    ) -> URL {
        let base = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent(appSupportFolderName, isDirectory: true)
    }

    public static func sharedGroupRoot(
        identifier: String = defaultSharedGroupID,
        fileManager: FileManager = .default
    ) -> URL? {
        sharedGroupRoot(
            identifier: identifier,
            fileManager: fileManager,
            sharedGroupRootResolver: defaultSharedGroupRoot
        )
    }

    static func sharedGroupRoot(
        identifier: String = defaultSharedGroupID,
        fileManager: FileManager = .default,
        sharedGroupRootResolver: SharedGroupRootResolver
    ) -> URL? {
        sharedGroupRootResolver(identifier, fileManager)
    }

    public static func sharedSettingsDefaults(
        sharedGroupIdentifier: String = defaultSharedGroupID
    ) -> UserDefaults {
        UserDefaults(suiteName: sharedGroupIdentifier) ?? .standard
    }

    /// Returns the shared App Group `Models` directory when available, otherwise
    /// `<Application Support>/<appSupportFolderName>/Models` for unsigned/dev builds.
    public static func modelsDirectory(
        sharedGroupIdentifier: String = defaultSharedGroupID,
        appSupportFolderName: String,
        fileManager: FileManager = .default
    ) -> URL {
        modelsDirectory(
            sharedGroupIdentifier: sharedGroupIdentifier,
            appSupportFolderName: appSupportFolderName,
            fileManager: fileManager,
            sharedGroupRootResolver: defaultSharedGroupRoot
        )
    }

    static func modelsDirectory(
        sharedGroupIdentifier: String = defaultSharedGroupID,
        appSupportFolderName: String,
        fileManager: FileManager = .default,
        sharedGroupRootResolver: SharedGroupRootResolver
    ) -> URL {
        let base = sharedGroupRoot(
            identifier: sharedGroupIdentifier,
            fileManager: fileManager,
            sharedGroupRootResolver: sharedGroupRootResolver
        )
            ?? appSupportDirectory(appSupportFolderName: appSupportFolderName, fileManager: fileManager)
        return base.appendingPathComponent("Models", isDirectory: true)
    }

    public static func legacyPerAppModelsDirectory(
        appSupportFolderName: String,
        fileManager: FileManager = .default
    ) -> URL {
        appSupportDirectory(appSupportFolderName: appSupportFolderName, fileManager: fileManager)
            .appendingPathComponent("Models", isDirectory: true)
    }

    public static func huggingFaceHubCacheDirectory(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        fileManager: FileManager = .default
    ) -> URL? {
        #if os(macOS)
        if let hubCache = nonEmptyEnvironmentValue("HF_HUB_CACHE", in: environment) {
            return URL(fileURLWithPath: hubCache, isDirectory: true).standardizedFileURL
        }

        if let hfHome = nonEmptyEnvironmentValue("HF_HOME", in: environment) {
            return URL(fileURLWithPath: hfHome, isDirectory: true)
                .appendingPathComponent("hub", isDirectory: true)
                .standardizedFileURL
        }

        let cacheRoot: URL
        if let xdgCacheHome = nonEmptyEnvironmentValue("XDG_CACHE_HOME", in: environment) {
            cacheRoot = URL(fileURLWithPath: xdgCacheHome, isDirectory: true)
        } else {
            cacheRoot = fileManager.homeDirectoryForCurrentUser
                .appendingPathComponent(".cache", isDirectory: true)
        }

        return cacheRoot
            .appendingPathComponent("huggingface", isDirectory: true)
            .appendingPathComponent("hub", isDirectory: true)
            .standardizedFileURL
        #else
        _ = environment
        _ = fileManager
        return nil
        #endif
    }

    private static func defaultSharedGroupRoot(identifier: String, fileManager: FileManager) -> URL? {
        fileManager.containerURL(forSecurityApplicationGroupIdentifier: identifier)
    }

    private static func nonEmptyEnvironmentValue(
        _ key: String,
        in environment: [String: String]
    ) -> String? {
        environment[key]?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty
    }
}

private extension String {
    var nilIfEmpty: String? {
        isEmpty ? nil : self
    }
}
