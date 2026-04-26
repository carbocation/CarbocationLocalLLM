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

    private static func defaultSharedGroupRoot(identifier: String, fileManager: FileManager) -> URL? {
        fileManager.containerURL(forSecurityApplicationGroupIdentifier: identifier)
    }
}
