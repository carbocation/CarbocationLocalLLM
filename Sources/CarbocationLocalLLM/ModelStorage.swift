import Foundation

public enum ModelStorage {
    public static let defaultSharedGroupID = "group.com.carbocation.shared"

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
        fileManager.containerURL(forSecurityApplicationGroupIdentifier: identifier)
    }

    /// Returns the shared App Group `Models` directory when available, otherwise
    /// `<Application Support>/<appSupportFolderName>/Models` for unsigned/dev builds.
    public static func modelsDirectory(
        sharedGroupIdentifier: String = defaultSharedGroupID,
        appSupportFolderName: String,
        fileManager: FileManager = .default
    ) -> URL {
        let base = sharedGroupRoot(identifier: sharedGroupIdentifier, fileManager: fileManager)
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
}

