import CarbocationLocalLLM

enum ModelLibraryPickerRowPresentation {
    static func showsDeleteControl(for model: InstalledModel) -> Bool {
        !model.isReadOnly
    }
}
