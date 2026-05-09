import CarbocationLocalLLM
import CarbocationLocalLLMRuntime
import CarbocationLocalLLMUI
import SwiftUI

public enum LocalLLMModelConfigurationContextControls: Hashable, Sendable {
    case visible
    case hidden
    case fixed(Int)
}

@MainActor
public struct LocalLLMModelConfigurationView: View {
    let library: ModelLibrary
    @Binding var selectedModelID: String
    let title: String
    let confirmTitle: String
    let confirmDisabled: Bool
    let systemModels: [LLMSystemModelOption]
    let curatedModels: [CuratedModel]
    let labelPolicy: ModelLibraryPickerLabelPolicy
    let defaults: UserDefaults
    let contextKeys: LlamaContextPreferenceKeys
    let contextControls: LocalLLMModelConfigurationContextControls
    let calibrationStore: LlamaContextCalibrationStore
    let configuration: LocalLLMEngineConfiguration
    let onModelDeleted: @MainActor (InstalledModel) -> Void
    let onConfirmSelection: (@MainActor (LLMModelSelection) -> Void)?

    @State var contextModeRaw: String
    @State var autoContextLimit: Int
    @State var autoContextLimitUsesMaximum: Bool
    @State var manualContext: Int
    @State var refreshToken = UUID()
    @State var activeCalibrationModelID: UUID?
    @State var calibrationProgress: LlamaContextCalibrationProgress?
    @State var calibrationTask: Task<Void, Never>?
    @State var calibrationErrorMessage: String?

    public init(
        library: ModelLibrary,
        selectedModelID: Binding<String>,
        title: String = "Configure Local Model",
        confirmTitle: String = "Use Selected Model",
        confirmDisabled: Bool = false,
        systemModels: [LLMSystemModelOption] = LocalLLMEngine.availableSystemModels(),
        curatedModels: [CuratedModel] = CuratedModelCatalog.all,
        labelPolicy: ModelLibraryPickerLabelPolicy = .default,
        defaults: UserDefaults = .standard,
        contextKeys: LlamaContextPreferenceKeys = LlamaContextPreferenceKeys(),
        contextControls: LocalLLMModelConfigurationContextControls = .visible,
        calibrationStore: LlamaContextCalibrationStore = .shared,
        configuration: LocalLLMEngineConfiguration = LocalLLMEngineConfiguration(),
        onModelDeleted: @escaping @MainActor (InstalledModel) -> Void = { _ in },
        onConfirmSelection: (@MainActor (LLMModelSelection) -> Void)? = nil
    ) {
        self.library = library
        self._selectedModelID = selectedModelID
        self.title = title
        self.confirmTitle = confirmTitle
        self.confirmDisabled = confirmDisabled
        self.systemModels = systemModels
        self.curatedModels = curatedModels
        self.labelPolicy = labelPolicy
        self.defaults = defaults
        self.contextKeys = contextKeys
        self.contextControls = contextControls
        self.calibrationStore = calibrationStore
        self.configuration = configuration
        self.onModelDeleted = onModelDeleted
        self.onConfirmSelection = onConfirmSelection
        self._contextModeRaw = State(initialValue: LlamaContextPolicy.currentMode(
            defaults: defaults,
            keys: contextKeys
        ).rawValue)
        self._autoContextLimit = State(initialValue: LlamaContextPolicy.autoContextLimit(
            defaults: defaults,
            keys: contextKeys
        ))
        self._autoContextLimitUsesMaximum = State(initialValue: LlamaContextPolicy.autoContextLimitUsesMaximum(
            defaults: defaults,
            keys: contextKeys
        ))
        self._manualContext = State(initialValue: LlamaContextPolicy.manualContext(
            defaults: defaults,
            keys: contextKeys
        ))
    }

    public var body: some View {
        VStack(spacing: 0) {
            ModelLibraryPickerView(
                library: library,
                selectedModelID: $selectedModelID,
                title: title,
                confirmTitle: confirmTitle,
                confirmDisabled: confirmDisabled,
                systemModels: systemModels,
                curatedModels: curatedModels,
                labelPolicy: labelPolicy,
                calibrationAdapter: pickerCalibrationAdapter,
                onModelDeleted: onModelDeleted,
                onConfirmSelection: { selection in
                    onConfirmSelection?(selection)
                }
            )

            if contextControls != .hidden {
                Divider()
                contextSection
                    .padding(20)
                    .id(refreshToken)
            }
        }
        .task {
            await refreshLibraryForContext()
            syncContextStateFromDefaults()
            applyFixedContextIfNeeded()
        }
        .onChange(of: selectedModelID) { oldValue, _ in
            preserveMaximumIntentIfNeeded(forSelectionID: oldValue)
            syncContextStateFromDefaults()
        }
        .onDisappear {
            cancelCalibration()
        }
    }
}
