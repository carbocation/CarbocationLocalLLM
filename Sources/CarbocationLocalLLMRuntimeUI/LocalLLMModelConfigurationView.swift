import CarbocationLocalLLM
import CarbocationLocalLLMRuntime
import CarbocationLocalLLMUI
import SwiftUI

public enum LocalLLMModelConfigurationContextControls: Hashable, Sendable {
    case visible
    case hidden
    case fixed(Int)
}

enum LocalLLMContextWindowTiers {
    static let defaultMinimumTier = 4_096

    static func candidates(
        trainingContext: Int,
        calibratedMaximum: Int?,
        defaultAutoCap: Int = LlamaContextPolicy.defaultAutoCap,
        minimumTier: Int = defaultMinimumTier
    ) -> [Int] {
        let knownTrainingContext = trainingContext > 0
            ? trainingContext
            : LlamaContextPolicy.unknownTrainingFallback
        let upperBound: Int
        if let calibratedMaximum, calibratedMaximum > 0 {
            upperBound = min(knownTrainingContext, calibratedMaximum)
        } else {
            upperBound = min(knownTrainingContext, defaultAutoCap)
        }
        let resolvedUpperBound = max(LlamaContextPolicy.minimumContext, upperBound)
        let visibleFloor = min(max(LlamaContextPolicy.minimumContext, minimumTier), resolvedUpperBound)
        let tiers = LlamaContextCalibrationAlgorithm
            .powerOfTwoTiers(upTo: resolvedUpperBound)
            .filter { $0 >= visibleFloor }
        return tiers.isEmpty ? [resolvedUpperBound] : tiers
    }

    static func nearestIndex(for value: Int, in candidates: [Int]) -> Int {
        guard let first = candidates.first else { return 0 }
        var bestIndex = 0
        var bestDistance = abs(first - value)
        for (index, candidate) in candidates.enumerated().dropFirst() {
            let distance = abs(candidate - value)
            if distance < bestDistance {
                bestIndex = index
                bestDistance = distance
            }
        }
        return bestIndex
    }
}

@MainActor
public struct LocalLLMModelConfigurationView: View {
    private let library: ModelLibrary
    @Binding private var selectedModelID: String
    private let title: String
    private let confirmTitle: String
    private let confirmDisabled: Bool
    private let systemModels: [LLMSystemModelOption]
    private let curatedModels: [CuratedModel]
    private let labelPolicy: ModelLibraryPickerLabelPolicy
    private let defaults: UserDefaults
    private let contextKeys: LlamaContextPreferenceKeys
    private let contextControls: LocalLLMModelConfigurationContextControls
    private let calibrationStore: LlamaContextCalibrationStore
    private let configuration: LocalLLMEngineConfiguration
    private let onModelDeleted: @MainActor (InstalledModel) -> Void
    private let onConfirmSelection: (@MainActor (LLMModelSelection) -> Void)?

    @State private var contextModeRaw: String
    @State private var autoContextLimit: Int
    @State private var autoContextLimitUsesMaximum: Bool
    @State private var manualContext: Int
    @State private var refreshToken = UUID()
    @State private var activeCalibrationModelID: UUID?
    @State private var calibrationProgress: LlamaContextCalibrationProgress?
    @State private var calibrationTask: Task<Void, Never>?
    @State private var calibrationErrorMessage: String?

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

    static func makeCalibrationAdapter(
        library: ModelLibrary,
        store: LlamaContextCalibrationStore,
        configuration: LocalLLMEngineConfiguration,
        onCalibrationStarted: @escaping @MainActor (_ model: InstalledModel) -> Void = { _ in },
        onCalibrationCompleted: @escaping @MainActor (
            _ model: InstalledModel,
            _ record: LlamaContextCalibrationRecord
        ) -> Void = { _, _ in }
    ) -> ModelLibraryPickerCalibrationAdapter {
        ModelLibraryPickerCalibrationAdapter(
            store: store,
            runtimeFingerprint: LocalLLMEngine.contextCalibrationRuntimeFingerprint(configuration: configuration),
            onCalibrationStarted: onCalibrationStarted,
            onCalibrationCompleted: onCalibrationCompleted,
            calibrate: { model, progress in
                try await LocalLLMEngine.calibrateContext(
                    for: model,
                    in: library,
                    store: store,
                    configuration: configuration
                ) { value in
                    await MainActor.run { progress(value) }
                }
            }
        )
    }

    @ViewBuilder
    private var contextSection: some View {
        switch contextControls {
        case .hidden:
            EmptyView()
        case .fixed(let context):
            fixedContextSection(context)
        case .visible:
            if let model = selectedInstalledModel {
                installedContextSection(model)
            } else if let option = selectedSystemModel {
                systemContextSection(option)
            } else if selectedSelection != nil {
                unresolvedContextSelectionSection
            } else {
                noContextSelectionSection
            }
        }
    }

    private var selectedSelection: LLMModelSelection? {
        LLMModelSelection(storageValue: selectedModelID)
    }

    private var selectedInstalledModel: InstalledModel? {
        guard case .installed(let id) = selectedSelection else { return nil }
        return library.model(id: id)
    }

    private var selectedSystemModel: LLMSystemModelOption? {
        systemModels.first { $0.id == selectedModelID }
    }

    private var pickerCalibrationAdapter: ModelLibraryPickerCalibrationAdapter? {
        guard contextControls == .visible else { return nil }
        return Self.makeCalibrationAdapter(
            library: library,
            store: calibrationStore,
            configuration: configuration,
            onCalibrationStarted: { model in
                prepareSelectedContextForCalibration(model)
            },
            onCalibrationCompleted: { model, _ in
                refreshSelectedContextAfterCalibration(model)
            }
        )
    }

    private func fixedContextSection(_ context: Int) -> some View {
        let sanitized = LlamaContextPolicy.sanitizedContext(context)
        return HStack(spacing: 10) {
            Label("Context Window", systemImage: "rectangle.expand.vertical")
            Spacer()
            Text("Fixed at \(sanitized.formatted()) tokens")
                .foregroundStyle(.secondary)
        }
        .font(.callout)
    }

    private var noContextSelectionSection: some View {
        HStack(spacing: 10) {
            Label("Context Window", systemImage: "rectangle.expand.vertical")
            Spacer()
            Text("Select a model to configure context.")
                .foregroundStyle(.secondary)
        }
        .font(.callout)
    }

    private var unresolvedContextSelectionSection: some View {
        HStack(spacing: 10) {
            Label("Context Window", systemImage: "rectangle.expand.vertical")
            Spacer()
            Text("Loading selected model context...")
                .foregroundStyle(.secondary)
        }
        .font(.callout)
    }

    private func systemContextSection(_ option: LLMSystemModelOption) -> some View {
        HStack(spacing: 10) {
            Label("Context Window", systemImage: "rectangle.expand.vertical")
            Spacer()
            Text("Managed by \(option.displayName) · \(option.contextLength.formatted()) tokens")
                .foregroundStyle(.secondary)
        }
        .font(.callout)
    }

    private func installedContextSection(_ model: InstalledModel) -> some View {
        let record = calibrationRecord(for: model)
        let candidates = LocalLLMContextWindowTiers.candidates(
            trainingContext: model.contextLength,
            calibratedMaximum: record?.maximumSupportedContext
        )
        let sliderValue = Binding<Double>(
            get: {
                if autoContextLimitTracksMaximum(in: candidates) {
                    return Double(max(0, candidates.count - 1))
                }
                return Double(LocalLLMContextWindowTiers.nearestIndex(
                    for: autoContextLimit,
                    in: candidates
                ))
            },
            set: { value in
                guard !candidates.isEmpty else { return }
                let index = min(max(0, Int(value.rounded())), candidates.count - 1)
                setAutoContextLimit(
                    candidates[index],
                    usesMaximum: index == candidates.count - 1
                )
            }
        )

        return VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                Label("Context Window", systemImage: "rectangle.expand.vertical")
                    .font(.headline)
                Spacer()
                contextStatus(record: record, model: model)
            }

            Picker("Mode", selection: Binding(
                get: { contextModeRaw },
                set: { setContextMode($0) }
            )) {
                Text("Auto").tag(LlamaContextMode.auto.rawValue)
                Text("Manual").tag(LlamaContextMode.manual.rawValue)
            }
            .pickerStyle(.segmented)

            if contextModeRaw == LlamaContextMode.manual.rawValue {
                manualContextControls(for: model)
            } else {
                if candidates.count > 1 {
                    Slider(
                        value: sliderValue,
                        in: 0...Double(candidates.count - 1),
                        step: 1
                    )
                }
                HStack {
                    Text(autoContextLimitText(in: candidates))
                    Spacer()
                    if let first = candidates.first, let last = candidates.last, first != last {
                        Text("\(first.formatted())-\(last.formatted())")
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)

                if record == nil {
                    Button("Calibrate to Unlock Larger Context Windows") {
                        startCalibration(model)
                    }
                    .disabled(activeCalibrationModelID != nil)
                    .buttonStyle(.borderless)
                }
            }

            if let progress = calibrationProgress,
               activeCalibrationModelID == model.id {
                calibrationProgressRow(progress)
            } else if let calibrationErrorMessage {
                Label(calibrationErrorMessage, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
    }

    private func contextStatus(record: LlamaContextCalibrationRecord?, model: InstalledModel) -> some View {
        let text: String
        if let record {
            text = "Calibrated max \(record.maximumSupportedContext.formatted())"
        } else if model.contextLength > 0 {
            text = "Model max \(model.contextLength.formatted())"
        } else {
            text = "Uncalibrated"
        }
        return Text(text)
            .font(.caption)
            .foregroundStyle(.secondary)
    }

    private func calibrationProgressRow(_ progress: LlamaContextCalibrationProgress) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            if let fraction = progress.fractionCompleted {
                ProgressView(value: fraction)
            } else {
                ProgressView()
            }
            HStack {
                Text(calibrationProgressText(progress))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Cancel", role: .destructive) {
                    cancelCalibration()
                }
                .font(.caption)
                .buttonStyle(.borderless)
            }
        }
    }

    private func calibrationRecord(for model: InstalledModel) -> LlamaContextCalibrationRecord? {
        calibrationStore.record(
            for: model,
            runtime: LocalLLMEngine.contextCalibrationRuntimeFingerprint(configuration: configuration)
        )
    }

    private func manualContextUpperBound(for model: InstalledModel) -> Int {
        max(
            LlamaContextPolicy.minimumContext,
            model.contextLength > 0 ? model.contextLength : 262_144
        )
    }

    private func manualContextControls(for model: InstalledModel) -> some View {
        let upperBound = manualContextUpperBound(for: model)
        let binding = Binding<Int>(
            get: { manualContext },
            set: { setManualContext($0, upperBound: upperBound) }
        )

        return HStack(spacing: 8) {
            Text("Context tokens")
            Spacer()
            TextField("Context tokens", value: binding, format: .number)
                .textFieldStyle(.roundedBorder)
                .multilineTextAlignment(.trailing)
                .monospacedDigit()
                .frame(width: 128)
                .contextNumericKeyboard()
            Stepper(
                "Adjust context tokens",
                value: binding,
                in: LlamaContextPolicy.minimumContext...upperBound,
                step: 512
            )
            .labelsHidden()
        }
    }

    private func resolvedAutoContextLimit(in candidates: [Int]) -> Int {
        guard !candidates.isEmpty else { return autoContextLimit }
        if autoContextLimitTracksMaximum(in: candidates), let last = candidates.last {
            return last
        }
        let index = LocalLLMContextWindowTiers.nearestIndex(for: autoContextLimit, in: candidates)
        return candidates[index]
    }

    private func autoContextLimitText(in candidates: [Int]) -> String {
        let resolved = resolvedAutoContextLimit(in: candidates)
        if autoContextLimitTracksMaximum(in: candidates) {
            return "Auto limit: Max (\(resolved.formatted()) tokens)"
        }
        return "Auto limit: \(resolved.formatted()) tokens"
    }

    private func autoContextLimitTracksMaximum(in candidates: [Int]) -> Bool {
        guard let last = candidates.last else { return autoContextLimitUsesMaximum }
        return autoContextLimitUsesMaximum
            || (hasExplicitAutoContextLimitPreference && autoContextLimit >= last)
    }

    private func setContextMode(_ rawValue: String) {
        contextModeRaw = rawValue
        defaults.set(rawValue, forKey: contextKeys.contextMode)
    }

    private func setAutoContextLimit(_ value: Int, usesMaximum: Bool) {
        let sanitized = LlamaContextPolicy.sanitizedContext(value)
        autoContextLimit = sanitized
        autoContextLimitUsesMaximum = usesMaximum
        defaults.set(sanitized, forKey: contextKeys.autoContextLimit)
        defaults.set(usesMaximum, forKey: contextKeys.autoContextLimitUsesMaximum)
    }

    private func setManualContext(_ value: Int, upperBound: Int? = nil) {
        let sanitized = LlamaContextPolicy.sanitizedContext(value)
        let bounded = upperBound.map { min(sanitized, max(LlamaContextPolicy.minimumContext, $0)) } ?? sanitized
        manualContext = bounded
        defaults.set(bounded, forKey: contextKeys.numCtx)
    }

    private func applyFixedContextIfNeeded() {
        guard case .fixed(let context) = contextControls else { return }
        let sanitized = LlamaContextPolicy.sanitizedContext(context)
        contextModeRaw = LlamaContextMode.manual.rawValue
        manualContext = sanitized
        defaults.set(LlamaContextMode.manual.rawValue, forKey: contextKeys.contextMode)
        defaults.set(sanitized, forKey: contextKeys.numCtx)
    }

    private func syncContextStateFromDefaults() {
        contextModeRaw = LlamaContextPolicy.currentMode(
            defaults: defaults,
            keys: contextKeys
        ).rawValue
        autoContextLimit = LlamaContextPolicy.autoContextLimit(
            defaults: defaults,
            keys: contextKeys
        )
        autoContextLimitUsesMaximum = LlamaContextPolicy.autoContextLimitUsesMaximum(
            defaults: defaults,
            keys: contextKeys
        )
        manualContext = LlamaContextPolicy.manualContext(
            defaults: defaults,
            keys: contextKeys
        )
    }

    private func refreshLibraryForContext() async {
        await library.refresh()
        normalizeSelectionForContext()
        refreshToken = UUID()
    }

    private func normalizeSelectionForContext() {
        if selectedModelID.isEmpty {
            selectedModelID = systemModels.first?.id ?? library.models.first?.id.uuidString ?? ""
            return
        }

        if selectedSystemModel == nil && selectedInstalledModel == nil {
            selectedModelID = systemModels.first?.id ?? library.models.first?.id.uuidString ?? ""
        }
    }

    private func preserveMaximumIntentIfNeeded(forSelectionID selectionID: String) {
        guard !autoContextLimitUsesMaximum,
              hasExplicitAutoContextLimitPreference,
              autoContextLimitTracksMaximum(forSelectionID: selectionID)
        else { return }

        autoContextLimitUsesMaximum = true
        defaults.set(true, forKey: contextKeys.autoContextLimitUsesMaximum)
    }

    private func prepareSelectedContextForCalibration(_ model: InstalledModel) {
        guard isSelectedInstalledModel(model) else { return }
        preserveMaximumIntentIfNeeded(forSelectionID: selectedModelID)
        syncContextStateFromDefaults()
    }

    private func refreshSelectedContextAfterCalibration(_ model: InstalledModel) {
        guard isSelectedInstalledModel(model) else { return }
        syncContextStateFromDefaults()
        refreshToken = UUID()
    }

    private func isSelectedInstalledModel(_ model: InstalledModel) -> Bool {
        guard case .installed(let id) = selectedSelection else { return false }
        return id == model.id
    }

    private func autoContextLimitTracksMaximum(forSelectionID selectionID: String) -> Bool {
        guard case .installed(let id) = LLMModelSelection(storageValue: selectionID),
              let model = library.model(id: id)
        else { return false }

        let record = calibrationRecord(for: model)
        let candidates = LocalLLMContextWindowTiers.candidates(
            trainingContext: model.contextLength,
            calibratedMaximum: record?.maximumSupportedContext
        )
        guard let last = candidates.last else { return false }
        return autoContextLimit >= last
    }

    private var hasExplicitAutoContextLimitPreference: Bool {
        defaults.object(forKey: contextKeys.autoContextLimit) != nil
    }

    private func startCalibration(_ model: InstalledModel) {
        guard activeCalibrationModelID == nil else { return }

        calibrationTask?.cancel()
        calibrationErrorMessage = nil
        activeCalibrationModelID = model.id
        calibrationProgress = LlamaContextCalibrationProgress(
            phase: .loadingModel,
            message: "Loading model"
        )

        let modelID = model.id
        let adapter = Self.makeCalibrationAdapter(
            library: library,
            store: calibrationStore,
            configuration: configuration,
            onCalibrationStarted: { model in
                prepareSelectedContextForCalibration(model)
            },
            onCalibrationCompleted: { model, _ in
                refreshSelectedContextAfterCalibration(model)
            }
        )
        calibrationTask = Task { @MainActor in
            do {
                _ = try await adapter.runCalibration(model) { progress in
                    guard activeCalibrationModelID == modelID else { return }
                    calibrationProgress = progress
                }
                guard !Task.isCancelled else { return }
                activeCalibrationModelID = nil
                calibrationProgress = nil
                calibrationTask = nil
                refreshToken = UUID()
            } catch is CancellationError {
                calibrationErrorMessage = "Calibration cancelled."
                activeCalibrationModelID = nil
                calibrationProgress = nil
                calibrationTask = nil
            } catch {
                calibrationErrorMessage = error.localizedDescription
                activeCalibrationModelID = nil
                calibrationProgress = nil
                calibrationTask = nil
            }
        }
    }

    private func cancelCalibration() {
        if activeCalibrationModelID != nil {
            calibrationErrorMessage = "Calibration cancelled."
        }
        calibrationTask?.cancel()
        calibrationTask = nil
        activeCalibrationModelID = nil
        calibrationProgress = nil
    }

    private func calibrationProgressText(_ progress: LlamaContextCalibrationProgress) -> String {
        var parts: [String] = []
        if let currentContext = progress.currentContext {
            parts.append("probing \(currentContext.formatted())")
        } else if let message = progress.message {
            parts.append(message)
        } else {
            parts.append(progress.phase.rawValue)
        }
        if let lastSuccessfulContext = progress.lastSuccessfulContext {
            parts.append("best \(lastSuccessfulContext.formatted())")
        }
        return parts.joined(separator: " · ")
    }
}

private extension View {
    @ViewBuilder
    func contextNumericKeyboard() -> some View {
#if os(iOS)
        keyboardType(.numberPad)
#else
        self
#endif
    }
}
