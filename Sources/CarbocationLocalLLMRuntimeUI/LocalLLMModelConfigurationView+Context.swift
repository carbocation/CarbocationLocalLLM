import CarbocationLocalLLM
import CarbocationLocalLLMRuntime
import CarbocationLocalLLMUI
import SwiftUI

extension LocalLLMModelConfigurationView {
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
    var contextSection: some View {
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

    var selectedSelection: LLMModelSelection? {
        LLMModelSelection(storageValue: selectedModelID)
    }

    var selectedInstalledModel: InstalledModel? {
        guard case .installed(let id) = selectedSelection else { return nil }
        return library.model(id: id)
    }

    var selectedSystemModel: LLMSystemModelOption? {
        systemModels.first { $0.id == selectedModelID }
    }

    var pickerCalibrationAdapter: ModelLibraryPickerCalibrationAdapter? {
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

    func fixedContextSection(_ context: Int) -> some View {
        let sanitized = LlamaContextPolicy.sanitizedContext(context)
        return HStack(spacing: 10) {
            Label("Context Window", systemImage: "rectangle.expand.vertical")
            Spacer()
            Text("Fixed at \(sanitized.formatted()) tokens")
                .foregroundStyle(.secondary)
        }
        .font(.callout)
    }

    var noContextSelectionSection: some View {
        HStack(spacing: 10) {
            Label("Context Window", systemImage: "rectangle.expand.vertical")
            Spacer()
            Text("Select a model to configure context.")
                .foregroundStyle(.secondary)
        }
        .font(.callout)
    }

    var unresolvedContextSelectionSection: some View {
        HStack(spacing: 10) {
            Label("Context Window", systemImage: "rectangle.expand.vertical")
            Spacer()
            Text("Loading selected model context...")
                .foregroundStyle(.secondary)
        }
        .font(.callout)
    }

    func systemContextSection(_ option: LLMSystemModelOption) -> some View {
        HStack(spacing: 10) {
            Label("Context Window", systemImage: "rectangle.expand.vertical")
            Spacer()
            Text("Managed by \(option.displayName) · \(option.contextLength.formatted()) tokens")
                .foregroundStyle(.secondary)
        }
        .font(.callout)
    }

    func installedContextSection(_ model: InstalledModel) -> some View {
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

    func contextStatus(record: LlamaContextCalibrationRecord?, model: InstalledModel) -> some View {
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

    func calibrationProgressRow(_ progress: LlamaContextCalibrationProgress) -> some View {
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

    func calibrationRecord(for model: InstalledModel) -> LlamaContextCalibrationRecord? {
        calibrationStore.record(
            for: model,
            runtime: LocalLLMEngine.contextCalibrationRuntimeFingerprint(configuration: configuration)
        )
    }

    func manualContextUpperBound(for model: InstalledModel) -> Int {
        max(
            LlamaContextPolicy.minimumContext,
            model.contextLength > 0 ? model.contextLength : 262_144
        )
    }

    func manualContextControls(for model: InstalledModel) -> some View {
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

    func resolvedAutoContextLimit(in candidates: [Int]) -> Int {
        guard !candidates.isEmpty else { return autoContextLimit }
        if autoContextLimitTracksMaximum(in: candidates), let last = candidates.last {
            return last
        }
        let index = LocalLLMContextWindowTiers.nearestIndex(for: autoContextLimit, in: candidates)
        return candidates[index]
    }

    func autoContextLimitText(in candidates: [Int]) -> String {
        let resolved = resolvedAutoContextLimit(in: candidates)
        if autoContextLimitTracksMaximum(in: candidates) {
            return "Auto limit: Max (\(resolved.formatted()) tokens)"
        }
        return "Auto limit: \(resolved.formatted()) tokens"
    }

    func autoContextLimitTracksMaximum(in candidates: [Int]) -> Bool {
        guard let last = candidates.last else { return autoContextLimitUsesMaximum }
        return autoContextLimitUsesMaximum
            || (hasExplicitAutoContextLimitPreference && autoContextLimit >= last)
    }

    func setContextMode(_ rawValue: String) {
        contextModeRaw = rawValue
        defaults.set(rawValue, forKey: contextKeys.contextMode)
    }

    func setAutoContextLimit(_ value: Int, usesMaximum: Bool) {
        let sanitized = LlamaContextPolicy.sanitizedContext(value)
        autoContextLimit = sanitized
        autoContextLimitUsesMaximum = usesMaximum
        defaults.set(sanitized, forKey: contextKeys.autoContextLimit)
        defaults.set(usesMaximum, forKey: contextKeys.autoContextLimitUsesMaximum)
    }

    func setManualContext(_ value: Int, upperBound: Int? = nil) {
        let sanitized = LlamaContextPolicy.sanitizedContext(value)
        let bounded = upperBound.map { min(sanitized, max(LlamaContextPolicy.minimumContext, $0)) } ?? sanitized
        manualContext = bounded
        defaults.set(bounded, forKey: contextKeys.numCtx)
    }

    func applyFixedContextIfNeeded() {
        guard case .fixed(let context) = contextControls else { return }
        let sanitized = LlamaContextPolicy.sanitizedContext(context)
        contextModeRaw = LlamaContextMode.manual.rawValue
        manualContext = sanitized
        defaults.set(LlamaContextMode.manual.rawValue, forKey: contextKeys.contextMode)
        defaults.set(sanitized, forKey: contextKeys.numCtx)
    }

    func syncContextStateFromDefaults() {
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

    func refreshLibraryForContext() async {
        await library.refresh()
        normalizeSelectionForContext()
        refreshToken = UUID()
    }

    func normalizeSelectionForContext() {
        if selectedModelID.isEmpty {
            selectedModelID = systemModels.first?.id ?? library.models.first?.id.uuidString ?? ""
            return
        }

        if selectedSystemModel == nil && selectedInstalledModel == nil {
            selectedModelID = systemModels.first?.id ?? library.models.first?.id.uuidString ?? ""
        }
    }

    func preserveMaximumIntentIfNeeded(forSelectionID selectionID: String) {
        guard !autoContextLimitUsesMaximum,
              hasExplicitAutoContextLimitPreference,
              autoContextLimitTracksMaximum(forSelectionID: selectionID)
        else { return }

        autoContextLimitUsesMaximum = true
        defaults.set(true, forKey: contextKeys.autoContextLimitUsesMaximum)
    }

    func prepareSelectedContextForCalibration(_ model: InstalledModel) {
        guard isSelectedInstalledModel(model) else { return }
        preserveMaximumIntentIfNeeded(forSelectionID: selectedModelID)
        syncContextStateFromDefaults()
    }

    func refreshSelectedContextAfterCalibration(_ model: InstalledModel) {
        guard isSelectedInstalledModel(model) else { return }
        syncContextStateFromDefaults()
        refreshToken = UUID()
    }

    func isSelectedInstalledModel(_ model: InstalledModel) -> Bool {
        guard case .installed(let id) = selectedSelection else { return false }
        return id == model.id
    }

    func autoContextLimitTracksMaximum(forSelectionID selectionID: String) -> Bool {
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

    var hasExplicitAutoContextLimitPreference: Bool {
        defaults.object(forKey: contextKeys.autoContextLimit) != nil
    }

    func startCalibration(_ model: InstalledModel) {
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

    func cancelCalibration() {
        if activeCalibrationModelID != nil {
            calibrationErrorMessage = "Calibration cancelled."
        }
        calibrationTask?.cancel()
        calibrationTask = nil
        activeCalibrationModelID = nil
        calibrationProgress = nil
    }

    func calibrationProgressText(_ progress: LlamaContextCalibrationProgress) -> String {
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
