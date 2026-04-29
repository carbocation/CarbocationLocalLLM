# CLLMDemoIOS

`CLLMDemoIOS` is a richer interactive iOS sandbox for exercising CarbocationLocalLLM by hand. It is *not* the same as the root project's `CLLMSmokeIOS` scheme — that one runs an automated JSON smoke test and prints `smoke: ok`. This demo is for exploratory use: pick a model, edit prompts, run, cancel, watch streaming events.

It includes:

- `ModelLibraryPickerView` for GGUF download/import/delete/selection
- Apple Intelligence system-model selection when available
- Editable system prompt and user prompt
- Run / Cancel buttons
- Free-form output pane and a streaming-event log

The demo uses the package's iOS llama defaults: CPU-only GGUF loading, memory mapping enabled, and a smaller initial context to keep first-run memory pressure predictable.

## Run

From the repository root, build the local multi-platform llama artifact:

```sh
Scripts/build-llama-xcframework.sh
```

Then open this example's standalone Xcode project:

```sh
open Examples/CLLMDemoIOS/CLLMDemoIOS.xcodeproj
```

Select the `CLLMDemoIOS` scheme and an iOS device or simulator, then run.

For command-line validation:

```sh
xcodebuild \
  -project Examples/CLLMDemoIOS/CLLMDemoIOS.xcodeproj \
  -scheme CLLMDemoIOS \
  -destination 'generic/platform=iOS' \
  -derivedDataPath .build/CLLMDemoIOSDerivedData \
  CODE_SIGNING_ALLOWED=NO \
  build
```

For the automated `smoke: ok` flow used in CI, use the root project's `CLLMSmokeMac` and `CLLMSmokeIOS` schemes instead. See the root README's "Smoke apps" section.
