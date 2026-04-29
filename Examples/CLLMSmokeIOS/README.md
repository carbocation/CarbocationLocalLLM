# CLLMSmokeIOS

`CLLMSmokeIOS` is a small iOS demo app for validating CarbocationLocalLLM in an app-style environment.

It includes:

- `ModelLibraryPickerView` for GGUF download/import/delete/selection
- Apple Intelligence system-model selection when available
- Prompt input and generation output
- Streaming event log

The demo uses the package's iOS llama defaults: CPU-only GGUF loading, memory mapping enabled, and a smaller initial context to keep first-run memory pressure predictable.

## Run

From the repository root, build the local multi-platform llama artifact:

```sh
Scripts/build-llama-xcframework.sh
```

Then open the top-level Xcode project from Finder or Xcode:

```text
CarbocationLocalLLM.xcodeproj
```

Select the `CLLMSmokeIOS` scheme and an iOS device or simulator.

For command-line validation:

```sh
xcodebuild \
  -project CarbocationLocalLLM.xcodeproj \
  -scheme CLLMSmokeIOS \
  -destination 'generic/platform=iOS' \
  -derivedDataPath .build/CLLMSmokeIOSDerivedData \
  CODE_SIGNING_ALLOWED=NO \
  build
```
