# Vision App Integration Guide

This library accepts image input only as in-memory request payloads. The app owns attachment storage, file access, encryption, filenames, attachment IDs, and history policy.

## Ownership Boundaries

The app stores attachments. The library does not persist image bytes, write them to model storage, cache them across requests, or manage attachment files. During a single request, the library may transiently copy, decode, normalize, tokenize, and evaluate image bytes.

Before calling the library, the app reads or decrypts the image bytes and creates an `LLMImageInput`:

```swift
let image = LLMImageInput.encoded(data: imageData, mimeType: "image/png")
let message = LLMChatMessage(role: .user, content: [
    .text("Describe this image."),
    .image(image)
])
```

`LLMImageInput` has no file path, filename, attachment ID, caption, or diagnostic metadata. Keep app attachment IDs and filenames in app models. If the model should know a filename, OCR result, dimensions, or caption, add that as adjacent `.text(...)` content.

## Conversation History

A later turn can inspect a prior image only if the app includes that image bytes again in the later request. The library does not remember prior image payloads after generation completes.

This means the app decides history policy:

- Include recent image bytes again when follow-up questions should reference them.
- Omit image bytes when the next turn should be text-only.
- Add app-owned OCR, captions, or summaries as text if that is the desired fallback.
- Reject or downscale large images according to the app's product and storage policy.

The library still validates every image it receives and returns clear errors for unsupported formats, invalid RGB layout, decode failures, tokenization failures, and context budget failures.

## Capability And Preflight

Check model capability before showing image controls or sending images:

```swift
let info = await engine.currentLoadedModelInfo()
guard info?.supportsVision == true else {
    // Hide image sending or show a model capability error.
    return
}
```

For GGUF model families that ship vision separately, install the text model and its companion
`mmproj` GGUF. Hugging Face downloads and Hugging Face cache discovery include matching
`mmproj` artifacts automatically when the repository exposes them. For local file import,
select both files, or keep the companion `mmproj*.gguf`/`*-mmproj.*.gguf` readable next to the
text model file.

Run preflight before generation:

```swift
let preflight = try await engine.preflight(messages: [message], options: options)
guard preflight.canGenerate else {
    // Ask the user to shorten context, remove images, or choose a larger context.
    return
}
```

Preflight validates image placement and data, counts text plus image context cost for vision-capable llama models, and reports errors before generation whenever possible.

## Format Rules

Encoded images can be PNG, JPEG, HEIC, or HEIF when platform ImageIO can decode them. If `mimeType` is omitted, the library sniffs by magic bytes. If `mimeType` is present and conflicts with sniffed content, the request fails. EXIF orientation is honored during decode and normalized to top-left row-major RGB.

Raw RGB images must be tightly packed `rgb8`:

- `width > 0`
- `height > 0`
- `data.count == width * height * 3`
- Row-major 8-bit RGB order
- No alpha, stride, planar data, or color profile interpretation

## Current Runtime Limits

For v1, image parts are accepted only in user messages. Prompt cache reuse, MTP acceleration, and tool generation are disabled for image requests. Apple Intelligence remains text-only and reports only `.text` support.
