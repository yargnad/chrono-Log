# Mnema Mnemosyne Experience & OCR Upgrade Proposal

## Goals

- Preserve live screen stream while keeping semantic-search results visible and interactive.
- Introduce on-device OCR using Microsoft TrOCR (base-printed ONNX) with embedded SentencePiece tokenizer.
- Provide privacy-first face blurring and convenient link extraction so memories are safe to review and easy to navigate.

## Status — December 2025

- ✅ **Dependencies & State**: `sentencepiece` crate added, `AppState` holds encoder/decoder sessions, tokenizer, and in-memory OCR cache.
- ✅ **Resource Loader**: Startup now loads TrOCR encoder + both decoder variants (`decoder_model_quantized.onnx`, `decoder_with_past_model_quantized.onnx`) plus `sentencepiece.bpe.model`, logging a warning if assets are missing.
- ✅ **OCR Backend**: `ocr_memory` Tauri command performs greedy decoding, auto-link extraction, and cache validation against LanceDB timestamps.
- 🚧 **UX Integration**: Carousel/detail panes still need to surface OCR text, link chips, and manual refresh actions.
- 🚧 **Documentation**: Wider contributor docs must spell out asset downloads, pkg-config requirements, and expected command responses.

## Frontend Experience

1. **Layout Refresh**
   - Shrink the live "monitor" feed at search results return and pin it left; show real-time status + capture controls.
   - Center the UI on a vertical carousel of memory thumbnails produced by search results.
   - Add a floating bottom details pane that shows metadata (timestamp, vector score) and OCR text for the currently highlighted memory.
2. **Carousel Interactions**
   - Click/arrow keys: move highlight, update detail pane and preview image in place (without rerunning search).
   - Double-click: open fullscreen modal for the selected memory, toggle face blur, and trigger OCR fetch.
   - Keep results persistent even as new screenshots stream in; pulling another search replaces the carousel but does not auto-clear.
3. **Details Pane Enhancements**
   - Display OCR text and auto-link detected URLs / file paths.
   - Include buttons for "Uncloak Faces" and "Re-run OCR" (manual override) plus copy-to-clipboard shortcuts.

## Resource Checklist

| Path | Required Files | Notes |
| --- | --- | --- |
| `mnema/src-tauri/resources/clip-vision.onnx` / `clip-text.onnx` | Already tracked in Git. | CLIP models used for capture + search. |
| `mnema/src-tauri/resources/trocr-base/encoder_model_quantized.onnx` | Download from `microsoft/trocr-base-printed` or export via `optimum-cli`. | DirectML-friendly encoder. |
| `mnema/src-tauri/resources/trocr-base/decoder_model_quantized.onnx` | “Decoder init” graph that seeds cache. | **New**: must be present alongside the with-past variant. |
| `mnema/src-tauri/resources/trocr-base/decoder_with_past_model_quantized.onnx` | Incremental decoder for fast greedy decoding. | Requires cached KV tensors from the init pass. |
| `mnema/src-tauri/resources/trocr-base/sentencepiece.bpe.model` | Tokenizer file. | Loaded through the `sentencepiece` crate. |

Suggested export command (runs outside the repo):

```bash
optimum-cli export vision-encoder-decoder \
  --model microsoft/trocr-base-printed \
  --task vision2seq-lm \
  --opset 17 \
  --framework pt \
  trocr-base
```

Copy the resulting ONNX + tokenizer files into the resource folder; keep them git-ignored per `.gitignore`.

> **Build prerequisite**: Install `pkg-config` (e.g., `choco install pkgconfiglite`) so `sentencepiece-sys` can discover system libraries on Windows.

## Backend Additions

1. **TrOCR Integration** *(implemented)*
   - Initialization now builds three DirectML sessions (encoder + two decoders) and loads the SentencePiece tokenizer into `AppState`.
   - Helper pipeline: preprocess screenshot → encoder session → decoder init session (seeds past cache) → iterative decoder-with-past passes until EOS/PAD → SentencePiece decode → whitespace/link cleanup.
   - `OcrCacheEntry` stores `{timestamp, text, links}` per memory ID so repeated calls short-circuit unless LanceDB reports a newer snapshot.
2. **OCR Command Contract**
   - `#[tauri::command] async fn ocr_memory(memory_id: String, file_path: String, timestamp: String)`
   - Returns `OcrResult { id, timestamp, text, links, refreshed }`, where `refreshed` indicates whether the cache was updated during the call.
   - Frontend should call this after selecting a memory card; pass through the LanceDB path/timestamp so the backend can validate freshness without another DB round-trip.
3. **Face Privacy**
   - Introduce a lightweight face-detection model (e.g., YuNet ONNX) to find faces and generate blurred overlays by default.
   - Expose `toggle_face_blur(memory_id, enabled)` so the frontend can switch between blurred/unblurred versions on demand.
4. **Caching Strategy**
   - Store `(timestamp, result)` for each memory in-memory; optionally persist to disk later.
   - Auto-refresh when new screenshot timestamp differs; provide manual "force re-OCR" path.

## Documentation & Tooling

- Update `README.md` with setup instructions for TrOCR assets (download size, placement), DirectML expectations, and the `pkg-config` prerequisite. **Status:** README now covers prerequisites + asset table; extend to CONTRIBUTING next.
- Document contribution expectations in `CONTRIBUTING.md` for working on carousel UI, OCR tuning, and privacy features.
- Note hardware requirements (Windows 11, DirectML-capable GPU/NPU) and fallback behavior when DirectML is unavailable.

## Sequence of Work

1. **Resource Setup**: Add TrOCR + tokenizer files, ignore them in Git, ensure Tauri bundle includes them.
2. **Backend Plumbing**: Extend `AppState`, load sessions/tokenizer, implement OCR cache + tauri command, add optional face-blur command.
3. **UI Revamp**: Build new layout + carousel, persistent results, fullscreen modal, details pane interactions.
4. **OCR/Privacy UX**: Hook frontend to new commands, display OCR text, linkify content, add face-blur toggle.
5. **Docs & QA**: Update README/CONTRIBUTING, test on Windows 11 with DirectML and CPU fallback, verify caching & privacy flows.
