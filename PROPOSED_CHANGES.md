# Mnema Mnemosyne Experience & OCR Upgrade Proposal

## Goals

- Preserve live screen stream while keeping semantic-search results visible and interactive.
- Introduce on-device OCR using Microsoft TrOCR (base-printed ONNX) with embedded SentencePiece tokenizer.
- Provide privacy-first face blurring and convenient link extraction so memories are safe to review and easy to navigate.

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

## Backend Additions

1. **TrOCR Integration**
   - Place TrOCR base ONNX weights + `spm.model` under `mnema/src-tauri/resources/trocr-base/` and ship them via `tauri.conf.json`.
   - Add `sentencepiece` crate; extend `AppState` with `trocr_session`, `trocr_tokenizer`, and an OCR cache map keyed by memory ID.
   - New helper flow: load PNG → preprocess to TrOCR tensor → run ONNX session (DirectML, fallback CPU) → decode tokens via SentencePiece → post-process text + links.
2. **OCR Command**
   - `#[tauri::command] async fn ocr_memory(id: String)`
     - Resolve snapshot path/timestamp from LanceDB entry.
     - Check cache; if timestamps differ, re-run OCR and refresh cache.
     - Return `{ text, links, timestamp, refreshed }`.
3. **Face Privacy**
   - Introduce a lightweight face-detection model (e.g., YuNet ONNX) to find faces and generate blurred overlays by default.
   - Expose `toggle_face_blur(memory_id, enabled)` so the frontend can switch between blurred/unblurred versions on demand.
4. **Caching Strategy**
   - Store `(timestamp, result)` for each memory in-memory; optionally persist to disk later.
   - Auto-refresh when new screenshot timestamp differs; provide manual "force re-OCR" path.

## Documentation & Tooling

- Update `README.md` with setup instructions for TrOCR assets (download size, placement) and explain OCR/face-blur behavior.
- Document contribution expectations in `CONTRIBUTING.md` for working on carousel UI, OCR tuning, and privacy features.
- Note hardware requirements (Windows 11, DirectML-capable GPU/NPU) and fallback behavior when DirectML is unavailable.

## Sequence of Work

1. **Resource Setup**: Add TrOCR + tokenizer files, ignore them in Git, ensure Tauri bundle includes them.
2. **Backend Plumbing**: Extend `AppState`, load sessions/tokenizer, implement OCR cache + tauri command, add optional face-blur command.
3. **UI Revamp**: Build new layout + carousel, persistent results, fullscreen modal, details pane interactions.
4. **OCR/Privacy UX**: Hook frontend to new commands, display OCR text, linkify content, add face-blur toggle.
5. **Docs & QA**: Update README/CONTRIBUTING, test on Windows 11 with DirectML and CPU fallback, verify caching & privacy flows.
