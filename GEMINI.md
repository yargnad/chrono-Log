# Mnema Terminology & Phase Plan

🏛️ Mnema Identity: Final Terminology

| Component | Term | Etymology / Meaning |
| --- | --- | --- |
| Application Name | Mnema | Memory, monument, permanent record |
| Phase 1 (Recall/Retrieval) | Mnemosyne | Titan goddess of memory |
| Safety Protocol | Xechno | To forget, to not remember (filters unwanted content) |
| Heuristic System | Heuristikon | The automaton that discovers |
| Focus / Emotion Tag | Pathos Tag | Tracks deep feeling / emotion |

📝 Updated Feature Plan

The roadmap now uses Mnema, Mnemosyne, Xechno, Heuristikon, and Pathos Tag everywhere. Phase 1 focuses on four pillars described below.

## Phase 1 — Mnemosyne (Retrieval Desk)

### 1. Persistent Recall UI (Layout & Interactivity)

This goal establishes the core functional interface for memory retrieval.

#### Frontend Requirements — Mnemosyne UI

- **Layout refresh:** Transition from the full-screen live feed to a split-panel view once search results arrive.
  - **Live stream:** Shrink the live feed and dock it to the top-left while keeping real-time updates via the `new-screenshot` event.
  - **Search carousel:** Dedicate the main area to a vertically scrolling carousel with thumbnail previews.
  - **Detail pane:** Float a pane (bottom or right) that surfaces metadata for the highlighted memory.
- **Interaction model:**
  - **Persistence:** Results remain pinned until a new search or manual clear occurs.
  - **Navigation:** Support mouse clicks and keyboard arrows for carousel traversal.
  - **Fullscreen modal:** Double-click a thumbnail to open fullscreen review tools (blur toggle, manual OCR, etc.).

### 2. On-Device OCR & Interaction (Coordinate-Based Recall)

This goal integrates low-latency, privacy-centric text recognition.

#### Backend Requirements — OCR Pipeline

- **Multi-stage pipeline:** `ocr_memory(id)` uses YuNet (text detection) followed by TrOCR (text recognition).
- **Data structure:**

```rust
pub struct OcrItem {
    text: String,
    x_min: u32,
    y_min: u32,
    width: u32,
    height: u32,
    item_type: String, // URL, FILE_PATH, PLAIN_TEXT
}
```

- **Post-processing:** Classify each `OcrItem` using regex-style heuristics for URLs, file paths, or plain text.

#### Frontend Requirements — OCR Interaction

- **Meta-Car:** A vertical carousel inside the detail pane listing `OcrItem` results.
- **Canvas overlays:** Layer an HTML canvas over the preview image to draw bounding boxes.
  1. Scale normalized coordinates (0–1000) to match the current image size.
  2. Highlight the box that aligns with the selected item in the Meta-Car.
- **Interactive hooks:**
  - Clicking a Meta-Car entry uses `shell:open` to launch URLs or file paths.
  - `Ctrl + Click` on a bounding box triggers the same `shell:open` call.

### 3. File Version Snapshot (Cryptographic Hashing)

This feature ensures the memory references a precise file version.

#### Backend Requirements — File Hashing

- **Hashing:** Use BLAKE3 for fast hashing.
- **Segmented strategy:**
  - `< 5 MB`: hash entire file.
  - `> 5 MB`: hash first and last 64 KB.
  - `> 2 GB`: skip, emit `FILE_TOO_LARGE`.
- **Capture loop integration:** Detect when the active window references a file path and hash immediately.
- **LanceDB schema:** Add `file_path` and `file_hash` fields.

#### Frontend Requirements — File Context

- Display both `file_path` and `file_hash` in the detail pane for the selected memory.

### 4. Privacy & Safety Core (Xechno Protocol & Face Blur)

This goal enforces the privacy covenant.

#### Backend Requirements — Xechno Protocol

- **Xechno protocol:** Run a local ONNX classifier per capture. If it flags sensitive content, set `is_saving_allowed = false` and skip persistence until conditions clear.
- **Face privacy:** Use YuNet to detect faces and store blurred thumbnails by default.
- **Uncloak feature:** `toggle_face_blur(memory_id)` streams the original image on demand.
- **Safety alert:** Emit a `safety-alert` event whenever capture pauses or resumes.

#### Frontend Requirements — Safety Experience

- **Alert banner:** Show a persistent warning (e.g., red bar) when a `safety-alert` pause event arrives, such as “🚨 Xechno Protocol Active: Recording paused until content is clear.”
- **Recording indicator:** Switch the status UI between “Recording” and “Paused.”
- **Uncloak toggle:** Provide a prominent button in the detail pane to call `toggle_face_blur`.

## 🌿 Git Strategy

- Create a focused branch such as `feature/branding-mnema` for the rename.
- Use a descriptive commit like `docs: Rename Phase 1 to Mnemosyne and implement Xechno terminology.`

## 📝 Files to Update

1. `ROADMAP.md` — adopt the Mnemosyne heading (completed).
2. `README.md` and `CONTRIBUTING.md` — enforce Mnema / Mnemosyne / Xechno wording consistently.
3. `mnema/src-tauri/src/lib.rs` — update log strings such as `println!("DEBUG: Starting Mnemosyne Core...");` and safety messaging.
4. `mnema/src/index.html` and `mnema/src/main.js` — refresh UI titles, alerts, and CTA copy with the new terminology.
