# Mnema Roadmap

Last updated: **2025-12-01**

## Phase 0 — Pulse (✅ Complete)

- Continuous capture loop using CLIP vision embeddings saved to LanceDB.
- Minimal Tauri monitor displaying the live stream and frame counter.
- Semantic search command wiring between the UI and Rust backend.

***

## Phase 1 — Mnemosyne Retrieval Desk (🚧 In Progress)

### Goals

- **Persistent Mnemosyne UI:** Live preview docked left, vertical carousel for search results, floating detail pane, fullscreen modal.
- **On-Device OCR & Interaction:** **Interactive, coordinate-based** OCR via Microsoft TrOCR + SentencePiece, supporting a clickable **Meta-Car** for links, files, and key phrases.
- **File Version Snapshot:** **Cryptographic content hashing** of active files (using segmented hashing for large files) to link memory records to the precise version of a file.
- **Xechno Privacy Protocol:** Blur faces in thumbnails by default with an "uncloak" toggle plus pause alerts when content blocks persistence.

### Status

- ✅ Design proposal documented in `PROPOSED_CHANGES.md` and README.
- ⬜ Bundle TrOCR resources + tokenizer under `src-tauri/resources`.
- ⬜ **Implement Segmented File Hashing** (BLAKE3) for active file versions (speed-optimized).
- ⬜ **Update LanceDB schema** to store `file_hash` and `file_path` in memory records.
- ⬜ **Implement OCR & Bounding Box Logic** using the lightweight YuNet ONNX detector (running via DirectML) combined with TrOCR to return `(x, y, w, h)` coordinates.
- ⬜ Implement OCR + caching command (`ocr_memory`).
- ⬜ Add face-detection helper, thumbnail blur toggle, and Xechno pause event.
- ⬜ Ship new Mnemosyne carousel/details UI and hook up OCR/blur interactions.

***

## Phase 2 — Canvas (📝 Planned)

- **Scriptable Overlay APIs** so collaborators can create widgets/agents atop the feed.
- Custom workflows (alerts, journaling, tagging, automations).
- Documentation + SDK samples for extending Mnema locally.

***

## Phase 3 — Distribution (📝 Planned)

- Hardened installer pipeline (MSIX/winget) with automatic updates.
- Release checklist + regression tests for Windows 11 builds.

---
    
---

## Phase 1 — Mnemosyne Retrieval Desk: Detailed Goal Breakdown

The following is a detailed breakdown of the four primary goals for Phase 1, outlining the required technical steps for both the Rust Backend and the JavaScript Frontend.

## 1. Persistent Recall UI (Layout & Interactivity)

This goal establishes the core functional interface for memory retrieval.

### Frontend (UI/UX) Requirements
* **Layout Refresh:** The application must transition from the initial full-screen live feed to a split-panel view upon search execution.
    * **Live Stream:** Shrink the live image feed and dock it to the top-left corner. It must continue to update in real-time (via the `new-screenshot` event) without disturbing search results.
    * **Search Carousel:** The main area will be occupied by a vertically scrolling carousel displaying thumbnail previews of search results.
    * **Detail Pane:** A floating panel (docked to the bottom or right) displays metadata for the **currently highlighted** memory.
* **Interaction Model:**
    * **Persistence:** Search results must remain displayed until a new search is initiated, or a manual "clear" command is given.
    * **Navigation:** Enable selection of memories in the carousel via mouse click and keyboard arrow keys.
    * **Fullscreen Modal:** Double-clicking a thumbnail opens a fullscreen, high-resolution view of the memory, used for in-depth review, blurring, and manual OCR fetching.

***

## 2. On-Device OCR & Interaction (Coordinate-Based Recall)

This goal integrates low-latency, privacy-centric text recognition and detection to create interactive memory elements.

### Backend (Rust Logic) Requirements
* **Multi-Stage AI Pipeline:** The `ocr_memory(memory_id: String)` command must implement a two-stage process:
    1.  **Text Detection:** Use the lightweight YuNet ONNX model (accelerated with DirectML) to identify all bounding boxes (`x, y, w, h`) containing text on the high-resolution screenshot.
    2.  **Text Recognition (TrOCR):** Crop the image for each detected bounding box and pass the cropped section to the TrOCR encoder/decoder session.
* **Data Structure:** Define the `OcrItem` struct for serialization:
    ```rust
    pub struct OcrItem {
        text: String,
        x_min: u32, y_min: u32, width: u32, height: u32,
        item_type: String, // "URL", "FILE_PATH", "PLAIN_TEXT"
    }
    ```
* **Post-Processing:** Implement logic to parse recognized text to classify its `item_type` (e.g., using regex to identify `http://` or file extension paths).

### Frontend (UI/UX) Requirements
* **Meta-Car:** Implement a **vertical carousel** within the detail pane, listing all `OcrItem`s for the selected memory.
* **Canvas Overlays:** The main image preview area must use a **layered HTML Canvas** over the `<img>` element. The JavaScript must be able to:
    1.  Scale the normalized coordinates (0-1000) from the backend to the current pixel dimensions of the image.
    2.  Draw and highlight the bounding box corresponding to the currently selected `OcrItem` in the Meta-Car.
* **Interactive Hooks:** Implement the following click handlers:
    * **Meta-Car Click:** Standard click on an item in the Meta-Car calls the Tauri `shell:open` API command with the extracted URL or file path.
    * **Bounding Box Click:** `Ctrl + Click` on the bounding box drawn on the canvas also calls the Tauri `shell:open` command for the corresponding link/path.

***

## 3. File Version Snapshot (Cryptographic Hashing)

This feature provides immutable version integrity for files present during the capture.

### Backend (Rust Logic) Requirements
* **Hashing Algorithm:** Implement the high-speed **BLAKE3** algorithm for hashing file contents.
* **Segmented Hashing Logic:** Implement a non-blocking, asynchronous file I/O strategy (`tokio::fs`) to prevent UI freezes:
    * If File Size **< 5 MB**: Hash the entire file content.
    * If File Size **> 5 MB**: Hash only the **first 64 KB** and the **last 64 KB** of the file to maintain integrity without performance cost.
    * If File Size **> 2 GB**: Skip hashing and log the event (`FILE_TOO_LARGE`).
* **Capture Loop Integration:** Augment the `start_screen_capture_loop` with a heuristic to detect if the active window title corresponds to a file path. If detected, calculate the hash.
* **Database Schema Update:** Modify the LanceDB schema to include:
    * `file_path`: Utf8 (The file's location during capture).
    * `file_hash`: Utf8 (The BLAKE3 version fingerprint, or "N/A").

### Frontend (UI/UX) Requirements
* **Detail Pane Display:** The floating detail pane must clearly display the captured `file_path` and `file_hash` for the selected memory.

***

## 4. Privacy Overlays (Face Blurring)

This feature ensures that sensitive visual data (faces) is obfuscated by default, maintaining the anti-Capitalist, privacy-first mandate.

### Backend (Rust Logic) Requirements

- **Face Detection Model:** Integrate the lightweight YuNet ONNX detector (also via DirectML) to locate faces on each captured image.
- **Image Processing:** Implement a blurring filter in the Rust backend's image processing pipeline.
- **Image Storage:** When a screenshot is saved, the detection logic must identify faces and store the **blurred version** of the image bytes as the primary thumbnail/preview used by the search results.
- **Command:** Implement the `toggle_face_blur(memory_id, enabled)` Tauri command. This command resolves the memory's original (unblurred) image data from disk and serves it back to the frontend on request.

### Frontend (UI/UX) Requirements — Privacy

- **Default State:** All thumbnails and preview images displayed in the carousel/detail pane must default to the **blurred** state.
- **Uncloak Toggle:** The details pane must include a prominent "Uncloak Faces" button that calls the `toggle_face_blur` command to swap the displayed image source from the blurred version to the original unblurred version.

***

## How to Help

- Tackle any unchecked tasks above and open a PR referencing the roadmap section.
- Propose additional milestones via issues if you have ideas for sensors, agents, or UX improvements.
