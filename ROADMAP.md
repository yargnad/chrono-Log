# ChronoLog Roadmap

Last updated: **2025-12-01**

## Phase 0 — Pulse (✅ Complete)

- Continuous capture loop using CLIP vision embeddings saved to LanceDB.
- Minimal Tauri monitor displaying the live stream and frame counter.
- Semantic search command wiring between the UI and Rust backend.

## Phase 1 — Recall Desk (🚧 In Progress)

### Goals

- Persistent recall UI: live preview docked left, vertical carousel for search results, floating detail pane, fullscreen modal.
- On-device OCR via Microsoft TrOCR (base-printed ONNX) + embedded SentencePiece tokenizer with auto-refreshing cache.
- Privacy overlays that blur faces in thumbnails by default with an "uncloak" toggle.

### Status

- ✅ Design proposal documented in `PROPOSED_CHANGES.md` and README.
- ⬜ Bundle TrOCR resources + tokenizer under `src-tauri/resources`.
- ⬜ Implement OCR + caching command (`ocr_memory`).
- ⬜ Add face-detection helper and thumbnail blur toggle.
- ⬜ Ship new carousel/details UI and hook up OCR/blur interactions.

## Phase 2 — Canvas (📝 Planned)

- Scriptable overlay APIs so collaborators can create widgets/agents atop the feed.
- Custom workflows (alerts, journaling, tagging, automations).
- Documentation + SDK samples for extending ChronoLog locally.

## Phase 3 — Distribution (📝 Planned)

- Hardened installer pipeline (MSIX/winget) with automatic updates.
- Release checklist + regression tests for Windows 11 builds.

## How to Help

- Tackle any unchecked tasks above and open a PR referencing the roadmap section.
- Propose additional milestones via issues if you have ideas for sensors, agents, or UX improvements.
