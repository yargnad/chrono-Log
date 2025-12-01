# Mnema

> **The antidote to corporate surveillance.** > *Part of [The Authentic Rebellion] suite of projects.*

![Status](https://img.shields.io/badge/Status-Pre--Alpha-red)
![License](https://img.shields.io/badge/License-AGPLv3-blue)
![Platform](https://img.shields.io/badge/Platform-Windows%2011-0078D6)

**Mnema** is a privacy-first, open-source alternative to Microsoft Recall. It is designed to act as your digital memory without compromising your dignity or data sovereignty.

Unlike proprietary solutions that treat your data as a commodity, Mnema runs entirely **offline**. It leverages modern local hardware (NPUs and GPUs) to capture, index, and analyze your screen activity, keeping 100% of the processing and storage on your own device.

## 🧠 Current Capabilities

- **Live Screen Stream**: Background capture loop (Rust + DirectML) emits a constant PNG feed to the Tauri UI so you can watch Mnema think in real time.
- **Vector Memory Store**: Every snapshot is embedded with CLIP and stored in LanceDB alongside timestamps and file paths for instant recall.
- **Semantic Search UI**: Ask natural-language questions from the Mnema desktop client; results stream back immediately even while the capture loop continues.
- **Offline-Only Pipeline**: No network calls, cloud storage, or telemetry. Models and data never leave your machine.

## 🔭 Near-Term Vision

The next milestone focuses on making recall usable, safe, and collaborative:

1. **Mnemosyne Retrieval Desk** – Persistent vertical carousel of search results, floating detail pane, and fullscreen viewer so results stay visible while capture continues.
2. **On-Device OCR** – Microsoft TrOCR (base printed) + embedded SentencePiece tokenizer to extract text/links from snapshots on-demand.
3. **Xechno Privacy Filters** – Default blurred faces in thumbnail previews with an "uncloak" toggle; full-resolution originals stay intact for face-based search.
4. **Smart Caching** – OCR outputs cached per memory (auto-refresh when the underlying screenshot changes) to keep responses instant.
5. **Heuristikon + Pathos Playground** – Documented hooks so contributors can extend the Heuristikon automation system or build Pathos Tag rituals on top of the memory stream.

## 🌟 Core Philosophy

We believe the world has become too motivated by capitalist gains at the expense of human dignity. Mnema is built on the following pillars:

- **Privacy is a Human Right:** No data leaves your machine. Ever.
- **Local-First:** We prioritize local processing power (NPU/GPU) over cloud dependency.
- **User Agency:** The interface is yours to mold. The frontend is fully scriptable and customizable.

## 🚀 Key Features

- **Hybrid Architecture:** A high-performance **Rust** backend handles the heavy lifting, while a lightweight web-based frontend provides a flexible user interface.
- **Hardware Acceleration:** Built on the **ONNX Runtime** and **DirectML**, Mnema specifically targets dedicated **NPUs** (Neural Processing Units) for efficient, low-power background processing, while leveraging **GPUs** for heavy on-demand tasks.
- **Hackable UI:** The frontend is designed for scripters. Users can modify the interface, create custom overlays, and build widgets using standard HTML, CSS, and JavaScript.
- **Semantic Search:** "Recall" past actions using natural language (e.g., *"Show me the article about pottery I was reading last Tuesday"*).

## 🛠️ Tech Stack

- **Core:** Rust 🦀
- **App Framework:** Tauri v2
- **AI/ML Runtime:** ONNX Runtime (with DirectML execution provider)
- **Database:** LanceDB (Local vector database for semantic search)
- **Frontend:** HTML/JS/CSS (Framework agnostic, designed for hackability)

## 📦 Installation & Development

*Note: Mnema is currently in active pre-alpha development.*

### Prerequisites

- **Windows 11** (Current primary target)
- [Rust](https://www.rust-lang.org/tools/install) (`rustup`)
- [Node.js](https://nodejs.org/) (LTS)
- Microsoft C++ Build Tools (via Visual Studio Installer)

### Setting up the Environment

1. Clone the repository:

    ```bash
    git clone [https://github.com/YOUR_USERNAME/mnema.git](https://github.com/YOUR_USERNAME/mnema.git)
    cd mnema
    ```

2. Install frontend dependencies:

    ```bash
    npm install
    ```

3. Run the application in development mode:

    ```bash
    npm run tauri dev
    ```

## 🗺️ Roadmap

- [x] **Phase 0 — Pulse**
    - [x] Continuous capture loop with CLIP embeddings + LanceDB storage.
    - [x] Minimal Tauri interface with semantic search invocation.
- [ ] **Phase 1 — Mnemosyne Retrieval Desk** *(in progress)*
    - [ ] Carousel-based results layout & fullscreen viewer.
    - [ ] TrOCR-powered OCR command with caching + linkification.
    - [ ] Xechno protocol integration: thumbnail face-blur overlays, manual uncloaking, and pause alerts.
- [ ] **Phase 2 — Canvas**
    - [ ] Scriptable overlay APIs, plugin surface for local agents.
    - [ ] Customizable workflows (alerts, journaling, tagging).
- [ ] **Phase 3 — Distribution**
    - [ ] Installers + `winget` package.
    - [ ] Hardened update channel and release QA.

## 🤝 Contributing

We welcome fellow travelers! Whether you are a Rustacean, a frontend wizard, or an AI enthusiast, your help is needed.

**Specific Help Needed:**

- **Rust/Systems Programming:** We are looking for assistance in optimizing the screen capture pipeline and NPU hardware abstraction layers.
- **AI/ML:** Help optimizing ONNX models for DirectML on Windows.

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📜 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** - ensuring that all modifications and network deployments remain open source. See the `LICENSE` file for details.
