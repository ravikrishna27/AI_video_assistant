# 🎬 AI Video Assistant

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C3C3C?logo=chainlink&logoColor=white)](https://www.langchain.com/)
[![Mistral AI](https://img.shields.io/badge/Mistral%20AI-mistral--small--latest-FD6F00)](https://mistral.ai/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Transform long videos, recordings, and team meetings into structured, searchable intelligence with automated transcription, executive summaries, decision tracking, and contextual RAG chat.**

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [File-by-File Breakdown](#-file-by-file-breakdown)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [Configuration (.env)](#-configuration-env)
- [Usage Guide](#-usage-guide)
  - [1. Web Interface (Streamlit)](#1-web-interface-streamlit)
  - [2. Command-Line Interface (CLI)](#2-command-line-interface-cli)
- [Troubleshooting & FAQs](#-troubleshooting--faqs)
- [Contributing & License](#-contributing--license)

---

## 💡 Overview

**AI Video Assistant** is an end-to-end multimodal pipeline designed to ingest video and audio media—from direct YouTube links or local file uploads—and extract high-value meeting intelligence. 

By orchestrating **Whisper**, **Sarvam AI**, **Mistral AI**, and **ChromaDB** using **LangChain Expression Language (LCEL)**, it automates:
1. High-fidelity speech-to-text conversion (English and Hinglish/code-mixed audio).
2. Hierarchical (Map-Reduce) document summarization.
3. Structured action item, key decision, and follow-up question extraction.
4. Semantic vector indexing for zero-hallucination interactive Q&A grounded strictly in the meeting transcript.

---

## ✨ Key Features

- **Multi-Source Ingestion:** Seamlessly download YouTube video audio using `yt-dlp` or process local media files (`.mp3`, `.wav`, `.m4a`, `.mp4`, `.mov`, `.mkv`, `.webm`).
- **Audio Standardization & Chunking:** Normalizes audio to 16 kHz mono WAV and segments large recordings into 10-minute processing blocks using `pydub` and `FFmpeg`.
- **Bilingual Speech-to-Text:**
  - **English:** High-accuracy local transcription powered by **OpenAI Whisper**.
  - **Hinglish / Indian Accents:** Automated chunk slicing into $\le 25\text{s}$ windows with **Sarvam AI Saaras v2.5** speech-to-text translation.
- **Hierarchical Summarization:** Map-Reduce chunking pipeline that processes lengthy transcripts without hitting LLM context limits.
- **Actionable Meeting Extraction:** Extracts:
  - **Action Items:** Task descriptions, assignees, and deadlines.
  - **Key Decisions:** Consensus and major choices made.
  - **Open Questions:** Unresolved discussions and pending queries.
- **Transcript-Grounded RAG Chat:** Vectorizes transcript chunks with `sentence-transformers/all-MiniLM-L6-v2` into **ChromaDB**, allowing users to chat with their meeting recordings with strict guardrails against hallucinations.
- **Dual Interfaces:** Modern dark-themed Streamlit dashboard with one-click `.txt` exports and a lightweight CLI utility.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph INGESTION["1. Media Ingestion & Audio Prep"]
        A[YouTube URL / Local Media] --> B{Source Type?}
        B -->|YouTube| C[yt-dlp Audio Downloader]
        B -->|Local File| D[Pydub File Converter]
        C --> E[Mono 16kHz WAV Audio]
        D --> E
        E --> F[Audio Chunking: 10-minute segments]
    end

    subgraph TRANSCRIPTION["2. Speech-to-Text Layer"]
        F --> G{Language Selected}
        G -->|English| H[OpenAI Whisper Engine]
        G -->|Hinglish| I[Sarvam AI API (Saaras v2.5)]
        H --> J[Consolidated Transcript]
        I --> J
    end

    subgraph INTELLIGENCE["3. LLM Processing (Mistral AI via LCEL)"]
        J --> K[Title Generator]
        J --> L[Map-Reduce Summarizer]
        J --> M[Insight Extractor]
        M --> M1[Action Items]
        M --> M2[Key Decisions]
        M --> M3[Open Questions]
    end

    subgraph RAG_PIPELINE["4. Vector Store & RAG Chat"]
        J --> N[Recursive Text Splitter (500 chars / 50 overlap)]
        N --> O[HuggingFace Embeddings (all-MiniLM-L6-v2)]
        O --> P[(ChromaDB Vector Store)]
        Q[User Query] --> R[Similarity Retriever (k=4)]
        P --> R
        R --> S[Context-Grounded Mistral Prompt]
        S --> T[Ground-Truth Answer]
    end

    subgraph INTERFACE["5. User Presentation"]
        K & L & M1 & M2 & M3 & T --> U[Streamlit Web App / CLI]
    end
```

---

## 📁 Repository Structure

```text
AI-Video-Assistant-main/
├── .gitignore                   # Ignores virtual environments, vector DBs, media, and .env
├── .env.example                 # Template for required environment variables
├── requirements.txt             # Complete Python package dependency manifest
├── README.md                    # Project documentation and architectural guide
├── main.py                      # Interactive Command-Line Interface (CLI)
├── app.py                       # Modern Streamlit Web Application
│
├── core/                        # Core AI and processing logic
│   ├── transcriber.py           # Speech-to-text pipeline (Whisper & Sarvam AI)
│   ├── summarizer.py            # Map-Reduce transcript summarization & title generation
│   ├── extractor.py             # Action item, key decision, and question extraction
│   ├── vector_store.py          # Chunk splitting, embedding, and ChromaDB vector store
│   └── rag_engine.py            # LangChain LCEL RAG QA retrieval and response generation
│
└── utils/                       # Supporting utilities
    └── audio_processor.py       # YouTube extraction, format conversion, and audio chunking
```

---

## 🔍 File-by-File Breakdown

| File Path | Primary Responsibility | Key Functions / Components |
|---|---|---|
| [`utils/audio_processor.py`](file:///C:/Users/Ravi/Desktop/AI-Video-Assistant-main/utils/audio_processor.py) | Audio extraction, conversion, and segmentation | `download_youtube_audio`: Uses `yt-dlp` to extract best audio.<br>`convert_to_wav`: Re-encodes audio to 16 kHz mono WAV via `pydub`.<br>`chunk_audio`: Splits long audio into 10-minute segments.<br>`process_input`: Unified input router. |
| [`core/transcriber.py`](file:///C:/Users/Ravi/Desktop/AI-Video-Assistant-main/core/transcriber.py) | Audio transcription and translation | `load_model`: Singleton loader for OpenAI Whisper.<br>`transcribe_chunk_whisper`: Local Whisper transcription.<br>`transcribe_chunk_sarvam`: Splits into 25s sub-pieces and invokes Sarvam AI.<br>`transcribe_all`: Orchestrates chunk-by-chunk transcription. |
| [`core/summarizer.py`](file:///C:/Users/Ravi/Desktop/AI-Video-Assistant-main/core/summarizer.py) | Transcript summarization and session naming | `split_transcript`: Splits long transcripts into 3,000-character segments.<br>`summarize`: Implements a Map-Reduce summary pattern using Mistral AI.<br>`generate_title`: Generates a punchy, professional meeting title ($\le 8$ words). |
| [`core/extractor.py`](file:///C:/Users/Ravi/Desktop/AI-Video-Assistant-main/core/extractor.py) | Structured information extraction | `extract_action_items`: Identifies tasks, assignees, and deadlines.<br>`extract_key_decisions`: Extracts agreed consensus and decisions.<br>`extract_questions`: Identifies unresolved topics and open queries. |
| [`core/vector_store.py`](file:///C:/Users/Ravi/Desktop/AI-Video-Assistant-main/core/vector_store.py) | Vector database indexing and retrieval | `get_embeddings`: Loads `sentence-transformers/all-MiniLM-L6-v2` on CPU.<br>`build_vector_store`: Chunks text (500 tokens / 50 overlap) and indexes into ChromaDB.<br>`get_retriever`: Returns a similarity-based vector retriever ($k=4$). |
| [`core/rag_engine.py`](file:///C:/Users/Ravi/Desktop/AI-Video-Assistant-main/core/rag_engine.py) | Context-grounded Q&A | `build_rag_chain`: Connects retriever and Mistral via LCEL runnable pipelines.<br>`ask_question`: Executes queries against the RAG chain with strict fallback rules. |
| [`app.py`](file:///C:/Users/Ravi/Desktop/AI-Video-Assistant-main/app.py) | Interactive web application | Complete Streamlit dashboard featuring custom dark CSS styling, real-time pipeline status trackers, multi-tab result displays, transcript download buttons, and interactive chat. |
| [`main.py`](file:///C:/Users/Ravi/Desktop/AI-Video-Assistant-main/main.py) | Terminal-based entry point | CLI script to execute the pipeline from the command prompt and start an interactive terminal chat session. |
| [`requirements.txt`](file:///C:/Users/Ravi/Desktop/AI-Video-Assistant-main/requirements.txt) | Environment dependencies | Pinpoints required packages for Streamlit, LangChain, Mistral AI, ChromaDB, Whisper, and audio tools. |

---

## ⚙️ Prerequisites

1. **Python 3.10 – 3.12** installed on your system.
2. **FFmpeg installed and available in your system PATH:**
   - **Windows:** Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) or install via winget:
     ```powershell
     winget install "Gyan.FFmpeg"
     ```
   - **macOS:**
     ```bash
     brew install ffmpeg
     ```
   - **Linux (Ubuntu/Debian):**
     ```bash
     sudo apt update && sudo apt install -y ffmpeg
     ```
3. **API Keys:**
   - **Mistral AI:** Required for title generation, summaries, extraction, and RAG. Obtain at [console.mistral.ai](https://console.mistral.ai/).
   - **Sarvam AI (Optional):** Required only if transcribing Hinglish / Indic audio. Obtain at [sarvam.ai](https://www.sarvam.ai/).

---

## 🚀 Installation & Setup

### 1. Navigate to the Project Directory

```powershell
cd C:\Users\Ravi\Desktop\AI-Video-Assistant-main
```

### 2. Create and Activate a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

*(If PowerShell script execution is restricted, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first)*

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on PyTorch / Whisper:** If you have an NVIDIA GPU and want GPU-accelerated Whisper transcription, install PyTorch with CUDA support from [pytorch.org](https://pytorch.org/). Otherwise, CPU execution will be used automatically.

---

## 🔑 Configuration (.env)

Create a file named `.env` in the root `AI-Video-Assistant-main/` directory by copying `.env.example`:

```powershell
Copy-Item .env.example .env
```

Open `.env` and fill in your actual credentials:

```env
# ============================================
# LLM & API KEYS
# ============================================
MISTRAL_API_KEY=your_mistral_api_key_here

# Optional: Needed only for Hinglish transcription
SARVAM_API_KEY=your_sarvam_api_key_here

# ============================================
# OPTIONAL MODEL OVERRIDES
# ============================================
# Whisper model size: tiny, base, small, medium, large
WHISPER_MODEL=small

# Sarvam translation model
SARVAM_STT_MODEL=saaras:v2.5
```

---

## 💻 Usage Guide

### 1. Web Interface (Streamlit)

Launch the web app:

```bash
streamlit run app.py
```

1. Open your browser at `http://localhost:8501`.
2. From the left sidebar:
   - Select your input type: **YouTube URL** or **Upload Audio / Video**.
   - Select transcription language (**English** or **Hinglish**).
3. Click **⚡ Analyse Media**.
4. Monitor the stage-by-stage progress bar.
5. Review results across the tabs:
   - **📊 Overview:** Executive summary + title.
   - **📝 Transcript:** Complete text output with download option.
   - **🎯 Meeting Insights:** Action items, key decisions, and open questions.
   - **💬 Ask AI:** Chat directly with your transcript.

### 2. Command-Line Interface (CLI)

Run `main.py` directly in your terminal:

```bash
python main.py
```

Follow the prompts to enter a YouTube link or file path, inspect the summary and insights, and interact with the terminal-based Q&A chat.

---

## 🛠️ Troubleshooting & FAQs

### Q: Why do I see red squiggly lines / "Import could not be resolved" in my code editor?
**A:** Your editor (e.g., VS Code or Antigravity) is using a global Python interpreter where dependencies are not installed. 
- Create and activate your virtual environment (`.venv`) as shown in the [Setup](#-installation--setup) section.
- In your editor, press `Ctrl + Shift + P` (or `Cmd + Shift + P` on Mac), search for **`Python: Select Interpreter`**, and choose `.\.venv\Scripts\python.exe`.

### Q: Error: `ffmpeg not found` or `FileNotFoundError: [WinError 2]` during audio processing
**A:** `pydub` and `yt-dlp` require the `ffmpeg` binary installed on your operating system. Ensure FFmpeg is installed and added to your system `PATH` variable. Verify in a new terminal with:
```powershell
ffmpeg -version
```

### Q: `ModuleNotFoundError: No module named 'whisper'`
**A:** Ensure `openai-whisper` is installed in your active virtual environment:
```powershell
pip install openai-whisper
```

### Q: Sarvam AI returns `400 Bad Request` or audio length error
**A:** Sarvam's synchronous STT API strictly enforces an audio duration limit of $\le 30$ seconds per call. The repository automatically splits audio chunks into 25-second sub-slices (`SARVAM_PIECE_SECONDS = 25`) to remain within safe thresholds.

---

## 📄 Contributing & License

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/YASH-1029/AI-Video-Assistant/issues).

Distributed under the **MIT License**. See `LICENSE` for more information.
