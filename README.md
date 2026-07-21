# CineBuddy AI: Interactive Video Companion 🍿🎬

CineBuddy AI is a real-time, interactive video companion application. Play any movie or TV episode, pause at any timestamp, and chat with an AI assistant that has real-time context of the video **only up to your current playback timestamp**.

Ask questions about characters, plot points, dialogue, visual scenes, or themes — and get instant, accurate explanations **without any spoilers** for parts you haven't watched yet!

---

## 🌟 Key Features

- **🎙️ Direct AI Audio Listening (No Subtitles Needed)**: Uses a native server-side **FFmpeg engine** to extract compressed audio slices on the fly. Gemini 2.0 Flash listens directly to spoken dialogue, background sounds, and character voices in real-time.
- **👁️ Visual Scene Analysis**: Extracts adaptive keyframe screenshots from the video using quadratic proximity sampling — densely sampling frames near the current timestamp for maximum context. The AI analyzes on-screen action, character expressions, props, visual text, and silent/visual storytelling.
- **🔒 Spoiler-Free AI Guardrails**: Strict prompt-engineered safety rules ensure the AI **never** reveals plot points, twists, or character fates beyond your current playback timestamp.
- **📝 Optional Subtitle Support**: Supports standard subtitle files (`.srt` and `.vtt`) for synced text captions alongside video playback. When loaded, subtitles replace audio extraction for transcript context.
- **🔑 Automatic `.env` API Key Loading**: Automatically parses your `.env` file on server launch — no need to paste your API key into the browser.
- **🔍 Smart Video File Search**: Automatically locates matching video files across common local directories (`Downloads`, `Videos`, `Desktop`, `Documents`, `data/`) using episode-aware path resolution (e.g., `S01E01` matching).
- **🎨 Ultra-Modern UI**: Sleek dark-mode dashboard styled with CSS glassmorphism, responsive grid layout, customizable AI personas (Helpful, Analytical, Sarcastic), and smooth micro-animations.
- **⚡ Zero External Dependencies**: Built entirely using Python's standard library (`http.server`, `urllib`, `subprocess`) — no `pip install` required.

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Details |
|---|---|
| **Python** | Python 3.8 or higher |
| **FFmpeg** | Must be installed and available in your system `PATH` |
| **Gemini API Key** | Free key from [Google AI Studio](https://aistudio.google.com/apikey) |
| **Browser** | Any modern browser (Chrome, Firefox, Edge, Safari) |

### Installation & Run

```bash
# 1. Clone the repository
git clone https://github.com/AnayYadav009/Frame2Story.git
cd Frame2Story

# 2. Create your .env file with your Gemini API Key
echo GEMINI_API_KEY=your_gemini_api_key_here > .env

# 3. Launch the companion server (no pip install needed!)
python companion_app/server.py

# 4. Open in your browser
#    → http://localhost:8000/
```

The server will automatically:
- Load your API key from `.env`
- Detect FFmpeg availability
- Serve the web UI at [http://localhost:8000/](http://localhost:8000/)

### Usage

1. Click **Select Video File** and choose any video (`.mp4`, `.mkv`, `.avi`, `.mov`, etc.).
2. CineBuddy will automatically register the video with the server's FFmpeg engine.
3. Play the video, **pause at any moment**, and ask questions in the chat sidebar.
4. *(Optional)* Load a `.srt` or `.vtt` subtitle file for synced text captions.
5. Configure your preferred AI persona and vision frame count in **Settings** (gear icon).

---

## 🏗️ Architecture

```text
┌──────────────────────────────────────────────────────────────────┐
│  Browser (index.html + app.js + style.css)                       │
│  ┌─────────────┐  ┌──────────────────┐  ┌─────────────────────┐ │
│  │ Video Player │  │ Subtitle Display │  │ Chat Sidebar        │ │
│  │ (HTML5)      │  │ (SRT/VTT sync)   │  │ (Gemini API proxy)  │ │
│  └──────┬───────┘  └────────┬─────────┘  └────────┬────────────┘ │
│         │   timestamp       │ context              │ question     │
└─────────┼───────────────────┼──────────────────────┼─────────────┘
          │                   │                      │
          ▼                   ▼                      ▼
┌──────────────────────────────────────────────────────────────────┐
│  Python Server (server.py)                                       │
│  ┌───────────────────┐  ┌─────────────────────────────────────┐ │
│  │ FFmpeg Engine      │  │ Gemini API Proxy                   │ │
│  │ • Audio slice      │  │ • Model fallback chain             │ │
│  │   (30s, 12kHz,     │  │   (2.0-flash → 2.0-flash-lite     │ │
│  │    mono, 32kbps)   │  │    → flash-latest)                 │ │
│  │ • Visual keyframes │  │ • Exponential backoff + retry      │ │
│  │   (480p, JPEG q=6, │  │ • API key validation               │ │
│  │    quadratic       │  │                                    │ │
│  │    sampling)       │  │                                    │ │
│  └───────────────────┘  └─────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```text
Frame2Story/
├── companion_app/
│   ├── index.html     # Dashboard layout & controls
│   ├── style.css      # Glassmorphic dark theme styles
│   ├── app.js         # Playback tracker, chat UI & API client
│   └── server.py      # HTTP server, FFmpeg engine & Gemini API proxy
├── data/              # Local video storage & pipeline artifacts (gitignored)
├── .env               # Environment configuration — GEMINI_API_KEY (gitignored)
├── .gitignore         # Git ignore rules
├── requirements.txt   # Dependency documentation (no pip packages needed)
└── README.md          # Project documentation
```

---

## ⚙️ Configuration

### Settings Panel (In-App)

| Setting | Description | Default |
|---|---|---|
| **Gemini API Key** | Can be set via `.env` file or in-app settings panel | From `.env` |
| **AI Persona** | `Helpful` / `Analytical` / `Sarcastic` — changes CineBuddy's personality | `Helpful` |
| **Vision Frames** | Number of visual keyframes extracted per query (0 = disabled, 1–10) | `5` |

### Environment Variables (`.env`)

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## 🛠️ Installing FFmpeg

**Windows** (using winget):
```bash
winget install FFmpeg
```

**Windows** (manual): Download from [ffmpeg.org/download.html](https://ffmpeg.org/download.html) and add the `bin/` folder to your system PATH.

**macOS**:
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian)**:
```bash
sudo apt install ffmpeg
```

Verify installation:
```bash
ffmpeg -version
```

---

## 📄 License

This project is for educational and personal use.
