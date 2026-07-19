# CineBuddy AI: Interactive Video Companion 🍿

CineBuddy AI is a real-time, interactive video companion application. Play any movie or TV episode, pause at any timestamp, and chat with an AI assistant that has real-time context of the video **only up to your current playback timestamp**.

Ask questions about characters, plot points, dialogue, or visual themes, and get instant explanations **without any spoilers** for parts of the video you haven't watched yet!

---

## 🌟 Key Features

- **Direct AI Audio Listening (No Subtitles Needed)**: Uses a native server-side **FFmpeg engine** to extract high-quality audio slices on the fly. Gemini 3.5 Flash listens directly to spoken dialogue and character voices in real-time.
- **Synced Subtitle Timeline (Optional)**: Supports standard subtitle files (`.srt` and `.vtt`) for synced text captions alongside video playback.
- **Spoiler-Free AI Guardrails**: Strict safety rules ensure the AI never reveals plot points, twists, or character fates beyond your current playback timestamp.
- **Automatic `.env` API Key Loading**: Automatically parses `.env` on server launch—no need to paste your API key into browser settings.
- **Smart System Video Search**: Automatically locates matching video files across local directories (`c:\Anay\Miscellaneous`, `data/`, `Downloads`, `Videos`, etc.) using episode-aware path resolution.
- **Ultra-Modern UI**: Sleek dark-mode dashboard styled with CSS glassmorphism, responsive grid layout, customizable personas (Helpful, Analytical, Sarcastic), and smooth micro-animations.
- **Zero Python External Dependencies**: Built entirely using Python's standard library (`http.server`, `urllib`, `subprocess`).

---

## 🚀 Quick Start

1. **Add Your Gemini API Key**:
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

2. **Launch the Companion Server**:
   ```bash
   python companion_app/server.py
   ```

3. **Open the Web Interface**:
   Navigate to [http://localhost:8000/](http://localhost:8000/) in your web browser.

4. **Load & Play Media**:
   - Click **Select Video File** and choose any video file (`.mp4`, `.mkv`, `.avi`, `.mov`, etc.).
   - Play the video, pause at any timestamp, and ask CineBuddy questions in the chat sidebar!

---

## 📁 Project Structure

```text
Frame2Story/
├── companion_app/
│   ├── index.html     # Dashboard layout & controls
│   ├── style.css      # Glassmorphic dark theme styles
│   ├── app.js         # Playback tracker & API client
│   └── server.py      # HTTP server, FFmpeg audio engine & Gemini proxy
├── data/              # Folder for local video projects & subtitles
├── .env               # Environment configuration (API keys)
├── .gitignore         # Git ignore rules
└── README.md          # Project documentation
```
