import os
import sys
import json
import base64
import subprocess
import urllib.request
import urllib.error
import time
import re
import socket
import shutil
from http.server import SimpleHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

# Resolve paths and change directory to project root
COMPANION_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(COMPANION_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
LOG_FILE = os.path.join(COMPANION_DIR, 'server.log')
os.chdir(PROJECT_ROOT)

ACTIVE_VIDEO_PATH = None

def log_msg(msg):
    """Log formatted timestamped messages to stdout and server.log file."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(formatted + '\n')
            f.flush()
    except Exception as e:
        print(f"Logging file error: {e}")

def load_env():
    """Manually parse .env file to load local environment keys."""
    env_path = os.path.join(PROJECT_ROOT, '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, _, value = line.partition('=')
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

# Automatically load environment variables on startup
load_env()

def extract_audio_slice_b64(video_path, end_sec, max_duration=30):
    """Extract audio slice up to end_sec using FFmpeg and return 12kHz mono 32k MP3 Base64 string."""
    if not video_path or not os.path.exists(video_path):
        return None
    
    start_sec = max(0.0, float(end_sec) - max_duration)
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_sec),
        '-to', str(end_sec),
        '-i', video_path,
        '-ar', '12000',
        '-ac', '1',
        '-b:a', '32k',
        '-f', 'mp3',
        '-'
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=15)
        if res.returncode == 0 and res.stdout and len(res.stdout) > 100:
            return base64.b64encode(res.stdout).decode('utf-8')
        elif res.returncode != 0 and res.stderr:
            log_msg(f"[FFmpeg Audio Error] Exit code {res.returncode}: {res.stderr.decode('utf-8', errors='replace')[:200]}")
    except Exception as e:
        log_msg(f"[FFmpeg Error] {e}")
    return None

def extract_visual_keyframes_b64(video_path, end_sec, num_frames=5, max_duration=30):
    """Extract adaptively-spaced visual keyframe JPEG Base64 strings with high density near end_sec."""
    if not video_path or not os.path.exists(video_path):
        return []
    
    end_sec = float(end_sec)
    start_sec = max(0.0, end_sec - max_duration)
    duration = end_sec - start_sec
    
    if duration <= 0:
        timestamps = [end_sec]
    elif num_frames <= 1:
        timestamps = [end_sec]
    else:
        # Quadratic proximity distribution: densely sample the final 10s before end_sec
        timestamps = []
        for i in range(num_frames):
            ratio = i / (num_frames - 1)
            t = end_sec - (((1.0 - ratio) ** 2.0) * duration)
            timestamps.append(round(t, 2))
            
    keyframes_b64 = []
    for ts in timestamps:
        cmd = [
            'ffmpeg', '-y',
            '-ss', f"{ts:.2f}",
            '-i', video_path,
            '-vframes', '1',
            '-vf', 'scale=480:-1',
            '-q:v', '6',
            '-f', 'image2',
            '-'
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, timeout=10)
            if res.returncode == 0 and res.stdout and len(res.stdout) > 100:
                keyframes_b64.append(base64.b64encode(res.stdout).decode('utf-8'))
            elif res.returncode != 0 and res.stderr:
                log_msg(f"[FFmpeg Keyframe Error] ts={ts}s exit {res.returncode}: {res.stderr.decode('utf-8', errors='replace')[:200]}")
        except Exception as e:
            log_msg(f"[FFmpeg Keyframe Error] ts={ts}s: {e}")
            
    return keyframes_b64

SEARCH_DIRS = [
    DATA_DIR,
    os.path.expanduser(os.path.join('~', 'Downloads')),
    os.path.expanduser(os.path.join('~', 'Videos')),
    os.path.expanduser(os.path.join('~', 'Desktop')),
    os.path.expanduser(os.path.join('~', 'Documents'))
]

VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.avi', '.webm', '.mov', '.flv', '.m4v', '.ts')
IGNORE_DIRS = {'node_modules', '.git', '.venv', 'venv', '__pycache__', 'AppData'}

def find_video_path(filename):
    if not filename:
        return None
    
    # Check exact or absolute path first
    if os.path.exists(filename):
        return os.path.abspath(filename)
        
    clean_name = os.path.basename(filename).strip().lower()
    base_name = os.path.splitext(clean_name)[0]
    
    # Extract episode code if present (e.g. s08e08, S08E01, etc.)
    ep_match = re.search(r's\d+e\d+', base_name)
    ep_code = ep_match.group(0) if ep_match else None

    fallback_matches = []
    
    for sdir in SEARCH_DIRS:
        if not os.path.exists(sdir):
            continue
        for root, dirs, files in os.walk(sdir):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            for f in files:
                if f.lower().endswith(VIDEO_EXTENSIONS):
                    f_lower = f.lower()
                    # 1. Match exact episode code if present
                    if ep_code and ep_code in f_lower:
                        return os.path.join(root, f)
                    # 2. Match exact base name without extension
                    if base_name == os.path.splitext(f_lower)[0]:
                        return os.path.join(root, f)
                    if base_name in f_lower or f_lower in base_name:
                        fallback_matches.append(os.path.join(root, f))
    if fallback_matches:
        return fallback_matches[0]
    return None

class CineBuddyHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        # Redirect root path to the companion app UI
        if self.path == '/' or self.path == '':
            self.send_response(302)
            self.send_header('Location', '/companion_app/')
            self.end_headers()
            return
        
        # Server environment config endpoint
        if self.path == '/api/config':
            has_key = bool(os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY'))
            has_ffmpeg = shutil.which('ffmpeg') is not None
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "hasApiKey": has_key,
                "hasFfmpeg": has_ffmpeg
            }).encode('utf-8'))
            return
        
        # Serve static files from project root directory
        try:
            super().do_GET()
        except ConnectionResetError:
            # Handle client disconnects gracefully during video streams
            pass

    def do_POST(self):
        global ACTIVE_VIDEO_PATH
        
        # Endpoint to register/upload active video for FFmpeg processing
        if self.path == '/api/register-video':
            content_type = self.headers.get('Content-Type', '')
            content_length = int(self.headers.get('Content-Length', 0))
            
            # Check if JSON payload (local filename match)
            if 'application/json' in content_type:
                req_data = json.loads(self.rfile.read(content_length).decode('utf-8'))
                filename = req_data.get('filename', '')
                
                # Search system paths for video file
                found_path = find_video_path(filename)
                if found_path:
                    ACTIVE_VIDEO_PATH = found_path
                else:
                    ACTIVE_VIDEO_PATH = None
                        
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "success": bool(ACTIVE_VIDEO_PATH),
                    "registeredPath": ACTIVE_VIDEO_PATH
                }).encode('utf-8'))
                return
            
            # Handle binary video file upload
            os.makedirs(DATA_DIR, exist_ok=True)
            saved_path = os.path.join(DATA_DIR, "active_video.mp4")
            
            # Write POST stream to file in chunks
            bytes_read = 0
            with open(saved_path, 'wb') as f:
                while bytes_read < content_length:
                    chunk_size = min(65536, content_length - bytes_read)
                    chunk = self.rfile.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    bytes_read += len(chunk)
            
            ACTIVE_VIDEO_PATH = saved_path
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "success": True,
                "registeredPath": ACTIVE_VIDEO_PATH
            }).encode('utf-8'))
            return

        # API Proxy endpoint to call Gemini API
        if self.path == '/api/chat':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            req_body = json.loads(post_data.decode('utf-8'))
            
            # Fetch API Key from env (server-side) or client request (fallback)
            api_key = os.environ.get('GEMINI_API_KEY') or req_body.get('apiKey')
            if not api_key:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": "Gemini API Key not found. Please set GEMINI_API_KEY in your .env file or input it in the Settings panel."
                }).encode('utf-8'))
                return
            
            contents = req_body.get('contents', [])
            timestamp = req_body.get('timestamp')
            has_subtitles = req_body.get('hasSubtitles', False)
            enable_vision = req_body.get('enableVision', True)
            
            # If video active, extract audio slice & visual keyframes via FFmpeg
            if ACTIVE_VIDEO_PATH and timestamp is not None:
                new_inline_parts = []
                
                # 1. Audio Slice Extraction
                if not has_subtitles:
                    log_msg(f"[Audio Extraction] Extracting audio slice up to {timestamp}s from '{os.path.basename(ACTIVE_VIDEO_PATH)}'")
                    audio_b64 = extract_audio_slice_b64(ACTIVE_VIDEO_PATH, timestamp)
                    if audio_b64:
                        audio_kb = round(len(audio_b64) / 1024, 2)
                        log_msg(f"[Audio Success] Extracted MP3 audio slice (~{audio_kb} KB Base64)")
                        new_inline_parts.append({
                            "inlineData": {
                                "mimeType": "audio/mp3",
                                "data": audio_b64
                            }
                        })

                # 2. Visual Keyframes Extraction
                if enable_vision:
                    num_frames = int(req_body.get('numFrames', 5))
                    log_msg(f"[Vision Extraction] Extracting {num_frames} adaptive visual keyframes up to {timestamp}s...")
                    keyframes = extract_visual_keyframes_b64(ACTIVE_VIDEO_PATH, timestamp, num_frames=num_frames)
                    if keyframes:
                        total_kf_kb = round(sum(len(k) for k in keyframes) / 1024, 2)
                        log_msg(f"[Vision Success] Extracted {len(keyframes)} visual JPEG keyframes (~{total_kf_kb} KB total)")
                        for kf in keyframes:
                            new_inline_parts.append({
                                "inlineData": {
                                    "mimeType": "image/jpeg",
                                    "data": kf
                                }
                            })

                # Inject inline parts into last user message
                if new_inline_parts and len(contents) > 0:
                    last_msg = contents[-1]
                    if last_msg.get('role') == 'user':
                        text_parts = [p for p in last_msg.get('parts', []) if 'inlineData' not in p]
                        last_msg['parts'] = new_inline_parts + text_parts

            # Candidate models for fallback sequence (official v1beta endpoints)
            MODELS = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-flash-latest"]
            
            gemini_payload = {
                "contents": contents,
                "systemInstruction": {
                    "parts": [{"text": req_body.get('systemInstruction', '')}]
                } if req_body.get('systemInstruction') else None
            }
            
            # Filter None/empty values
            gemini_payload = {k: v for k, v in gemini_payload.items() if v is not None}
            payload_bytes = json.dumps(gemini_payload).encode('utf-8')
            log_msg(f"[API Request] Payload size: {round(len(payload_bytes)/1024, 2)} KB")
            
            last_error_code = 500
            last_error_content = None

            for model_name in MODELS:
                gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
                model_failed = False

                for attempt in range(3):  # Retry up to 3 times per model
                    try:
                        log_msg(f"[API Call] Requesting model '{model_name}' (attempt {attempt + 1}/3)...")
                        start_time = time.time()
                        req = urllib.request.Request(
                            gemini_url,
                            data=payload_bytes,
                            headers={'Content-Type': 'application/json'},
                            method='POST'
                        )
                        with urllib.request.urlopen(req) as response:
                            res_data = response.read()
                            elapsed = round(time.time() - start_time, 2)
                            log_msg(f"[API Success] Model '{model_name}' responded in {elapsed}s (HTTP 200)")
                            self.send_response(200)
                            self.send_header('Content-Type', 'application/json')
                            self.end_headers()
                            self.wfile.write(res_data)
                            return
                    except urllib.error.HTTPError as e:
                        last_error_code = e.code
                        last_error_content = e.read().decode('utf-8')
                        
                        # Immediate return if API Key itself is invalid
                        if "API_KEY_INVALID" in last_error_content:
                            log_msg(f"[API Error] Invalid Gemini API Key detected (HTTP {e.code})")
                            self.send_response(e.code)
                            self.send_header('Content-Type', 'application/json')
                            self.end_headers()
                            self.wfile.write(last_error_content.encode('utf-8'))
                            return
                        
                        # 429 (Too Many Requests / Quota Exceeded) or 503 (Service Unavailable)
                        if e.code in (429, 503):
                            log_msg(f"[API Warning] Model '{model_name}' returned HTTP {e.code} (attempt {attempt + 1}/3)")
                            if attempt < 2:
                                backoff = (attempt + 1) * 2.0  # 2.0s, 4.0s backoff
                                log_msg(f"[API Retry] Waiting {backoff}s backoff before retrying '{model_name}'...")
                                time.sleep(backoff)
                                continue
                            else:
                                log_msg(f"[API Fallback] Retries exhausted for '{model_name}'. Trying next fallback model...")
                                model_failed = True
                                break
                        else:
                            log_msg(f"[API Warning] Model '{model_name}' returned HTTP {e.code}. Trying next model...")
                            model_failed = True
                            break
                    except Exception as e:
                        log_msg(f"[API Exception] Model '{model_name}' failed with error: {e}")
                        last_error_code = 500
                        last_error_content = json.dumps({"error": str(e)})
                        model_failed = True
                        break

                if model_failed:
                    continue

            # If all models & attempts failed
            log_msg(f"[API Failure] All model fallbacks failed. Returning status HTTP {last_error_code}")
            self.send_response(last_error_code)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            if last_error_content:
                self.wfile.write(last_error_content.encode('utf-8'))
            else:
                self.wfile.write(json.dumps({"error": "All Gemini model endpoints failed. High demand or quota spike."}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()


def run(port=8000):
    server_address = ('127.0.0.1', port)
    
    # Enable SO_REUSEADDR and multi-threading to handle concurrent requests without blocking
    class ReusableHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True
        def server_bind(self):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            super().server_bind()

    httpd = ReusableHTTPServer(server_address, CineBuddyHandler)
    log_msg("=================================================================")
    log_msg("CineBuddy: Interactive Video Companion Server Running")
    log_msg(f"Local Access URL: http://localhost:{port}/")
    log_msg("=================================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log_msg("Stopping server...")
        httpd.server_close()
        sys.exit(0)

if __name__ == '__main__':
    run()
