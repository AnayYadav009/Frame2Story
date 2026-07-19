import os
import sys
import json
import base64
import subprocess
import urllib.request
import urllib.error
from http.server import SimpleHTTPRequestHandler, HTTPServer

# Resolve paths and change directory to project root
COMPANION_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(COMPANION_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
os.chdir(PROJECT_ROOT)

ACTIVE_VIDEO_PATH = None

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

def extract_audio_slice_b64(video_path, end_sec, max_duration=180):
    """Extract audio slice up to end_sec using FFmpeg and return 16kHz mono WAV Base64 string."""
    if not video_path or not os.path.exists(video_path):
        return None
    
    start_sec = max(0.0, float(end_sec) - max_duration)
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_sec),
        '-to', str(end_sec),
        '-i', video_path,
        '-ar', '16000',
        '-ac', '1',
        '-f', 'wav',
        '-'
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=15)
        if res.returncode == 0 and res.stdout and len(res.stdout) > 100:
            return base64.b64encode(res.stdout).decode('utf-8')
    except Exception as e:
        print("FFmpeg extraction error:", e)
    return None

SEARCH_DIRS = [
    DATA_DIR,
    r'c:\Anay\Miscellaneous',
    r'c:\Anay',
    os.path.expanduser(r'~\Downloads'),
    os.path.expanduser(r'~\Videos'),
    os.path.expanduser(r'~\Desktop'),
    os.path.expanduser(r'~\Documents')
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
    import re
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
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "hasApiKey": has_key,
                "hasFfmpeg": True
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
            
            # If no subtitles or forced audio context, extract audio slice via FFmpeg if video is available
            if ACTIVE_VIDEO_PATH and timestamp is not None and not has_subtitles:
                audio_b64 = extract_audio_slice_b64(ACTIVE_VIDEO_PATH, timestamp)
                if audio_b64 and len(contents) > 0:
                    last_msg = contents[-1]
                    if last_msg.get('role') == 'user':
                        # Inject audio inlineData into Gemini request
                        audio_part = {
                            "inlineData": {
                                "mimeType": "audio/wav",
                                "data": audio_b64
                            }
                        }
                        # Prepend audio part to user parts
                        last_msg['parts'] = [audio_part] + [p for p in last_msg.get('parts', []) if 'inlineData' not in p]

            # Prepare Gemini request parameters
            gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={api_key}"
            
            gemini_payload = {
                "contents": contents,
                "systemInstruction": {
                    "parts": [{"text": req_body.get('systemInstruction', '')}]
                } if req_body.get('systemInstruction') else None
            }
            
            # Filter None/empty values
            gemini_payload = {k: v for k, v in gemini_payload.items() if v is not None}
            
            try:
                # Issue direct POST call using urllib
                req = urllib.request.Request(
                    gemini_url,
                    data=json.dumps(gemini_payload).encode('utf-8'),
                    headers={'Content-Type': 'application/json'},
                    method='POST'
                )
                with urllib.request.urlopen(req) as response:
                    res_data = response.read()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(res_data)
            except urllib.error.HTTPError as e:
                err_content = e.read().decode('utf-8')
                self.send_response(e.code)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(err_content.encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def run(port=8000):
    load_env()
    server_address = ('', port)
    
    # Enable SO_REUSEADDR to avoid 'Address already in use' errors during fast restarts
    class ReusableHTTPServer(HTTPServer):
        def server_bind(self):
            import socket
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            super().server_bind()

    httpd = ReusableHTTPServer(server_address, CineBuddyHandler)
    print("=================================================================")
    print("CineBuddy: Interactive Video Companion Server Running")
    print(f"Local Access URL: http://localhost:{port}/")
    print("=================================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()
        sys.exit(0)

if __name__ == '__main__':
    run()
