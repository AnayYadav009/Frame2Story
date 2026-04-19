import cv2
import threading

class VideoReader:
    """Reusable video reader that maintains an open cv2.VideoCapture handle.
    
    Supports random access seeking and safe resource cleanup via the 
    context manager pattern. Thread-safe for serial reads via a lock.
    """
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None
        self._lock = threading.Lock()
        self._closed = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        if self._closed:
            raise RuntimeError("Cannot re-open a closed VideoReader. Create a new instance.")
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video: {self.video_path}")
        return self

    def close(self):
        with self._lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self._closed = True

    def clone(self):
        """Returns a new VideoReader instance for the same video.
        
        Useful for multi-threaded access where each thread needs its own handle.
        """
        return VideoReader(self.video_path)

    def get_frame(self, frame_index: int):
        """Seek to and return frame at frame_index."""
        if self._closed or self.cap is None:
            raise RuntimeError("VideoReader is not open. Use 'with' or call open().")
        
        with self._lock:
            # Optimize: only seek if we aren't already there
            current = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current != frame_index:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            
            success, frame = self.cap.read()
            if not success:
                return None
            return frame

def read_video_properties(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    if fps <= 0:
        video.release()
        raise ValueError("Invalid video FPS; cannot compute duration")

    duration = frame_count / fps
    video.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration": int(duration),
        "duration_seconds": float(duration),
        "resolution": (width, height),
    }

def get_frame_at_index(video_path, frame_index):
    """Legacy helper. For performance, use VideoReader class instead."""
    with VideoReader(video_path) as reader:
        frame = reader.get_frame(frame_index)
        if frame is None:
            raise Exception(f"Could not read frame at index {frame_index}")
        return frame

def time_to_frame(time_sec, fps):
    return int(time_sec * fps)

def save_frame(frame, output_path):
    cv2.imwrite(output_path, frame)