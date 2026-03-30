import cv2

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
    
    cap = cv2.VideoCapture(video_path)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    
    success, frame = cap.read()
    
    cap.release()
    
    if not success:
        raise Exception("Couldn't read frames")
    
    return frame

def time_to_frame(time_sec, fps):
    return int(time_sec * fps)

def save_frame(frame, output_path):
    cv2.imwrite(output_path, frame)


if __name__ == "__main__":
    video_path = "data/sample_video.mp4"

    # Quick manual check when this module is run directly.
    info = read_video_properties(video_path)

    print("FPS:", info["fps"])
    print("Frame Count:", info["frame_count"])
    print("Duration:", info["duration"])

    frame_index = time_to_frame(5, info["fps"])
    frame = get_frame_at_index(video_path, frame_index)
    save_frame(frame, "data/frame_5sec.jpg")

    print("Frame saved successfully.")