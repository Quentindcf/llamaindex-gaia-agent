'''helper functions for handling videos'''

import tempfile
from typing import List
import yt_dlp
import cv2

def download_youtube_video(url: str) -> str:
    '''locally downloads a youtube video'''
    temp_dir = tempfile.mkdtemp()
    filepath = os.path.join(temp_dir, 'video.mp4')

    ydl_opts = {
        'outtmpl': filepath,
        'format': 'mp4/best',
        'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return filepath


def extract_video_frames(path, output_dir="data/frames", fps=0.5) -> List[str]:
    """
    Extract frames from a video at the specified FPS
    (e.g., 0.5 = one frame every 2 seconds).
    Default of fps=0.5 is good. Only reduce FPS when .5 would generate too many frames
    Saves frames to disk and returns list of file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(video_fps / fps) if video_fps > 0 else int(1 / fps)
    count = 0
    saved_frames = 0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved_frames}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
            saved_frames += 1
        count += 1

    cap.release()
    return frames
