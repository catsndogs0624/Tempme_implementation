import cv2
from PIL import Image

def load_video(video_path, frame_interval=5):
    """Loads video frames from a given file path.
    Args:
        video_path (str): Path to the video file.
        frame_interval (int): Extract every N-th frame.
    Returns:
        list: List of PIL image frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        count += 1
    cap.release()
    return frames