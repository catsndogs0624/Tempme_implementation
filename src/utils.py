import cv2
from PIL import Image

'''
프레임 추출 함수 (비디오 → 이미지)
'''

def extract_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR → RGB 변환
            frames.append(Image.fromarray(frame))
    
    cap.release()
    return frames