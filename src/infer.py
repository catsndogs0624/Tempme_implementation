import torch
from transformers import CLIPProcessor, CLIPModel
from peft import get_peft_model, LoraConfig, TaskType
from PIL import Image
import cv2
import numpy as np


def extract_frames(video_path, num_frames=8):
    """
    동영상 파일에서 num_frames 개의 프레임을 균등하게 추출하는 함수.
    :param video_path: 동영상 파일 경로
    :param num_frames: 추출할 프레임 수
    :return: PIL 이미지 리스트
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    # 동영상 전체에서 균등하게 num_frames 인덱스 선택
    indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # OpenCV는 BGR 포맷이므로 RGB로 변환
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)
            frames.append(pil_img)
        else:
            print(f"Frame {idx} 추출 실패")
    cap.release()
    return frames

if __name__ == '__main__':
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    # 2. 텍스트와 동영상(동영상 파일 경로 지정)
    text_query = "bird"
    video_path = "data/a.MOV"  # 실제 동영상 파일 경로로 변경

    # 3. 동영상에서 프레임 추출 (예: 8 프레임)
    frames = extract_frames(video_path, num_frames=8)

    # 4. CLIP 입력 데이터 전처리 (텍스트와 이미지 프레임 모두 처리)
    inputs = processor(text=[text_query], images=frames, return_tensors="pt", padding=True)

    # 5. 모델 추론: 텍스트와 이미지 임베딩 획득
    with torch.no_grad():
        outputs = model(**inputs)

    # 6. 출력: 이미지-텍스트 간 유사도 로짓 계산
    # outputs.logits_per_image: 각 이미지(프레임)와 텍스트 쌍의 유사도 로짓, shape [num_images, 1]
    similarities = outputs.logits_per_image

    # 7. 비디오 대표 임베딩: 여러 프레임의 유사도 평균
    video_text_similarity = similarities.mean(dim=0)
    print("Video-Text Similarity Score:", video_text_similarity.item())
