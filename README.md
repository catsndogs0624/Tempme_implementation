# CLIP ViT + LoRA + TEMPME for Video-Text Retrieval

## 프로젝트 개요
이 프로젝트는 **Hugging Face의 CLIP ViT 모델**을 기반으로 **LoRA (Low-Rank Adaptation) 및 TEMPME (Temporal Token Merging) 기법을 적용하여 비디오-텍스트 검색을 최적화**하는 모델입니다.

### 핵심 기능
- **CLIP ViT (`openai/clip-vit-base-patch32`) 백본 활용**  
  - 기존 CLIP 모델을 사용하여 텍스트-비디오 유사도 계산  
- **LoRA 적용 (파라미터 효율적 학습)**  
  - `q_proj`, `v_proj`에 LoRA 적용하여 모델 학습 최적화  
- **TEMPME 기법 추가 (토큰 병합 최적화)**  
  - **ImgMe Block**: 개별 프레임 내 유사한 토큰 병합  
  - **Cross-Clip Merging**: 인접한 클립 간 유사한 토큰 병합  
  - **Intra-Clip Merging**: 동일한 클립 내에서 유사한 토큰 병합  
- **비디오-텍스트 유사도 검색**  
  - CLIP ViT 모델을 기반으로 비디오와 텍스트 간 유사도 점수 계산  

---

## 📦 설치 방법

### 1️⃣ 필수 라이브러리 설치
```bash
pip install -r requirements.txt
