import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPProcessor, CLIPTokenizer, CLIPModel
import utils

# ==============================
# 1. 프레임 추출 함수 (비디오 → 이미지)
# ==============================
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


class CrossClipMerging(nn.Module):
    def __init__(self, embed_dim, merge_ratio=0.5):
        super(CrossClipMerging, self).__init__()
        self.embed_dim = embed_dim
        self.merge_ratio = merge_ratio

    def forward(self, clip1_embeddings, clip2_embeddings):
        batch_size, num_tokens, embed_dim = clip1_embeddings.shape
        similarity_matrix = F.cosine_similarity(
            clip1_embeddings.unsqueeze(2), clip2_embeddings.unsqueeze(1), dim=-1
        )
        num_tokens_to_keep = int(num_tokens * self.merge_ratio)
        _, top_indices = torch.topk(similarity_matrix, k=num_tokens_to_keep, dim=-1, largest=True)

        top_indices = top_indices[:, :, 0]
        expanded_indices = top_indices.unsqueeze(-1).repeat(1, 1, embed_dim)
        selected_clip1 = torch.gather(clip1_embeddings, 1, expanded_indices)
        selected_clip2 = torch.gather(clip2_embeddings, 1, expanded_indices)
        merged_clip_embeddings = (selected_clip1 + selected_clip2) / 2
        return merged_clip_embeddings

class IntraClipMerging(nn.Module):
    def __init__(self, embed_dim, merge_ratio=0.5):
        super(IntraClipMerging, self).__init__()
        self.embed_dim = embed_dim
        self.merge_ratio = merge_ratio

    def forward(self, clip_embeddings):
        batch_size, num_tokens, embed_dim = clip_embeddings.shape
        similarity_matrix = F.cosine_similarity(
            clip_embeddings.unsqueeze(2), clip_embeddings.unsqueeze(1), dim=-1
        )
        num_tokens_to_keep = int(num_tokens * self.merge_ratio)
        _, top_indices = torch.topk(similarity_matrix, k=num_tokens_to_keep, dim=-1, largest=True)
        selected_embeddings = torch.gather(clip_embeddings, 1, top_indices.unsqueeze(-1).expand(-1, -1, embed_dim))
        merged_clip_embeddings = selected_embeddings.mean(dim=1, keepdim=True)
        return merged_clip_embeddings

class ImgMeBlock(nn.Module):
    def __init__(self, embed_dim, reduction_ratio=0.5):
        super(ImgMeBlock, self).__init__()
        self.embed_dim = embed_dim
        self.reduction_ratio = reduction_ratio
        self.token_importance = nn.Linear(embed_dim, 1)

    def forward(self, token_embeddings):
        batch_size, num_tokens, embed_dim = token_embeddings.shape
        token_scores = self.token_importance(token_embeddings).squeeze(-1)
        token_weights = F.softmax(token_scores, dim=-1)
        num_tokens_to_keep = int(num_tokens * (1 - self.reduction_ratio))
        _, top_indices = torch.topk(token_weights, num_tokens_to_keep, dim=-1, largest=True)
        merged_embeddings = torch.gather(token_embeddings, 1, top_indices.unsqueeze(-1).expand(-1, -1, embed_dim))
        return merged_embeddings

class TEMPMEBlock(nn.Module):
    def __init__(self, embed_dim, img_reduction=0.5, cross_merge_ratio=0.5, intra_merge_ratio=0.5):
        super(TEMPMEBlock, self).__init__()
        self.imgme_block = ImgMeBlock(embed_dim, reduction_ratio=img_reduction)
        self.cross_clip_merging = CrossClipMerging(embed_dim, merge_ratio=cross_merge_ratio)
        self.intra_clip_merging = IntraClipMerging(embed_dim, merge_ratio=intra_merge_ratio)

    def forward(self, clip_embeddings_list):
        processed_clips = [self.imgme_block(clip) for clip in clip_embeddings_list]
        cross_merged_clips = [
            self.cross_clip_merging(processed_clips[i], processed_clips[i + 1])
            for i in range(len(processed_clips) - 1)
        ]
        final_clips = [self.intra_clip_merging(clip) for clip in cross_merged_clips]
        return final_clips
    
def main():
    video_frames = utils.load_video("./data/dog.mp4")
    tempme = TEMPMEBlock()
    processed_video = tempme(video_frames)
    #print(f"{processed_video.size()}")
    #caption = generate_video_caption(processed_video,tempme)
    #print(caption)


if __name__ == '__main__':
    main()