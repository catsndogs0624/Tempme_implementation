import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPProcessor, CLIPTokenizer, CLIPModel
import utils



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
    """
    TEMPME 논문의 ClipMe Block 중 두 번째 단계인 Intra-Clip Merging.
    - 개별 클립 내에서 유사한 토큰을 병합하여 연산량 감소.
    """
    def __init__(self, embed_dim, merge_ratio=0.5):
        """
        Args:
            embed_dim (int): 입력 토큰 임베딩 차원.
            merge_ratio (float): 병합할 토큰의 비율 (0.5 = 50% 병합).
        """
        super(IntraClipMerging, self).__init__()
        self.embed_dim = embed_dim
        self.merge_ratio = merge_ratio

    def forward(self, clip_embeddings):
        """
        Args:
            clip_embeddings (Tensor): [Batch, Num_Tokens, Embed_Dim] (현재 클립의 토큰)
        Returns:
            merged_clip_embeddings (Tensor): 병합된 클립 토큰 출력
        """
        batch_size, num_tokens, embed_dim = clip_embeddings.shape

        # 1. 코사인 유사도 계산 (각 토큰 간 유사도 행렬)
        similarity_matrix = F.cosine_similarity(
            clip_embeddings.unsqueeze(2),  # [B, N, 1, D]
            clip_embeddings.unsqueeze(1),  # [B, 1, N, D]
            dim=-1
        )  # 결과: [B, N, N] (클립 내 토큰 간 유사도 행렬)

        # 2. 가장 유사한 토큰 찾기 (가장 큰 유사도 값)
        _, top_indices = torch.topk(similarity_matrix, k=int(num_tokens * self.merge_ratio), dim=-1, largest=True)

        # 3. 병합할 토큰 선택 및 가중 평균 연산
        selected_embeddings = torch.gather(clip_embeddings, 1, top_indices.unsqueeze(-1).expand(-1, -1, embed_dim))
        merged_clip_embeddings = selected_embeddings.mean(dim=1, keepdim=True)  # 평균 병합

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
    """ TEMPME 전체 블록 """
    def __init__(self, embed_dim, img_reduction=0.5, cross_merge_ratio=0.5, intra_merge_ratio=0.5):
        super(TEMPMEBlock, self).__init__()
        self.imgme_block = ImgMeBlock(embed_dim, reduction_ratio=img_reduction)
        self.cross_clip_merging = CrossClipMerging(embed_dim, merge_ratio=cross_merge_ratio)
        self.intra_clip_merging = IntraClipMerging(embed_dim, merge_ratio=intra_merge_ratio)

    def forward(self, clip_embeddings_list):
        """
        Args:
            clip_embeddings_list (list of Tensors): [Clip1, Clip2, ..., ClipN] (각 클립의 토큰)
        Returns:
            processed_clips (list of Tensors): 병합이 적용된 클립 리스트
        """
        # Step 1: 각 클립의 토큰에 ImgMe Block 적용
        processed_clips = [self.imgme_block(clip) for clip in clip_embeddings_list]

        # Step 2: Cross-Clip Merging 적용
        cross_merged_clips = []
        for i in range(len(processed_clips) - 1):
            merged_clip = self.cross_clip_merging(processed_clips[i], processed_clips[i + 1])
            cross_merged_clips.append(merged_clip)

        # Step 3: Intra-Clip Merging 적용
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