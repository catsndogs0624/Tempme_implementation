import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPProcessor, CLIPTokenizer, CLIPModel
from PIL import Image


def cross_clip_merging(tokens, reduction_ratio):
    """Merges similar tokens in a cross-clip merging manner."""
    
    # Ensure tokens is a 2D tensor (N, D)
    tokens = tokens.view(-1, tokens.shape[-1])  # Convert to (N, D)
    
    N, D = tokens.shape
    num_merge = int(N * reduction_ratio)

    if num_merge == 0:
        return tokens

    # Compute similarity matrix
    similarity = F.cosine_similarity(tokens.unsqueeze(1), tokens.unsqueeze(0), dim=-1)
    values, indices = torch.topk(similarity, num_merge, dim=1)  # Shape: (N, num_merge)

    # Merge selected tokens by averaging
    for i in range(num_merge):
        a, b = indices[i, 0], indices[i, 1]  # Extract first two indices

        tokens[a] = (tokens[a] * 0.5 + tokens[b] * 0.5)  # Weighted averaging
        tokens[b] = tokens[a]

    return tokens[:N - num_merge]



def intra_clip_merging(tokens, reduction_ratio):
    """Merges similar tokens within a clip, focusing on reducing spatial and temporal redundancy."""
    
    # Ensure tokens is a 2D tensor (N, D)
    tokens = tokens.view(-1, tokens.shape[-1])  # Convert to (N, D)
    
    N, D = tokens.shape
    num_merge = int(N * reduction_ratio)

    if num_merge == 0:
        return tokens

    # Compute similarity matrix
    similarity = F.cosine_similarity(tokens.unsqueeze(1), tokens.unsqueeze(0), dim=-1)
    values, indices = torch.topk(similarity, num_merge, dim=1)  # Shape: (N, num_merge)

    # Merge selected tokens by averaging
    for i in range(num_merge):
        a, b = indices[i, 0], indices[i, 1]  # Extract first two indices

        tokens[a] = (tokens[a] * 0.6 + tokens[b] * 0.4)  # Weighted averaging
        tokens[b] = tokens[a]

    return tokens[:N - num_merge]


def img_me_block(tokens, reduction_ratio):
    """Merges similar tokens within an individual frame to reduce spatial redundancy."""
    
    # Ensure tokens is a 2D tensor (N, D)
    tokens = tokens.view(-1, tokens.shape[-1])  # Convert to (N, D)
    
    N, D = tokens.shape
    num_merge = int(N * reduction_ratio)

    if num_merge == 0:
        return tokens

    # Compute similarity matrix
    similarity = F.cosine_similarity(tokens.unsqueeze(1), tokens.unsqueeze(0), dim=-1)
    values, indices = torch.topk(similarity, num_merge, dim=1)  # Shape: (N, num_merge)

    # Merge selected tokens by averaging
    for i in range(num_merge):
        a, b = indices[i, 0], indices[i, 1]  # Extract first two indices

        tokens[a] = (tokens[a] * 0.5 + tokens[b] * 0.5)
        tokens[b] = tokens[a]

    return tokens[:N - num_merge]

class TempMe(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", reduction_ratio=0.5):
        """Video Processor using CLIP-ViT as the backbone."""
        super().__init__()
        self.clip_model = CLIPVisionModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.cross_clip_merging = cross_clip_merging
        self.intra_clip_merging = intra_clip_merging
        self.img_me_block = img_me_block
        self.attn = nn.MultiheadAttention(self.clip_model.config.hidden_size, 8, batch_first=True)
        self.ln1 = nn.LayerNorm(self.clip_model.config.hidden_size)
        self.ln2 = nn.LayerNorm(self.clip_model.config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(self.clip_model.config.hidden_size, self.clip_model.config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(self.clip_model.config.hidden_size * 4, self.clip_model.config.hidden_size)
        )
        self.reduction_ratio = reduction_ratio

    def forward(self, video_frames):
        """Process input video frames using CLIP-ViT.
        Args:
            video_frames (list of PIL images): Input video frames.
        Returns:
            torch.Tensor: Processed video features with reduced redundancy.
        """
        inputs = self.processor(images=video_frames, return_tensors="pt")
        with torch.no_grad():
            video_features = self.clip_model(**inputs).last_hidden_state  # Shape: (B, N, D)
        
        tokens = video_features.view(video_features.shape[0], -1, video_features.shape[-1])
        
        # Apply Image Merging Block
        tokens = self.img_me_block(tokens, self.reduction_ratio)
        
        # Apply Cross-clip merging
        tokens = self.cross_clip_merging(tokens, self.reduction_ratio)
        
        # Apply Attention Mechanism
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.ln1(tokens + attn_out)
        
        # Apply Intra-clip merging
        tokens = self.intra_clip_merging(tokens, self.reduction_ratio / 2)
        
        # Apply Feedforward
        ffn_out = self.ffn(tokens)
        tokens = self.ln2(tokens + ffn_out)
        
        return tokens