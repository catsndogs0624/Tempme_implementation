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
