import numpy as np 
import torch 

def chamfer_distance(pred, target): 
    """
    pred, target: (B, N, 3)
    For each point in pred, find the nearest neighbour in target 
    and vice versa 

    Returns mean squared nearest-neighbour distance in both directions
    """
    # Pairwise squared distances: (B, N, 3)
    diff = pred.unsqueeze(2) - target.unsqueeze(1) # (B, N, M, 3)
    distance = torch.sum(diff ** 2, dim=-1)        # (B, N, M)

    pred_to_target = torch.min(distance, dim=2)[0] # (B, N)
    target_to_pred = torch.min(distance, dim=1)[0] # (B, M)

    return pred_to_target.mean() + target_to_pred.mean()