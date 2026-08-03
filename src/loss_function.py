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
"""
Chamfer distance is a nearest-neighbor metric, it doesn't require uniform density coverage, 
just that every point (in both sets) has something close by. 
A thin, well-placed wireframe threading through the object's edges and corners can genuinely satisfy both directions reasonably well 
if the "backbone" of the frame passes close enough to most of the scattered surface points, 
because point density on the frame is high (many points clustered onto the same lines) 
while gaps between frame-points and stray surface points can still be relatively small in this normalized coordinate space. 
It's not cheating the math, it's a real local minimum that Chamfer distance doesn't strongly discourage, 
since it doesn't have any built-in notion of "surface area" or "coverage," only nearest-point proximity.
"""

"""
TO fix the coverage problem, 
1) Density/repulsion penalty
2) Density-aware chamfer variant 
3) Earth Mover's Distance 
"""


def repulsion_loss(pred, k=4): 
    """
    Penalise predicted points that cluster too close to their
    neighbours, thereby directly the model from bunching many 
    points onto the same thin wireframe line 
    """
    # pred : (B, N, 3)
    diff = pred.unsqueeze(2) - pred.unsqueeze(1) # (B, N, N, 3)
    dist = torch.sum(diff ** 2, dim=-1) # (B, N, N)
    dist = dist + torch.eye(pred.shape[1], device=pred.device) * 1e6 # ignore self distance
    knn_dist, _ = torch.topk(dist, k, dim=2, largest=False) #(B, N, k)
    # Penalise points whose nearest neighbours are too close (i.e. clustering)

    return torch.exp(-knn_dist).mean()

def chamfer_distance_knn(pred, target, k=8): # Changed k to be 8 instead of 3 -- changed to 3 again to see if it is better or not -- 8 is more denser but 3 also works perfectly

    """
    pred, target = (B, N, 3)
    Instead of matching only single nearest neighbour, this averages
    over the k nearest neighnours in each direction. This makes it 
    harder for the model to "cheat" by tracing a thin wireframe, since
    a sparse line onluy has a few geniunely close pponts to match
    against per query, spreading points across the full surface scores
    better under a k-NN average than under a single-nearest-neighbour
    match
    """
    diff = pred.unsqueeze(2) - target.unsqueeze(1) # (B, N, M, 3)
    distance = torch.sum(diff ** 2, dim=-1) # (B, N, M)

    # k-nearest neighbours in each direction, averaged
    pred_to_target = torch.topk(distance, k, dim=2, largest=False)[0].mean(dim=2) # (B, N)
    target_to_pred = torch.topk(distance, k, dim=1, largest=False)[0].mean(dim=1) # (B, M)

    return pred_to_target.mean() + target_to_pred.mean()


