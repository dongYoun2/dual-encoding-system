import torch
import utils


def cosine_similarity(x: torch.Tensor, y: torch.Tensor):  # (B_x, E), (B_y, E)
    x_normed = utils.l2_normalize(x)  # (B_x, E)
    y_normed = utils.l2_normalize(y)  # (B_y, E)

    sim = torch.mm(x_normed, y_normed.T) # (B_x, B_y)

    return sim


def cosine_distance(x, y):
    sim = cosine_similarity(x, y)
    dist = 1 - sim

    return dist


def jaccard_similarity(x: torch.Tensor, y: torch.Tensor):   # (B_x, E), (B_y, E)
    x_batch_size = x.shape[0]
    y_batch_size = y.shape[0]

    x_ = x.unsqueeze(1).expand(-1, y_batch_size, -1)    # (B_x, B_y, E)
    y_ = y.unsqueeze(0).expand(x_batch_size, -1, -1)    # (B_x, B_y, E)

    intersection = torch.min(x_, y_).sum(dim=-1)    # (B_x, B_y)
    union = torch.max(x_, y_).sum(dim=-1) # (B_x, B_y)

    sim = intersection / union

    return sim