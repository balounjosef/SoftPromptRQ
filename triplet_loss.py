import math

from torch import nn
import torch
import torch.nn.functional as F


def dist_func(x1, x2):
    # result = - torch.sum(x1*x2, dim=-1)
    result = 1 - F.cosine_similarity(x1, x2, dim=-1)
    return result


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = dist_func(anchor, positive)
        distance_negative = dist_func(anchor, negative)

        # # saled dot prod. att.
        # tmp = torch.cat([torch.unsqueeze(distance_positive, dim=-1), torch.unsqueeze(distance_negative, dim=-1)], dim=-1)
        # tmp = tmp / math.sqrt(anchor.shape[-1])
        # tmp = torch.softmax(tmp, dim=-1)
        #
        # distance_positive = tmp[:, 0]
        # distance_negative = tmp[:, 1]

        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean(), distance_positive, distance_negative