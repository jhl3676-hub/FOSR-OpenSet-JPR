import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableCovCenterLoss(nn.Module):
    """
    Center Loss with learnable centers and noise consistency regularization.
    """

    def __init__(self, num_classes, feat_dim, noise_margin=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.noise_margin = noise_margin
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, features, labels, noise_features=None, center_weight=1.0, boundary_weight=1.0):
        # Distance to centers
        diff = features.unsqueeze(1) - self.centers.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=2)

        # Use negative distance as logits for CrossEntropy
        logits = -dist_sq
        ce_center_loss = self.ce_loss(logits, labels)

        # Noise Consistency Loss
        if noise_features is not None:
            noise_diff = noise_features.unsqueeze(1) - self.centers.unsqueeze(0)
            noise_dist_sq = (noise_diff ** 2).sum(dim=2)

            # Gather distances for the correct class
            pos_dist = dist_sq.gather(1, labels.view(-1, 1)).squeeze(1)
            pos_noise_dist = noise_dist_sq.gather(1, labels.view(-1, 1)).squeeze(1)

            noise_consistency = F.relu(torch.abs(pos_noise_dist - pos_dist) - self.noise_margin).mean()
        else:
            noise_consistency = torch.tensor(0.0, device=features.device)

        return center_weight * ce_center_loss + boundary_weight * noise_consistency


class DistanceCrossEntropyLoss(nn.Module):
    """Auxiliary loss for Autoencoder training."""

    def __init__(self, num_classes, feat_dim):
        super(DistanceCrossEntropyLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        dists = torch.cdist(features, self.centers, p=2)
        logits = -dists
        return F.cross_entropy(logits, labels)