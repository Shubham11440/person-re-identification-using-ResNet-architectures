# src/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer."""
    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        # Use PyTorch's log_softmax for better numerical stability
        log_probs = F.log_softmax(inputs, dim=1)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss

class TripletLoss(nn.Module):
    """
    Triplet loss with hard positive/negative mining.
    Reference: https://github.com/KaiyangZhou/deep-person-reid
    """
    def __init__(self, margin=0.3, distance_metric='euclidean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.distance_metric = distance_metric

    def forward(self, inputs, targets):
        n = inputs.size(0) # Batch size

        # Compute distance matrix
        if self.distance_metric == 'euclidean':
            dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
            dist = dist + dist.t()
            dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        elif self.distance_metric == 'cosine':
            # Cosine similarity to distance
            inputs_norm = F.normalize(inputs, p=2, dim=1)
            sim = torch.matmul(inputs_norm, inputs_norm.t())
            dist = 1 - sim
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][~mask[i]].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

class CombinedLoss(nn.Module):
    """
    Combines CrossEntropy and Triplet Loss for joint optimization.
    """
    def __init__(self, num_classes, margin=0.3, epsilon=0.1, ce_weight=1.0, tri_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.ce_loss = CrossEntropyLabelSmooth(num_classes, epsilon)
        self.triplet_loss = TripletLoss(margin)
        self.ce_weight = ce_weight
        self.tri_weight = tri_weight

    def forward(self, part_logits, global_features, targets):
        # Cross-Entropy loss for each part
        ce_losses = [self.ce_loss(logits, targets) for logits in part_logits]
        total_ce_loss = sum(ce_losses) / len(ce_losses)

        # Triplet loss on the global feature descriptor
        triplet_loss = self.triplet_loss(global_features, targets)

        # Combine the losses
        combined_loss = (self.ce_weight * total_ce_loss) + (self.tri_weight * triplet_loss)
        return combined_loss