"""
Loss functions for person re-identification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """Triplet loss for person re-identification."""
    
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: Feature vectors of anchor samples
            positive: Feature vectors of positive samples  
            negative: Feature vectors of negative samples
        """
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class CrossEntropyLoss(nn.Module):
    """Cross-entropy loss for classification."""
    
    def __init__(self, label_smoothing=0.1):
        super(CrossEntropyLoss, self).__init__()
        self.label_smoothing = label_smoothing
        
    def forward(self, logits, targets):
        """
        Args:
            logits: Model predictions
            targets: Ground truth labels
        """
        return F.cross_entropy(logits, targets, 
                             label_smoothing=self.label_smoothing)


class CombinedLoss(nn.Module):
    """Combined loss function for person re-identification."""
    
    def __init__(self, triplet_weight=1.0, ce_weight=1.0, margin=0.3, 
                 label_smoothing=0.1):
        super(CombinedLoss, self).__init__()
        self.triplet_weight = triplet_weight
        self.ce_weight = ce_weight
        
        self.triplet_loss = TripletLoss(margin=margin)
        self.ce_loss = CrossEntropyLoss(label_smoothing=label_smoothing)
        
    def forward(self, features, logits, targets, anchor_idx, pos_idx, neg_idx):
        """
        Args:
            features: Feature vectors
            logits: Classification logits
            targets: Ground truth labels
            anchor_idx: Indices of anchor samples
            pos_idx: Indices of positive samples
            neg_idx: Indices of negative samples
        """
        # Classification loss
        ce_loss = self.ce_loss(logits, targets)
        
        # Triplet loss
        anchor_features = features[anchor_idx]
        pos_features = features[pos_idx]
        neg_features = features[neg_idx]
        triplet_loss = self.triplet_loss(anchor_features, pos_features, neg_features)
        
        total_loss = (self.ce_weight * ce_loss + 
                     self.triplet_weight * triplet_loss)
        
        return total_loss, ce_loss, triplet_loss
