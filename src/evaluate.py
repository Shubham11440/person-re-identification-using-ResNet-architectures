"""
Evaluation script for person re-identification model.
"""

import torch
import numpy as np
from sklearn.metrics import average_precision_score
import time
from tqdm import tqdm

from models import create_model
from datasets import PersonReIDDataset, get_transforms, create_dataloader
from utils import load_checkpoint


def extract_features(model, dataloader, device):
    """Extract features from the model."""
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Extracting features'):
            images = images.to(device)
            batch_features = model.extract_features(images)
            
            features.append(batch_features.cpu().numpy())
            labels.append(targets.numpy())
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    return features, labels


def compute_distance_matrix(query_features, gallery_features):
    """Compute distance matrix between query and gallery features."""
    # Using Euclidean distance
    q_norm = np.linalg.norm(query_features, axis=1, keepdims=True)
    g_norm = np.linalg.norm(gallery_features, axis=1, keepdims=True)
    
    query_features = query_features / q_norm
    gallery_features = gallery_features / g_norm
    
    # Compute cosine similarity and convert to distance
    similarity = np.dot(query_features, gallery_features.T)
    distance = 1 - similarity
    
    return distance


def evaluate_ranking(distance_matrix, query_labels, gallery_labels, max_rank=50):
    """Evaluate ranking performance."""
    num_query = distance_matrix.shape[0]
    
    # Sort gallery samples by distance for each query
    indices = np.argsort(distance_matrix, axis=1)
    
    # Compute CMC and mAP
    cmc = np.zeros(max_rank)
    ap_scores = []
    
    for i in range(num_query):
        query_label = query_labels[i]
        
        # Get sorted gallery labels for this query
        sorted_gallery_labels = gallery_labels[indices[i]]
        
        # Find matches
        matches = (sorted_gallery_labels == query_label)
        
        # Skip if no matches
        if not np.any(matches):
            continue
            
        # CMC computation
        first_match_idx = np.where(matches)[0][0]
        for k in range(first_match_idx, max_rank):
            cmc[k] += 1
            
        # mAP computation
        num_matches = np.sum(matches)
        if num_matches > 0:
            sorted_distances = distance_matrix[i][indices[i]]
            y_true = matches.astype(int)
            y_score = -sorted_distances  # Negative distance as score
            ap = average_precision_score(y_true, y_score)
            ap_scores.append(ap)
    
    # Normalize CMC
    cmc = cmc / num_query
    
    # Compute mAP
    mAP = np.mean(ap_scores) if ap_scores else 0.0
    
    return cmc, mAP


def evaluate_model(config):
    """Main evaluation function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create model
    model = create_model(
        model_name=config['model_name'],
        num_classes=config['num_classes'],
        feature_dim=config['feature_dim']
    )
    model.to(device)
    
    # Load checkpoint
    checkpoint = load_checkpoint(config['checkpoint_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded checkpoint from epoch {checkpoint["epoch"]}')
    
    # Create datasets and dataloaders
    transform = get_transforms(is_training=False)
    
    query_dataset = PersonReIDDataset(config['query_data_dir'], transform)
    gallery_dataset = PersonReIDDataset(config['gallery_data_dir'], transform)
    
    query_loader = create_dataloader(query_dataset, config['batch_size'], False)
    gallery_loader = create_dataloader(gallery_dataset, config['batch_size'], False)
    
    # Extract features
    print('Extracting query features...')
    query_features, query_labels = extract_features(model, query_loader, device)
    
    print('Extracting gallery features...')
    gallery_features, gallery_labels = extract_features(model, gallery_loader, device)
    
    # Compute distance matrix
    print('Computing distance matrix...')
    distance_matrix = compute_distance_matrix(query_features, gallery_features)
    
    # Evaluate ranking
    print('Evaluating ranking...')
    cmc, mAP = evaluate_ranking(distance_matrix, query_labels, gallery_labels)
    
    # Print results
    print('\\nEvaluation Results:')
    print(f'mAP: {mAP:.4f}')
    print(f'Rank-1: {cmc[0]:.4f}')
    print(f'Rank-5: {cmc[4]:.4f}')
    print(f'Rank-10: {cmc[9]:.4f}')
    
    return {
        'mAP': mAP,
        'cmc': cmc,
        'rank1': cmc[0],
        'rank5': cmc[4],
        'rank10': cmc[9]
    }


if __name__ == '__main__':
    # Example configuration
    config = {
        'model_name': 'resnet50',
        'num_classes': 1000,
        'feature_dim': 2048,
        'batch_size': 32,
        'query_data_dir': 'data/query',
        'gallery_data_dir': 'data/gallery',
        'checkpoint_path': 'checkpoints/best_model.pth'
    }
    
    results = evaluate_model(config)
