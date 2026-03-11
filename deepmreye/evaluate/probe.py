import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score


        
def compute_probe_metrics(targets, predictions):
    """
    Calculates Euclidean Error, Pearson correlation (r), and R^2 score.
    targets: [B, 2] (Ground truth X, Y)
    predictions: [B, 2] (Predicted X, Y)
    """
    if torch.is_tensor(targets):
        tgt_np = targets.detach().cpu().numpy()
    else:
        tgt_np = targets
        
    if torch.is_tensor(predictions):
        pred_np = predictions.detach().cpu().numpy()
    else:
        pred_np = predictions
    
    # 1. Take care of NaN in XYs (mask out invalid labels)
    nan_indices = np.any(np.isnan(tgt_np), axis=1)
    tgt_np = tgt_np[~nan_indices]
    pred_np = pred_np[~nan_indices]
    
    if len(tgt_np) < 1:
        # Batch had entirely missing ground truth data
        return {
            "euclidean_error": np.nan,
            "pearson_r_x": np.nan,
            "pearson_r_y": np.nan,
            "r2_x": np.nan,
            "r2_y": np.nan
        }

    # 2. Euclidean Distance
    euclidean_distances = np.sqrt(np.sum((tgt_np - pred_np) ** 2, axis=1))
    avg_euclidean = np.mean(euclidean_distances)
    
    # 3. Pearson correlation R
    r_x, _ = pearsonr(tgt_np[:, 0], pred_np[:, 0]) if len(tgt_np) > 1 else (np.nan, 1.0)
    r_y, _ = pearsonr(tgt_np[:, 1], pred_np[:, 1]) if len(tgt_np) > 1 else (np.nan, 1.0)
    
    # 4. R^2 Score
    r2_x = r2_score(tgt_np[:, 0], pred_np[:, 0]) if len(tgt_np) > 1 else np.nan
    r2_y = r2_score(tgt_np[:, 1], pred_np[:, 1]) if len(tgt_np) > 1 else np.nan
    
    return {
        "euclidean_error": avg_euclidean,
        "pearson_r_x": r_x,
        "pearson_r_y": r_y,
        "r2_x": r2_x,
        "r2_y": r2_y
    }
