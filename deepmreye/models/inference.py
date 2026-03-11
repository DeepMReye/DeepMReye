import torch
import torch.nn as nn
from deepmreye.models.jepa import JEPAModel

class DeepMReyeJEPAInference(nn.Module):
    """
    Final end-to-end model bridging the pre-trained unsupervised JEPA 
    Context Encoder to the linear probing regression coordinates.
    """
    def __init__(self, embed_dim=256, encoder_depth=4, num_heads=8):
        super().__init__()
        # Load the core architecture (specifically for its Context Encoder)
        self.encoder = JEPAModel(embed_dim=embed_dim, encoder_depth=encoder_depth, num_heads=num_heads)
        
        # Linear Probing Head (can be swapped with deepmreye.evaluate.probe.LinearProbe)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # X, Y output coordinates
        )
        
    def forward(self, x):
        """
        x: [B, X, Y, Z, T] standard batch
        """
        # 1. Patcher Sequence Generation
        seq, N_S, N_T = self.encoder.patcher(x)
        
        # 2. Context Latent Retrieval (No masking in inference mode)
        B = x.shape[0]
        c_idx_full = torch.arange(N_S * N_T, device=x.device).unsqueeze(0).expand(B, -1)
        latent_sequence = self.encoder.forward_context(seq, c_idx_full, N_S, N_T)
        
        # 3. Target Regression Coordinate Prediction
        latent_mean = latent_sequence.mean(dim=1)
        xy_coords = self.regressor(latent_mean)
        
        return latent_mean, xy_coords
