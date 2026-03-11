#!/usr/bin/env python3
import os
import glob
import torch
import torch.nn as nn
import argparse
import numpy as np
from pathlib import Path
import sys

# Add deepmreye to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepmreye.preprocess import run_participant, get_masks
from deepmreye.models.inference import DeepMReyeJEPAInference

def process_file(file_path, dme_template, eyemask_big, eyemask_small, x_edges, y_edges, z_edges):
    print(f"\nProcessing {file_path}")
    masked_eye, _, _ = run_participant(
        fp_func=file_path,
        dme_template=dme_template,
        eyemask_big=eyemask_big,
        eyemask_small=eyemask_small,
        x_edges=x_edges,
        y_edges=y_edges,
        z_edges=z_edges,
        replace_with=0,
        transforms=["Affine", "Affine", "SyNAggro"],
        save_path=None,
        as_pickle=False,
        save_overview=False,
        dataset_name=os.path.basename(file_path)
    )
    return masked_eye

def main():
    parser = argparse.ArgumentParser(description="DeepMReye JEPA Inference")
    parser.add_argument("--input", type=str, required=True, help="Path to a .nii.gz file or a directory containing .nii.gz files.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to full pretrained weights (optional hook)")
    parser.add_argument("--out", type=str, default="inference_results.npy", help="Path to save output coordinates and latents")
    args = parser.parse_args()
    
    input_path = Path(args.input).resolve()
    
    files_to_process = []
    if input_path.is_file():
        files_to_process.append(str(input_path))
    elif input_path.is_dir():
        files_to_process.extend(glob.glob(str(input_path / "**" / "*_bold.nii.gz"), recursive=True))
    
    if not files_to_process:
        print(f"No valid BOLD files found targeting: {input_path}")
        return
        
    print("Loading ANTs Registration Templates...")
    eyemask_small, eyemask_big, dme_template, mask_np, x_edges, y_edges, z_edges = get_masks()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Deploying model to {device}")
    
    model = DeepMReyeJEPAInference(latent_dim=128).to(device)
    model.eval()
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading generic JEPA weights from {args.checkpoint}...")
        try:
            model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
        except Exception as e:
            print(f"Warning: Failed strict weight matching. {e}")
            
    results = {}
    
    for f in files_to_process:
        try:
            # 1. Pipeline Alignment & Extraction
            data_np = process_file(f, dme_template, eyemask_big, eyemask_small, x_edges, y_edges, z_edges)
            
            # 2. Reformatting to Tensor BxD...
            tensor_data = torch.from_numpy(data_np).float().to(device)
            if tensor_data.ndim == 4: # Add batch channel if (T, H, W, D)
                tensor_data = tensor_data.unsqueeze(0).unsqueeze(0)
            elif tensor_data.ndim == 5:
                tensor_data = tensor_data.unsqueeze(0)
                
            # Usually input expects (Batch=Time, Channels=1, H, W, D) inside 3D CNN architectures
            if tensor_data.shape[0] == 1 and tensor_data.ndim == 6:
                tensor_data = tensor_data.squeeze(0) # (T, C, H, W, D)
                
            # 3. Model Inference Map
            with torch.no_grad():
                latents, coords = model(tensor_data)
                
            results[os.path.basename(f)] = {
                'coordinates': coords.cpu().numpy(),
                'latents': latents.cpu().numpy()
            }
            
            print(f" -> Success! Output Sequence Coordinates [X, Y]: shape {coords.shape}")
            print(f" -> High-dimensional User Export Latent Representation: shape {latents.shape}")
            
        except Exception as e:
            print(f"Error mapping file {f}: {e}")
            
    if results:
        np.save(args.out, results)
        print(f"\nSaved predicted coordinates and Dense JEPA Latent spaces to: {args.out}")

if __name__ == "__main__":
    main()
