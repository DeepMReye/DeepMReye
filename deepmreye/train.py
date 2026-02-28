import os
from os.path import join
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

from deepmreye import architecture
from deepmreye.util import data_generator, util
from deepmreye.config import DeepMReyeConfig


class DeepMReye:
    """Unified API for DeepMReye Model Training and Inference."""
    
    def __init__(self, config=None, device=None):
        if config is None:
            config = DeepMReyeConfig()
        elif isinstance(config, dict):
            config = DeepMReyeConfig(**config)
            
        self.config = config
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.scheduler = None

    def build(self, input_shape):
        """Initializes the PyTorch model architecture."""
        self.model = architecture.DeepMReyeModel(input_shape, self.config)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        
        # Step decay scheduler: decay by factor 0.9 every (epochs // 10) epochs roughly? 
        # The original used a custom decay. Let's approximate it with StepLR
        step_size = max(1, self.config.epochs // 10) 
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=0.9)

    def fit(self, training_generator, testing_generator=None, verbose=1):
        """Trains the model."""
        if self.model is None:
            # Try to infer input shape from the first batch
            sample_batch = next(iter(training_generator))
            X_sample, _ = sample_batch
            # Shape is (Batch, Channels, X, Y, Z) - we need (X, Y, Z)
            input_shape = X_sample.shape[2:]
            self.build(input_shape)
            
        for epoch in range(self.config.epochs):
            self.model.train()
            train_loss = 0
            train_steps = 0
            
            # Use tqdm if verbose
            pbar = tqdm(training_generator, desc=f"Epoch {epoch+1}/{self.config.epochs}", disable=(verbose==0))
            
            for X, y in pbar:
                if train_steps >= self.config.steps_per_epoch:
                    break
                    
                X, y = X.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                out_regression, out_confidence = self.model(X)
                
                loss_euclidean, loss_confidence = architecture.compute_standard_loss(out_confidence, y, out_regression)
                
                total_loss = (self.config.loss_euclidean * loss_euclidean) + (self.config.loss_confidence * loss_confidence)
                
                total_loss.backward()
                self.optimizer.step()
                
                train_loss += total_loss.item()
                train_steps += 1
                
                pbar.set_postfix({"loss": total_loss.item()})
                
            self.scheduler.step()
            
            if testing_generator and self.config.validation_steps > 0:
                val_loss = self.evaluate(testing_generator, steps=self.config.validation_steps)
                if verbose > 0:
                    print(f"Epoch {epoch+1} - Train Loss: {train_loss/train_steps:.4f} - Val Loss: {val_loss:.4f}")

    def evaluate(self, generator, steps=None):
        """Evaluates the model and returns total loss."""
        self.model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for X, y in generator:
                if steps and val_steps >= steps:
                    break
                    
                X, y = X.to(self.device), y.to(self.device)
                out_regression, out_confidence = self.model(X)
                
                loss_euclidean, loss_confidence = architecture.compute_standard_loss(out_confidence, y, out_regression)
                total_loss = (self.config.loss_euclidean * loss_euclidean) + (self.config.loss_confidence * loss_confidence)
                
                val_loss += total_loss.item()
                val_steps += 1
                
        return val_loss / val_steps if val_steps > 0 else 0

    def predict(self, X, batch_size=16, verbose=0):
        """Runs inference on the provided data."""
        self.model.eval()
        
        # If X is a numpy array
        if isinstance(X, np.ndarray):
             X = torch.from_numpy(X).float()
             
        # Make sure X has right dims. It might be (Batch, X, Y, Z, Channels) from legacy
        if X.dim() == 5 and X.shape[-1] == 1:
            X = X.permute(0, 4, 1, 2, 3)
            
        dataset = torch.utils.data.TensorDataset(X)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_preds = []
        all_confs = []
        
        with torch.no_grad():
            for batch in loader:
                batch_X = batch[0].to(self.device)
                pred_reg, pred_conf = self.model(batch_X)
                all_preds.append(pred_reg.cpu().numpy())
                all_confs.append(pred_conf.cpu().numpy())
                
        return np.concatenate(all_preds, axis=0), np.concatenate(all_confs, axis=0)

    def save_weights(self, filepath):
        """Saves PyTorch model weights."""
        if self.model:
            torch.save(self.model.state_dict(), filepath)

    def load_weights(self, filepath):
        """Loads PyTorch model weights."""
        # Need to know input shape to build model if not built. 
        # Usually user should build first or we load state dict directly if built.
        if self.model:
            self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        else:
            raise RuntimeError("Model must be built before loading weights. Call build(input_shape) first.")


# =========================================================================================
# Legacy Helper Functions (Maintained for backward compatibility with scripts)
# =========================================================================================

def train_model(
    dataset,
    generators,
    opts,
    clear_graph=True,
    save=False,
    model_path="./",
    workers=4,
    use_multiprocessing=True,
    models=None,
    return_untrained=False,
    verbose=0,
):
    (
        training_generator,
        testing_generator,
        single_testing_generators,
        single_testing_names,
        single_training_generators,
        single_training_names,
        full_testing_list,
        full_training_list,
    ) = generators

    runner = DeepMReye(config=opts)
    
    if models is not None:
        runner.model = models[0] # assume models is (model, model) tuple
    else:
        # Infer input shape from generator
        X_sample, _ = next(iter(training_generator))
        input_shape = X_sample.shape[2:]
        runner.build(input_shape)
        
    if return_untrained:
        return (runner.model, runner.model)

    if verbose > 0:
        print(f"Subjects in training set: {len(single_training_generators)}, "
              f"Subjects in test set: {len(single_testing_generators)}")
              
    runner.fit(training_generator, testing_generator, verbose=verbose)
    
    if save:
        runner.save_weights(join(model_path, f"modelinference_{dataset}.pt"))
        
    return (runner.model, runner.model)


def evaluate_model(dataset, model, generators, save=False, model_path="./", model_description="", verbose=0, **args):
    (
        training_generator,
        testing_generator,
        single_testing_generators,
        single_testing_names,
        single_training_generators,
        single_training_names,
        full_testing_list,
        full_training_list,
    ) = generators
    
    evaluation, scores = dict(), dict()
    
    # Needs a config, we'll just instantiate a runner with default and the existing model
    runner = DeepMReye()
    runner.model = model
    
    for idx, subj in enumerate(full_testing_list):
        # We need the PyTorch-compatible get_all_subject_data
        X, real_y = data_generator.get_all_subject_data(subj)
        
        pred_y, euc_pred = runner.predict(X, verbose=verbose - 2, batch_size=16)
        
        # evaluation dict stores real target and predictions
        evaluation[subj] = {"real_y": real_y.numpy() if isinstance(real_y, torch.Tensor) else real_y, 
                            "pred_y": pred_y, 
                            "euc_pred": euc_pred}

        # Quantify predictions
        df_scores = util.get_model_scores(evaluation[subj]["real_y"], pred_y, euc_pred, **args)
        scores[subj] = df_scores

        if verbose > 0:
            print(f"{util.color.BOLD}{idx + 1} / {len(single_testing_names)} - Model Performance for {subj}{util.color.END}")
            if verbose > 1:
                pd.set_option("display.width", 120)
                pd.options.display.float_format = "{:.3f}".format
                print(df_scores)
            else:
                print(
                    f"Default: r={df_scores[('Pearson', 'Mean')]['Default']:.3f}, "
                    f"subTR: r={df_scores[('Pearson', 'Mean')]['Default subTR']:.3f}, "
                    f"Euclidean Error: {df_scores[('Eucl. Error', 'Mean')]['Default']:.3f}°"
                )
            print("\n")

    if save:
        np.save(join(model_path, f"results{model_description}_{dataset}.npy"), evaluation)

    return (evaluation, scores)
