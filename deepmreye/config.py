from typing import Optional, Callable
from pydantic import BaseModel, Field

from deepmreye.util.util import mish

class DeepMReyeConfig(BaseModel):
    """Configuration class for the DeepMReye model and training."""
    
    # Model Architecture Options
    kernel: int = Field(default=3, description="Kernel size for all convolutional layers")
    filters: int = Field(default=32, description="Number of filters in convolutional layers")
    multiplier: int = Field(default=2, description="How much does the number of filters increase in each layer")
    depth: int = Field(default=4, description="Maximum number of layers")
    dropout_rate: float = Field(default=0.1, description="Dropout ratio for fully connected layers")
    num_dense: int = Field(default=2, description="Number of fully connected layers")
    num_fc: int = Field(default=1024, description="Number of units in fully connected layer")
    gaussian_noise: float = Field(default=0.0, description="How much gaussian noise is added (unit = standard deviation)")
    groups: int = Field(default=8, description="Number of groups to normalize across (see GroupNorm)")
    inner_timesteps: int = Field(default=10, description="Default number of subTR samples which are being reconstructed")
    
    # Loss Weights
    loss_euclidean: float = Field(default=1.0, description="Loss weight for euclidean distance")
    loss_confidence: float = Field(default=0.1, description="Loss weight for uncertainty measure")
    
    # Training Options
    lr: float = Field(default=0.00002, description="Learning rate")
    epochs: int = Field(default=25, description="Number of epochs")
    steps_per_epoch: int = Field(default=1500, description="Number of steps per training epoch")
    validation_steps: int = Field(default=1500, description="Number of steps per validation epoch")
    train_test_split: float = Field(default=0.6, description="Default proportion of train (60%)-test(40%) split")
    batch_size: int = Field(default=8, description="Batch size used for training the model")
    mixed_batches: bool = Field(default=True, description="If true, each batch contains samples across participants")
    mc_dropout: bool = Field(default=False, description="If true, monte carlo dropout is used")
    
    # Augmentation Options
    rotation_x: float = Field(default=5.0, description="Augmentation parameter, rotation in x-axis")
    rotation_y: float = Field(default=5.0, description="Augmentation parameter, rotation in y-axis")
    rotation_z: float = Field(default=5.0, description="Augmentation parameter, rotation in z-axis")
    shift: int = Field(default=4, description="Augmentation parameter, shift in all axes")
    zoom: float = Field(default=0.15, description="Augmentation parameter, zoom in all axes")
    
    # Activations
    activation: str = "mish"
    bottleneck_activation: str = "mish"

    class Config:
        arbitrary_types_allowed = True
