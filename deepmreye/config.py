from typing import Optional, Callable
from dataclasses import dataclass, field

@dataclass
class DeepMReyeConfig:
    """
    Configuration for DeepMReye paths and models.
    """
    # Centralized storage directory 
    data_dir: str = field(default="./data", metadata={"description": "Central directory where HDF5, reports, and block datasets are stored."})
    
    train_test_split: float = field(default=0.6, metadata={"description": "Default proportion of train (60%)-test(40%) split"})
    batch_size: int = field(default=8, metadata={"description": "Batch size used for training the model"})
    mixed_batches: bool = field(default=True, metadata={"description": "If true, each batch contains samples across participants"})
    mc_dropout: bool = field(default=False, metadata={"description": "If true, monte carlo dropout is used"})
    
    # Augmentation Options
    rotation_y: float = field(default=5.0, metadata={"description": "Augmentation parameter, rotation in y-axis"})
    rotation_z: float = field(default=5.0, metadata={"description": "Augmentation parameter, rotation in z-axis"})
    shift: int = field(default=4, metadata={"description": "Augmentation parameter, shift in all axes"})
    zoom: float = field(default=0.15, metadata={"description": "Augmentation parameter, zoom in all axes"})
    
    # Activations
    activation: str = "mish"
    bottleneck_activation: str = "mish"

    class Config:
        arbitrary_types_allowed = True
