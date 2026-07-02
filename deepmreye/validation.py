import ants
import logging

class MissingTRError(Exception):
    """Exception raised when a valid TR (Repetition Time) cannot be found in the image header."""
    pass

def validate_and_extract_tr(file_path):
    """
    Reads the NIfTI header to validate the presence of a valid Repetition Time (TR).
    
    Args:
        file_path (str or Path): Path to the functional NIfTI image.
        
    Returns:
        float: The extracted Repetition Time (TR).
        
    Raises:
        MissingTRError: If the TR is missing (dimension < 4) or invalid (<= 0).
    """
    try:
        # Use ants to read the image and get the header/spacing
        # While image_read loads the data, it is robust and guarantees we get 
        # the same spacing that will be used during registration.
        img = ants.image_read(str(file_path))
    except Exception as e:
        raise ValueError(f"Failed to read image at {file_path}: {e}")
        
    spacing = img.spacing
    
    if len(spacing) < 4:
        raise MissingTRError(f"Missing TR: Image only has {len(spacing)} dimensions in spacing (expected at least 4).")
        
    tr = float(spacing[3])
    
    if tr <= 0.0:
        raise MissingTRError(f"Invalid TR: Expected positive TR, got {tr}.")
        
    # Sometimes 1.0 is a default placeholder, but it could be valid. We only strictly reject <= 0.
    
    return tr
