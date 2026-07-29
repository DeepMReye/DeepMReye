import ants
import logging

class MissingTRError(Exception):
    """Exception raised when a valid TR (Repetition Time) cannot be found in the image header."""
    pass


# Plausible repetition times for a BOLD sequence, in seconds. Headers in the
# wild carry values that cannot be a TR at all -- 0.044 s (23 Hz) and 10 s both
# appear in the OpenNeuro corpus, 0.6% of subjects in total.
#
# This band used to not exist, and did not need to: TR was recorded but never
# read by the model, so a wrong one cost nothing. It is load-bearing now that
# TR conditions the temporal encoding -- a header saying 0.044 s does not just
# get ignored, it teaches the model that this window spans 4 seconds when it
# spans 200. Wrong metadata is worse than absent metadata once you condition on
# it.
MIN_PLAUSIBLE_TR = 0.3
MAX_PLAUSIBLE_TR = 5.0


def is_plausible_tr(tr):
    """Whether a repetition time is usable as a conditioning signal."""
    return tr is not None and MIN_PLAUSIBLE_TR <= float(tr) <= MAX_PLAUSIBLE_TR


def validate_and_extract_tr(file_path, strict=True):
    """
    Reads the NIfTI header to validate the presence of a valid Repetition Time (TR).

    Args:
        file_path (str or Path): Path to the functional NIfTI image.
        strict (bool): Also require the TR to fall inside the plausible band
            (``MIN_PLAUSIBLE_TR``..``MAX_PLAUSIBLE_TR``). Pass False to accept
            any positive TR, e.g. to re-examine subjects rejected by the band.

    Returns:
        float: The extracted Repetition Time (TR).

    Raises:
        MissingTRError: If the TR is missing (dimension < 4), invalid (<= 0),
            or -- when ``strict`` -- outside the plausible range.
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

    # Sometimes 1.0 is a default placeholder, but it could be valid.
    if strict and not is_plausible_tr(tr):
        raise MissingTRError(
            f"Implausible TR: got {tr} s, outside the {MIN_PLAUSIBLE_TR}-{MAX_PLAUSIBLE_TR} s "
            f"band for a BOLD sequence. The header is almost certainly wrong, and TR is used "
            f"as a conditioning signal, so a wrong value is worse than a missing one."
        )

    return tr
