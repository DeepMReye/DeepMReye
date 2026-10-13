"""From a functional run (NIfTI) to the eye block the decoder reads.

Three steps, the same ones that produced every participant of the published
corpus, so a new run lands in exactly the distribution the basis was fitted on:

1. **Register** the run's mean volume to the DeepMReye template in three stages
   (affine on the whole volume, affine inside a large eye mask, then SyN inside
   the small eye mask), and apply the composed transform to every volume.
2. **Crop** the eye region: a fixed ``[47, 29, 18]`` box around both eyeballs,
   with every voxel outside the 14,236-voxel eye mask set to zero.
3. **Normalise**: z-score every voxel over time, then every volume over space,
   and clip at 5 SD.

ANTs (``antspyx``) does the registration; it takes a minute or two per run.
"""
import warnings
from pathlib import Path

import numpy as np

MASK_DIR = Path(__file__).resolve().parent / "masks"

# The three registration stages, in order: whole volume, large eye mask, small
# eye mask. The corpus was extracted with exactly these.
DEFAULT_TRANSFORMS = ("Affine", "Affine", "SyNAggro")

# Plausible repetition times for a BOLD sequence, in seconds. Headers in the wild
# carry values that cannot be a TR (0.044 s and 10 s both occur on OpenNeuro).
MIN_PLAUSIBLE_TR = 0.3
MAX_PLAUSIBLE_TR = 5.0


class MissingTRError(Exception):
    """The image header carries no usable repetition time."""


def is_plausible_tr(tr):
    return tr is not None and MIN_PLAUSIBLE_TR <= float(tr) <= MAX_PLAUSIBLE_TR


def read_tr(img, strict=True):
    """The repetition time from a 4D image's header (an ``ants`` image or a path)."""
    import ants

    if not hasattr(img, "spacing"):
        img = ants.image_read(str(img))
    if len(img.spacing) < 4:
        raise MissingTRError(f"the image has {len(img.spacing)} dimensions; expected a 4D run")
    tr = float(img.spacing[3])
    if tr <= 0.0:
        raise MissingTRError(f"invalid TR in the header: {tr}")
    if strict and not is_plausible_tr(tr):
        raise MissingTRError(f"implausible TR in the header: {tr} s (outside "
                             f"{MIN_PLAUSIBLE_TR}-{MAX_PLAUSIBLE_TR} s); pass tr= explicitly")
    return tr


def get_masks(mask_dir=MASK_DIR):
    """The template, the two eye masks, and the crop edges of the small one."""
    import ants

    mask_dir = Path(mask_dir)
    eyemask_small = ants.image_read(str(mask_dir / "eyemask_small.nii"))
    eyemask_big = ants.image_read(str(mask_dir / "eyemask_big.nii"))
    template = ants.image_read(str(mask_dir / "dme_template.nii"))
    return {"template": template, "eyemask_big": eyemask_big, "eyemask_small": eyemask_small,
            "edges": mask_edges(eyemask_small.numpy())}


def mask_edges(mask):
    """``(x_edges, y_edges, z_edges)`` of the eye mask, the two eyes split along x."""
    idx = np.where(mask == 1)
    gap = np.argmax(np.diff(idx[0]))
    middle = (idx[0][gap] + idx[0][gap + 1]) // 2
    left, right = idx[0][idx[0] < middle], idx[0][idx[0] > middle]
    x_edges = (np.max(left), np.min(left), np.max(right), np.min(right))
    y_edges = (np.max(idx[1]), np.min(idx[1]))
    z_edges = (np.max(idx[2]), np.min(idx[2]))
    return x_edges, y_edges, z_edges


def register(func, masks, transforms=DEFAULT_TRANSFORMS, metric="GC"):
    """Register a 4D ``ants`` image to the template, one stage per mask."""
    import ants

    template = masks["template"]
    for mask, kind in zip((None, masks["eyemask_big"], masks["eyemask_small"]), transforms):
        reg = ants.registration(
            fixed=template,
            moving=ants.get_average_of_timeseries(func),
            aff_random_sampling_rate=1,
            type_of_transform=kind,
            mask=mask,
            aff_metric=metric,
            aff_sampling=512,
            aff_iterations=(200, 200, 200, 10),
            aff_smoothing_sigmas=(0, 0, 0, 0),
        )
        func = ants.apply_transforms(fixed=template, moving=func,
                                     transformlist=reg["fwdtransforms"], imagetype=3)
    return func


def crop_eyes(registered, masks):
    """The raw ``[47, 29, 18, T]`` eye block of a registered run, non-eye voxels zeroed."""
    data = registered.numpy() if hasattr(registered, "numpy") else np.array(registered)
    mask = masks["eyemask_small"].numpy()
    (x1, x0, x3, x2), (y1, y0), (z1, z0) = masks["edges"]
    data = data.copy()
    data[mask < 1, ...] = 0
    left = data[x0:x1, y0:y1, z0:z1, ...]
    right = data[x2:x3, y0:y1, z0:z1, ...]
    return np.concatenate((right, left)).astype(np.float32)


def normalize(block, std_cut=5):
    """Z-score each voxel over time, then each volume over space, clip at ``std_cut`` SD.

    Zero voxels (outside the mask or the field of view) are left at zero.
    """
    img = np.transpose(np.asarray(block, dtype=np.float32), axes=(3, 0, 1, 2)).copy()
    zero = img == 0
    img[zero] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        # Voxels that are zero in every volume are all-NaN here; they go back to 0.
        warnings.simplefilter("ignore", RuntimeWarning)
        img = (img - np.nanmean(img, axis=0)) / np.nanstd(img, axis=0)
        img = np.array([(x - np.nanmean(x)) / np.nanstd(x) for x in img])
    if std_cut is not None:
        sd = np.nanstd(img)
        np.clip(img, -std_cut * sd, std_cut * sd, out=img)
    img[~np.isfinite(img)] = 0
    img[zero] = 0
    return np.transpose(img, axes=(1, 2, 3, 0))


def extract_eye_block(func, masks=None, transforms=DEFAULT_TRANSFORMS, qa_png=None, tr=None):
    """A functional run -> ``(eye_block [47, 29, 18, T], tr)``.

    ``func`` is a path to a 4D NIfTI (or any format ANTs reads) or an ``ants``
    image. ``qa_png`` writes the QA thumbnail of the registration there. ``tr``
    overrides the header's repetition time.
    """
    import ants

    masks = masks or get_masks()
    img = func if hasattr(func, "spacing") else ants.image_read(str(func))
    tr = float(tr) if tr is not None else read_tr(img)
    registered = register(img, masks, transforms)
    raw = crop_eyes(registered, masks)
    if qa_png is not None:
        from deepmreye import thumbnail

        whole_brain = ants.get_average_of_timeseries(registered).numpy()
        thumbnail.save(thumbnail.from_arrays(whole_brain, masks["eyemask_small"].numpy(), raw),
                       qa_png)
    return normalize(raw), tr
