"""A small QA image of a participant's registered eyes.

The one question to ask of a preprocessed run before trusting its gaze is
whether both eyeballs are inside the eye mask. This writes a ~20 KB PNG that
answers it: the registered brain at the eyes' level (z = -30 in template space)
with the eye mask drawn over it, then the extracted eye block seen from two
sides.

It is built from the *raw* registered volumes. Normalisation z-scores every
voxel across time, so the temporal mean of a stored eye block is flat and
renders as noise; :func:`from_block` (for a block that is already normalised)
shows the temporal standard deviation instead.
"""

import numpy as np

# Panel height in pixels. The eye block is 47 x 29 x 18, so panels are tiny;
# they get scaled up to this with nearest-neighbour, which keeps voxel edges
# crisp rather than smearing them into a blur.
PANEL_HEIGHT = 132

# Gap between panels, in the background colour.
GUTTER = 4

# The slice the whole QA workflow has always looked at: z=-30 in template
# space, where both eyeballs sit. `plot_subject_report` annotates it as such.
Z_SLICE = 15


def _to_uint8(a):
    """Scale to 0-255 on a robust range, so one hot voxel cannot flatten the rest."""
    a = np.asarray(a, dtype=np.float32)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros(a.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, 1), np.percentile(finite, 99)
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def _overlay_mask(gray, mask):
    """Grayscale slice with the eye mask painted red, as in the report."""
    rgb = np.stack([gray] * 3, axis=-1)
    hot = np.asarray(mask) > 0
    if hot.any():
        rgb[hot, 0] = np.clip(rgb[hot, 0].astype(np.int16) + 160, 0, 255).astype(np.uint8)
        rgb[hot, 1] = (rgb[hot, 1] * 0.3).astype(np.uint8)
        rgb[hot, 2] = (rgb[hot, 2] * 0.3).astype(np.uint8)
    return rgb


def _compose(panels, height=PANEL_HEIGHT, gutter=GUTTER):
    """Lay panels out left to right at a common height."""
    from PIL import Image

    images = []
    for p in panels:
        arr = np.asarray(p)
        img = Image.fromarray(arr if arr.ndim == 3 else np.stack([arr] * 3, -1))
        width = max(1, round(img.width * height / img.height))
        images.append(img.resize((width, height), Image.NEAREST))

    total = sum(i.width for i in images) + gutter * (len(images) - 1)
    sheet = Image.new("RGB", (total, height), (0, 0, 0))
    x = 0
    for img in images:
        sheet.paste(img, (x, 0))
        x += img.width + gutter
    return sheet


def _panels(wb_slice, mask_slice, eye_mean):
    """The three views, from the whole-brain slice and the eye block's temporal mean.

    ``eye_mean`` is [X, Y, Z]. Collapsing Z gives the axial view where both
    eyeballs appear side by side -- the panel that answers the QA question --
    and collapsing Y gives a sagittal view that shows whether the bounding box
    clipped them (label 3).
    """
    return [
        _overlay_mask(_to_uint8(np.flipud(wb_slice)), np.flipud(mask_slice)),
        _to_uint8(eye_mean.mean(axis=2).T),
        _to_uint8(eye_mean.mean(axis=1).T),
    ]


def from_block(eye_block, height=PANEL_HEIGHT):
    """Build a thumbnail from a stored, already-normalized eye block.

    For participants that never went through registration here -- the gaze
    labeled datasets arrived as ``.npz`` exports -- there is no whole-brain
    volume or mask to draw, so this renders the eye panels alone and the strip
    is two panels rather than three.

    It shows the temporal *standard deviation*, not the mean. Normalization
    z-scores every voxel across time, which flattens the temporal mean to noise
    (measured: SD map std 0.50 against temporal mean std 0.06 on the same
    block), while the variance map keeps the eyeballs clearly outlined.
    """
    block = np.asarray(eye_block, dtype=np.float32)
    volume = block.std(axis=3) if block.ndim == 4 else block
    return _compose([_to_uint8(volume.mean(axis=2).T), _to_uint8(volume.mean(axis=1).T)],
                    height=height)


def from_arrays(whole_brain, mask, masked_eye, height=PANEL_HEIGHT):
    """Build the thumbnail during extraction.

    ``whole_brain`` is the registered 3D mean volume, ``mask`` the eye mask in
    the same space, and ``masked_eye`` the raw 4D eye block *before*
    normalization -- see the module docstring for why that ordering matters.
    """
    whole_brain = np.asarray(whole_brain)
    mask = np.asarray(mask)
    eye_mean = np.asarray(masked_eye, dtype=np.float32).mean(axis=3)
    return _compose(_panels(whole_brain[:, :, Z_SLICE], mask[:, :, Z_SLICE], eye_mean),
                    height=height)


def save(image, path):
    """Write a thumbnail, atomically, so a killed job leaves no half PNG."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".png.tmp")
    image.save(tmp, format="PNG", optimize=True)
    tmp.replace(path)
    return path
