"""Small QA thumbnails: the image you actually judge a subject on.

Every extracted participant gets a ``<subject>.png`` next to its
``<subject>.h5``. It is a strip of panels -- the registered brain at z=-30 with
the eye mask drawn over it, then the extracted eye block seen from two sides --
which is enough to answer the only question QA asks: are the eyeballs in there.

This exists because the full HTML report is ~5 MB and the thumbnail is ~20 KB.
At the QA sample of 1779 subjects the reports cost 8 GB, which was tolerable; at
full extraction they would cost well over 100 GB, which is not, and no one is
going to open twenty thousand Plotly pages anyway. The report is still available
per subject (``--report html``) when a specific case needs the histogram and the
timecourses; the thumbnail is what gets produced by default and shipped.

Two entry points produce byte-identical output, so the corpus is uniform even
though half of it was extracted before thumbnails existed:

- :func:`from_arrays` -- during extraction, straight from the volumes in hand.
- :func:`from_report` -- backfill, by reading the arrays back out of an
  existing report.

Both take *raw* (pre-normalization) volumes. That matters: normalization
z-scores every voxel across time, so the temporal mean of a stored ``eye_block``
is approximately zero everywhere and renders as noise. The anatomy is only
visible before that step.
"""
import base64
import json
import re
import zlib

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


# ---------------------------------------------------------------------------
# Backfill: recover the same arrays from a report that already exists.
# ---------------------------------------------------------------------------

# `plot_subject_report` emits its heatmaps in a fixed order. Indices 4/5 are the
# z=-30 brain and its mask; 7/8/9 are the eye block collapsed along each axis.
_TRACE_WB_Z, _TRACE_MASK_Z = 4, 5
_TRACE_EYE_X, _TRACE_EYE_Y, _TRACE_EYE_Z = 7, 8, 9
_MIN_TRACES = 10


def _plotly_traces(html):
    """The trace array from a report's ``Plotly.newPlot`` call.

    Plotly embeds heatmap ``z`` as base64 rather than JSON numbers, so the
    arrays cannot be read without walking the script -- there is no lighter
    handle on this data than the report itself. Several ``newPlot`` calls can
    appear; the real one is the array with every trace in it.
    """
    for script in re.findall(r"<script.*?>(.*?)</script>", html, re.DOTALL):
        if "Plotly.newPlot" not in script:
            continue
        for match in re.finditer(r"Plotly\.newPlot", script):
            start = script.find("[", match.start())
            if start == -1:
                continue
            depth, end = 0, -1
            for i in range(start, len(script)):
                if script[i] == "[":
                    depth += 1
                elif script[i] == "]":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end == -1:
                continue
            try:
                parsed = json.loads(script[start:end + 1])
            except ValueError:
                continue
            if isinstance(parsed, list) and len(parsed) >= _MIN_TRACES:
                return parsed
    return None


def _trace_matrix(trace):
    """Decode one heatmap trace's ``z`` into an array."""
    z = trace.get("z")
    if isinstance(z, list):
        return np.array(z, dtype=np.float32)
    if not isinstance(z, dict) or "bdata" not in z:
        return None
    raw = base64.b64decode(z["bdata"])
    try:
        raw = zlib.decompress(raw)
    except zlib.error:
        pass  # Plotly only compresses above a size threshold.
    arr = np.frombuffer(raw, dtype=np.dtype(z.get("dtype", "f4")))
    shape = z.get("shape")
    if shape:
        arr = arr.reshape(tuple(int(x) for x in str(shape).split(",")))
    return arr


def from_report(html, height=PANEL_HEIGHT):
    """Rebuild the thumbnail from an existing HTML report.

    Returns ``None`` if the report does not carry the expected traces, so a
    malformed one is skipped rather than producing a misleading image.
    """
    traces = _plotly_traces(html)
    if traces is None:
        return None

    wb = _trace_matrix(traces[_TRACE_WB_Z])
    mask = _trace_matrix(traces[_TRACE_MASK_Z])
    if wb is None or mask is None:
        return None

    # The report stores the eye block already collapsed, and transposed the way
    # it was plotted. Undo that so both entry points feed `_compose` the same
    # orientation.
    eye_z = _trace_matrix(traces[_TRACE_EYE_Z])   # mean over Z, [X, Y]
    eye_y = _trace_matrix(traces[_TRACE_EYE_Y])   # mean over Y, plotted [Z, X]
    if eye_z is None or eye_y is None:
        return None

    panels = [
        _overlay_mask(_to_uint8(np.flipud(wb)), np.flipud(mask)),
        _to_uint8(eye_z.T),
        _to_uint8(eye_y),
    ]
    return _compose(panels, height=height)


def save(image, path):
    """Write a thumbnail, atomically, so a killed job leaves no half PNG."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".png.tmp")
    image.save(tmp, format="PNG", optimize=True)
    tmp.replace(path)
    return path
