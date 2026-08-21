"""The eye-tracking scan's file pattern.

This is worth a test because both of its failure modes are silent. Too narrow
and whole datasets vanish from the survey with no error -- an earlier version
required `.tsv`/`.asc` and so missed ds001840 (24 participants) and ds004283,
both of which ship real simultaneous gaze as EyeLink `.edf`. Too broad and it
invents datasets that do not exist: a substring scan for "eye" reports ds004529
as 34 paired participants on the strength of 204 files named
`s001-fp_no_Eyelink.log`, which are stimulus logs for the condition run
*without* the tracker.

Every key below is a real key from the bucket.
"""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from scan_eyetracking_datasets import ET_RX, EXT_RX  # noqa: E402

MATCHES = [
    # BIDS physio, the common form
    "ds004158/sub-01/ses-01/func/sub-01_ses-01_task-rest_run-01_recording-eye1_physio.tsv.gz",
    "ds000113/sub-01/ses-movie/func/sub-01_ses-movie_task-movie_run-1_recording-eyegaze_physio.tsv.gz",
    "ds001242/sub-01/func/sub-01_task-fearlearning_recording-eye_physio.tsv.gz",
    # raw EyeLink ASCII, shipped under sourcedata
    "ds006642/sourcedata/sub-01/ses-001/func/sub-01_ses-001_run-001_task-backtothefuture_eyelinkraw.asc.gz",
    # raw EyeLink binary -- needs a converter, but must still be *found*
    "ds004283/sub-01/ses-02/func/sub-01_ses-02_task-lokicat_run-01_recording-eyetracking_physio.EDF",
    "ds004283/sub-01/ses-02/func/sub-01_ses-02_task-lokicat_recording-eyetracking-luminance_physio.EDF",
    "ds001840/sub-04/eyetrack/sub-04_task-viewclips_eyetrack.edf",
    "ds007305/sub-004/beh/sub-004_task-foodlottery_eyetrack.edf",
]

NON_MATCHES = [
    # "Eyelink" in the name, but it is a stimulus log for the no-tracker
    # condition. This is the false positive that put ds004529 in the old survey.
    "ds004529/derivatives/preprocessing/FRAPPS/sub-01/func/s001-fp_no_Eyelink.log",
    "ds004529/derivatives/preprocessing/FRAPPS/sub-01/func/s001-fp_no_Eyelink_onsets.mat",
    # other physio recordings that are not gaze
    "ds000113/sub-01/func/sub-01_task-x_recording-cardresp_physio.tsv.gz",
    "ds000113/sub-06/func/sub-06_task-x_recording-motion_physio.tsv.gz",
    # documentation and code that mention the tracker
    "ds000113/sourcedata/code/rawdata_conversion/convert_eyelink",
    "ds000113/sourcedata/code/overlay_gaze_on_video",
    # the sidecar, not the recording
    "ds007305/sub-004/beh/sub-004_task-foodlottery_eyetrack.json",
]


@pytest.mark.parametrize("key", MATCHES)
def test_recognised_eye_tracking_keys(key):
    assert ET_RX.search(key), f"missed a real recording: {key}"


@pytest.mark.parametrize("key", NON_MATCHES)
def test_rejected_keys(key):
    assert not ET_RX.search(key), f"false positive: {key}"


def test_edf_is_matched_so_it_can_be_reported_as_blocked():
    """`.edf` needs a converter, which is a reason to flag it, not to skip it."""
    key = "ds001840/sub-04/eyetrack/sub-04_task-viewclips_eyetrack.edf"
    assert ET_RX.search(key)
    assert EXT_RX.search(key).group(1).lower() == "edf"


def test_extension_is_recovered_for_every_recognised_key():
    for key in MATCHES:
        m = EXT_RX.search(key)
        assert m, key
        assert m.group(1).lower() in {"tsv", "tsv.gz", "asc", "asc.gz", "edf"}


def test_pattern_is_anchored_at_the_end_of_the_key():
    """A recording name appearing mid-path is a directory, not the file."""
    assert not ET_RX.search(
        "ds001840/sub-04/sub-04_task-viewclips_eyetrack.edf/notes.txt")
