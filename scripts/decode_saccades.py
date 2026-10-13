"""Archived saccade studies, decoded trial by trial with no eye tracking.

Three OpenNeuro datasets from one lab ran saccade tasks in the scanner and
publish each trial's target position and timing in their ``events.tsv``, but
no gaze traces: ``ds000119`` (visually guided saccades and antisaccades in the
same runs), ``ds000120`` and ``ds000121`` (antisaccades). Their participants
are in the unlabelled corpus, so the decoded gaze comes from
``scripts/decode_corpus.py`` and no label of theirs was ever used.

For every trial: the change in decoded horizontal gaze from the 1.5 s before
the target appears to 0.3-1.8 s after it, against the target's horizontal
position, and the event-locked trace around it at sub-TR resolution. A guided
saccade looks at the target and an antisaccade looks away from it, so the two
should correlate with opposite signs.

    python scripts/decode_saccades.py     # fetches events.tsv from OpenNeuro once (cached)
"""
import argparse
import io
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deepmreye.datasource import resolve  # noqa: E402

DATASETS = {"ds000119": "cue_position", "ds000120": "stim_dot_position", "ds000121": "stim_dot_position"}
CENTRE = 320.0                  # screen x of the fixation cross, in the tasks' own units
BASELINE, RESPONSE = (-1.5, 0.0), (0.3, 1.8)   # seconds from target onset
LAGS = np.round(np.arange(-3.0, 6.0001, 0.15), 2)


def trial_type(dataset, row):
    """``guided`` or ``anti`` for the trials the dataset records a target for."""
    if dataset == "ds000119":
        return {"VGSTARG": "guided", "ANTITARG": "anti"}.get(str(row["trial_type"]).upper())
    return "anti"


def events(dataset, source_key, cache):
    """The run's events.tsv, from OpenNeuro on first use."""
    import pandas as pd

    path = cache / f"{Path(source_key).name.replace('_bold.nii.gz', '_events.tsv')}"
    if not path.exists():
        from deepmreye.pipeline import make_s3_client

        key = source_key.replace("_bold.nii.gz", "_events.tsv")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(make_s3_client().get_object(Bucket="openneuro.org", Key=key)["Body"].read())
    ev = pd.read_csv(io.BytesIO(path.read_bytes()), sep="\t")
    ev.columns = [c.lower() for c in ev.columns]
    return ev


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--decoded", default="results/decoded/corpus_gaze.h5")
    ap.add_argument("--out", default="results/decoded/saccades.json")
    ap.add_argument("--events", default="results/decoded/events")
    a = ap.parse_args()

    import pandas as pd
    from scipy.stats import pearsonr

    root = resolve(None, download=False, quiet=True)
    index = pd.read_parquet(Path(root) / "index.parquet").set_index(["dataset", "subject"])
    f = h5py.File(a.decoded, "r")
    meta, offsets, gaze = json.loads(f.attrs["meta"]), f["offsets"][:], f["gaze"]
    trials, traces = [], {}
    for i, m in enumerate(meta):
        if m["dataset"] not in DATASETS:
            continue
        g = gaze[offsets[i]:offsets[i + 1]]                       # [T, 10, 2]
        t = (np.arange(len(g))[:, None] + np.arange(10)[None, :] / 10.0).reshape(-1) * m["tr"]
        x = g[..., 0].reshape(-1)
        x = (x - x.mean()) / x.std()                             # each run in its own z-units
        ev = events(m["dataset"], index.loc[(m["dataset"], m["subject"]), "source_key"], Path(a.events))
        position = DATASETS[m["dataset"]]
        for _, row in ev[ev[position].notna()].iterrows():
            kind, onset = trial_type(m["dataset"], row), float(row["onset"])
            if kind is None or onset + LAGS[0] < 0 or onset + LAGS[-1] > t[-1]:
                continue
            base = x[(t >= onset + BASELINE[0]) & (t < onset + BASELINE[1])].mean()
            shift = x[(t >= onset + RESPONSE[0]) & (t < onset + RESPONSE[1])].mean() - base
            outcome = str(row.get("score_txt", "")).upper()
            trials.append({"dataset": m["dataset"], "subject": m["subject"], "agreement": m["agreement"],
                           "type": kind, "outcome": outcome, "target": float(row[position]) - CENTRE,
                           "shift": float(shift)})
            side = "left" if row[position] < CENTRE else "right"
            traces.setdefault(f"{m['dataset']}|{kind}|{outcome}|{side}", []).append(
                (np.interp(onset + LAGS, t, x) - base).tolist())

    summary = {}
    df = pd.DataFrame(trials)
    for (ds, kind, outcome), g in df.groupby(["dataset", "type", "outcome"]):
        if len(g) >= 4:
            r = pearsonr(g.target, g["shift"])
            per = [pearsonr(s.target, s["shift"])[0] for _, s in g.groupby("subject") if len(s) > 4]
            summary[f"{ds}|{kind}|{outcome}"] = {
                "trials": int(len(g)), "participants": int(g.subject.nunique()),
                "r": float(r[0]), "p": float(r[1]), "per_participant_r": [float(v) for v in per]}
    out = {"lags": LAGS.tolist(), "baseline": BASELINE, "response": RESPONSE,
           "summary": summary, "trials": trials,
           "traces": {k: np.mean(v, axis=0).tolist() for k, v in traces.items()},
           "trace_sem": {k: (np.std(v, axis=0) / np.sqrt(len(v))).tolist() for k, v in traces.items()},
           "trace_counts": {k: len(v) for k, v in traces.items()}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"))
    for k, v in summary.items():
        print(f"  {k:<40} trials {v['trials']:4d}  r {v['r']:+.2f}  per participant "
              f"{np.round(v['per_participant_r'], 2).tolist()}")
    print(f"[+] {a.out}")


if __name__ == "__main__":
    main()
