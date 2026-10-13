"""What the data are: OpenNeuro datasets screened, QA-labelled and passed, the
corpus's acquisitions, and the labelled datasets."""
import json
import shutil
from pathlib import Path

import h5py

from deepmreye.labels import (
    LBL_EYES, LBL_EYES_CUT, LBL_EYES_FAINT, LBL_NO_EYES_BAD_TRANSFORM, LBL_NO_EYES_GOOD_TRANSFORM,
    LBL_UNLABELED, qa_status,
)
from deepmreye.unsupervised import EVALUATION_SOURCES, unlabeled_subjects

QA_NAMES = {LBL_EYES: "eyes visible", LBL_EYES_FAINT: "eyes visible, faint",
            LBL_EYES_CUT: "eyes visible, cut off", LBL_NO_EYES_GOOD_TRANSFORM: "no eyes, registration fine",
            LBL_NO_EYES_BAD_TRANSFORM: "no eyes, registration failed"}
# One participant per QA outcome for a figure.
QA_EXAMPLES = {LBL_EYES: ("ds000006", "sub-01"), LBL_NO_EYES_GOOD_TRANSFORM: ("ds000017", "sub-1"),
               LBL_NO_EYES_BAD_TRANSFORM: ("ds000138", "sub-01")}


def corpus_stats(ctx, out):
    import pandas as pd

    root = Path(ctx.root)
    registry = root / "datasets.h5"
    _m, _bases, meta = ctx.basis()
    used = [tuple(s.split("/", 1)) for s in meta["subjects"]]
    eligible = unlabeled_subjects(root)
    gate = qa_status(registry)
    with h5py.File(registry, "r") as f:
        flat = {(d, s): int(f[d][s].attrs.get("approved", LBL_UNLABELED))
                for d in f if not d.startswith("dsL") for s in f[d]}
    flat = {k: v for k, v in flat.items() if v != LBL_UNLABELED}

    def by_status(rows):
        acc = {d for d, *_ in rows}
        return {st: {"datasets": sum(gate.get(d) == st for d in acc),
                     "participants": sum(gate.get(d) == st for d, *_ in rows)}
                for st in ("pass", "fail", "not labelled", "skipped")}

    openneuro = [d for d in gate if not d.startswith("dsL")]
    corpus_ds = sorted({d for d, _s in used})
    idx = pd.read_parquet(root / "index.parquet").set_index(["dataset", "subject"])
    corpus = idx.loc[used]
    labelled = idx[idx.has_labels].reset_index()
    record = {
        "funnel": {
            "openneuro datasets": len(openneuro),
            "skipped (nothing extracted to label)": sum(gate[d] == "skipped" for d in openneuro),
            "QA-labelled": sum(gate[d] in ("pass", "fail") for d in openneuro),
            "pass": sum(gate[d] == "pass" for d in openneuro),
            "fail": sum(gate[d] == "fail" for d in openneuro),
        },
        "qa_labels": {QA_NAMES[v]: sum(x == v for x in flat.values()) for v in QA_NAMES},
        "evaluation_sources_excluded": EVALUATION_SOURCES,
        "eligible": {"datasets": len({d for d, *_ in eligible}), "participants": len(eligible),
                     "by_status": by_status(eligible)},
        "corpus": {"datasets": len(corpus_ds), "participants": len(used), "volumes": int(meta["n_trs"]),
                   "by_status": by_status([(d, s) for d, s in used]),
                   "participants_per_dataset": [sum(d == a for d, _s in used) for a in corpus_ds],
                   "tr": corpus.repetition_time.astype(float).tolist(),
                   "volumes_per_run": corpus.n_trs.astype(int).tolist()},
        "labelled": [{"dataset": d, "participants": int(len(g)), "tr": float(g.repetition_time.median()),
                      "volumes_per_participant": float(g.n_trs.median())}
                     for d, g in labelled.groupby("dataset")],
        "qa_examples": {QA_NAMES[v]: f"qa_examples/{d}_{s}.png" for v, (d, s) in QA_EXAMPLES.items()},
    }
    (Path(out).parent / "qa_examples").mkdir(exist_ok=True)
    for v, (d, s) in QA_EXAMPLES.items():
        if flat.get((d, s)) == v:
            shutil.copy(root / d / f"{s}.png", Path(out).parent / "qa_examples" / f"{d}_{s}.png")
    json.dump(record, open(out, "w"), indent=1)
    print(json.dumps({k: record[k] for k in ("funnel", "eligible")}, indent=1))
    print(json.dumps({k: v for k, v in record["corpus"].items() if k in ("datasets", "participants", "by_status")}))
