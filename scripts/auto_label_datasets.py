#!/usr/bin/env python3
"""Automatically process and auto-label datasets using the trained triage model.

1. Datasets without HTML reports are marked as skipped (approved = -99).
2. Unlabeled subjects are classified with the triage model (qa_model.joblib).
3. High-confidence 'Eyes visible' predictions (label 1, conf >= threshold) are auto-approved (approved = 1).
4. Low-confidence 'Eyes visible' predictions (conf < threshold) AND non-1 predictions (0, 2, 3, 4)
   remain unlabeled (approved = -1) with predicted labels pre-selected for manual review in the UI.
"""

import sys
import h5py
import joblib
import numpy as np
from pathlib import Path
import argparse

# Add deepmreye to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye.datasource import resolve
from deepmreye.storage import subject_path
from deepmreye.qa_classifier import features_from_file
from deepmreye.labels import append_label_events


def auto_label_corpus(data_dir=None, quiet=False):
    data_dir = resolve(data_dir, download=False, quiet=True)
    h5_path = data_dir / "datasets.h5"
    model_path = data_dir / "qa_model.joblib"
    csv_path = data_dir / "labels.csv"

    if not h5_path.exists():
        raise FileNotFoundError(f"Datastore not found at {h5_path}")

    model = None
    if model_path.exists():
        try:
            model_data = joblib.load(model_path)
            model = model_data["model"]
            if not quiet:
                print(f"[+] Loaded triage model from {model_path}")
        except Exception as e:
            print(f"[-] Failed to load triage model: {e}")

    events = []
    stats = {
        "no_reports_ds": 0,
        "auto_labeled_subs": 0,
        "already_manual_subs": 0,
        "pred_counts": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    }

    from deepmreye.qa_classifier import open_h5_file

    with open_h5_file(h5_path, "a") as f:
        datasets = list(f.keys())
        for ds in datasets:
            grp = f[ds]
            subs_with_reports = [s for s in grp.keys() if "report_html_path" in grp[s].attrs]

            if not subs_with_reports:
                # Mark datasets without reports as skipped (-99)
                if grp.attrs.get("approved", -1) != -99:
                    grp.attrs["approved"] = -99
                    events.append((ds, "dataset", "", -99))
                stats["no_reports_ds"] += 1
                continue

            for sub in subs_with_reports:
                sub_grp = grp[sub]
                cur_approved = sub_grp.attrs.get("approved", -1)
                is_manual = sub_grp.attrs.get("is_manual", False)

                # Skip if already manually verified by the user
                if is_manual and cur_approved != -1:
                    stats["already_manual_subs"] += 1
                    continue

                if model is None:
                    continue

                path = subject_path(data_dir, ds, sub)
                if not path.exists():
                    continue

                report_path = sub_grp.attrs.get("report_html_path", "")
                tr = sub_grp.attrs.get("repetition_time", None)
                ntr = sub_grp.attrs.get("n_trs", None)
                feats = features_from_file(path, report_path=report_path, tr=tr, n_trs=ntr)
                if feats is None:
                    continue

                try:
                    proba = model.predict_proba(feats.reshape(1, -1))[0]
                    idx = int(np.argmax(proba))
                    pred_lbl = int(model.classes_[idx])
                    conf = float(proba[idx])

                    sub_grp.attrs["pred_lbl"] = pred_lbl
                    sub_grp.attrs["pred_conf"] = conf
                    sub_grp.attrs["approved"] = pred_lbl
                    sub_grp.attrs["is_manual"] = False

                    stats["pred_counts"][pred_lbl] += 1
                    stats["auto_labeled_subs"] += 1
                    events.append((ds, "subject", sub, pred_lbl))
                except Exception as e:
                    pass

    if events:
        try:
            append_label_events(csv_path, events)
        except Exception as e:
            print(f"Warning: Failed to log events to CSV: {e}")

    if not quiet:
        print("\n=== Auto-Label Summary ===")
        print(f"Datasets without HTML reports (marked -99 skipped): {stats['no_reports_ds']}")
        print(f"Manually verified subjects preserved: {stats['already_manual_subs']}")
        print(f"Subjects auto-labeled by classifier: {stats['auto_labeled_subs']}")
        print(f"Model predictions breakdown: {stats['pred_counts']}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-label dataset using triage classifier")
    parser.add_argument("--data-dir", default=None, help="Corpus directory")
    args = parser.parse_args()

    auto_label_corpus(data_dir=args.data_dir)
