#!/usr/bin/env python3
"""Train the eye-detection triage model and rank unlabeled subjects.

Two uses, neither of which approves anything:

1. **Label the uncertain ones first.** ``--rank`` writes unlabeled subjects
   ordered by how unsure the model is, so your labeling effort goes where it
   changes the model most instead of walking accession numbers in order.
2. **Screen the full download.** After a dataset is approved, extraction pulls
   in every subject and nobody looks at them. ``--flag`` lists the ones that
   look like no-eyes so you can check them.

Dataset approval remains manual. See ``deepmreye/qa_classifier.py`` for why this
is not the automatic gate that was previously removed.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye import qa_classifier as qac
from deepmreye.storage import iter_subjects, subject_path


def train_and_save_classifier(data_dir, registry=None, out_path=None, manual_only=True):
    """Train the QA triage classifier and save to out_path.

    Returns a dict with status and summary metrics.
    """
    data_dir = Path(data_dir).resolve()
    X, y, keys = qac.build_training_set(data_dir, registry, manual_only=manual_only)
    if len(X) < 5 and manual_only:
        X, y, keys = qac.build_training_set(data_dir, registry, manual_only=False)

    if len(X) == 0:
        return {
            "success": False,
            "message": "No labeled subjects with extracted data found. Label some subjects first.",
            "num_samples": 0
        }

    unique_classes = set(y)
    num_datasets = len(set(d for d, _ in keys))

    if len(unique_classes) < 2:
        return {
            "success": False,
            "message": f"Only 1 label class present ({len(X)} subjects). Label subjects of at least 2 different classes before training.",
            "num_samples": len(X),
            "num_datasets": num_datasets
        }

    groups = [ds for ds, _ in keys]
    model, scores = qac.train(X, y, groups=groups)

    out = Path(out_path) if out_path else data_dir / "qa_model.joblib"
    import joblib
    joblib.dump({"model": model, "features": qac.FEATURE_NAMES}, out)

    cv_mean = float(scores.mean()) if scores is not None else None
    cv_std = float(scores.std()) if scores is not None else None

    msg = f"Retrained on {len(X)} subjects across {num_datasets} datasets."
    if cv_mean is not None:
        msg += f" CV Accuracy: {cv_mean*100:.1f}% (±{cv_std*100:.1f}%)."

    return {
        "success": True,
        "message": msg,
        "num_samples": len(X),
        "num_datasets": num_datasets,
        "num_classes": len(unique_classes),
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "out_path": out,
        "model": model,
        "X": X,
        "y": y,
        "keys": keys
    }


def main():
    parser = argparse.ArgumentParser(description="Train the QA triage classifier.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--out", default=None,
                        help="Where to save the fitted model. Defaults to "
                             "<data-dir>/qa_model.joblib, which is exactly where the "
                             "labeling UI looks for it, so a plain run makes the next "
                             "`deepmreye qa` session pre-select predictions.")
    parser.add_argument("--rank", action="store_true",
                        help="Write unlabeled subjects ordered by model uncertainty.")
    parser.add_argument("--flag", action="store_true",
                        help="List subjects the model predicts have no eyes.")
    parser.add_argument("--flag-threshold", type=float, default=0.5,
                        help="P(eyes) below this is flagged for review.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    print("Collecting labeled subjects & training classifier...")
    res = train_and_save_classifier(data_dir, registry=args.registry, out_path=args.out)
    print(res["message"])
    if not res["success"]:
        return

    model = res["model"]
    keys = res["keys"]
    scores = res.get("cv_mean")

    order = np.argsort(model.feature_importances_)[::-1]
    print("\nFeature importance:")
    for i in order[:6]:
        print(f"  {qac.FEATURE_NAMES[i]:<20} {model.feature_importances_[i]:.3f}")

    print(f"\nSaved model to {res['out_path']}")
    print("Restart the labeling UI (or click Retrain in the UI) to pick it up.")

    if not (args.rank or args.flag):
        return

    labeled = {(ds, sub) for ds, sub in keys}
    todo = [(ds, sub, p) for ds, sub, p in iter_subjects(data_dir) if (ds, sub) not in labeled]
    if not todo:
        print("\nNo unlabeled extracted subjects.")
        return

    print(f"\nScoring {len(todo)} unlabeled subjects...")
    from deepmreye.pipeline import APPROVED_LABELS
    classes = list(model.classes_)
    eye_idx = [i for i, c in enumerate(classes) if int(c) in APPROVED_LABELS]

    rows = []
    for ds, sub, path in todo:
        feats = qac.features_from_file(path)
        if feats is None:
            continue
        proba = model.predict_proba(feats.reshape(1, -1))[0]
        # p_eyes sums every "has eyes" class (clean + cut off), which is the
        # quantity that decides whether a dataset survives QA.
        p_eyes = float(sum(proba[i] for i in eye_idx))
        top = int(classes[int(np.argmax(proba))])
        rows.append((ds, sub, p_eyes, top, float(max(proba))))

    if args.rank:
        # Least confident first: probability nearest 0.5 is where a label helps most.
        # Least confident first: a low top-class probability is where your
        # label teaches the model the most.
        ranked = sorted(rows, key=lambda r: r[4])
        out = data_dir / "qa_triage_order.csv"
        with open(out, "w") as f:
            f.write("dataset,subject,p_eyes,predicted,confidence\n")
            for ds, sub, p, top, conf in ranked:
                f.write(f"{ds},{sub},{p:.4f},{top},{conf:.4f}\n")
        print(f"[+] Labeling order (least confident first) -> {out}")
        for ds, sub, p, top, conf in ranked[:10]:
            print(f"    {ds}/{sub}  predicted={qac.LABEL_NAMES.get(top, top)} conf={conf:.2f}")

    if args.flag:
        flagged = sorted((r for r in rows if r[2] < args.flag_threshold), key=lambda r: r[2])
        out = data_dir / "qa_flagged.csv"
        with open(out, "w") as f:
            f.write("dataset,subject,p_eyes,predicted,confidence\n")
            for ds, sub, p, top, conf in flagged:
                f.write(f"{ds},{sub},{p:.4f},{top},{conf:.4f}\n")
        print(f"\n[+] {len(flagged)} subjects flagged as likely no-eyes -> {out}")
        print("    These are NOT removed. Review them in the QA UI.")


if __name__ == "__main__":
    main()
