"""Roll every voxelnet trial JSON into one comparable table.

Written because the trial log was being maintained by hand, which does not survive "we will
run MANY more". Every run writes `{"args": ..., "results": {fold: {...,"history":[...]}}}`,
so the record already exists in machine-readable form; this reads all of them and emits

  results/subtr/trials.csv        one row per (trial, fold) -- config knobs, outcome, shape
  results/subtr/trials_epochs.csv one row per (trial, fold, epoch) -- the full curves

and prints a ranked summary. Nothing here re-runs or re-scores anything: if a number is not
in a JSON it is not in the table, so the table cannot drift from what was actually run.

The columns worth reading together are `fit_r_at_best`, `val_r_best` and `test_r`. Their
differences separate the two failure modes that need opposite fixes:

  fit high, val low   -> overfitting; regularise, augment, or get more acquisitions
  fit low,  val low   -> underfitting; train longer, raise lr, or cut regularisation

`gap` is `fit_r_at_best - val_r_best`. `es_fired` records whether *early stopping* ended the
run, which is NOT the same as convergence and must not be read as such: trial04 stopped on
patience at epoch 11 looking converged, and trial06 -- the identical configuration on a
150-epoch cosine schedule -- went on to improve val r from 0.6350 to 0.6874. Patience on a
metric that swings +-0.05 epoch to epoch fires on noise. `hit_cap` is the honest warning: the
run used its whole epoch budget, so its configuration is untested beyond that budget.

    python scripts/summarize_voxelnet_trials.py
    python scripts/summarize_voxelnet_trials.py --glob 'results/subtr/trial*.json'
"""
import argparse
import csv
import json
from pathlib import Path

KNOBS = ("note", "encoder", "rank", "width", "hidden", "dropout", "lags", "chunk",
         "batch_chunks", "epochs", "steps_per_epoch", "lr", "weight_decay", "cosine",
         "patience", "val_datasets", "val_subjects", "noise", "vox_dropout", "shift",
         "mixup", "init_encoder", "seed")


def rows_for(path):
    try:
        d = json.loads(Path(path).read_text())
    except (json.JSONDecodeError, OSError):
        return [], []
    args = d.get("args", {})
    res = d.get("results", {})
    if not isinstance(res, dict):
        return [], []
    flat, curves = [], []
    for fold, r in res.items():
        if not isinstance(r, dict):
            continue
        hist = r.get("history", []) or []
        best_ep, fit_at_best, val_best = None, None, None
        for h in hist:
            v = h.get("val_r")
            if v is None:
                continue
            if val_best is None or v > val_best:
                val_best, best_ep, fit_at_best = v, h.get("epoch"), h.get("fit_r")
            curves.append({"trial": Path(path).stem, "fold": fold, **h})
        last_ep = hist[-1]["epoch"] if hist else None
        pat = args.get("patience")
        es_fired = (None if (best_ep is None or last_ep is None or pat is None)
                    else bool(last_ep - best_ep >= pat))
        cap = args.get("epochs")
        hit_cap = (None if (last_ep is None or cap is None) else bool(last_ep >= cap - 1))
        inc, test = r.get("incumbent"), r.get("net")
        flat.append({
            "trial": Path(path).stem,
            "fold": fold,
            "incumbent": inc,
            "test_r": test,
            "delta": (None if (inc is None or test is None) else round(test - inc, 4)),
            "val_r_best": val_best,
            "fit_r_at_best": fit_at_best,
            "gap": (None if (fit_at_best is None or val_best is None)
                    else round(fit_at_best - val_best, 4)),
            "best_epoch": best_ep,
            "last_epoch": last_ep,
            "es_fired": es_fired,
            "hit_cap": hit_cap,
            "n_test": r.get("n_test"),
            **{k: args.get(k) for k in KNOBS},
        })
    return flat, curves


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--glob", default="results/subtr/trial*.json")
    p.add_argument("--out-dir", default="results/subtr")
    args = p.parse_args()

    flat, curves = [], []
    for f in sorted(Path().glob(args.glob)):
        a, b = rows_for(f)
        flat += a
        curves += b
    if not flat:
        raise SystemExit(f"[!] no trial JSONs matched {args.glob}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "trials.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)
    if curves:
        keys = sorted({k for c in curves for k in c})
        with (out / "trials_epochs.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(curves)

    def fmt(v, n=4):
        return "--" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))

    print(f"{len(flat)} (trial, fold) rows -> {out/'trials.csv'}")
    print(f"{len(curves)} epoch rows -> {out/'trials_epochs.csv'}\n")
    hdr = (f"{'trial':<34} {'fold':<24} {'test':>7} {'inc':>7} {'delta':>7} "
           f"{'val*':>7} {'fit*':>7} {'gap':>7} {'ep':>7} {'stop':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(flat, key=lambda q: (-(q["test_r"] or -9), q["trial"])):
        print(f"{r['trial']:<34} {r['fold']:<24} {fmt(r['test_r']):>7} "
              f"{fmt(r['incumbent']):>7} {fmt(r['delta']):>7} {fmt(r['val_r_best']):>7} "
              f"{fmt(r['fit_r_at_best']):>7} {fmt(r['gap']):>7} "
              f"{str(r['best_epoch']) + '/' + str(r['last_epoch']):>7} "
              f"{('cap' if r['hit_cap'] else 'patnce'):>6}")
    capped = [r for r in flat if r["hit_cap"]]
    if capped:
        print(f"\n[!] {len(capped)} run(s) used their entire epoch budget -- untested beyond it:")
        for r in capped:
            print(f"    {r['trial']} / {r['fold']} (best {r['best_epoch']}, cap {r['epochs']})")
    late = [r for r in flat if r["best_epoch"] is not None and r["last_epoch"]
            and r["best_epoch"] > 0.6 * r["last_epoch"]]
    if late:
        print(f"\n[!] {len(late)} run(s) peaked in their final 40% of epochs -- suspect the "
              f"schedule, not the configuration:")
        for r in late:
            print(f"    {r['trial']} / {r['fold']} (best {r['best_epoch']} of {r['last_epoch']})")


if __name__ == "__main__":
    main()
