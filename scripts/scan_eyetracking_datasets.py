#!/usr/bin/env python3
"""Find OpenNeuro datasets that ship eye tracking paired with a functional run.

Two questions, and the second is the cheap one:

1. **Across all of OpenNeuro** -- which accessions record gaze during a BOLD
   run? That is how `dsL07`..`dsL12` were found.
2. **Among the accessions already extracted into this corpus** -- which of
   *those* record gaze? This is the question worth asking first, because those
   participants already have coregistered eye blocks on disk. Adding their
   labels costs one TSV download each and a `--labels-only` relabel: no
   download of the BOLD, no ANTs run. Everything else in the ingest path is
   already paid for.

Pairing is per *subject*, not per dataset: a dataset can record gaze in a
behavioural session and BOLD in another, which is why `ds005166` was excluded.
A subject counts only if some eye-tracking file and some functional run both
carry its label. That is necessary, not sufficient -- whether the two are
*simultaneous* still needs reading the dataset, and whether they are *aligned*
still needs `verify_gaze_sync.py`.

    python scripts/scan_eyetracking_datasets.py --scope corpus
    python scripts/scan_eyetracking_datasets.py --scope all --out results/eyetracking_scan.json
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from deepmreye.datasource import resolve  # noqa: E402
from deepmreye.pipeline import BUCKET_NAME, make_s3_client  # noqa: E402

# Eye-tracking file shapes seen across OpenNeuro. `recording-eye*_physio` is the
# BIDS form; `.asc` is a raw EyeLink export, which several datasets ship under
# `sourcedata/` instead (ds006642 does, and it is the best fold we have).
#
# **`.edf` has to be in here even though nothing can read it yet.** It is
# EyeLink's proprietary binary, so it needs `edf2asc` or a reader like
# `eyelinkio`/`pyedfread` before the ingest can touch it -- but leaving it out
# of the *scan* silently loses whole datasets, which is exactly what happened:
# an earlier version of this pattern missed ds001840 (24 participants) and
# ds004283, both of which ship real simultaneous gaze as `.edf`. Find them
# first, decide whether to convert them second. `format` in the output says
# which are blocked on a converter.
#
# Matching on the extension rather than on "does the key contain 'eye'" is what
# keeps ds004529 out: it has 204 keys mentioning Eyelink, all of them
# `s001-fp_no_Eyelink.log` -- stimulus logs for the condition *without* the
# tracker. A substring scan reports it as 34 paired participants. It has none.
ET_RX = re.compile(
    r"(_recording-eye[A-Za-z0-9-]*_physio\.(tsv(\.gz)?|asc(\.gz)?|edf)$"
    r"|_eyetrack\w*\.(tsv(\.gz)?|asc(\.gz)?|edf)$"
    r"|_eyegaze\w*\.(tsv(\.gz)?|asc(\.gz)?|edf)$"
    r"|eyelinkraw\.(asc(\.gz)?|edf)$"
    r"|_eyelink\w*\.(asc(\.gz)?|edf)$)", re.IGNORECASE)
EXT_RX = re.compile(r"\.(tsv\.gz|tsv|asc\.gz|asc|edf)$", re.IGNORECASE)
BOLD_RX = re.compile(r"_bold\.nii\.gz$")
SUB_RX = re.compile(r"(sub-[A-Za-z0-9]+)")
TASK_RX = re.compile(r"task-([A-Za-z0-9]+)")

# Already resolved, so they are reported as such rather than as fresh leads.
KNOWN = {
    "ds006833": "ingested as dsL07_deepmreye_calib",
    "ds000113": "ingested as dsL08_studyforrest_movie",
    "ds006642": "ingested as dsL11_backtothefuture (4 of 39 extracted)",
    "ds004158": "ingested as dsL12_rest",
    "ds001242": "RETIRED -- per-subject trigger jitter; kept unlabeled",
    "ds007532": "REJECTED -- per-run trigger jitter; kept unlabeled",
    "ds005166": "excluded -- eye tracking recorded in /beh/, not in the scanner",
    "ds004926": "excluded -- spinal cord FOV, 1D pupil only",
    "ds001107": "excluded -- 100% of it is inside ds000113",
    "ds001473": "excluded -- 100% of it is inside ds000113",
}


def list_accessions(s3):
    pg = s3.get_paginator("list_objects_v2")
    out = []
    for page in pg.paginate(Bucket=BUCKET_NAME, Delimiter="/"):
        for p in page.get("CommonPrefixes", []):
            name = p["Prefix"].rstrip("/")
            if re.fullmatch(r"ds\d{6}", name):
                out.append(name)
    return sorted(out)


def scan_one(ds):
    """``(dataset, record)`` -- eye-tracking and BOLD subjects for one accession."""
    s3 = make_s3_client()
    pg = s3.get_paginator("list_objects_v2")
    et_subs, bold_subs, tasks, examples = set(), set(), set(), []
    fmts, where = set(), set()
    n_et = 0
    try:
        for page in pg.paginate(Bucket=BUCKET_NAME, Prefix=f"{ds}/"):
            for o in page.get("Contents", []):
                k = o["Key"]
                if ET_RX.search(k):
                    n_et += 1
                    e = EXT_RX.search(k)
                    if e:
                        fmts.add(e.group(1).lower())
                    # Where the recording sits is the difference between
                    # "simultaneous with the scan" and "a separate behavioural
                    # session". ds005166 was excluded for exactly this: its
                    # antisaccade gaze is under /beh/ while the BOLD is a
                    # flanker task under /func/.
                    where.add("func" if "/func/" in k
                              else "beh" if "/beh/" in k
                              else "sourcedata" if "/sourcedata/" in k
                              else "other")
                    m = SUB_RX.search(k)
                    if m:
                        et_subs.add(m.group(1))
                    t = TASK_RX.search(k)
                    if t:
                        tasks.add(t.group(1))
                    if len(examples) < 3:
                        examples.append(k)
                elif BOLD_RX.search(k):
                    m = SUB_RX.search(k)
                    if m:
                        bold_subs.add(m.group(1))
    except Exception as e:                      # noqa: BLE001
        return ds, {"error": str(e)[:120]}
    paired = sorted(et_subs & bold_subs)
    # Listable is not readable. ds008507 enumerates 418 gaze recordings and
    # returns AccessDenied on every one of them -- embargoed, or staged but not
    # released. Without this probe it reads as the most promising lead in the
    # survey. One HEAD against a real recording settles it.
    readable = None
    if examples:
        try:
            s3.head_object(Bucket=BUCKET_NAME, Key=examples[0])
            readable = True
        except Exception:                       # noqa: BLE001
            readable = False
    return ds, {"n_et_files": n_et, "n_et_sub": len(et_subs), "readable": readable,
                "n_bold_sub": len(bold_subs), "n_paired_sub": len(paired),
                "paired": paired[:8], "tasks": sorted(tasks)[:6],
                "format": sorted(fmts), "where": sorted(where),
                "examples": examples}


def content_overlap(datasets):
    """Which of these accessions are re-releases of each other.

    OpenNeuro carries the same acquisition under several accessions, and this is
    not a curiosity -- ingesting two of them puts the *same participants* on both
    sides of a leave-one-dataset-out split, which silently inflates every
    cross-dataset number in the project. ds001107 was caught this way; ds001473
    was not, and is 100% contained in ds000113.

    Names and paths do not detect it (ds001107 re-lays out its tree, so it
    shares no relative path with ds000113 while sharing every byte). S3 hands
    back a size and an ETag per object for free, which for a non-multipart
    upload is the MD5 -- so a set intersection over (size, etag) finds identical
    content wherever it has been moved to.
    """
    sig = {}
    for ds in datasets:
        s3 = make_s3_client()
        pg = s3.get_paginator("list_objects_v2")
        blobs = set()
        for page in pg.paginate(Bucket=BUCKET_NAME, Prefix=f"{ds}/"):
            for o in page.get("Contents", []):
                if o["Key"].endswith(("_bold.nii.gz", "physio.tsv.gz")):
                    blobs.add((o["Size"], o["ETag"]))
        sig[ds] = blobs

    pairs = []
    names = sorted(sig)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = sig[a] & sig[b]
            if not shared:
                continue
            smaller = min(len(sig[a]), len(sig[b])) or 1
            pairs.append((a, b, len(shared), len(shared) / smaller))
    return sig, sorted(pairs, key=lambda t: -t[3])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scope", choices=("corpus", "all"), default="corpus",
                   help="corpus: only accessions already extracted here (their "
                        "eye blocks exist, so labels are nearly free). "
                        "all: every OpenNeuro accession.")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default="results/eyetracking_scan.json")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-dedupe", action="store_true",
                   help="Skip the re-release check over the hits. It costs one "
                        "extra listing each and is the only thing that catches "
                        "the same acquisition published under two accessions.")
    a = p.parse_args()

    s3 = make_s3_client()
    if a.scope == "corpus":
        data_dir = Path(a.data_dir or resolve(None, download=False, quiet=True))
        targets = sorted({d.name for d in data_dir.iterdir()
                          if d.is_dir() and re.fullmatch(r"ds\d{6}", d.name)})
        print(f"[*] corpus {data_dir}: {len(targets)} extracted accessions")
    else:
        targets = list_accessions(s3)
        print(f"[*] OpenNeuro: {len(targets)} accessions")
    if a.limit:
        targets = targets[:a.limit]

    rows = {}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (ds, rec) in enumerate(ex.map(scan_one, targets), 1):
            rows[ds] = rec
            if i % 50 == 0:
                print(f"    {i}/{len(targets)} scanned", flush=True)

    hits = {d: r for d, r in rows.items() if r.get("n_paired_sub", 0) > 0}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))

    print(f"\n[+] {len(hits)} of {len(rows)} accessions pair eye tracking with BOLD"
          f"   ({sum(r['n_paired_sub'] for r in hits.values())} participants)\n")
    print(f"{'dataset':<11}{'paired':>7}  {'format':<12}{'where':<18}"
          f"{'tasks':<30}status")
    for ds, r in sorted(hits.items(), key=lambda kv: -kv[1]["n_paired_sub"]):
        status = KNOWN.get(ds, "NEW -- not examined")
        if r.get("readable") is False:
            status = "NOT PUBLIC -- listable, every object AccessDenied"
        print(f"{ds:<11}{r['n_paired_sub']:7d}  {','.join(r['format'])[:11]:<12}"
              f"{','.join(r['where'])[:17]:<18}{','.join(r['tasks'])[:28]:<30}{status}")
    blocked = [d for d, r in hits.items()
               if d not in KNOWN and r.get("readable") is not False
               and set(r["format"]) == {"edf"}]
    if blocked:
        print(f"\n[*] blocked on an EDF reader (edf2asc / eyelinkio / pyedfread): "
              + ", ".join(sorted(blocked)))
    denied = [d for d, r in hits.items() if r.get("readable") is False]
    if denied:
        print("[*] listable but not downloadable, so not a lead: "
              + ", ".join(sorted(denied)))
    beh_only = [d for d, r in hits.items()
                if d not in KNOWN and r.get("readable") is not False
                and "func" not in r["where"]]
    if beh_only:
        print(f"[*] gaze is not under /func/, so check it is simultaneous with "
              f"the scan before ingesting: " + ", ".join(sorted(beh_only)))
    errs = {d: r["error"] for d, r in rows.items() if "error" in r}
    if errs:
        print(f"\n[!] {len(errs)} accessions failed to list: "
              + ", ".join(list(errs)[:5]))
    if not a.no_dedupe and len(hits) > 1:
        print("\n[*] checking the hits for re-releases of one another...",
              flush=True)
        _, pairs = content_overlap(sorted(hits))
        dupes = [t for t in pairs if t[3] >= 0.05]
        if dupes:
            print(f"{'pair':<26}{'identical blobs':>16}  share of the smaller")
            for x, y, n, frac in dupes:
                mark = "  <-- INGEST ONLY ONE" if frac >= 0.5 else ""
                print(f"{x + ' / ' + y:<26}{n:16d}  {frac:>6.0%}{mark}")
            for x, y, n, frac in dupes:
                rows.setdefault(x, {}).setdefault("overlaps", []).append(
                    {"with": y, "identical_blobs": n, "share": round(frac, 3)})
                rows.setdefault(y, {}).setdefault("overlaps", []).append(
                    {"with": x, "identical_blobs": n, "share": round(frac, 3)})
            out.write_text(json.dumps(rows, indent=1))
        else:
            print("    none share content.")

    print(f"\n[+] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
