#!/bin/bash
# Size and submit the extraction array from the current manifest.
#
# The array size has to match the manifest, and the manifest changes every time
# you stage more work, so computing it by hand is a good way to silently drop
# subjects off the end. This reads the manifest and submits accordingly.
#
#   ./slurm/submit_extraction.sh [max_parallel_tasks]

set -euo pipefail

# Derive the repo root from this script so a clone anywhere works.
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
STAGING_DIR="${STAGING_DIR:-/leonardo_scratch/fast/AIFAC_S07_154/mfrey/dme/staging}"
MANIFEST="${MANIFEST:-$STAGING_DIR/manifest.jsonl}"
MAX_PARALLEL="${1:-64}"

cd "$REPO_DIR"
mkdir -p logs

if [[ ! -f "$MANIFEST" ]]; then
    echo "No manifest at $MANIFEST. Run slurm/stage_downloads.py first." >&2
    exit 1
fi

N=$(grep -c . "$MANIFEST")
if [[ "$N" -eq 0 ]]; then
    echo "Manifest is empty -- nothing to extract." >&2
    exit 0
fi

# One task per ~40 subjects keeps each well inside the 4 h wall clock at the
# measured ~1 min/subject, while leaving headroom for slower long runs.
TASKS=$(( (N + 39) / 40 ))
[[ "$TASKS" -gt "$MAX_PARALLEL" ]] && TASKS=$MAX_PARALLEL
[[ "$TASKS" -lt 1 ]] && TASKS=1

echo "$N subjects to extract across $TASKS array tasks (~$(( (N + TASKS - 1) / TASKS )) each)"

# The controller has been intermittently unreachable here: sinfo/sbatch hang
# rather than erroring. Fail fast with a clear message instead of blocking.
if ! timeout 30 sinfo -h -o "%P" >/dev/null 2>&1; then
    echo "ERROR: the Slurm controller is not responding (sinfo timed out)." >&2
    echo "       This has been intermittent on Leonardo; retry later." >&2
    exit 1
fi

# Site-specific and validated up front: a bad partition or missing account
# fails at submit with an unhelpful message.
#
# The compute budget is AIFAC_S07_154 — the same allocation as the scratch
# space, NOT the EUHPC_D21_101 that appears in the repo's /leonardo_work path.
# Submitting under the latter fails with "invalid account or expired budget".
ACCOUNT="${SLURM_ACCOUNT:-AIFAC_S07_154}"
PARTITION="${SLURM_PARTITION:-}"

# A partition existing is not the same as this account being allowed to use it:
# AIFAC_S07_154 has no allocation on dcgp_usr_prod, and sbatch reports that as
# "invalid account or expired budget", which sends you chasing the wrong thing.
# So probe with --test-only, which actually exercises the account+partition
# pair, rather than just checking the partition exists.
if [[ -z "$PARTITION" ]]; then
    for p in boost_usr_prod dcgp_usr_prod lrd_all_serial; do
        if sinfo -h -p "$p" -o "%P" 2>/dev/null | grep -q . && \
           timeout 45 sbatch --test-only -A "$ACCOUNT" -p "$p" --time=00:10:00 \
               --nodes=1 --ntasks=1 --wrap="true" 2>&1 | grep -q "^sbatch: Job"; then
            PARTITION="$p"
            break
        fi
        echo "  $p: not usable by account $ACCOUNT, trying next" >&2
    done
fi

if [[ -z "$PARTITION" ]]; then
    echo "ERROR: could not determine a usable partition. Available:" >&2
    sinfo -h -o "  %P" | sort -u >&2
    echo "Set SLURM_PARTITION=<name> and rerun." >&2
    exit 1
fi

echo "submitting to partition '$PARTITION' under account '$ACCOUNT'"

sbatch --account="$ACCOUNT" \
       --partition="$PARTITION" \
       --array="0-$((TASKS - 1))%${MAX_PARALLEL}" \
       --export=ALL,MANIFEST="$MANIFEST" \
       slurm/extract_array.sbatch
