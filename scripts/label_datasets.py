#!/usr/bin/env python3
import os
import h5py
from flask import Flask, render_template_string, request, redirect, url_for, Response
import argparse
from pathlib import Path
import sys

# Add deepmreye to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import deepmreye.config as cfg
from deepmreye.labels import append_label_events

app = Flask(__name__)
config = cfg.DeepMReyeConfig()

# Fallback H5 PATH based on config, overridden by args later
H5_PATH = str(Path(config.data_dir).resolve() / "datasets.h5")
# Set by run_labeler; when true the UI never reaches for HuggingFace.
NO_DOWNLOAD = False

# Optional triage model (scripts/train_qa_classifier.py --out). When present the
# UI pre-selects its prediction so labeling becomes confirm-and-next instead of
# read-and-choose. The confidence is always shown: a preselected wrong answer is
# easy to rubber-stamp, and seeing "confidence 55%" is what stops that.
_MODEL = None


def load_model(path=None):
    """Load the triage model if one has been trained. Never fatal."""
    global _MODEL
    if path is None:
        path = Path(H5_PATH).parent / "qa_model.joblib"
    try:
        import joblib
        _MODEL = joblib.load(path)
        print(f"Loaded triage model from {path} - predictions will be pre-selected.")
    except Exception:
        _MODEL = None
        print("No triage model found; labeling without pre-selection. "
              "Train one with scripts/train_qa_classifier.py --out qa_model.joblib")
    return _MODEL


def predict_for(dataset, subjects):
    """Return ({subject: label}, {subject: confidence}) for a dataset's subjects."""
    if _MODEL is None:
        return {}, {}
    import numpy as np
    from deepmreye.qa_classifier import features_from_file
    from deepmreye.storage import subject_path

    data_dir = Path(H5_PATH).parent
    preds, confs = {}, {}
    for sub in subjects:
        feats = features_from_file(subject_path(data_dir, dataset, sub))
        if feats is None:
            continue
        try:
            proba = _MODEL["model"].predict_proba(feats.reshape(1, -1))[0]
            idx = int(np.argmax(proba))
            preds[sub] = int(_MODEL["model"].classes_[idx])
            confs[sub] = float(proba[idx])
        except Exception:
            continue
    return preds, confs

def get_status():
    if not os.path.exists(H5_PATH):
        return [], 0, 0
    with h5py.File(H5_PATH, 'r') as f:
        datasets = list(f.keys())
        total = len(datasets)
        unlabeled = []
        skipped_count = 0
        for ds in datasets:
            if f[ds].attrs.get('approved', 0) == -99:
                skipped_count += 1
                continue
            
            subs_with_reports = [s for s in f[ds].keys() if 'report_html_path' in f[ds][s].attrs]
            if not subs_with_reports:
                skipped_count += 1
                continue
                
            if any(f[ds][sub].attrs.get('approved', -1) == -1 for sub in subs_with_reports):
                unlabeled.append(ds)
                
        labeled = total - len(unlabeled) - skipped_count
        return unlabeled, labeled, total

@app.route('/')
def index():
    unlabeled, labeled_count, total_count = get_status()
    
    if total_count == 0:
        return "<h1>No datasets found in datasets.h5. Run compile_openneuro.py first!</h1>"
        
    if len(unlabeled) == 0:
        return f"<h1>All {total_count} datasets labeled! 🎉 You can safely close this window.</h1>"
    
    current_ds = unlabeled[0]
    
    # Extract subjects and their reports
    subjects = []
    desc = "{}"
    with h5py.File(H5_PATH, 'r') as f:
        grp = f[current_ds]
        desc = grp.attrs.get('dataset_description', '{}')
        for sub in grp.keys():
            if 'report_html_path' in grp[sub].attrs:
                subjects.append(sub)
                
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Label Datasets</title>
        <style>
            body { font-family: sans-serif; margin: 0; padding: 0; display: flex; flex-direction: column; height: 100vh; background: #222; margin-top: 10px; color: #eee; }
            #header { padding: 10px 20px; background: #333; display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #555; }
            .btn-yes { background-color: #28a745; border:none; border-radius:4px; font-weight:bold; color:white; padding: 10px 15px; cursor: pointer;}
            .btn-yes:hover { background-color: #218838; }
            .info { font-size: 14px; color: #bbb; margin-top: 5px; }
            h2 { margin: 0; }
            .reports-container { display: flex; flex-direction: row; flex-grow: 1; overflow: hidden; }
            .report-box { flex: 1; display: flex; flex-direction: column; border-right: 2px solid #555; }
            .report-title { background: #444; padding: 5px; text-align: center; font-weight: bold; }
            .label-controls { background: #333; padding: 15px; display: flex; flex-direction: column; gap: 10px; border-top: 2px solid #555;}
            .radio-group { display: flex; flex-direction: column; gap: 8px; font-size: 15px;}
            .pred { font-size: 13px; color: #8ab4f8; margin-top: 6px; }
            .lowconf { color: #f0ad4e; font-weight: bold; }
            /* Iframe scaling wrapper */
            .iframe-wrapper { position: relative; flex-grow: 1; overflow: hidden; }
            iframe { 
                position: absolute; 
                top: 0; left: 0; 
                width: 200%; 
                height: 200%; 
                transform: scale(0.5); 
                transform-origin: 0 0; 
                border: none; 
                background: #fff; 
            }
        </style>
    </head>
    <body>
        <form method="POST" action="{{ url_for('submit_label') }}" style="display: flex; flex-direction: column; height: 100%;">
            <div id="header">
                <div>
                    <h2>Labeling: {{ dataset_name }}</h2>
                    <div class="info">Progress: {{ labeled_count }} / {{ total_count }} labeled</div>
                </div>
                <div id="buttons" style="display:flex; gap:10px;">
                    <input type="hidden" name="dataset" value="{{ dataset_name }}">
                    <button type="submit" name="action" value="save" class="btn-yes">Save Subject Labels & Next</button>
                    <button type="submit" name="action" value="skip" style="background:#555; border:none; border-radius:4px; font-weight:bold; color:white; padding: 10px 15px; cursor: pointer;">Skip Entire Dataset</button>
                </div>
            </div>
            
            <div class="reports-container">
                {% for sub in subjects %}
                <div class="report-box">
                    <div class="report-title">Subject: {{ sub }}</div>
                    <div class="iframe-wrapper">
                        <iframe src="{{ url_for('serve_report', dataset=dataset_name, subject=sub) }}"></iframe>
                    </div>
                    <div class="label-controls">
                        <strong>Rate {{ sub }}:</strong>
                        <div class="radio-group">
                            <label><input type="radio" name="label_{{ sub }}" value="1" required
                                {% if preds.get(sub) == 1 %}checked{% endif %}> Eyes visible</label>
                            <label><input type="radio" name="label_{{ sub }}" value="3"
                                {% if preds.get(sub) == 3 %}checked{% endif %}> Eyes visible but cut off</label>
                            <label><input type="radio" name="label_{{ sub }}" value="0"
                                {% if preds.get(sub) == 0 %}checked{% endif %}> No eyes - Bad transform</label>
                            <label><input type="radio" name="label_{{ sub }}" value="2"
                                {% if preds.get(sub) == 2 %}checked{% endif %}> No eyes - Good transform</label>
                        </div>
                        {% if sub in preds %}
                        <div class="pred">predicted by model &middot; confidence {{ "%.0f"|format(confs[sub] * 100) }}%
                            {% if confs[sub] < 0.7 %}<span class="lowconf">&mdash; unsure, please check</span>{% endif %}
                        </div>
                        {% endif %}
                    </div>
                </div>
                {% endfor %}
                {% if not subjects %}
                <div style="padding: 20px;">No HTML reports found for this dataset. You may reject or skip.</div>
                {% endif %}
            </div>
        </form>
    </body>
    </html>
    """
    
    preds, confs = predict_for(current_ds, subjects)

    return render_template_string(
        html_template,
        dataset_name=current_ds,
        subjects=subjects,
        labeled_count=labeled_count,
        total_count=total_count,
        preds=preds,
        confs=confs,
    )

def _find_report(dataset, subject):
    """Locate a subject's QA report on this machine.

    The registry stores the absolute path from wherever extraction ran, which
    is a cluster path that does not exist on a laptop. So the local layout
    (<data_dir>/<dataset>/<subject>/report_*.html) is authoritative and the
    recorded path is only a fallback -- that keeps one registry usable from
    both machines.
    """
    data_dir = Path(H5_PATH).parent
    local = sorted((data_dir / dataset / subject).glob("*.html"))
    if local:
        return local[0]

    try:
        with h5py.File(H5_PATH, "r") as f:
            recorded = f[dataset][subject].attrs.get("report_html_path", "")
        if recorded and os.path.exists(recorded):
            return Path(recorded)
    except Exception:
        pass
    return None


@app.route('/reports/<dataset>/<subject>')
def serve_report(dataset, subject):
    path = _find_report(dataset, subject)

    if path is None and not NO_DOWNLOAD:
        # Reports total more than the eye blocks do, so they are pulled per
        # dataset as labeling reaches them rather than all up front.
        try:
            from deepmreye.datasource import ensure_reports
            ensure_reports(Path(H5_PATH).parent, [dataset], quiet=True)
            path = _find_report(dataset, subject)
        except Exception as e:
            return f"Report not available locally and download failed: {e}", 404

    if path is None:
        return "Report not found on disk.", 404

    try:
        return Response(path.read_text(errors="replace"), mimetype="text/html")
    except Exception as e:
        return str(e), 404

@app.route('/submit', methods=['POST'])
def submit_label():
    dataset_name = request.form.get('dataset')
    action = request.form.get('action')
    
    if dataset_name:
        events = []  # (dataset, scope, subject, label) mirrored to the CSV backup
        try:
            with h5py.File(H5_PATH, 'a') as f:
                if dataset_name in f:
                    if action == 'skip':
                        f[dataset_name].attrs['approved'] = -99
                        events.append((dataset_name, 'dataset', '', -99))
                    else:
                        for sub in f[dataset_name].keys():
                            if 'report_html_path' in f[dataset_name][sub].attrs:
                                lbl = request.form.get(f'label_{sub}')
                                if lbl is not None:
                                    f[dataset_name][sub].attrs['approved'] = int(lbl)
                                    events.append((dataset_name, 'subject', sub, int(lbl)))
        except Exception as e:
            print(f"Failed to update HDF5: {e}")

        if events:
            try:
                append_label_events(Path(H5_PATH).parent / "labels.csv", events)
            except Exception as e:
                print(f"Failed to write label backup CSV: {e}")

    return redirect(url_for('index'))

def run_labeler(h5_path=None, data_dir=None, port=5050, no_download=False):
    """Launch the Flask labeling UI against the given registry."""
    global H5_PATH, NO_DOWNLOAD
    NO_DOWNLOAD = no_download
    if h5_path is not None:
        H5_PATH = str(h5_path)
    elif data_dir is not None:
        H5_PATH = str(Path(data_dir).resolve() / "datasets.h5")

    load_model()
    print("Starting Flask server for dataset labeling...")
    print(f"Target HDF5: {H5_PATH}")
    print(f"Open http://127.0.0.1:{port} in your browser to begin.")
    app.run(host='0.0.0.0', port=port, threaded=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dataset Labeling UI")
    parser.add_argument("--h5", type=str, default=None, help="Path to HDF5 Datastore")
    parser.add_argument("--data-dir", type=str, default=None, help="Central data directory (uses <data-dir>/datasets.h5)")
    parser.add_argument("--port", type=int, default=5050, help="Port to run the app on")
    args = parser.parse_args()

    run_labeler(h5_path=args.h5, data_dir=args.data_dir, port=args.port)
