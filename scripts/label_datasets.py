import argparse
import io
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np
from flask import (Flask, Response, jsonify, redirect, render_template_string, request,
                   send_file, url_for)

# Add deepmreye to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye import thumbnail
from deepmreye.labels import append_label_events
from deepmreye.pipeline import thumbnail_path

app = Flask(__name__)


# Fallback H5 PATH based on config, overridden by args later
H5_PATH = str(Path("./data").resolve() / "datasets.h5")
# Set by run_labeler; when true the UI never reaches for HuggingFace.
NO_DOWNLOAD = False


def open_h5_file(path, mode="r"):
    """Open HDF5 file safely without file locking issues."""
    try:
        return h5py.File(path, mode, locking=False)
    except (TypeError, ValueError):
        return h5py.File(path, mode)

# Optional triage model (scripts/train_qa_classifier.py --out). When present the
# UI pre-selects its prediction so labeling becomes confirm-and-next instead of
# read-and-choose. The confidence is always shown.
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
        print("No triage model found; labeling without pre-selection.")
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
    with open_h5_file(H5_PATH, "r") as f:
        ds_grp = f.get(dataset)
        for sub in subjects:
            sub_path = subject_path(data_dir, dataset, sub)
            if not sub_path.exists():
                continue
            report_path = ds_grp[sub].attrs.get("report_html_path", "") if (ds_grp and sub in ds_grp) else ""
            tr = ds_grp[sub].attrs.get("repetition_time", None) if (ds_grp and sub in ds_grp) else None
            ntr = ds_grp[sub].attrs.get("n_trs", None) if (ds_grp and sub in ds_grp) else None
            feats = features_from_file(sub_path, report_path=report_path, tr=tr, n_trs=ntr)
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


def get_dataset_info():
    """Retrieve details for all datasets in datasets.h5 for navigation and statistics."""
    if not os.path.exists(H5_PATH):
        return [], {}, {"total": 0, "labeled": 0, "unlabeled": 0, "skipped": 0}
        
    datasets = []
    details = {}
    counts = {"total": 0, "labeled": 0, "unlabeled": 0, "skipped": 0}

    with open_h5_file(H5_PATH, 'r') as f:
        datasets = list(f.keys())
        counts["total"] = len(datasets)

        for ds in datasets:
            grp = f[ds]
            approved_attr = grp.attrs.get('approved', -1)
            is_skipped = (approved_attr == -99)
            
            raw_desc = grp.attrs.get('dataset_description', '{}')
            ds_title = ""
            try:
                desc_json = json.loads(raw_desc)
                ds_title = desc_json.get('Name', '')
            except Exception:
                ds_title = ""

            subs_with_reports = _reviewable_subjects(grp, ds)

            sub_labels = {}
            sub_confs = {}
            sub_manual = {}
            sub_meta = {}
            for sub in subs_with_reports:
                s_grp = grp[sub]
                sub_labels[sub] = s_grp.attrs.get('approved', -1)
                sub_confs[sub] = float(s_grp.attrs.get('pred_conf', 1.0))
                sub_manual[sub] = bool(s_grp.attrs.get('is_manual', False))
                tr = s_grp.attrs.get('repetition_time', None)
                ntr = s_grp.attrs.get('n_trs', None)
                sub_meta[sub] = {
                    "tr": float(tr) if tr is not None else None,
                    "ntr": int(ntr) if ntr is not None else None
                }

            min_conf = min(sub_confs.values()) if sub_confs else 1.0
            mean_conf = (sum(sub_confs.values()) / len(sub_confs)) if sub_confs else 1.0
            all_manual = all(sub_manual.values()) if sub_manual else False

            if is_skipped or not subs_with_reports:
                status = "skipped"
                counts["skipped"] += 1
            elif any(lbl == -1 for lbl in sub_labels.values()):
                status = "unlabeled"
                counts["unlabeled"] += 1
            else:
                status = "labeled"
                counts["labeled"] += 1

            details[ds] = {
                "title": ds_title,
                "status": status,
                "approved_attr": approved_attr,
                "subjects": subs_with_reports,
                "sub_labels": sub_labels,
                "sub_confs": sub_confs,
                "sub_manual": sub_manual,
                "sub_meta": sub_meta,
                "min_conf": min_conf,
                "mean_conf": mean_conf,
                "all_manual": all_manual,
                "total_subs": len(subs_with_reports),
                "labeled_subs": sum(1 for lbl in sub_labels.values() if lbl != -1)
            }

    return datasets, details, counts


@app.route('/')
def index():
    all_datasets, details, counts = get_dataset_info()

    if counts["total"] == 0:
        return "<h1>No datasets found in datasets.h5. Run compile_openneuro.py first!</h1>"

    filter_mode = request.args.get('filter', 'all')
    sort_mode = request.args.get('sort', 'conf_asc')
    target_ds = request.args.get('ds')

    # Exclude datasets without HTML reports / skipped datasets from active list unless in skipped mode
    if filter_mode == 'unlabeled':
        active_datasets = [ds for ds in all_datasets if details[ds]["status"] == "unlabeled"]
    elif filter_mode == 'labeled':
        active_datasets = [ds for ds in all_datasets if details[ds]["status"] == "labeled"]
    elif filter_mode == 'skipped':
        active_datasets = [ds for ds in all_datasets if details[ds]["status"] in ("skipped", "no_reports")]
    else:  # 'all'
        active_datasets = [ds for ds in all_datasets if details[ds]["status"] != "skipped"]

    if not active_datasets:
        active_datasets = [ds for ds in all_datasets if details[ds]["status"] != "skipped"] or all_datasets

    # Sort active datasets based on selected sort mode
    if sort_mode == 'conf_asc':
        # Lowest confidence first (Least sure datasets at the top of the queue!)
        active_datasets.sort(key=lambda d: details[d]["min_conf"])
    elif sort_mode == 'conf_desc':
        # Highest confidence first
        active_datasets.sort(key=lambda d: details[d]["min_conf"], reverse=True)
    elif sort_mode == 'name':
        active_datasets.sort()

    current_idx = 0
    if target_ds and target_ds in active_datasets:
        current_idx = active_datasets.index(target_ds)
    elif target_ds and target_ds in all_datasets:
        filter_mode = 'all'
        active_datasets = [ds for ds in all_datasets if details[ds]["status"] != "skipped"]
        if sort_mode == 'conf_asc':
            active_datasets.sort(key=lambda d: details[d]["min_conf"])
        elif sort_mode == 'conf_desc':
            active_datasets.sort(key=lambda d: details[d]["min_conf"], reverse=True)
        elif sort_mode == 'name':
            active_datasets.sort()
        current_idx = active_datasets.index(target_ds) if target_ds in active_datasets else 0
    else:
        current_idx = 0

    current_ds = active_datasets[current_idx]
    prev_ds = active_datasets[current_idx - 1] if current_idx > 0 else active_datasets[-1]
    next_ds = active_datasets[current_idx + 1] if current_idx < len(active_datasets) - 1 else active_datasets[0]

    ds_info = details[current_ds]
    subjects = ds_info["subjects"]

    preds, confs = predict_for(current_ds, subjects)
    # Merge stored confidences from datasets.h5 if not dynamically computed
    for sub in subjects:
        if sub not in confs and sub in ds_info["sub_confs"]:
            confs[sub] = ds_info["sub_confs"][sub]

    msg = request.args.get('msg', '')
    msg_type = request.args.get('msg_type', 'info')

    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Label Datasets - {{ dataset_name }} ({{ current_idx + 1 }}/{{ active_datasets|length }})</title>
        <style>
            * { box-sizing: border-box; }
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 0; display: flex; flex-direction: column; height: 100vh; background: #121214; color: #f4f4f5; }
            
            #header { padding: 10px 20px; background: #1c1c1f; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #27272a; flex-wrap: wrap; gap: 10px; }
            .header-left { display: flex; align-items: center; gap: 10px; }
            .header-center { display: flex; align-items: center; gap: 10px; }
            .header-right { display: flex; align-items: center; gap: 10px; }

            #sub-bar { padding: 6px 20px; background: #18181b; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #27272a; font-size: 13px; color: #a1a1aa; }
            .ds-title { font-weight: 500; color: #e4e4e7; text-overflow: ellipsis; overflow: hidden; white-space: nowrap; max-width: 450px; }

            .view-mode-group { display: inline-flex; background: #27272a; border-radius: 6px; padding: 2px; gap: 2px; }
            .view-btn { background: transparent; border: none; color: #a1a1aa; padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: 500; cursor: pointer; transition: all 0.15s ease; }
            .view-btn.active, .view-btn:hover { background: #3f3f46; color: #fff; }

            h2 { margin: 0; font-size: 16px; color: #fff; font-weight: 600; display: flex; align-items: center; gap: 6px; }
            .ds-select { background: #27272a; color: #fff; border: 1px solid #3f3f46; border-radius: 6px; padding: 6px 10px; font-size: 13px; outline: none; cursor: pointer; max-width: 250px; }
            .filter-select { background: #27272a; color: #fff; border: 1px solid #3f3f46; border-radius: 6px; padding: 6px 10px; font-size: 13px; font-weight: 500; outline: none; cursor: pointer; }
            .sort-select { background: #0284c7; color: #fff; border: none; border-radius: 6px; padding: 6px 10px; font-size: 13px; font-weight: 600; outline: none; cursor: pointer; }

            .nav-btn { background: #27272a; color: #f4f4f5; border: 1px solid #3f3f46; border-radius: 6px; padding: 6px 12px; font-weight: 500; text-decoration: none; font-size: 13px; cursor: pointer; transition: all 0.15s ease; display: inline-flex; align-items: center; gap: 4px; }
            .nav-btn:hover { background: #3f3f46; color: #fff; }
            
            .btn-save { background-color: #16a34a; border: none; border-radius: 6px; font-weight: 600; color: white; padding: 8px 14px; cursor: pointer; font-size: 14px; transition: background 0.15s ease; }
            .btn-save:hover { background-color: #15803d; }
            .btn-save-prev { background-color: #0284c7; border: none; border-radius: 6px; font-weight: 600; color: white; padding: 8px 14px; cursor: pointer; font-size: 14px; }
            .btn-save-prev:hover { background-color: #0369a1; }
            .btn-secondary { background-color: #3f3f46; border: none; border-radius: 6px; font-weight: 600; color: white; padding: 8px 12px; cursor: pointer; font-size: 13px; }
            .btn-secondary:hover { background-color: #52525b; }
            .btn-skip { background-color: #d97706; border: none; border-radius: 6px; font-weight: 600; color: white; padding: 8px 12px; cursor: pointer; font-size: 13px; }
            .btn-skip:hover { background-color: #b45309; }
            .btn-push { background-color: #2563eb; border: none; border-radius: 6px; font-weight: 600; color: white; padding: 8px 14px; cursor: pointer; font-size: 13px; display: inline-flex; align-items: center; gap: 5px; }
            .btn-push:hover { background-color: #1d4ed8; }

            .badge { padding: 3px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; text-transform: uppercase; }
            .badge-autolabeled { background: #0284c7; color: #fff; }
            .badge-manual { background: #16a34a; color: #fff; }
            .badge-skipped { background: #ef4444; color: #fff; }

            .info { font-size: 13px; color: #a1a1aa; }
            
            .banner { padding: 8px 16px; font-size: 14px; font-weight: 500; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #3f3f46; }
            .banner-success { background: #14532d; color: #bbf7d0; }
            .banner-error { background: #7f1d1d; color: #fecaca; }
            .banner-info { background: #1e3a8a; color: #bfdbfe; }

            .reports-container { display: flex; flex-direction: row; flex-grow: 1; overflow: hidden; background: #000; }
            .reports-container.layout-stacked { flex-direction: column; overflow-y: auto; }

            .report-box { flex: 1; display: flex; flex-direction: column; border-right: 1px solid #27272a; min-width: 0; height: 100%; position: relative; }
            .report-box:last-child { border-right: none; }
            .reports-container.layout-stacked .report-box { border-right: none; border-bottom: 2px solid #27272a; height: auto; min-height: 600px; }

            .report-title-bar { background: #1c1c1f; padding: 8px 14px; display: flex; justify-content: space-between; align-items: center; font-weight: 600; font-size: 14px; border-bottom: 1px solid #27272a; color: #e4e4e7; }
            .meta-pill { font-size: 11px; background: #27272a; color: #a1a1aa; padding: 2px 7px; border-radius: 4px; font-weight: normal; margin-left: 8px; }
            .open-link { color: #38bdf8; font-size: 12px; text-decoration: none; font-weight: 500; display: inline-flex; align-items: center; gap: 3px; }
            .open-link:hover { text-decoration: underline; }

            .label-controls { background: #18181b; padding: 12px 16px; display: flex; flex-direction: column; gap: 8px; border-top: 1px solid #27272a; }
            .radio-group { display: flex; flex-direction: column; gap: 6px; font-size: 14px; }
            .radio-group label { display: flex; align-items: center; gap: 8px; cursor: pointer; padding: 4px 6px; border-radius: 4px; transition: background 0.1s ease; }
            .radio-group label:hover { background: #27272a; }
            
            .status-tag-manual { display: inline-flex; align-items: center; gap: 4px; font-size: 12px; color: #4ade80; background: #064e3b; border: 1px solid #16a34a; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
            .status-tag-auto { display: inline-flex; align-items: center; gap: 4px; font-size: 12px; color: #38bdf8; background: #082f49; border: 1px solid #0284c7; padding: 3px 8px; border-radius: 4px; font-weight: 500; }
            
            .conf-pill { font-size: 12px; font-weight: 600; padding: 3px 8px; border-radius: 4px; display: inline-flex; align-items: center; gap: 4px; }
            .conf-low { background: #7f1d1d; color: #fecaca; border: 1px solid #b91c1c; }
            .conf-mid { background: #713f12; color: #fef08a; border: 1px solid #a16207; }
            .conf-high { background: #14532d; color: #bbf7d0; border: 1px solid #15803d; }

            .iframe-wrapper { position: relative; flex-grow: 1; overflow: hidden; background: #18181b; }
            iframe { position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none; background: #18181b; }

            .shortcuts-help { font-size: 12px; color: #a1a1aa; padding: 6px 12px; background: #121214; border-top: 1px solid #27272a; display: flex; justify-content: space-between; align-items: center; }
            kbd { background: #27272a; color: #fff; padding: 2px 5px; border-radius: 3px; font-size: 11px; font-family: monospace; }
        </style>
    </head>
    <body>
        {% if msg %}
        <div class="banner banner-{{ msg_type }}">
            <span>{{ msg }}</span>
            <button onclick="this.parentElement.style.display='none'" style="background:none; border:none; color:inherit; cursor:pointer; font-weight:bold;">&times;</button>
        </div>
        {% endif %}

        <form id="qa-form" method="POST" action="{{ url_for('submit_label') }}" style="display: flex; flex-direction: column; height: 100%;">
            <input type="hidden" name="dataset" value="{{ dataset_name }}">
            <input type="hidden" name="filter" value="{{ filter_mode }}">
            <input type="hidden" name="sort" value="{{ sort_mode }}">
            <input type="hidden" id="target-ds-field" name="target_ds" value="">
            <input type="hidden" id="action-field" name="action" value="save_next">

            <div id="header">
                <div class="header-left">
                    <div style="display: flex; gap: 4px; background: #18181b; padding: 2px; border-radius: 6px; border: 1px solid #27272a; margin-right: 8px;">
                        <a href="/" style="padding: 4px 8px; border-radius: 4px; background: #27272a; color: #fff; text-decoration: none; font-size: 11px; font-weight: 600;">🔍 Detailed QA View</a>
                        <a href="/rapid" style="padding: 4px 8px; border-radius: 4px; color: #a1a1aa; text-decoration: none; font-size: 11px; font-weight: 600;">⚡ Rapid Audit</a>
                    </div>
                    <h2>
                        <span>Dataset {{ current_idx + 1 }}/{{ active_datasets|length }}:</span>
                        <select class="ds-select" onchange="submitAndJump(this.value)">
                            {% for ds_name in active_datasets %}
                                {% set info = details[ds_name] %}
                                <option value="{{ ds_name }}" {% if ds_name == dataset_name %}selected{% endif %}>
                                    ({{ "%.0f"|format(info.min_conf * 100) }}%) {{ ds_name }}
                                </option>
                            {% endfor %}
                        </select>
                    </h2>
                    
                    <select class="sort-select" onchange="changeSort(this.value)" title="Change Queue Sorting Order">
                        <option value="conf_asc" {% if sort_mode == 'conf_asc' %}selected{% endif %}>📶 Confidence: Lowest &rarr; Highest</option>
                        <option value="conf_desc" {% if sort_mode == 'conf_desc' %}selected{% endif %}>📶 Confidence: Highest &rarr; Lowest</option>
                        <option value="name" {% if sort_mode == 'name' %}selected{% endif %}>🔤 Name (A-Z)</option>
                    </select>

                    <select class="filter-select" onchange="changeFilter(this.value)" title="Filter Queue View">
                        <option value="all" {% if filter_mode == 'all' %}selected{% endif %}>🌐 All Active Datasets ({{ counts.total - counts.skipped }})</option>
                        <option value="unverified" {% if filter_mode == 'unverified' %}selected{% endif %}>🤖 Auto-Labeled ({{ counts.unlabeled }})</option>
                        <option value="manual" {% if filter_mode == 'manual' %}selected{% endif %}>✓ Manually Verified ({{ counts.labeled }})</option>
                        <option value="skipped" {% if filter_mode == 'skipped' %}selected{% endif %}>⊘ Skipped / No Reports ({{ counts.skipped }})</option>
                    </select>
                    
                    <span class="badge badge-{{ ds_info.status }}">{{ ds_info.status }}</span>
                </div>

                <div class="header-center">
                    <button type="submit" onclick="setAction('save_prev')" class="nav-btn" title="Save current dataset & Go to Previous Dataset (Left Arrow)">&larr; Prev</button>
                    <span class="info">Min Conf: <strong>{{ "%.1f"|format(ds_info.min_conf * 100) }}%</strong></span>
                    <button type="submit" onclick="setAction('save_next')" class="nav-btn" title="Save current dataset & Go to Next Dataset (Right Arrow)">Next &rarr;</button>
                </div>

                <div class="header-right">
                    <button type="submit" onclick="setAction('save_prev')" class="btn-save-prev" title="Save & Go to Previous Dataset">&larr; Save & Prev</button>
                    <button type="submit" onclick="setAction('save_next')" class="btn-save" title="Save & Go to Next Dataset (Ctrl+Enter)">Save & Next &rarr;</button>
                    
                    {% if ds_info.status == 'skipped' %}
                    <button type="submit" onclick="setAction('unskip')" class="btn-secondary">Unskip</button>
                    {% else %}
                    <button type="submit" onclick="setAction('skip_next')" class="btn-skip" title="Skip Dataset">Skip Dataset</button>
                    {% endif %}
                    
                    <button type="button" onclick="triggerPushHF()" class="btn-push" title="Push updated datasets.h5 and labels.csv to HuggingFace">🚀 Push Labels to HF</button>
                </div>
            </div>

            <div id="sub-bar">
                <div class="ds-title" title="{{ ds_info.title }}">
                    {% if ds_info.title %}<strong>{{ ds_info.title }}</strong>{% else %}OpenNeuro Corpus Dataset{% endif %}
                </div>
                <div style="display: flex; align-items: center; gap: 15px;">
                    <span>View Layout:</span>
                    <div class="view-mode-group">
                        <button type="button" class="view-btn active" onclick="setLayout('split')">↔ Side-by-Side</button>
                        {% for sub in subjects %}
                        <button type="button" class="view-btn" onclick="setLayout('sub-{{ loop.index0 }}')">👁 {{ sub }}</button>
                        {% endfor %}
                        <button type="button" class="view-btn" onclick="setLayout('stacked')">↕ Stacked</button>
                    </div>
                </div>
            </div>
            
            <div id="reports-container" class="reports-container">
                {% for sub in subjects %}
                {% set cur_lbl = ds_info.sub_labels.get(sub, 1) %}
                {% set sub_conf = confs.get(sub, 1.0) %}
                {% set is_manual = ds_info.sub_manual.get(sub, False) %}
                {% set meta = ds_info.sub_meta.get(sub, {}) %}
                <div class="report-box" id="box-{{ loop.index0 }}">
                    <div class="report-title-bar">
                        <div>
                            <span>Subject: {{ sub }}</span>
                            {% if meta.tr %}<span class="meta-pill">TR: {{ meta.tr }}s</span>{% endif %}
                            {% if meta.ntr %}<span class="meta-pill">{{ meta.ntr }} vols</span>{% endif %}
                        </div>
                        <a href="{{ url_for('serve_report', dataset=dataset_name, subject=sub) }}" target="_blank" class="open-link" title="Open interactive HTML report in a new tab">
                            Open Full Report ↗
                        </a>
                    </div>
                    <div class="iframe-wrapper">
                        <iframe src="{{ url_for('serve_report', dataset=dataset_name, subject=sub) }}" loading="lazy"></iframe>
                    </div>
                    <div class="label-controls">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong>Rate {{ sub }}:</strong>
                                {% if is_manual %}
                                <span class="status-tag-manual">✓ Manually Verified</span>
                                {% else %}
                                <span class="status-tag-auto">🤖 Classifier Pre-labeled</span>
                                {% endif %}
                            </div>
                            
                            {% set pct = sub_conf * 100 %}
                            {% if pct < 70 %}
                            <span class="conf-pill conf-low">⚠️ {{ "%.1f"|format(pct) }}% Confidence</span>
                            {% elif pct < 90 %}
                            <span class="conf-pill conf-mid">⚠️ {{ "%.1f"|format(pct) }}% Confidence</span>
                            {% else %}
                            <span class="conf-pill conf-high">✓ {{ "%.1f"|format(pct) }}% Confidence</span>
                            {% endif %}
                        </div>
                        
                        <div class="radio-group" data-subject="{{ sub }}">
                            <label><input type="radio" name="label_{{ sub }}" value="1" required
                                {% if cur_lbl == 1 %}checked{% endif %}> <kbd>1</kbd> Eyes visible</label>
                            <label><input type="radio" name="label_{{ sub }}" value="4"
                                {% if cur_lbl == 4 %}checked{% endif %}> <kbd>4</kbd> Eyes visible but faint</label>
                            <label><input type="radio" name="label_{{ sub }}" value="3"
                                {% if cur_lbl == 3 %}checked{% endif %}> <kbd>3</kbd> Eyes visible but cut off</label>
                            <label><input type="radio" name="label_{{ sub }}" value="0"
                                {% if cur_lbl == 0 %}checked{% endif %}> <kbd>0</kbd> No eyes - Bad transform</label>
                            <label><input type="radio" name="label_{{ sub }}" value="2"
                                {% if cur_lbl == 2 %}checked{% endif %}> <kbd>2</kbd> No eyes - Good transform</label>
                        </div>
                    </div>
                </div>
                {% endfor %}
                {% if not subjects %}
                <div style="padding: 40px; text-align: center; color: #f87171; width: 100%;">
                    <div style="font-size: 16px; font-weight: 600; margin-bottom: 8px;">⚠️ No HTML reports found for this dataset</div>
                    <div style="font-size: 13px; color: #a1a1aa; max-width: 500px; margin: 0 auto 15px auto;">
                        No coregistered eye reports exist for this dataset (e.g. download failed or invalid headers). 
                        Clicking <strong>Save & Next</strong> will automatically mark this dataset as skipped (<code>-99</code>) and remove it from training.
                    </div>
                </div>
                {% endif %}
            </div>
        </form>

        <form id="push-hf-form" method="POST" action="{{ url_for('push_hf') }}" style="display:none;">
            <input type="hidden" name="current_ds" value="{{ dataset_name }}">
        </form>

        <div class="shortcuts-help">
            <span><strong>Shortcuts:</strong> <kbd>1</kbd> Eyes | <kbd>4</kbd> Eyes faint | <kbd>3</kbd> Eyes cut off | <kbd>0</kbd> No eyes (bad) | <kbd>2</kbd> No eyes (good) | <kbd>&larr;</kbd> <kbd>&rarr;</kbd> Save & Prev/Next | <kbd>Ctrl</kbd>+<kbd>Enter</kbd> Save & Next</span>
            <span>Dataset: <code>{{ dataset_name }}</code> &middot; Min Conf: <code>{{ "%.1f"|format(ds_info.min_conf * 100) }}%</code></span>
        </div>

        <script>
            function setAction(actionName) {
                document.getElementById('action-field').value = actionName;
            }

            function submitAndJump(targetDs) {
                document.getElementById('target-ds-field').value = targetDs;
                document.getElementById('action-field').value = 'save_to_ds';
                document.getElementById('qa-form').submit();
            }

            function changeFilter(newFilter) {
                window.location.href = "{{ url_for('index') }}?ds={{ dataset_name }}&sort={{ sort_mode }}&filter=" + newFilter;
            }

            function changeSort(newSort) {
                window.location.href = "{{ url_for('index') }}?ds={{ dataset_name }}&filter={{ filter_mode }}&sort=" + newSort;
            }

            function triggerPushHF() {
                if (confirm("Push current datasets.h5 and labels.csv to HuggingFace (DeepMReye/eyeballs)?")) {
                    document.getElementById('push-hf-form').submit();
                }
            }

            function setLayout(mode) {
                const container = document.getElementById('reports-container');
                const boxes = document.querySelectorAll('.report-box');
                const btns = document.querySelectorAll('.view-btn');

                btns.forEach(b => b.classList.remove('active'));

                if (mode === 'stacked') {
                    container.className = 'reports-container layout-stacked';
                    boxes.forEach(b => b.style.display = 'flex');
                    document.querySelector('.view-btn[onclick*="stacked"]').classList.add('active');
                } else if (mode.startsWith('sub-')) {
                    container.className = 'reports-container';
                    const idx = parseInt(mode.replace('sub-', ''));
                    boxes.forEach((b, i) => {
                        b.style.display = (i === idx) ? 'flex' : 'none';
                    });
                    document.querySelector(`.view-btn[onclick*="${mode}"]`).classList.add('active');
                } else {
                    // Split view
                    container.className = 'reports-container';
                    boxes.forEach(b => b.style.display = 'flex');
                    document.querySelector('.view-btn[onclick*="split"]').classList.add('active');
                }
            }

            document.addEventListener('keydown', function(e) {
                if (e.target.tagName === 'INPUT' && e.target.type === 'text') return;
                
                const radioGroups = document.querySelectorAll('.radio-group');
                if (['1', '4', '3', '0', '2'].includes(e.key)) {
                    radioGroups.forEach(group => {
                        const input = group.querySelector(`input[value="${e.key}"]`);
                        if (input) input.checked = true;
                    });
                }

                if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                    e.preventDefault();
                    setAction('save_next');
                    document.getElementById('qa-form').submit();
                }

                if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'SELECT') {
                    if (e.key === 'ArrowLeft') {
                        e.preventDefault();
                        setAction('save_prev');
                        document.getElementById('qa-form').submit();
                    } else if (e.key === 'ArrowRight') {
                        e.preventDefault();
                        setAction('save_next');
                        document.getElementById('qa-form').submit();
                    }
                }
            });
        </script>
    </body>
    </html>
    """

    return render_template_string(
        html_template,
        dataset_name=current_ds,
        active_datasets=active_datasets,
        all_datasets=all_datasets,
        details=details,
        counts=counts,
        filter_mode=filter_mode,
        sort_mode=sort_mode,
        current_idx=current_idx,
        prev_ds=prev_ds,
        next_ds=next_ds,
        ds_info=ds_info,
        subjects=subjects,
        preds=preds,
        confs=confs,
        msg=msg,
        msg_type=msg_type
    )


def _reviewable_subjects(grp, dataset):
    """Subjects of a dataset that have something for a human to look at.

    Used to be "has a report_html_path attribute". Full extraction writes only
    the ~20 KB thumbnail, so a subject with a PNG and no report is perfectly
    reviewable -- gating on the report alone would hide every subject extracted
    after the switch.
    """
    data_dir = Path(H5_PATH).parent
    return [s for s in grp.keys()
            if 'report_html_path' in grp[s].attrs
            or thumbnail_path(data_dir, dataset, s).exists()]


def _find_report(dataset, subject):
    """Locate a subject's QA report on this machine."""
    data_dir = Path(H5_PATH).parent
    local = sorted((data_dir / dataset / subject).glob("*.html"))
    if local:
        return local[0]

    try:
        with open_h5_file(H5_PATH, "r") as f:
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
        try:
            from deepmreye.datasource import ensure_reports
            ensure_reports(Path(H5_PATH).parent, [dataset], quiet=True)
            path = _find_report(dataset, subject)
        except Exception as e:
            return f"Report not available locally and download failed: {e}", 404

    if path is None:
        return "Report not found on disk.", 404

    try:
        html_content = path.read_text(errors="replace")
        # Inject custom styling into report to seamlessly blend with dark UI and handle responsive resizing
        custom_style = """
        <style>
            body { background-color: #121214 !important; color: #eee !important; margin: 0; padding: 0; }
            .plotly-graph-div { margin: 0 auto; max-width: 100% !important; }
        </style>
        <script>
            window.addEventListener('resize', function() {
                if (window.Plotly) {
                    var plots = document.querySelectorAll('.plotly-graph-div');
                    plots.forEach(function(p) { Plotly.Plots.resize(p); });
                }
            });
        </script>
        """
        if "</head>" in html_content:
            html_content = html_content.replace("</head>", f"{custom_style}</head>")
        else:
            html_content = f"{custom_style}{html_content}"

        return Response(html_content, mimetype="text/html")
    except Exception as e:
        return str(e), 404


@app.route('/submit', methods=['POST'])
def submit_label():
    dataset_name = request.form.get('dataset')
    action = request.form.get('action', 'save_next')
    target_ds_param = request.form.get('target_ds', '')
    filter_mode = request.form.get('filter', 'all')
    sort_mode = request.form.get('sort', 'conf_asc')
    
    all_datasets, details, _ = get_dataset_info()

    if filter_mode == 'unlabeled':
        active_datasets = [ds for ds in all_datasets if details[ds]["status"] == "unlabeled"]
    elif filter_mode == 'labeled':
        active_datasets = [ds for ds in all_datasets if details[ds]["status"] == "labeled"]
    elif filter_mode == 'skipped':
        active_datasets = [ds for ds in all_datasets if details[ds]["status"] in ("skipped", "no_reports")]
    else:
        active_datasets = [ds for ds in all_datasets if details[ds]["status"] != "skipped"]

    if not active_datasets:
        active_datasets = all_datasets

    if sort_mode == 'conf_asc':
        active_datasets.sort(key=lambda d: details[d]["min_conf"])
    elif sort_mode == 'conf_desc':
        active_datasets.sort(key=lambda d: details[d]["min_conf"], reverse=True)
    elif sort_mode == 'name':
        active_datasets.sort()

    current_idx = active_datasets.index(dataset_name) if dataset_name in active_datasets else 0
    prev_ds = active_datasets[current_idx - 1] if current_idx > 0 else active_datasets[-1]
    next_ds = active_datasets[current_idx + 1] if current_idx < len(active_datasets) - 1 else active_datasets[0]

    if dataset_name:
        events = []  # (dataset, scope, subject, label) mirrored to the CSV backup
        try:
            with open_h5_file(H5_PATH, 'a') as f:
                if dataset_name in f:
                    subs_with_reports = _reviewable_subjects(f[dataset_name], dataset_name)

                    if action in ('skip_next', 'skip') or not subs_with_reports:
                        # Datasets without HTML reports cannot be visually QA'd and are automatically skipped (-99)
                        f[dataset_name].attrs['approved'] = -99
                        events.append((dataset_name, 'dataset', '', -99))
                    elif action == 'unskip':
                        f[dataset_name].attrs['approved'] = -1
                        events.append((dataset_name, 'dataset', '', -1))
                    else:
                        for sub in subs_with_reports:
                            lbl = request.form.get(f'label_{sub}')
                            if lbl is not None:
                                f[dataset_name][sub].attrs['approved'] = int(lbl)
                                f[dataset_name][sub].attrs['is_manual'] = True
                                events.append((dataset_name, 'subject', sub, int(lbl)))
        except Exception as e:
            print(f"Failed to update HDF5: {e}")

        if events:
            try:
                append_label_events(Path(H5_PATH).parent / "labels.csv", events)
            except Exception as e:
                print(f"Failed to write label backup CSV: {e}")

    if action == 'save_prev':
        target = prev_ds
    elif action == 'save_to_ds' and target_ds_param:
        target = target_ds_param
    elif action in ('save_stay', 'unskip'):
        target = dataset_name
    else:  # save_next, skip_next
        target = next_ds

    return redirect(url_for('index', ds=target, filter=filter_mode, sort=sort_mode))


@app.route('/push_hf', methods=['POST'])
def push_hf():
    current_ds = request.form.get('current_ds', '')
    try:
        from scripts.sync_labels import push
        from deepmreye.datasource import DEFAULT_REPO
        data_dir = Path(H5_PATH).parent
        push(data_dir, DEFAULT_REPO)
        msg = f"Successfully pushed datasets.h5 and labels.csv to HuggingFace ({DEFAULT_REPO})!"
        msg_type = "success"
    except Exception as e:
        msg = f"Push to HuggingFace failed: {e}"
        msg_type = "error"

    return redirect(url_for('index', ds=current_ds, msg=msg, msg_type=msg_type))


RAPID_AUDIT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapid Visual Audit - deepmreye</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #09090b; color: #f4f4f5; min-height: 100vh; display: flex; flex-direction: column; }

        header { background: #121215; border-bottom: 1px solid #27272a; padding: 12px 24px; display: flex; align-items: center; justify-content: space-between; position: sticky; top: 0; z-index: 100; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }
        .nav-tabs { display: flex; gap: 4px; background: #18181b; padding: 3px; border-radius: 8px; border: 1px solid #27272a; }
        .nav-tab { padding: 6px 14px; border-radius: 6px; text-decoration: none; color: #a1a1aa; font-size: 13px; font-weight: 500; transition: all 0.15s; }
        .nav-tab.active, .nav-tab:hover { background: #27272a; color: #fff; }
        .nav-tab.active { font-weight: 600; }

        .stats-bar { display: flex; gap: 16px; align-items: center; font-size: 13px; font-weight: 500; }
        .stat-pill { background: #18181b; border: 1px solid #27272a; padding: 4px 10px; border-radius: 6px; }
        .stat-pill strong { color: #fff; }
        .stat-kept strong { color: #4ade80; }
        .stat-removed strong { color: #f87171; }

        .controls { display: flex; gap: 12px; align-items: center; }
        .search-input { background: #18181b; border: 1px solid #27272a; color: #fff; padding: 6px 12px; border-radius: 6px; font-size: 13px; outline: none; width: 220px; }
        .search-input:focus { border-color: #3b82f6; }
        .push-btn { background: #2563eb; color: #fff; border: none; padding: 6px 14px; border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer; transition: background 0.15s; }
        .push-btn:hover { background: #1d4ed8; }

        main { flex-grow: 1; padding: 20px; max-width: 1800px; margin: 0 auto; width: 100%; }
        
        .grid-container { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 16px; }

        .card { background: #18181b; border: 2px solid #16a34a; border-radius: 10px; overflow: hidden; cursor: pointer; user-select: none; transition: transform 0.1s, border-color 0.15s, opacity 0.15s; position: relative; }
        .card:hover { transform: translateY(-2px); }
        .card.removed { border-color: #dc2626; opacity: 0.55; }

        .card-header { background: #121215; padding: 8px 12px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #27272a; }
        .card-title { font-weight: 700; font-size: 14px; color: #fff; }
        
        .status-badge { font-size: 10px; font-weight: 700; text-transform: uppercase; padding: 2px 6px; border-radius: 4px; letter-spacing: 0.5px; }
        .badge-kept { background: #14532d; color: #4ade80; border: 1px solid #16a34a; }
        .badge-removed { background: #7f1d1d; color: #f87171; border: 1px solid #dc2626; }

        .card-body { display: flex; padding: 8px; gap: 8px; background: #09090b; }
        .sub-view { flex: 1; display: flex; flex-direction: column; align-items: center; position: relative; }
        .sub-label { font-size: 10px; font-weight: 600; color: #a1a1aa; margin-bottom: 4px; text-transform: uppercase; }
        .zview-img { width: 100%; height: auto; aspect-ratio: 109 / 91; background: #000; border-radius: 4px; image-rendering: crisp-edges; object-fit: contain; }

        .remove-overlay { position: absolute; inset: 0; background: rgba(220, 38, 38, 0.25); display: flex; align-items: center; justify-content: center; pointer-events: none; opacity: 0; transition: opacity 0.15s; }
        .card.removed .remove-overlay { opacity: 1; }
        .remove-icon { font-size: 64px; color: #ef4444; text-shadow: 0 0 12px rgba(0,0,0,0.8); font-weight: bold; }

        .loading-spinner { text-align: center; padding: 60px; font-size: 18px; color: #a1a1aa; }
    </style>
</head>
<body>
    <header>
        <div class="nav-tabs">
            <a href="/" class="nav-tab">🔍 Detailed QA View</a>
            <a href="/rapid" class="nav-tab active">⚡ Rapid Visual Audit (Both Eyes)</a>
        </div>

        <div class="stats-bar">
            <div class="stat-pill">Qualifying Datasets: <strong id="cnt-total">0</strong></div>
            <div class="stat-pill stat-kept">Kept: <strong id="cnt-kept">0</strong></div>
            <div class="stat-pill stat-removed">Marked Removed: <strong id="cnt-removed">0</strong></div>
        </div>

        <div class="controls">
            <input type="text" class="search-input" id="search-box" placeholder="Search dataset ID..." oninput="filterGrid()">
            <form method="POST" action="/push_hf" style="display:inline;">
                <input type="hidden" name="current_ds" value="">
                <button type="submit" class="push-btn">Push Labels to HF</button>
            </form>
        </div>
    </header>

    <main>
        <div id="loading" class="loading-spinner">Loading dataset Z-views...</div>
        <div id="grid-container" class="grid-container" style="display:none;"></div>
    </main>

    <script>
        let datasetList = [];

        async function loadDatasets() {
            try {
                const res = await fetch('/api/rapid_datasets');
                datasetList = await res.json();
                renderGrid();
            } catch (err) {
                document.getElementById('loading').innerText = 'Failed to load datasets: ' + err;
            }
        }

        function updateCounters() {
            const total = datasetList.length;
            const kept = datasetList.filter(d => d.approved).length;
            const removed = total - kept;

            document.getElementById('cnt-total').innerText = total;
            document.getElementById('cnt-kept').innerText = kept;
            document.getElementById('cnt-removed').innerText = removed;
        }

        function renderGrid() {
            const container = document.getElementById('grid-container');
            container.innerHTML = '';
            
            datasetList.forEach(item => {
                const card = document.createElement('div');
                card.className = 'card' + (item.approved ? '' : ' removed');
                card.id = 'card-' + item.dataset;
                card.onclick = () => toggleDataset(item.dataset);

                card.innerHTML = `
                    <div class="card-header">
                        <span class="card-title">${item.dataset}</span>
                        <span class="status-badge ${item.approved ? 'badge-kept' : 'badge-removed'}" id="badge-${item.dataset}">
                            ${item.approved ? 'KEPT' : 'REMOVED'}
                        </span>
                    </div>
                    <div class="card-body">
                        <div class="sub-view">
                            <span class="sub-label">${item.sub1}</span>
                            <img class="zview-img" src="${item.sub1_zview}" loading="lazy" alt="${item.sub1}">
                        </div>
                        <div class="sub-view">
                            <span class="sub-label">${item.sub2}</span>
                            <img class="zview-img" src="${item.sub2_zview}" loading="lazy" alt="${item.sub2}">
                        </div>
                    </div>
                    <div class="remove-overlay">
                        <div class="remove-icon">✕</div>
                    </div>
                `;

                container.appendChild(card);
            });

            document.getElementById('loading').style.display = 'none';
            container.style.display = 'grid';
            updateCounters();
        }

        function filterGrid() {
            const q = document.getElementById('search-box').value.toLowerCase().trim();
            datasetList.forEach(item => {
                const card = document.getElementById('card-' + item.dataset);
                if (card) {
                    const match = item.dataset.toLowerCase().includes(q);
                    card.style.display = match ? 'block' : 'none';
                }
            });
        }

        async function toggleDataset(ds) {
            const item = datasetList.find(d => d.dataset === ds);
            if (!item) return;

            item.approved = !item.approved;

            const card = document.getElementById('card-' + ds);
            const badge = document.getElementById('badge-' + ds);

            if (item.approved) {
                card.classList.remove('removed');
                badge.className = 'status-badge badge-kept';
                badge.innerText = 'KEPT';
            } else {
                card.classList.add('removed');
                badge.className = 'status-badge badge-removed';
                badge.innerText = 'REMOVED';
            }

            updateCounters();

            try {
                await fetch('/api/toggle_dataset_approval', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ dataset: ds, approved: item.approved })
                });
            } catch (err) {
                console.error('Failed to save toggle state for ' + ds, err);
            }
        }

        window.onload = loadDatasets;
    </script>
</body>
</html>
"""


@app.route('/zview/<dataset>/<subject>.png')
def serve_zview_png(dataset, subject):
    """Serve a subject's QA thumbnail.

    Normally this is a file read: extraction writes ``<dataset>/<subject>.png``
    beside the participant's HDF5. The fallback renders it from the HTML report
    on demand, for a corpus whose thumbnails have not been backfilled yet
    (``scripts/backfill_thumbnails.py``) -- that path parses ~5 MB of embedded
    base64 per subject, which is exactly what the stored PNG exists to avoid.
    """
    path = thumbnail_path(Path(H5_PATH).parent, dataset, subject)
    if path.exists():
        return send_file(path, mimetype="image/png")

    report_path = _find_report(dataset, subject)
    if report_path is None or not report_path.exists():
        return "No thumbnail or report found", 404

    try:
        image = thumbnail.from_report(report_path.read_text(errors="replace"))
    except Exception as e:
        return f"Error rendering thumbnail: {e}", 500
    if image is None:
        return "Report carries no usable slice data", 404

    # Cache it so the expensive parse happens once per subject, not per view.
    try:
        thumbnail.save(image, path)
    except OSError:
        pass  # read-only corpus: serve it anyway, just without caching.

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Response(buf.getvalue(), mimetype="image/png")


@app.route('/rapid')
def rapid_audit():
    return render_template_string(RAPID_AUDIT_HTML)


@app.route('/api/rapid_datasets')
def api_rapid_datasets():
    if not os.path.exists(H5_PATH):
        return jsonify([])

    result = []
    with open_h5_file(H5_PATH, 'r') as f:
        for ds in f.keys():
            grp = f[ds]
            approved_attr = grp.attrs.get('approved', -1)

            if approved_attr == -99:
                continue

            # The gaze-labeled datasets are not QA candidates: eye tracking was
            # recorded alongside the scan, so the eyeballs are in frame by
            # construction. They are skipped here so a stray click in the grid
            # cannot mark ground truth as no-eyes.
            if grp.attrs.get('labeled', False):
                continue

            subs_with_reports = _reviewable_subjects(grp, ds)
            if not subs_with_reports:
                continue

            sub_labels = {s: grp[s].attrs.get('approved', -1) for s in subs_with_reports}

            all_eyes_present = all(lbl in (1, 3, 4) for lbl in sub_labels.values())
            all_removed = all(lbl == 0 for lbl in sub_labels.values())

            if all_eyes_present or (all_removed and grp.attrs.get('toggled_in_rapid', False)):
                sub1 = subs_with_reports[0] if len(subs_with_reports) > 0 else ""
                sub2 = subs_with_reports[1] if len(subs_with_reports) > 1 else sub1

                result.append({
                    "dataset": ds,
                    "approved": all_eyes_present,
                    "sub1": sub1,
                    "sub2": sub2,
                    "sub1_zview": f"/zview/{ds}/{sub1}.png" if sub1 else "",
                    "sub2_zview": f"/zview/{ds}/{sub2}.png" if sub2 else "",
                })

    return jsonify(result)


@app.route('/api/toggle_dataset_approval', methods=['POST'])
def api_toggle_dataset_approval():
    data = request.get_json(force=True)
    ds_name = data.get('dataset')
    approved = data.get('approved', True)

    if not ds_name:
        return jsonify({"status": "error", "message": "Missing dataset"}), 400

    events = []
    try:
        with open_h5_file(H5_PATH, 'a') as f:
            if ds_name in f:
                grp = f[ds_name]
                grp.attrs['toggled_in_rapid'] = True
                subs_with_reports = _reviewable_subjects(grp, ds_name)

                target_lbl = 1 if approved else 0
                for sub in subs_with_reports:
                    grp[sub].attrs['approved'] = target_lbl
                    grp[sub].attrs['is_manual'] = True
                    events.append((ds_name, 'subject', sub, target_lbl))
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    if events:
        try:
            append_label_events(Path(H5_PATH).parent / "labels.csv", events)
        except Exception as e:
            print(f"Failed to write backup CSV: {e}")

    return jsonify({"status": "ok", "dataset": ds_name, "approved": approved})


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
