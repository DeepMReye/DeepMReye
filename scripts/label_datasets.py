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

app = Flask(__name__)
config = cfg.DeepMReyeConfig()

# Fallback H5 PATH based on config, overridden by args later
H5_PATH = str(Path(config.data_dir).resolve() / "datasets.h5")

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
                            <label><input type="radio" name="label_{{ sub }}" value="1" required> Yes (Approved)</label>
                            <label><input type="radio" name="label_{{ sub }}" value="0"> No eyes - Bad transform</label>
                            <label><input type="radio" name="label_{{ sub }}" value="2"> No eyes - Good transform</label>
                        </div>
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
    
    return render_template_string(
        html_template, 
        dataset_name=current_ds, 
        subjects=subjects,
        labeled_count=labeled_count,
        total_count=total_count
    )

@app.route('/reports/<dataset>/<subject>')
def serve_report(dataset, subject):
    try:
        with h5py.File(H5_PATH, 'r') as f:
            html_path = f[dataset][subject].attrs.get('report_html_path', '')
            if os.path.exists(html_path):
                with open(html_path, 'r') as html_f:
                    return Response(html_f.read(), mimetype='text/html')
            return "Report file not found strictly on disk.", 404
    except Exception as e:
        return str(e), 404

@app.route('/submit', methods=['POST'])
def submit_label():
    dataset_name = request.form.get('dataset')
    action = request.form.get('action')
    
    if dataset_name:
        try:
            with h5py.File(H5_PATH, 'a') as f:
                if dataset_name in f:
                    if action == 'skip':
                        f[dataset_name].attrs['approved'] = -99
                    else:
                        for sub in f[dataset_name].keys():
                            if 'report_html_path' in f[dataset_name][sub].attrs:
                                lbl = request.form.get(f'label_{sub}')
                                if lbl is not None:
                                    f[dataset_name][sub].attrs['approved'] = int(lbl)
        except Exception as e:
            print(f"Failed to update HDF5: {e}")
            
    return redirect(url_for('index'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dataset Labeling UI")
    parser.add_argument("--h5", type=str, default=H5_PATH, help="Path to HDF5 Datastore")
    parser.add_argument("--port", type=int, default=5050, help="Port to run the app on")
    args = parser.parse_args()
    
    H5_PATH = args.h5
    
    print("Starting Flask server for dataset labeling...")
    print(f"Target HDF5: {H5_PATH}")
    print(f"Open http://127.0.0.1:{args.port} in your browser to begin.")
    app.run(host='0.0.0.0', port=args.port, threaded=True)
