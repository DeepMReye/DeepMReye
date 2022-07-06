# Import modules and add library to path
import sys
sys.path.insert(0, "/home/marx/Documents/Github/DeepMReye")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Change to os.environ["CUDA_VISIBLE_DEVICES"] = "" if you dont have access to a GPU
import numpy as np 
import pandas as pd
import pickle

# DeepMReye imports
from deepmreye import architecture, train, analyse, preprocess
from deepmreye.util import util, data_generator, model_opts

# Initialize plotly figures
from plotly.offline import init_notebook_mode 
init_notebook_mode(connected = True)


# Define paths to functional data
experiment_folder = '/home/neuro/experiment_folder/' # Replace this path to your downloaded files
functional_data = experiment_folder + 'functional_data/'
processed_data = experiment_folder + 'processed_data/'
gaze_data = experiment_folder + 'gaze_data/'
model_weights = experiment_folder + 'model_weights/'
mask_path = functional_data + 'masks/'

# Get participants from functional folder
participants = os.listdir(functional_data) # (if needed, remove single participants with participants.remove('participant01') or recreate participants list)

# Preload masks to save time within participant loop
(eyemask_small, eyemask_big, dme_template, mask, x_edges, y_edges, z_edges) = preprocess.get_masks(data_path=mask_path)

# Loop across participants and extract eye mask
for participant in participants:
    if participant.startswith('s'):
        print('Running participant {}'.format(participant))
        participant_folder = functional_data + participant
        for run in os.listdir(participant_folder):
            if run.startswith('run'):
                fp_func = participant_folder + os.path.sep + run # Filepath to functional
                preprocess.run_participant(fp_func, dme_template, eyemask_big, eyemask_small, x_edges, y_edges, z_edges)

# Combine processed masks with labels
for participant in participants:
    if participant.startswith('s'):
        print('Running participant {}'.format(participant))
        participant_folder = functional_data + participant
        participant_data, participant_labels, participant_ids = [], [], []
        for run_idx, run in enumerate(os.listdir(participant_folder)):
            if not run.endswith(".p"):
                continue
            # Load mask and normalize it
            this_mask = participant_folder + os.path.sep + run
            this_mask = pickle.load(open(this_mask, 'rb'))
            this_mask = preprocess.normalize_img(this_mask)
        
            # If experiment has no labels use dummy labels
            this_label = np.zeros((this_mask.shape[3], 10, 2)) # 10 is the number of subTRs used in the pretrained weights, 2 is XY
        
            # Check if each functional image has a corresponding label. Note that mask has time as third dimension
            if not this_mask.shape[3] == this_label.shape[0]:
                print('WARNING --- Skipping Subject {} Run {} --- Wrong alignment (Mask {} - Label {}).'.format(subj, run_number, this_mask.shape, this_label.shape))
                continue
        
            # Store across runs
            participant_data.append(this_mask)
            participant_labels.append(this_label)
            participant_ids.append(([participant]*this_label.shape[0], [run_idx]*this_label.shape[0]))
    
        # Save participant file
        preprocess.save_data(participant + 'no_label', participant_data, participant_labels, participant_ids, processed_data, center_labels=False)


# Define paths to example dataset
datasets = [processed_data + p for p in os.listdir(processed_data) if 'no_label' in p]

# Load data from one participant to showcase input/output
X, y = data_generator.get_all_subject_data(datasets[0])
print('Input: {}, Output: {}'.format(X.shape, y.shape))

fig = analyse.visualise_input_data(X, y, bg_color="rgb(255,255,255)", ylim=[-11, 11])
fig.show()

opts = model_opts.get_opts()
test_participants = [processed_data + p for p in os.listdir(processed_data) if 'no_label' in p]
generators = data_generator.create_generators(test_participants, test_participants)
generators = (*generators, test_participants, test_participants) # Add participant list

model_weights = model_weights + 'datasets_1to5.h5'

# Get untrained model and load with trained weights
(model, model_inference) = train.train_model(dataset='example_data', generators=generators, opts=opts, return_untrained=True)
model_inference.load_weights(model_weights)

(evaluation, scores) = train.evaluate_model(dataset='example_data', model=model_inference, generators=generators,
                                            save=False, model_path=experiment_folder, model_description='', verbose=2, percentile_cut=80)

fig = analyse.visualise_predictions_slider(evaluation, scores, color="rgb(0, 150, 175)", bg_color="rgb(255,255,255)", ylim=[-11, 11])
fig.show()