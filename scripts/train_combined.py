# System imports
import os
from argparse import ArgumentParser

# DeepMReye imports
from deepmreye import train
from deepmreye.util import data_generator, model_opts
from deepmreye.util.util import CLI_OPTIONS

# --------------------------------------------------------------------------------
# ------------------------Evaluate model on all datasets--------------------------
# --------------------------------------------------------------------------------

# Parse input arguments
parser = ArgumentParser()
for val, item in CLI_OPTIONS.items():
    parser.add_argument(*item.cli, dest=val, **item.kwargs)
args = parser.parse_args()

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"

# Create data generators from given path
parsed_datasets = args.datasets.split(",")
if "jbj" in parsed_datasets:
    parsed_datasets = ["jbj"] + parsed_datasets
else:
    parsed_datasets = parsed_datasets[0] + parsed_datasets
datasets = [args.dataset_path + ds + os.path.sep for ds in parsed_datasets]
opts = model_opts.get_opts()
generators = data_generator.create_leaveoneout_generators(
    datasets,
    batch_size=opts["batch_size"],
    augment_list=((opts["rotation_x"], opts["rotation_y"], opts["rotation_z"]), opts["shift"], opts["zoom"]),
    mixed_batches=True,
)

# User options which are different than the default
# are adjusted *within* the script (uncomment line below) not at command line
opts["epochs"] = 100

# Train model and save weights into weights_path
(model, model_inference) = train.train_model(
    dataset="_".join(parsed_datasets),
    generators=generators[0],
    opts=opts,
    return_untrained=False,
    verbose=1,
    save=True,
    model_path=args.weights_path,
)
