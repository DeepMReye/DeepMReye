
.PHONY: all clean

subject       = functional_data/sub-NDARAA948VFH
gaze_data     = gaze_data/stim_vals.csv
model_weights = model_weights/datasets_1to5.h5

all: $(model_weights) $(subject)/run1_sub-NDARAA948VFH.nii $(gaze_data)

clean: clean_output
	rm -rf functional_data
	rm -rf gaze_data
	rm -rf model_weights

clean_output:
	rm -rf processed_data
	rm -rf functional_data/sub-NDARAA948VFH/*.p
	rm -rf functional_data/sub-NDARAA948VFH/*.html

$(gaze_data):
	mkdir -p gaze_data
	wget https://osf.io/download/s92u7/ -O gaze_data/stim_vals.csv

$(subject)/run1_sub-NDARAA948VFH.nii:
	wget https://files.de-1.osf.io/v1/resources/mrhk9/providers/osfstorage/612d10f8ab8bca001eedc8b8/?zip=
	mv index.html?zip= sub-NDARAA948VFH.zip
	mkdir -p functional_data/sub-NDARAA948VFH
	unzip sub-NDARAA948VFH.zip -d functional_data/sub-NDARAA948VFH
	rm sub-NDARAA948VFH.zip

$(model_weights):
	mkdir -p model_weights
	wget https://osf.io/download/23t5v/ -O model_weights/datasets_1to5.h5
