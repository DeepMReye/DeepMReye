#! /bin/bash

# download data and model weights

# TODO
# consider using only python (e.g pooch) instead of bash to make it portable on windows

# download one participant from OSF: sub-NDARAA948VFH
wget https://files.de-1.osf.io/v1/resources/mrhk9/providers/osfstorage/612d10f8ab8bca001eedc8b8/?zip=
mv index.html?zip= sub-NDARAA948VFH.zip

mkdir -p functional_data/sub-NDARAA948VFH

unzip sub-NDARAA948VFH.zip -d functional_data/sub-NDARAA948VFH

# download pre-trained weights
mkdir -p model_weights

wget https://osf.io/download/23t5v/ -O model_weights/datasets_1to5.h5
