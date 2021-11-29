# CONTRIBUTING

## Docker recipe

Made using neurodocker

```bash
docker run --rm repronim/neurodocker:0.7.0 generate docker \
    --base debian:bullseye-slim \
    --pkg-manager apt \
    --install "git wget" \
    --miniconda \
        version="latest" \
        create_env="deepmreye" \
        conda_install="python=3.7 pip" \
        pip_install="git+https://github.com/DeepMReye/DeepMReye.git" \
        activate="true" \
    --run "mkdir -p /inputs/models" \
    --run "wget https://osf.io/cqf74/download -O /inputs/models/dataset1_guided_fixations.h5" \
    --output Dockerfile
```

Build image

```bash
docker build --tag deepmreye:0.1.0 --file Dockerfile .
```

Run it

```bash
docker run -it --rm \
    deepmreye:0.1.0
```