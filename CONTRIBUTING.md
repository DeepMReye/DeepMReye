# CONTRIBUTING

## Docker recipe

Made using neurodocker

```bash
docker run --rm repronim/neurodocker:0.7.0 generate docker \
    --base debian:bullseye-slim \
    --pkg-manager apt \
    --install "git wget" \
    --user=neuro \
    --workdir /home \
    --miniconda \
        version="latest" \
        create_env="deepmreye" \
        conda_install="python=3.7 pip" \
        pip_install="git+https://github.com/DeepMReye/DeepMReye.git" \
        activate="true" \
    --env LD_LIBRARY_PATH="/opt/miniconda-latest/envs/neuro:$LD_LIBRARY_PATH" \
    --run-bash "source activate deepmreye" \
    --user=root \
    --run 'chmod 777 -Rf /home' \
    --run 'chown -R neuro /home' \
    --user=neuro \
    --run "mkdir -p /home/inputs/models" \
    --run "wget https://osf.io/cqf74/download -O /home/inputs/models/dataset1_guided_fixations.h5" \
    --run 'mkdir -p ~/.jupyter && echo c.NotebookApp.ip = \"0.0.0.0\" > ~/.jupyter/jupyter_notebook_config.py' > Dockerfile
```

Build image

```bash
docker build --tag deepmreye:0.1.0 --file Dockerfile .
```

Run it

```bash
docker run -it -p 8888:8888 --rm \
    deepmreye:0.1.0
```
