################################################################################
# 	DOCKER

.PHONY: Dockerfile 
Dockerfile:
	docker run --rm repronim/neurodocker:0.7.0 generate docker \
		--base python:3.9.12-slim-buster \
		--pkg-manager apt \
		--install "wget build-essential" \
		--copy deepmreye setup.py README.md requirements.txt /deepmreye/ \
		--workdir /deepmreye \
		--run "pip install ." \
		--run "pip install jupyterlab" \
		--user neuro \
		--workdir /home/neuro \
		--copy notebooks /home/neuro/notebooks \
		--run "mkdir -p /home/neuro/models" \
		--run "wget https://osf.io/cqf74/download -O /home/neuro/dataset1_guided_fixations.h5" \
		--expose 8888 > Dockerfile

docker_build: Dockerfile
	docker build . -t deepmreye:latest

docker_run: docker_build
	mkdir -p $$PWD/notebooks
	docker run -it --rm \
		--publish 8888:8888 \
		--volume $$PWD/notebooks:/home/neuro/notebooks \
		deepmreye:latest \
			jupyter-lab --no-browser --ip 0.0.0.0