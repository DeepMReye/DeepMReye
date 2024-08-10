.PHONY: clean clean-build clean-pyc
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

dist: clean ## builds source and wheel package
	python -m build

release_test:
	twine upload --repository testpypi dist/*

release: dist ## package and upload a release (requires the twine package)
	twine upload dist/*



.PHONY: Dockerfile
Dockerfile:
	docker run --rm repronim/neurodocker:0.7.0 generate docker \
		--base python:3.9.12-slim-buster \
		--pkg-manager apt \
		--install "wget build-essential" \
		--run "pip install jupyterlab deepmreye" \
		--user neuro \
		--workdir /home/neuro \
		--copy notebooks /home/neuro/notebooks \
		--run "mkdir -p /home/neuro/models" \
		--run "wget https://osf.io/download/mr87v/ -O /home/neuro/dataset1to6.h5" \
		--expose 8888 > Dockerfile

Dockerfile_dev:
	docker run --rm repronim/neurodocker:0.7.0 generate docker \
		--base python:3.9.12-slim-buster \
		--pkg-manager apt \
		--install "wget build-essential" \
		--copy deepmreye setup.py setup.cfg MANIFEST.in README.md /deepmreye/ \
		--workdir /deepmreye \
		--run "pip install ." \
		--run "pip install jupyterlab" \
		--user neuro \
		--workdir /home/neuro \
		--copy notebooks /home/neuro/notebooks \
		--run "mkdir -p /home/neuro/models" \
		--run "wget https://osf.io/download/mr87v/ -O /home/neuro/dataset1to6.h5" \
		--expose 8888 > Dockerfile_dev

docker_build: Dockerfile
	docker build . --tag deepmreye:latest

docker_dev_build: Dockerfile_dev
	docker build . --file Dockerfile_dev --tag deepmreye:dev

docker_run: docker_build
	mkdir -p $$PWD/notebooks
	docker run -it --rm \
		--publish 8888:8888 \
		--volume $$PWD/notebooks:/home/neuro/notebooks \
		deepmreye:latest \
			jupyter-lab --no-browser --ip 0.0.0.0
