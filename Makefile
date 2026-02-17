.PHONY: help install install-dev test test-cov lint format type-check security clean docs
# Makefile for building a custom local docker image and launching a JupyterLab from the containerized image
# Docker image can be built by running "make build" in shell (DO THIS ONLY ONCE)
# Run "make local" to open JupyterLab from containerized image

# Name to give the image
IMG_NAME = era-renewables

# "make local"
# Launch local JupyerLab from containerized instance of IMG_NAME
# Open JupyterLab by clicking the bottom link in the terminal output
local:
	@echo "Opening JupyterLab from runtime instance of $(IMG_NAME)"
	@echo "Copy/paste the last link in the terminal output into your web browser to open JupyterLab"
	docker run -t --rm --volume "$(PWD)":/home/jovyan \
	-e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID) -e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY) \
	-p 8888:8888 $(IMG_NAME) jupyter lab --ip 0.0.0.0

# "make build"
# YOU ONLY NEED TO DO THIS ONCE
# Create the docker image locally using the Dockerfile in the current repository
# Output from this command can be found in the logfile build.log
build:
	@echo "Building docker image named $(IMG_NAME) from Dockerfile"
	@echo "A conda environment will be created in the image using a conda-lock.yml file if it exists. If not, the environment.yml will be used."
	@echo "Pip dependencies will be installed from the requirements.txt if it exists."
	@echo "Output from the build will be saved in a log file build.log"
	docker build -t $(IMG_NAME) .  &> build.log

# "make conda-lock"
# Make conda-lock.yml from environment.yml
# You must have conda-lock installed
conda-lock:
	@echo "Creating multi-platform conda-lock.yml from environment.yml"
	@echo "Will fail if conda-lock package is not installed"
	conda-lock -f environment.yml -p osx-64 -p linux-64

# BONUS: "make run"
# Run a python script in a runtime instance (container) of the image
# Customize the path below to add your own script
# Add any additional arguments for python script to ARGS
#PY_SCRIPT = $(script)
#ARGS = $(args)
run:
	@echo "Running python script $(PY_SCRIPT) in runtime instance of $(IMG_NAME)"
	docker run -t --rm --volume "$(PWD)":/home/jovyan \
	-e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID) -e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY) \
	$(IMG_NAME) python $(PY_SCRIPT) $(ARGS)



help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install-mamba: ## Install mamba package manager
	@if ! command -v mamba >/dev/null 2>&1; then \
		conda install -y mamba -c conda-forge; \
		eval "$(mamba shell hook --shell ${SHELL##*/})"; \
		mamba activate; \
	else \
		echo "mamba already installed"; \
	fi

install-mamba-env:
	@if ! command -v mamba >/dev/null 2>&1; then \
		echo "mamba is not installed. Please run 'make install-mamba' first."; \
		exit 1; \
	fi
	mamba env create -f environment.yml

install-pkg: ## Install the package
	pip install .

install-pkg-dev: ## Install the package with development dependencies
	pip install .
	pre-commit install

test: ## Run tests
	python -m pytest

test-cov: ## Run tests with coverage
	python -m pytest --cov=src --cov-report=html --cov-report=term-missing

lint: ## Run all linters
	ruff check src/ tests/ --fix

format: ## Format code
	black src/ tests/
	isort src/ tests/
	ruff check --fix src/ tests/

security: ## Run security checks
	bandit -r src/
	detect-secrets scan --baseline .secrets.baseline

pre-commit: ## Run all pre-commit hooks
	pre-commit run --all-files

clean: ## Clean up temporary files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type f -name "*.pyc" -delete
	find . -type d -name __pycache__ -delete

docs: ## Generate documentation
	@echo "Documentation generation not implemented yet"

write-env:
	mamba env export --no-builds > environment.yml

install-precommit:
	pre-commit install

setup: install-mamba install-mamba-env
all: clean install-pkg-dev install-precommit format lint pre-commit  ## Run full development setup and checks
