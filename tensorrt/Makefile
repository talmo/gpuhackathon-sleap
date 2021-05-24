# Makefile for building and running the container.
#
# Usage:
# 	make
#	make build
#	make jlab
#	make bash
# 
# Ref: https://gist.github.com/mpneuried/0594963ad38e68917ef189b4e6a269db

# CONFIG
APP_NAME="trt"
PORT=8888

# HELP
# This will output the help for each task
# thanks to https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
.PHONY: help

help: ## This help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.DEFAULT_GOAL := help


# DOCKER TASKS
# Build the container
build: ## Build the container
	docker build -t $(APP_NAME) .

build-nc: ## Build the container without caching
	docker build --no-cache -t $(APP_NAME) .

jlab: ## Run container and launch a jupyter notebook
	docker run --gpus all -it -w /mnt -v $(PWD):/mnt -p $(PORT):$(PORT) $(APP_NAME) jlab

bash: ## Run container and launch a bash session
	docker run --gpus all -it -w /mnt -v $(PWD):/mnt -p $(PORT):$(PORT) $(APP_NAME) bash
