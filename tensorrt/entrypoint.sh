#!/bin/bash
# https://gist.github.com/xkortex/84226629c2a1286120bf139b93bca9bf

# cache the value because the shell hook step will remove it
_CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-tf}"

__conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup

# Restore our "indended" default env
conda activate "${_CONDA_DEFAULT_ENV}"
# This just logs the output to stderr for debugging. 
>&2 echo "ENTRYPOINT: CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV}"

exec "$@"