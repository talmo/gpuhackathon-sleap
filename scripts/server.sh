#!/bin/bash

singularity exec --nv singularity/triton-sleap-21.03.sif tritonserver --model-repository=/gpuhackathon-sleap/triton/model_repository --backend-config=tensorflow,version=2
