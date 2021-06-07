# Setting up Triton Inference Server with SLEAP

Make sure you're set up to use Triton on your host machine -- specifically make sure you're set up to NVIDIA containers(NVIDIA Container Toolkit) as described in the [Triton quickstart guide](https://github.com/triton-inference-server/server/blob/main/docs/quickstart.md)

Clone this repo: 
```bash
git clone https://github.com/talmo/gpuhackathon-sleap
cd gpuhackathon-sleap/te
```

## [1_triton_setup.ipynb](1_triton_setup.ipyn)

This note book shows the basic setup for Triton -- the most important command is simply:

```bash
docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v${path_to_model_repository}:/models nvcr.io/nvidia/tritonserver:21.05-py3 tritonserver --model-repository=/models --backend-config=tensorflow,version=2
```

Note the tensorflow version selection to fix compatability with SLEAP models and our model repository directory. For instructions on setting up the model repo and configs check out the notebook itself or the [Triton github page](https://github.com/triton-inference-server/server).


### Metrics with Promtheus

```bash
./prometheus --config.file=prometheus.yml
```

### ngrok Serving

While the easiest way to test Triton is to just use the default local serving, we can also expose our server to the public. One easy way to do this is using a service like [ngrok](https://ngrok.com/).

To expose our triton server (:8001) and the default prometheus port (:9090) which we'll look more at later, we can start ngrok pointing the the config found in [ngrok_config.yml](ngrok_config.yml)

```bash
./ngrok start --config=ngrok_config.yml all
```

## Colab Example

Now that you have Triton and ngrok set up set up, check out this colab notebook for an example of remote inference: [SLEAP_TRT_Triton](https://colab.research.google.com/drive/1JyiPudH4y3gdcGL9c9pPAvUBM8Yf5Jwc). 




