# Setting up the TensorRT container (`trt`)

Clone the repo:
```
git clone https://github.com/talmo/gpuhackathon-sleap
cd gpuhackathon-sleap/tensorrt
```

- [Install GPU drivers](installing_drivers.md) if you haven't yet.
- [Install docker](installing_docker.md) if you haven't yet.

To build the container just run:
```
docker build -t trt .
```
You can re-run this command after making changes to the `Dockerfile` or `environment.yml`.

What this does is:
- Start with a Ubuntu 18.04 NVIDIA development image with **cudatoolkit 11.2.1** and **cudnn 8**. These are the precise versions compatible with `tensorflow==2.5.0` that's prebuilt on PyPI.
- Install miniconda and create a conda environment called `trt` from `environment.yml`
- Install `nvidia-tensorrt==7.2.3.4` which is a prebuilt wheel compatible with the 11.2 and cudnn 8 as needed by tensorflow. See the [archived docs](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-722/quick-start-guide/index.html#installing-pip) for this version.
- Setup jupyter lab and some other utilities


# Running the container

To just drop into a bash terminal run:
```
docker run --gpus all -it -w /mnt -v $PWD:/mnt trt:latest bash
```
And activate installed environment:
```
(base) root@a7d64731005b:/mnt# conda activate trt
(trt) root@a7d64731005b:/mnt# which python
/root/miniconda3/envs/trt/bin/python
```

To start a jupyter lab session and run the notebooks:
```
docker run --gpus all -it -w /mnt -v $PWD:/mnt -p 8888:8888 trt:latest jlab
```

This will print out a link that you can paste in your browser to access it:
```
(base) talmo@talmo-desktop-ubuntu:~/gpuhackathon-sleap/tensorrt$ docker run --gpus all -it -w /mnt -v $PWD:/mnt -p 8888:8888 trt:latest jlab
[I 2021-05-23 00:58:22.502 ServerApp] jupyterlab | extension was successfully linked.
[I 2021-05-23 00:58:22.508 ServerApp] Writing notebook server cookie secret to /root/.local/share/jupyter/runtime/jupyter_cookie_secret
[I 2021-05-23 00:58:22.659 ServerApp] nbclassic | extension was successfully linked.
[I 2021-05-23 00:58:22.675 ServerApp] nbclassic | extension was successfully loaded.
[I 2021-05-23 00:58:22.675 LabApp] JupyterLab extension loaded from /root/miniconda3/envs/trt/lib/python3.6/site-packages/jupyterlab
[I 2021-05-23 00:58:22.675 LabApp] JupyterLab application directory is /root/miniconda3/envs/trt/share/jupyter/lab
[I 2021-05-23 00:58:22.677 ServerApp] jupyterlab | extension was successfully loaded.
[I 2021-05-23 00:58:22.678 ServerApp] Serving notebooks from local directory: /mnt
[I 2021-05-23 00:58:22.678 ServerApp] Jupyter Server 1.8.0 is running at:
[I 2021-05-23 00:58:22.678 ServerApp] http://38cdf497e15c:8888/lab?token=bb3f3b34a2625cd91c308ffaddfe370acfe2fc7271994b55
[I 2021-05-23 00:58:22.678 ServerApp]     http://127.0.0.1:8888/lab?token=bb3f3b34a2625cd91c308ffaddfe370acfe2fc7271994b55
[I 2021-05-23 00:58:22.678 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[W 2021-05-23 00:58:22.681 ServerApp] No web browser found: could not locate runnable browser.
[C 2021-05-23 00:58:22.681 ServerApp] 
    
    To access the server, open this file in a browser:
        file:///root/.local/share/jupyter/runtime/jpserver-24-open.html
    Or copy and paste one of these URLs:
        http://38cdf497e15c:8888/lab?token=bb3f3b34a2625cd91c308ffaddfe370acfe2fc7271994b55
        http://127.0.0.1:8888/lab?token=bb3f3b34a2625cd91c308ffaddfe370acfe2fc7271994b55

```

The working directory for both of these is `/mnt` which will point to the current directory (`gpuhackathon-sleap/tensorrt`) so you can persist files with the VM from here.


# Running the notebooks

1. Start the container with jupyter lab: `docker run --gpus all -it -w /mnt -v $PWD:/mnt -p 8888:8888 trt:latest jlab`
2. `download_data.ipynb`: Downloads data and trained models to `data/`
3. `convert_models.ipynb`: Converts trained models to [`SavedModel`](https://www.tensorflow.org/guide/saved_model) and [TensorRT optimized engine](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-722/quick-start-guide/index.html#framework-integration) formats.
4. `benchmark_models.ipynb`: Run basic benchmarking and plot results
