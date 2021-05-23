# Installing NVIDIA docker

Following [these instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) to install docker + nvidia-docker:

Install base docker:
```
curl https://get.docker.com | sh
sudo systemctl --now enable docker
```

Cool note for later: [Multi-Instance GPU (MIG)](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html) -- Adds the ability to partition A100s into instances so multiple users can share the GPU :)


Get distribution string:
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
echo "distribution: '$distribution'"  # 'ubuntu20.04'
```

Setup some auth stuff:
```
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```

Install NVIDIA docker for GPU support:
```
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

*(Optional)*: Removing sudo requirement ([ref](https://docs.docker.com/engine/install/linux-postinstall/)):
```
sudo groupadd docker
sudo usermod -aG docker $USER
```
Then logout, log back in (no need to restart).

To test if worked:
```
docker run --gpus all --rm nvidia/cuda:10.2-base nvidia-smi
```

You can find other relevant images to use here:
* TensorFlow images: https://hub.docker.com/r/tensorflow/tensorflow/tags/?page=1&ordering=last_updated
* NVIDIA images: https://hub.docker.com/r/nvidia/cuda/tags?page=1&ordering=last_updated