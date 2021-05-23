FROM nvidia/cuda:11.2.1-cudnn8-devel-ubuntu18.04
RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# Setup miniconda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

# Create an environment (named "trt")
COPY environment.yml /tmp
RUN conda env create -f /tmp/environment.yml

# Make conda run commands use the env:
# https://pythonspeed.com/articles/activate-conda-dockerfile/
SHELL ["conda", "run", "-n", "trt", "/bin/bash", "-c"]

# Fix some bash/docker stuff
# https://gist.github.com/xkortex/84226629c2a1286120bf139b93bca9bf
ENV CONDA_DEFAULT_ENV=trt
RUN conda init bash && echo 'conda activate "${CONDA_TARGET_ENV:-trt}"' >>  ~/.bashrc
COPY entrypoint.sh /usr/local/bin/
RUN chmod u+x /usr/local/bin/entrypoint.sh
SHELL ["/usr/local/bin/entrypoint.sh", "/bin/bash", "-c"]

# Setup our jupyter lab shortcut
# https://stackoverflow.com/questions/49024624/how-to-dockerize-jupyter-lab
COPY jlab.sh /usr/local/bin/
RUN chmod u+x /usr/local/bin/jlab.sh
RUN ln -s /usr/local/bin/jlab.sh /usr/local/bin/jlab
EXPOSE 8888


#FROM tf AS jlab
#ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "tf", "jupyter", "lab","--ip=0.0.0.0","--allow-root"]

# Build with:
# docker build .

# Run with:
# docker run --gpus all -it -w /mnt -v $PWD:/mnt -e HOST_PERMS="$(id -u):$(id -g)" tf:latest
# docker run --gpus all -it -w /mnt -v $PWD:/mnt tf:latest
# docker run --gpus all -it -w /mnt -v $PWD:/mnt jlab:latest

# docker run --gpus all -it -w /mnt -v $PWD:/mnt tf:latest bash
# docker run --gpus all -it -w /mnt -v $PWD:/mnt tf:latest jlab