# Docker for Deep Learning
In this repository, I include a series of tutorials on how to deploy ML algorithms within Docker. 

*Key points for best practice**

1. I think whenever possible use the [nvidia/cuda](https://hub.docker.com/r/nvidia/cuda) image if you need to run PyTorch with GPUs within a container.

## Tut.1: **download_data_container_tutorial**, simple mnist AE with docker
In this first tutorial, I show how to use Docker to train a simple Vanilla auto-encoder (AE) on the mnist dataset. 
The purpose of this tutorial is twofold **1)** Show how to build a docker container with a ML dataset (mnist) directly inside it, rather than downloading the data everytime I run the container (slow). **2)** Show how to access on your local machine any data that has been saved on the container (e.g., by using `torch.save`). 

- There are two main ways to build a docker container with a dataset inside:

1. (used in this tutorial) Create a `get_data` function separate from the main training process. This allow you to run this function inside the `Dockerfile` so that the data is downloaded directly inside the container when you built it. Insert the following code inside the Dockerfile to run a `get_data` function inside a `get_mnist.py` file (which dowloads the data) when building the cotainer:
```
# Set environment variable, used below to determine the dir where mnist data is dowloaded in the container
ENV DATA_PATH="/data"

RUN python -c "from get_mnist import get_data; get_data('${DATA_PATH}')"
```

2. Dowload the dataset on you local machine and load the repository with the dataset onto the container when building it. It is best to do this with `VOLUME` since this creates a mount point which is tagged as holding externally mounted volume from the local host or other containers, thus keeping track of any local changes to the dataset. 
```
VOLUME /Users/LOCAL_PATH_TO_DATASET
```

- To access any data that has been saved on the container from your local machine, you can use volumes. The most straight forward way to do this is just to make your local code dir as a volume when running the container (e.g., `docker run --rm -v /$(pwd):/app ae_v1`), in this way any change that the container makes to this dir (e.g., by saving a file/dir) will be reflected on your local machine `pwd` dir. This is also useful when you want to edit any file locally and reflect the changes in the container without re-mounting the image everytime.

**Note**: I also include two docker files to show two approaches to run applications in Docker, **1)** `Dockerfile` which takes the pytorch image directly from Docker Hub. **2)** `Dockerfile_pip` which import the python base image and then install pytorch using pip and a `requirements.txt` file. 


## Tut.2: Docker with conda
In this second tutorial, I show how to use Docker to train the same Vanilla auto-encoder (AE) on the mnist dataset as the first tutorial, but using a conda environemnt inside the container (recommended approach).

**NOTE** I first install `pip` inside my conda env and then I use this verison of `pip` to install the python dependecies (e.g., torch etc.) via a `requirements.txt` file. 
I tried also installing the python dependencies directly via `conda`, however, I am encountering some issues of the packages not being found (I suspect this is to do with my mac M1 for which Conda does not provide the packages).
This doesn't really matter as long as I use the pip verison installed in the conda environemnt and NOT! the system `pip` (i.e., by installing pip inside the conda env).


## Tut.3: **docker_nvidia**:
In this tutorial I run PyTorch with CUDA inside a container, while creating a conda environment.
To do so optimally, I take the official nvidia container image, which should ensure everything is optimized.
This requires (?) installing the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
To install it, I followed this [tutorial](https://medium.com/@u.mele.coding/a-beginners-guide-to-nvidia-container-toolkit-on-docker-92b645f92006).
After that and based on NVIDIA official configurations, I run,
```
# Configure the container runtime by using the nvidia-ctk command
sudo nvidia-ctk runtime configure --runtime=docker
# Restart the Docker daemon:
sudo systemctl restart docker
```

