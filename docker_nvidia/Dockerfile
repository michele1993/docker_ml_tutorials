ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=11.8.0
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION}

# Set environment variables for non-interactive installation
# This ensures you will not be prompted for inputs during installation
ENV DEBIAN_FRONTEND=noninteractive

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities.
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*


# Create a working directory.
RUN mkdir /app
WORKDIR /app


# Install conda
# Set up the Conda environment
ENV CONDA_AUTO_UPDATE_CONDA=false 
ENV PATH=/opt/conda/bin:$PATH

COPY environment.yml /app/environment.yml
COPY requirements.txt /app/requirements.txt

RUN curl -sLo ~/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
 && chmod +x ~/miniforge.sh \
 && ~/miniforge.sh -b -p /opt/conda \
 && rm ~/miniforge.sh \
 && conda env create -f environment.yml \
 && rm /app/environment.yml /app/requirements.txt\
 && conda clean -ya

#SHELL ["conda", "run", "-n", "docker_nvidia", "/bin/bash", "-c"]
# Demonstrate the environment is working by loading a shipped Conda package
#RUN echo "Make sure torch is installed:"
#RUN python -c "import torch"
#ENTRYPOINT ["/bin/bash", "-c", "ls -la /opt/conda/condabin > /app/directory_contents.txt"] 


# Run python script within the conda env
#ENTRYPOINT ["conda", "run", "-n","docker_nvidia",  "python"]
# Deafult shows if cuda avilable if no input provide
CMD ["conda", "run", "-n","docker_nvidia",  "python", "-c", "import torch; print(torch.cuda.current_device())"] 
