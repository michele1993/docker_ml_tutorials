FROM nvidia/cuda:11.0.3-base-ubuntu20.04

RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "docker_nvidia", "/bin/bash", "-c"]

CMD ["python", "-c", "import torch; torch.cuda.current_device()"] 
