# Import miniconda parent image
FROM continuumio/miniconda3

# Set environment variable, used below to determine the dir where mnist data is dowloaded in the container
#NOTE the data is not saved in the working directory (/app) this may be helpful because if we mount our local working dir
# to the container working dir /app with volumes we don't want the data onto our local working dir.
ENV DATA_PATH="/data"

# Set the container working directory 
WORKDIR /app

# Copy current directory onto container working dir
COPY . .

# Create the Conda env
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "docker_conda_tutorial", "/bin/bash", "-c"]
# Demonstrate the environment is working by loading a shipped Conda package
RUN echo "Make sure torch is installed:"
RUN python -c "import torch"

# Run the get_data script to dowload data onto container specified dir
# This should throw an error if env is not activated since pytorch shouldn't be present
RUN python -c "from get_mnist import get_data; get_data('${DATA_PATH}')"

# # The code to run when container is started:
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "docker_conda_tutorial", "python", "main.py"]

