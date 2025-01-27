# Use the official pytorch image from the Docker Hub without using pip
FROM pytorch/pytorch

# Set environment variable, used below to determine the dir where mnist data is dowloaded in the container
#NOTE the data is not saved in the working directory (/app) this may be helpful because if we mount our local working dir
# to the container working dir /app with volumes we don't want the data onto our local working dir.
ENV DATA_PATH="/data" 

# Set the container working directory 
WORKDIR /app

# Copy current directory onto container working dir
COPY . .

# Run the get_data script to dowload data onto container specified dir
RUN python -c "from get_mnist import get_data; get_data('${DATA_PATH}')"

# Run main.py when the container launches
CMD ["python", "main.py"]
