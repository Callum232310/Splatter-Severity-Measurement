# Use NVIDIA L4T base image compatible with JetPack version on Jetson Orin Nano
FROM nvcr.io/nvidia/l4t-jetpack:r35.2.1

# Install necessary system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /miniconda.sh \
    && bash /miniconda.sh -b -p /miniconda \
    && rm /miniconda.sh

# Add Miniconda to PATH
ENV PATH=/miniconda/bin:${PATH}

# Set the working directory in the container
WORKDIR /usr/src/app

# Install Mamba for faster package installation
RUN conda install mamba -n base -c conda-forge

# Increase pip's default timeout to 10 minutes
ENV PIP_DEFAULT_TIMEOUT=600

# Install line-profiler and psutil directly using Conda
RUN mamba install -c conda-forge psutil

# Copy the environment file into the container
COPY splatterEnv4_export.yml ./

#Use the environment file to create the Conda environment
RUN mamba env create -f splatterEnv4_export.yml

#Install python and set environment path to make sure it is accessible
RUN mamba install -n splatterEnv4 python=3.8
ENV PYTHONPATH="${PYTHONPATH}:/usr/lib/python3.8/dist-packages"

#Install some packages to fix dependency errors
RUN conda install -n splatterEnv4 libffi=3.3
RUN conda install -n splatterEnv4 seaborn
RUN conda install -n splatterEnv4 -c conda-forge pycuda
RUN conda install -n splatterEnv4 -c conda-forge numpy==1.23.*
RUN conda run -n splatterEnv4 python -m pip install tensorflow-aarch64
RUN conda run -n splatterEnv4 python -m pip install typing-extensions --upgrade
RUN apt-get update && apt-get install -y libglib2.0-0
RUN apt-get install -y libglib2.0-dev


# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "splatterEnv4", "/bin/bash", "-c"]

#Use the fixed version of __init__.py
COPY __init__.py /usr/lib/python3.8/dist-packages/tensorrt/__init__.py


# Copy everything else into the container
COPY . .

# Ensure model weights are in the right directory
COPY bestcw.pt /usr/src/app/models/
COPY bestcw.engine /usr/src/app/models/

# Run API script to ensure the environment is activated
CMD ["conda", "run", "--no-capture-output", "-n", "splatterEnv4", "python", "fastAPI.py"]

# Expose the port the app runs on
EXPOSE 8000

# Define the environment variable to specify the model path (fixes an error)
ENV MODELS_DIR=/usr/src/app/models

# Define the lib directory for access to library files (fixes an error)
ENV LD_LIBRARY_PATH=/miniconda/envs/splatterEnv4/lib:$LD_LIBRARY_PATH

#Define the python path to ensure python works properly (fixes an error)
ENV PYTHONPATH=/miniconda/envs/splatterEnv4/lib/python3.8/site-packages:$PYTHONPATH


