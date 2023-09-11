FROM 812206152185.dkr.ecr.us-west-2.amazonaws.com/latch-base-cuda:9c8f-main

# conda envs
ENV CONDA_PREFIX="/root/miniconda3/envs/alphafold"
ENV CONDA_DEFAULT_ENV="alphafold"
ENV PATH="/root/miniconda3/envs/alphafold/bin:/root/miniconda3/bin:${PATH}"

# miniconda3 instllation
RUN apt-get update && apt install --upgrade \
    && apt install aria2 cmake wget curl libfontconfig1-dev git -y\
    && curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && conda init bash

# activate for following run commands
RUN conda create --name alphafold python==3.9
SHELL ["conda", "run", "-n", "alphafold", "/bin/bash", "-c"]

RUN conda install -c conda-forge cudatoolkit=11.4

# patch cuda libcusolver for cuda 11.4 -- nvidia bug wants to find deprecated library solver
RUN apt-key del 7fa2af80
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && dpkg -i cuda-keyring_1.0-1_all.deb
RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-get update && apt-get install libcusolver-11-0 -y
RUN ln /usr/local/cuda-11.0/lib64/libcusolver.so.10 /usr/local/cuda-11.0/lib64/libcusolver.so.11
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda-11.0/lib64"

# mmseqs2
RUN git clone https://github.com/soedinglab/MMseqs2.git && \
    cd MMseqs2 && mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=. .. && \
    make && make install && \
    export PATH=$(pwd)/bin/:$PATH

# ColabFold
RUN pip install latch
RUN pip install alphafold-colabfold
RUN pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install --upgrade jaxlib
RUN conda install -c conda-forge -c bioconda kalign2=2.04 hhsuite=3.3.0 openmm=7.5.1 pdbfixer numpy=1.22.4

RUN pip install --upgrade latch
COPY ColabFold /root/ColabFold
RUN pip install -e /root/ColabFold

ENV PATH="/root/MMseqs2/build/bin:/root/miniconda3/envs/alphafold/bin:/root/miniconda3/bin:${PATH}"

RUN pip uninstall protobuf -y
RUN pip install --no-binary protobuf protobuf

ENV TF_FORCE_UNIFIED_MEMORY="1"
ENV XLA_PYTHON_CLIENT_MEM_FRACTION="4.0"

COPY wf /root/wf
ARG tag
ENV FLYTE_INTERNAL_IMAGE $tag
WORKDIR /root

