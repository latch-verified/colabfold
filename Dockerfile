FROM 812206152185.dkr.ecr.us-west-2.amazonaws.com/latch-base-cuda:cb86-main

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
RUN conda create --name alphafold python==3.10
SHELL ["conda", "run", "-n", "alphafold", "/bin/bash", "-c"]

RUN conda update -n base conda -y
RUN conda install -c conda-forge python=3.10 openmm==7.7.0 pdbfixer -y

RUN conda install -c conda-forge -c bioconda kalign2=2.04 hhsuite=3.3.0 mmseqs2=14.7e284 -y
RUN pip install --upgrade pip
RUN pip install --no-warn-conflicts "colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold" tensorflow==2.12.0
RUN pip install https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.4.14+cuda11.cudnn86-cp310-cp310-manylinux2014_x86_64.whl
RUN pip install jax==0.4.14 chex==0.1.6 biopython==1.79

# mmseqs2
RUN git clone https://github.com/soedinglab/MMseqs2.git && \
    cd MMseqs2 && mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=. .. && \
    make && make install

# ColabFold
RUN pip install --upgrade latch
COPY ColabFold /root/ColabFold
RUN pip install -e /root/ColabFold

ENV PATH="/root/MMseqs2/build/bin:/root/miniconda3/envs/alphafold/bin:/root/miniconda3/bin:${PATH}"

RUN pip uninstall protobuf -y
RUN pip install --no-binary protobuf protobuf==3.20

ENV TF_FORCE_UNIFIED_MEMORY="1"
ENV XLA_PYTHON_CLIENT_MEM_FRACTION="4.0"

COPY wf /root/wf
ARG tag
ENV FLYTE_INTERNAL_IMAGE $tag
WORKDIR /root
