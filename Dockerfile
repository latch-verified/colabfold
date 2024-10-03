# DO NOT CHANGE
from 812206152185.dkr.ecr.us-west-2.amazonaws.com/latch-base:fe0b-main

workdir /tmp/docker-build/work/

shell [ \
    "/usr/bin/env", "bash", \
    "-o", "errexit", \
    "-o", "pipefail", \
    "-o", "nounset", \
    "-o", "verbose", \
    "-o", "errtrace", \
    "-O", "inherit_errexit", \
    "-O", "shift_verbose", \
    "-c" \
]
env TZ='Etc/UTC'
env LANG='en_US.UTF-8'

COPY ColabFold /root/ColabFold
COPY alphafold /root/alphafold

WORKDIR /root/ColabFold
RUN pip install -e .
WORKDIR /root/alphafold
RUN pip install -e .
WORKDIR /root/ColabFold
RUN pip install -e .
RUN pip install -q "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

ARG tag
ENV FLYTE_INTERNAL_IMAGE $tag
WORKDIR /root


# Latch SDK
# DO NOT REMOVE
run pip install latch==2.52.3
run mkdir /opt/latch


# Copy workflow data (use .dockerignore to skip files)

copy . /root/


# Latch workflow registration metadata
# DO NOT CHANGE
arg tag
# DO NOT CHANGE
env FLYTE_INTERNAL_IMAGE $tag

workdir /root
