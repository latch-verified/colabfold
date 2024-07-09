FROM 812206152185.dkr.ecr.us-west-2.amazonaws.com/50_wf_init_colabfold_mmseqs2_wf:v1.5.2-476ecb

COPY wf /root/wf

ARG tag
ENV FLYTE_INTERNAL_IMAGE $tag
WORKDIR /root
