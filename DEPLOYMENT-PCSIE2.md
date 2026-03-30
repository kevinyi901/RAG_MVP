# RAG MVP Deployment Guide

on g5 instance, run docker compose to start the app

`docker compose -f containers/docker-compose.pcsie2.yml up`


### models

the '/models' path contains models that must be hydrated from a separate 'models' container that is not used in docker compose. 

[insert PCSIE2 path]

### cache

also, there are some cached tokens we must pull

curl -o o200k_base.tiktoken https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken
curl -o cl100k_base.tiktoken https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken

and add to models/cache/tiktoken/

these are then refered to in env var

TIKTOKEN_ENCODINGS_BASE=/models/cache/tiktoken/

## dev note: 

this is 'airgap lite' - as in, the containers (and models) in PCSIE are commits from a running instance on bare metal RHEL8 instance, as the 'scripts/download_for_airgap.sh' is failing. 