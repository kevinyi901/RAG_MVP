# RAG MVP Deployment Guide

1. load the dockers as needed

`docker load -i myimage.tar` per : 

    - containerfile-patched.tar 
    - vllm-openai-latest-patched.tar
    - rag-postgres.tar


2. tag if needed to ensure they're called what the application expects:

    - containerfile-patched.tar -> containerfile-patched
    - vllm-openai-latest-patched.tar -> vllm/vllm-openai:latest-patched
    - rag-postgres.tar -> pgvector/pgvector:pg16

3. run docker compose to start the app

docker compose -f containers/docker-compose.stage.yml up