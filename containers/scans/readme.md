# container overview

4 containers from docker-compose-dev

vllm/vllm-openai       latest            aec43437a777 
pgvector/pgvector      pg16              8ed3192326bb
containers-streamlit   latest            31017dea5099
containers-api         latest            bc94bb2b097f

became 3 containers in docker-compose-staging  (after patching)

image: vllm/vllm-openai:latest-patched
image: containerfile-patched
image: pgvector/pgvector:pg16

Zero high or critical findings. 

Details below, or just skip to 'Deployment-Staging-Quickstart' to run the app

# vllm/vllm-openai 

- vllm-report.html

had 12 high, we patched, there are zero:

- vllm-patch-report.html

# pgvector/pgvector

is already ok, zero critical or high. 

- pgvector-report.html

# containers-streamlit

- containers-streamlit-report.html

has 2 high - yet in looking, they're the same CVE and are actually medium.

https://avd.aquasec.com/nvd/2026/cve-2026-0861/

This is a really arcane error and likely not worth the squeeze to mitigate.

# container-api

- containers-api-report.html

has 2 high - yet in looking, they're the same CVE and are actually medium.

https://avd.aquasec.com/nvd/2026/cve-2026-0861/

This is a really arcane error and likely not worth the squeeze to mitigate.

# let's mitigate anyway

we were able to patch the upstream python and re-build the app containers, and update docker-compose-staging to work .. verified working. 

python                  3.12.11-slim-patched
then became 
containerfile-patched

and now we have 

containers-patch-report.html

so we now have zero critical or high findings. 

zero. that's nice.

So there are zero critical or high issues in any of the 4 containers. 

QED. 
