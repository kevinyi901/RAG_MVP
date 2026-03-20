# container overview

4 containers: 

vllm/vllm-openai       latest            aec43437a777 
pgvector/pgvector      pg16              8ed3192326bb
containers-streamlit   latest            31017dea5099
containers-api         latest            bc94bb2b097f



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

so we now have

containers-patch-report.html

zero. that's nice.

So there are zero critical or high issues in any of the 4 containers. 

QED. 
