# Security Scan

These are notes & logs as we do security scanning for this repo. 

per

https://www.trmc.osd.mil/bitbucket/projects/JMIE/repos/data-dept/browse/docker/security.md

`pip install trivy`

or use docker: 

`docker run aquasec/trivy m-int`

or do local rhel

`rpm -ivh https://github.com/aquasecurity/trivy/releases/download/v0.50.4/trivy_0.50.4_Linux-64bit.rpm`

for html output: 

`wget https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/html.tpl`

then 

`docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v $(pwd):/host aquasec/trivy image --format template --severity HIGH,CRITICAL -o /host/containers-streamlit.html containers-streamlit`

or more legibly: 

    docker run --rm \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v $HOME/trivy-cache:/root/.cache/ \
    -v $(pwd):/host \
    aquasec/trivy:latest image \
    --severity HIGH,CRITICAL \
    --format template \
    --template "@contrib/html.tpl" \
    --output /host/localhost/rag-vllm  \
    localhost/rag-vllm 

for each of the containers : 

    containers-api           latest                    1fa8c8c08977   27 hours ago    6.6GB
    containers-streamlit     latest                    44c9cde59d94   27 hours ago    6.6GB
    pgvector/pgvector        pg16                      4c0c0efbd40e   10 days ago     438MB
    localhost/rag-postgres   latest                    4c0c0efbd40e   10 days ago     438MB
    aquasec/trivy            latest                    e0b9ad5c73e3   2 weeks ago     184MB
    vllm/vllm-openai         latest                    8f7f6d447794   2 weeks ago     20.1GB
    localhost/rag-vllm       latest  


all worked - some are redundant. 

as for the last one - 

localhost/rag-vllm sitll needs scanning - i keep getting a 'no space left on device' thought I have plenty of space. 

`docker system prune -a`

re-running... 


# RHEL TRIVY SCAN TIPS


### Basic high-priority scan
trivy image --severity HIGH,CRITICAL --ignore-unfixed [IMAGE_NAME]

### Scan using an exception file
trivy image --ignore-unfixed --severity HIGH,CRITICAL --ignorefile .trivyignore [IMAGE_NAME]



## NB: 

Trivy maps Red Hat's "Impact" metric to severity. If a CVE is high in NVD but low in Red Hat, Trivy reflects the lower, more accurate rating.

### Key Workflow Actions to Reduce Noise 

- Filter Unfixed Vulnerabilities: Use --ignore-unfixed in CI/CD to ignore CVEs that lack a Red Hat patch.
- Focus on Severity: Run scans with --severity HIGH,CRITICAL to focus on actionable risks.
- Manage Exceptions: Create a .trivyignore file or use --ignore-policy (Rego) to document and ignore false positives or accepted risks.
- Use Precise Detection: Implement --detection-priority precise to reduce false positives by trusting OS vendor advisories over upstream sources.
- Optimize Base Image: Use minimal RHEL base images (e.g., UBI-minimal) to reduce the initial package count.
- Scan Frequently: Integrate into PR workflows to catch issues early, rather than relying on daily container registry scans.



# actually mitigating the high and critical findings

I'm used to doing this in Ubunutu, RHEL is different enough I need to go back to first principles to find the best way....

