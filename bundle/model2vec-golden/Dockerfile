# Model2Vec's own ids and vectors for a model, to hold turbo's static
# models against (golden.py, core/tests/static_parity.rs, docs/static.md).
# A test tool only: no bundle is made with it.
#
#   docker build -t turbo-model2vec-golden bundle/model2vec-golden
FROM python:3.12-slim
# Model2Vec installs from its git repository: setuptools_scm finds its
# version and its subpackages from the git metadata, which an archive lacks.
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
COPY golden.py /
ENV HF_HUB_OFFLINE=1 HOME=/tmp
ENTRYPOINT ["python", "/golden.py"]
