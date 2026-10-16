# The reference SaT implementation (wtpsplit, MIT) on PyTorch for the CPU,
# for the fixtures analysis/sat compares with. It runs only while the
# fixtures are made, with no network, on files already on disk.
#
#   docker build -t turbo-sat-reference analysis/sat/reference
#
# The base image is named by its digest; conda.lock names every conda-forge
# package by URL and MD5 and requirements.txt every package from PyPI by
# SHA-256. analysis/sat/README.md says how the fixtures are made.
FROM mambaorg/micromamba@sha256:ce2026639d612fbe837a35ca9d513292a8d776a24a5e6c82cccf9dddf45ad3d2

COPY conda.lock requirements.txt /env/
RUN micromamba install -y -n base -f /env/conda.lock \
 && micromamba clean -a -y
RUN /opt/conda/bin/python -m pip install --no-cache-dir --no-deps --no-build-isolation --require-hashes -r /env/requirements.txt
COPY sat_reference.py /sat_reference.py
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HOME=/tmp
ENTRYPOINT ["/opt/conda/bin/python", "/sat_reference.py"]
