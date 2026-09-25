# Hailo's Dataflow Compiler, which compiles a bundle's ONNX export into a
# HEF (hef_compile.py). It runs only while a bundle is made, with no
# network, on files the bundle tool fetched and hashed.
#
# The compiler's wheel is not in this repository: its licence does not
# allow redistribution. Download it from Hailo's Developer Zone (Software
# Downloads, Dataflow Compiler, Linux x86_64) into this directory, then:
#
#   docker build --build-arg WHEEL=<the wheel's file name> -t turbo-hailo-dfc bundle/hailo
#   docker image inspect --format '{{.Id}}' turbo-hailo-dfc
#
# The recipe pins the image as turbo-hailo-dfc@sha256:<that id>, in the
# HEF artifact's produced_by.container.
FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e
ARG WHEEL
RUN apt-get update \
    && apt-get install -y --no-install-recommends graphviz graphviz-dev build-essential pkg-config libgl1 cmake git \
    && rm -rf /var/lib/apt/lists/*
COPY ${WHEEL} /tmp/
RUN pip install --no-cache-dir /tmp/${WHEEL} tokenizers==0.22.2 && rm /tmp/${WHEEL}
COPY hef_compile.py /
ENV HOME=/tmp
WORKDIR /tmp
ENTRYPOINT ["python", "/hef_compile.py"]
