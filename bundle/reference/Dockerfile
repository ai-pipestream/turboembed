# The reference pipeline a bundle's vectors come from: the upstream
# library, fp32 on CPU. It runs only while a bundle is made, with no
# network, on files the bundle tool fetched and hashed. The same image
# converts the F32 ONNX file to the F16 one the recipe asks for
# (onnx_f16.py), writes a PyTorch checkpoint as safetensors
# (bin_to_safetensors.py), and converts the export to a static-shape
# OpenVINO IR for the npu backend (onnx_to_openvino_ir.py), each run
# with --entrypoint python.
#
#   docker build -t turbo-reference bundle/reference
#   docker image inspect --format '{{.Id}}' turbo-reference
#
# The recipe pins the image as turbo-reference@sha256:<that id>.
#
# The base is pinned by digest (the multi-arch index of python:3.12-slim,
# Debian trixie), so a rebuild starts from the same layers.
FROM python:3.12-slim-trixie@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r /requirements.txt
COPY reference.py rerank.py qa_reference.py onnx_f16.py bin_to_safetensors.py onnx_to_openvino_ir.py /
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HOME=/tmp
ENTRYPOINT ["python", "/reference.py"]
