# Pinned wtpsplit (SaT) peer for the analysis benchmark (#1521). The
# base image is pinned by digest; torch (CPU build), wtpsplit, and the
# sat-3l weights are baked in at build time so runs need no network.
FROM python@sha256:12316debbfb86890f6b2d825820612741ba527ad5a44e3902759141d572bfe8f

ARG WTPSPLIT_VERSION=2.2.2
ARG TORCH_VERSION=2.6.0

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    "torch==${TORCH_VERSION}"
RUN pip install --no-cache-dir "wtpsplit==${WTPSPLIT_VERSION}"
RUN python -c "from wtpsplit import SaT; SaT('sat-3l').split('One. Two.')"
ENV HF_HUB_OFFLINE=1

COPY predict.py /predict.py
ENTRYPOINT ["python", "/predict.py"]
