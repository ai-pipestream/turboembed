# Pinned spaCy transformer peer for the analysis benchmark (#1521):
# en_core_web_trf needs spacy-transformers and torch, so this is the
# spaCy peer container plus a CPU torch pin. The base image is pinned
# by digest; spaCy, torch, and the model by exact version.
FROM python@sha256:12316debbfb86890f6b2d825820612741ba527ad5a44e3902759141d572bfe8f

ARG SPACY_VERSION=3.8.7
ARG MODEL=en_core_web_trf
ARG MODEL_VERSION=3.8.0
ARG TORCH_VERSION=2.6.0

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    "torch==${TORCH_VERSION}"
# click is pinned explicitly: spaCy's CLI imports it, but current
# typer resolves to a slim build that no longer provides it.
RUN pip install --no-cache-dir "spacy==${SPACY_VERSION}" "click==8.1.8" \
    "https://github.com/explosion/spacy-models/releases/download/${MODEL}-${MODEL_VERSION}/${MODEL}-${MODEL_VERSION}-py3-none-any.whl"

COPY predict.py /predict.py
ENTRYPOINT ["python", "/predict.py"]
