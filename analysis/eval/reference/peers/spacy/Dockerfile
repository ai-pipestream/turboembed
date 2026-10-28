# Pinned spaCy peer for the analysis benchmark (#1521). The base image
# is pinned by digest, spaCy and the model by exact version; the model
# wheel comes from the spaCy model releases so nothing floats.
FROM python@sha256:12316debbfb86890f6b2d825820612741ba527ad5a44e3902759141d572bfe8f

ARG SPACY_VERSION=3.8.7
ARG MODEL=en_core_web_sm
ARG MODEL_VERSION=3.8.0

# click is pinned explicitly: spaCy's CLI imports it, but current
# typer resolves to a slim build that no longer provides it.
RUN pip install --no-cache-dir "spacy==${SPACY_VERSION}" "click==8.1.8" \
    "https://github.com/explosion/spacy-models/releases/download/${MODEL}-${MODEL_VERSION}/${MODEL}-${MODEL_VERSION}-py3-none-any.whl"

COPY predict.py /predict.py
ENTRYPOINT ["python", "/predict.py"]
