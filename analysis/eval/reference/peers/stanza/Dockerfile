# Pinned Stanza peer for the analysis benchmark (#1521). The base image
# is pinned by digest; torch (CPU build), Stanza, and the default
# models for STANZA_LANG are baked in at build time so runs need no
# network. The peer runs as shipped: each language's default
# packages, no treebank override.
FROM python@sha256:12316debbfb86890f6b2d825820612741ba527ad5a44e3902759141d572bfe8f

ARG STANZA_VERSION=1.10.1
ARG TORCH_VERSION=2.6.0
ARG STANZA_LANG=en

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    "torch==${TORCH_VERSION}"
RUN pip install --no-cache-dir "stanza==${STANZA_VERSION}"
RUN python -c "import stanza; stanza.download('${STANZA_LANG}', model_dir='/stanza', processors='tokenize,mwt,pos,lemma,depparse,ner', verbose=False)"

COPY predict.py /predict.py
COPY ner_predict.py /ner_predict.py
ENTRYPOINT ["python", "/predict.py"]
