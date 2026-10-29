# Pinned lingua peer for the language benchmark (#1521, #1606). The
# base image is pinned by digest (the same Python image the other
# peers use); lingua-language-detector and its language models are
# baked in at build time so runs need no network. The peer runs as
# shipped: all 75 languages, accurate mode.
FROM python@sha256:12316debbfb86890f6b2d825820612741ba527ad5a44e3902759141d572bfe8f

ARG LINGUA_VERSION=2.2.0

RUN pip install --no-cache-dir "lingua-language-detector==${LINGUA_VERSION}"

COPY lang_predict.py /lang_predict.py
ENTRYPOINT ["python", "/lang_predict.py"]
