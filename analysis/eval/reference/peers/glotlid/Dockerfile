# Pinned GlotLID v3 peer for the language benchmark (#1525). The base
# image, fastText-turbo and hub client are pinned. The local-only weights
# are fetched at build time only after explicit acceptance and verified
# byte for byte; runs need no network and only predictions are recorded.
FROM python@sha256:12316debbfb86890f6b2d825820612741ba527ad5a44e3902759141d572bfe8f

ARG FASTTEXT_TURBO_VERSION=0.1.0
ARG HF_HUB_VERSION=1.31.0
ARG CHECKPOINT=cis-lmu/glotlid
ARG REVISION=baa20d139151a26a71a20eddabf30d4de7329fc7
ARG MODEL_SHA256=a818b6bd42a628ab47d3dfc1578c7ea615c45381f3494c42535e31e8c4cafc9e
ARG TURBO_ACCEPT_TERMS=

RUN pip install --no-cache-dir "fasttext-turbo==${FASTTEXT_TURBO_VERSION}" \
    "huggingface_hub==${HF_HUB_VERSION}"
COPY fetch_model.py /fetch_model.py
RUN TURBO_ACCEPT_TERMS="${TURBO_ACCEPT_TERMS}" python /fetch_model.py \
    "${CHECKPOINT}" "${REVISION}" "${MODEL_SHA256}" /glotlid-model

COPY lang_predict.py /lang_predict.py
ENTRYPOINT ["python", "/lang_predict.py", "/glotlid-model/model.bin"]
