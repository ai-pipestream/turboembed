# Pinned GLiNER peer for the entity benchmark (#1606). The base image
# is pinned by digest; torch (CPU build), the gliner package, and the
# checkpoint files at the zoo-pinned revision are baked in at build
# time so runs need no network. The checkpoint is local-only by its
# training data (docs/licences.md): this image is a local peer runner,
# never shipped, and only its predictions are recorded. Its fetch is
# gated on explicit acceptance of the dataset card's terms: build
# with --build-arg TURBO_ACCEPT_TERMS=urchade/gliner_small-v2.1
# (fetch_checkpoint.py refuses, before any download, without it).
FROM python@sha256:12316debbfb86890f6b2d825820612741ba527ad5a44e3902759141d572bfe8f

ARG GLINER_VERSION=0.2.21
ARG TORCH_VERSION=2.6.0
# transformers and huggingface_hub are pinned to the releases current
# when gliner 0.2.21 was published; transformers 5 breaks its
# from_pretrained signature.
ARG TRANSFORMERS_VERSION=4.46.3
ARG HF_HUB_VERSION=0.26.5
ARG CHECKPOINT=urchade/gliner_small-v2.1
ARG REVISION=4e091416cf7c3481db542c2a3d26156916f3a47f
# Empty by default: the checkpoint download below refuses until the
# builder records acceptance by naming the checkpoint (see header).
ARG TURBO_ACCEPT_TERMS=

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    "torch==${TORCH_VERSION}"
RUN pip install --no-cache-dir "gliner==${GLINER_VERSION}" \
    "transformers==${TRANSFORMERS_VERSION}" "huggingface_hub==${HF_HUB_VERSION}"
COPY fetch_checkpoint.py /fetch_checkpoint.py
RUN TURBO_ACCEPT_TERMS="${TURBO_ACCEPT_TERMS}" python /fetch_checkpoint.py "${CHECKPOINT}" "${REVISION}" /gliner-model
# The checkpoint's gliner_config.json names microsoft/deberta-v3-small
# as its encoder and tokenizer; bake it at the licence-checked revision
# and point the config's copy at the local directory, so no run
# reaches the hub. Only its config and tokenizer files are read: the
# encoder weights come from the GLiNER checkpoint itself.
ARG TOKENIZER=microsoft/deberta-v3-small
ARG TOKENIZER_REVISION=a36c739020e01763fe789b4b85e2df55d6180012
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${TOKENIZER}', revision='${TOKENIZER_REVISION}', local_dir='/tokenizer', allow_patterns=['config.json', 'spm.model', 'tokenizer_config.json', 'special_tokens_map.json'])"
RUN python -c "import json; p='/gliner-model/gliner_config.json'; c=json.load(open(p)); c['model_name']='/tokenizer'; json.dump(c, open(p,'w'), indent=2)"

COPY ner_predict.py /ner_predict.py
ENTRYPOINT ["python", "/ner_predict.py"]
