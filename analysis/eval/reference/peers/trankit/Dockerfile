# Pinned Trankit peer for the analysis benchmark (#1521). Trankit
# 1.1.2 requires torch <= 2.0.1, which has no wheels for current
# Python, so the base is a digest-pinned Python 3.10 and the rest of
# the set is pinned to era-consistent versions. The English pipeline
# (xlm-roberta-base embedding) is baked in at build time.
FROM python@sha256:6ff506466f8b1e981719b468cc0023b8ef000b983cdec0eafe119d01f379e31d

ARG TRANKIT_VERSION=1.1.2
ARG TORCH_VERSION=2.0.1+cpu

ENV HF_HOME=/hf TRANKIT_HOME=/trankit-cache
# Current pip discards era wheels over distribution-name casing in
# their metadata and falls back to sdists it cannot build; pin pip
# to the era release the set was resolvable with.
RUN pip install --no-cache-dir "pip==24.0"
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    "torch==${TORCH_VERSION}"
RUN pip install --no-cache-dir \
    "trankit==${TRANKIT_VERSION}" \
    "transformers==4.39.3" "adapters==0.2.0" "tokenizers==0.15.2" \
    "numpy==1.26.4" "huggingface_hub==0.23.4" "sentencepiece==0.2.0" "sacremoses==0.1.1" \
    "langid==1.1.6" "protobuf==4.25.3"
# Trankit downloads its models from a single university host over
# plain HTTP; where that host is unreachable, the same zip from the
# uonlp mirror on the Hub, pinned by revision, fills the cache layout
# its downloader would have produced (unzip plus the .downloaded
# marker), and the warm-up run below proves the pipeline loads it.
ARG TRANKIT_MIRROR_REVISION=1add34e3909949dbb98523ed6cfcb4e9393e2c8e
RUN python -c "\
import urllib.request, zipfile, os; \
d='/trankit-cache/xlm-roberta-base/english'; os.makedirs(d, exist_ok=True); \
url='https://huggingface.co/uonlp/trankit/resolve/${TRANKIT_MIRROR_REVISION}/models/v1.0.0/xlm-roberta-base/english.zip'; \
z=os.path.join(d,'english.zip'); urllib.request.urlretrieve(url, z); \
zipfile.ZipFile(z).extractall(d); os.remove(z); \
open(os.path.join(d,'english.downloaded'),'w').close()"
RUN python -c "from trankit import Pipeline; Pipeline('english', cache_dir='/trankit-cache', gpu=False)('One. Two.')"
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

COPY predict.py /predict.py
ENTRYPOINT ["python", "/predict.py"]
