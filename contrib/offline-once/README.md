# Offline one-offs (not part of runtime or fetch)

**Not part of runtime or fetch.** Nothing in this directory is invoked by
CI, `make test`, `make fetch-*`, or `make verify-*`. The serving binaries
and the Rust fetcher (`cargo run -p inferstream-fetch`) do not import it.

## `export_ovms_embeddings.py`

One-off OpenVINO IR export for Intel OVMS DAG pipelines. There is no
prebuilt IR to download: the graph (pooling + L2 normalize baked in) is
produced from a pinned Hugging Face revision with torch / transformers /
openvino / openvino-tokenizers.

Offline **verify** of already-exported artifacts is Rust:

```bash
make verify-embeddings-intel [ALIASES=…] [OVMS_DIR=…] [HF_TOK_DIR=…]
# or:
cargo run -p inferstream-fetch -- --ovms --all --verify-only \
    --out /work/models/ovms-embedder --hf-out "$HOME/ovms-models"
```

Export / re-pin (maintainers, Intel host with an export venv):

```bash
uv venv --python 3.12 /tmp/ov-export-venv
uv pip install --python /tmp/ov-export-venv/bin/python \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch transformers openvino openvino-tokenizers sentencepiece protobuf
/tmp/ov-export-venv/bin/python contrib/offline-once/export_ovms_embeddings.py \
    --all --out /work/models/ovms-embedder --hf-out "$HOME/ovms-models"
# re-pin hashes after a deliberate toolchain or revision bump:
/tmp/ov-export-venv/bin/python contrib/offline-once/export_ovms_embeddings.py \
    --all --update-manifest --out /work/models/ovms-embedder --hf-out "$HOME/ovms-models"
```

Then register pipelines as in `docs/adding-ovms-embedding-pipelines.md`.
