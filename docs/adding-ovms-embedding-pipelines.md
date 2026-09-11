# Adding an OVMS embedding pipeline (Intel arch)

How to bring a new embedding alias live on an Intel host that fronts the
OpenVINO Model Server (OVMS). On krick-1 that server is the Docker container
`protomolt-embedder-ovms` (`openvino/model_server:latest-gpu`), which mounts
`/work/models/ovms-embedder` at `/models` and serves:

- gRPC (KServe OIP V2) on the container bridge IP, port 8000 — what the
  `ovms` backend talks to (`INFERSTREAM_OVMS_ENDPOINT`, catalog `endpoint`).
- REST on `127.0.0.1:8091` — use `GET /v1/config` to list served models and
  their states.

Only aliases with a real OVMS pipeline resolve on `intel`. As of the OVMS
expansion pass, that is every embedding alias in the catalog: minilm,
mpnet, minilm-l12, bge-small/base/large, bge-m3, e5-small/base/large,
gte-small/base, and nomic-embed-text — all as `<name>_pipeline` DAGs with
the embedding model on the Battlemage GPU (`target_device: GPU`). Never map
an alias to a pipeline that does not exist — startup will accept it but
every Embed will fail.

## Anatomy of a pipeline

Each live alias is an OVMS DAG pipeline: string input `strings` →
`tokenizer_<name>` (openvino_tokenizer, CPU) → `embedding_<name>`
(OpenVINO IR of the sentence-transformer, GPU) → output
`sentence_embedding`. The inferstream `ovms` backend depends on exactly that
contract: one string input named `strings`, one float output named
`sentence_embedding`.

On disk (host side, `/work/models/ovms-embedder/`):

```
tokenizer_<name>/1/openvino_tokenizer.{xml,bin}   # from openvino_tokenizers
embedding_<name>/1/openvino_model.{xml,bin}       # OpenVINO IR export
config-gpu.json                                   # model + pipeline registry
```

Plus a plain HuggingFace tokenizer copy for the local Tokenize/Detokenize
RPCs, outside the mount: `~/ovms-models/hf_tokenizer_<name>/tokenizer.json`.

## Steps

1. **Export + stage + hash with the committed tooling.** For every alias in
   the catalog this is one command (it exports the FP16 IR with pooling +
   L2 normalize baked in, converts the tokenizer, copies the HF
   `tokenizer.json`, and verifies SHA-256 against
   `models/manifests/ovms-embeddings.json`):

   ```bash
   make fetch-embeddings-intel ALIASES=<alias> \
       PYTHON=/path/to/export-venv/bin/python
   ```

   The venv needs torch (CPU is fine), transformers, openvino,
   openvino-tokenizers, sentencepiece — see the docstring of
   `scripts/export_ovms_embeddings.py`. For a NEW alias, add it to `SPECS`
   in that script (repo, pooling per the model card, dims), export with
   `--update-manifest`, and commit the manifest diff.

   Hard-won details the script already handles — keep them if you ever
   export by hand:

   - The graph must expose `sentence_embedding` (pooled + normalized) as an
     output; `optimum-cli --task feature-extraction` alone does NOT pool.
   - Load the torch model with `dtype=float32`. Repos that ship fp16
     weights (thenlper/gte-*) otherwise trace to an fp16-IO graph and OVMS
     answers Embed with FP16, which the inferstream backend rejects.
   - The converted tokenizer's string input must be renamed to
     `Parameter_1` (convert_tokenizer auto-numbers it, and the DAG maps the
     request's `strings` to that exact input name).
   - Architectures whose remote code cannot be torch-traced (NomicBERT's
     data-dependent rotary-cache branches) are imported from the repo's
     official ONNX export instead, with pooling grafted on via OpenVINO
     graph surgery (`source: "onnx"` in `SPECS`); their DAG additionally
     maps `token_type_ids` tokenizer → model.

2. **(covered by step 1)** — tokenizer conversion and staging are part of
   the export script; outputs must be `input_ids` and `attention_mask`
   (the DAG maps them by these names).

3. **(covered by step 1)** — files land under `/work/models/ovms-embedder/`
   (version subdirectory `1/` is required by OVMS) and
   `~/ovms-models/hf_tokenizer_<name>/`.

4. **Register in `config-gpu.json`**: add two `model_config_list` entries
   (tokenizer on `CPU`, embedding on `GPU`) and one `pipeline_config_list`
   entry named `<name>_pipeline` cloned from `minilm_pipeline` with the model
   names swapped. Validate the JSON (`jq . config-gpu.json`) before touching
   the container.

5. **Reload OVMS — no restart needed.** The container is shared (protomolt
   and other services use it); never `docker restart` it for an additive
   change. The krick-1 OVMS runs with the default 1-second config poll, so
   saving `config-gpu.json` triggers a reload on its own; you can also force
   one:

   ```bash
   curl -s -X POST http://127.0.0.1:8091/v1/config/reload
   ```

   **Gotcha:** OVMS does NOT re-read model files that changed in place
   inside an unchanged version directory (`1/`). If you re-export an
   artifact for a model OVMS has already loaded, either bump the version
   directory, or do a remove/re-add cycle: write a config without that
   model's entries, reload, restore the full config, reload again. Existing
   serving pipelines are untouched throughout. Then confirm every pipeline
   (old and new) is `AVAILABLE`:

   ```bash
   curl -s http://127.0.0.1:8091/v1/config | jq -r \
     'to_entries[] | "\(.key): \(.value.model_version_status[0].state)"'
   ```

   Confirm GPU placement from the container log (each embedding model must
   say `with target device: GPU`), and, live, from the xe DRM client
   counters — the compute engine cycles attributable to the OVMS process
   only move when it is inferring:

   ```bash
   docker logs protomolt-embedder-ovms 2>&1 | grep 'target device: GPU'
   docker exec protomolt-embedder-ovms sh -c \
     'grep -h drm-cycles-ccs /proc/1/fdinfo/*'   # sample before/after Embed
   ```

6. **Map the alias in the catalog** (`config/catalog.toml`) under
   `[models.<alias>.intel]`:

   ```toml
   [models.<alias>.intel]
   backend = "ovms"
   endpoint = "http://172.22.0.2:8000"          # container bridge IP:gRPC port
   upstream_model = "<name>_pipeline"
   tokenizer_dir = "/home/krickert/ovms-models/hf_tokenizer_<name>"
   ```

   The bridge IP can drift across container recreation; check with
   `docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}} ' protomolt-embedder-ovms`
   and override per host with `INFERSTREAM_OVMS_ENDPOINT` rather than
   hardcoding a new IP.

7. **Serve and verify.** Add the alias to `serve` in `config/intel.toml`,
   rebuild/restart `inferstream-intel`, then:

   ```bash
   scripts/smoke-embeddings.sh 127.0.0.1:8461 change-me <alias>
   ```

   Check the reported dimension against the model card (e.g. 384 for MiniLM
   family, 768 for mpnet/bge-base, 1024 for bge-large/e5-large) and compare a
   couple of vectors against the sentence-transformers reference if possible
   (`testdata/reference_embeddings/README.md`).

## Why this is not done casually

Registering a pipeline means restarting or reloading a shared production
container and exporting new IR artifacts to the GPU. A bad export (wrong
output name, missing pooling) fails only at inference time, and a malformed
`config-gpu.json` can take down the existing minilm/mpnet pipelines. Do it
deliberately, one model at a time, re-running the smoke for all served
aliases afterwards.
