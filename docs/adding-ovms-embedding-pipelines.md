# Adding an OVMS embedding pipeline (Intel arch)

How to bring a new embedding alias live on an Intel host that fronts the
OpenVINO Model Server (OVMS). On krick-1 that server is the Docker container
`protomolt-embedder-ovms` (`openvino/model_server:latest-gpu`), which mounts
`/work/models/ovms-embedder` at `/models` and serves:

- gRPC (KServe OIP V2) on the container bridge IP, port 8000 — what the
  `ovms` backend talks to (`INFERSTREAM_OVMS_ENDPOINT`, catalog `endpoint`).
- REST on `127.0.0.1:8091` — use `GET /v1/config` to list served models and
  their states.

Only aliases with a real OVMS pipeline resolve on `intel` (currently
`minilm` → `minilm_pipeline` and `mpnet` → `mpnet_pipeline`). Never map an
alias to a pipeline that does not exist — startup will accept it but every
Embed will fail.

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

1. **Export the model to OpenVINO IR.** From a venv with `optimum-intel`
   (krick-1 has `~/ovms-venv`):

   ```bash
   optimum-cli export openvino \
     --model sentence-transformers/<model> --task feature-extraction \
     /tmp/ov-export/<name>
   ```

   If the export does not already pool to a single vector, the embedding
   model must expose a `sentence_embedding` output (mean pooling +
   normalization inside the graph, as the existing minilm/mpnet exports do).
   Verify with `ovc`/netron that the output name matches; the pipeline output
   alias in `config-gpu.json` must be `sentence_embedding`.

2. **Convert the tokenizer** with `openvino_tokenizers`
   (`convert_tokenizer <model> --with-detokenizer -o ...`), producing
   `openvino_tokenizer.{xml,bin}`. Its outputs must be `input_ids` and
   `attention_mask` (the DAG maps them by these names).

3. **Stage the files** under `/work/models/ovms-embedder/` following the
   layout above (version subdirectory `1/` is required by OVMS). Copy the HF
   `tokenizer.json` to `~/ovms-models/hf_tokenizer_<name>/`.

4. **Register in `config-gpu.json`**: add two `model_config_list` entries
   (tokenizer on `CPU`, embedding on `GPU`) and one `pipeline_config_list`
   entry named `<name>_pipeline` cloned from `minilm_pipeline` with the model
   names swapped. Validate the JSON (`jq . config-gpu.json`) before touching
   the container.

5. **Reload OVMS.** The container is shared (protomolt and other services use
   it), so prefer a config reload over a restart if the running OVMS was
   started with `--file_system_poll_wait_seconds` > 0; otherwise restart
   during a quiet window: `docker restart protomolt-embedder-ovms`, then
   confirm every pipeline (old and new) is `AVAILABLE`:

   ```bash
   curl -s http://127.0.0.1:8091/v1/config | jq -r \
     'to_entries[] | "\(.key): \(.value.model_version_status[0].state)"'
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
