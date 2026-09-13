# OVMS embedding pipelines — out of scope

**Removed.** inferstream no longer fronts OpenVINO Model Server (OVMS)
over gRPC. There is no `backend = "ovms"` kind, no `crates/backend-ovms`
client, and no catalog or `config/intel.toml` path that talks to a Model
Server.

Intel embeddings are **in-process OpenVINO GenAI only**:
`backend = "openvino"` → C++ `TextEmbeddingPipeline` in
`crates/backend-openvino`. Walkthrough:
[`docs/intel-genai-embed.md`](intel-genai-embed.md).

```bash
make fetch-ov-genai
scripts/build-intel.sh
scripts/run-intel.sh --config config/intel.toml
```

A host may still run an OVMS Docker container (for example on Machine B).
inferstream does not depend on it, start it, or stop it.

Historical OpenVINO IR export (tokenizer + model XML/BIN that can be
copied into a GenAI dir) lives under `contrib/offline-once/` and is **not**
invoked by Make, CI, or `inferstream-fetch`. See that README.
