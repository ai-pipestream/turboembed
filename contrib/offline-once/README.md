# Offline one-offs (historical IR tooling)

**Not part of runtime or fetch.** Nothing in this directory is invoked by
CI, `make test`, `make fetch-*`, or `make verify-*`. The serving binaries
and the Rust fetcher (`cargo run -p inferstream-fetch`) do not import it.

This is **historical OpenVINO IR export** — useful only when a GenAI
directory needs `openvino_tokenizer.xml/.bin` (or a model IR) that Hugging
Face does not publish. inferstream does **not** serve OVMS. Do not add
`backend = "ovms"`; Intel embeds are in-process GenAI
(`docs/intel-genai-embed.md`).

The hash pin for the old DAG layout is `models/manifests/ovms-embeddings.json`
(not wired to `inferstream-fetch` or Make). Copy tokenizer IR into
`models/ov/<alias>/` next to `openvino_model.xml` if a catalog alias lacks
a public tokenizer pair.

## `export_ovms_embeddings.py`

One-off OpenVINO IR export (torch / transformers / openvino /
openvino-tokenizers). **No Python on the inferstream path** — this script
is maintainer-only, on an Intel host with an export venv, when you need
IR that `make fetch-ov-genai` cannot download.

```bash
uv venv --python 3.12 /tmp/ov-export-venv
uv pip install --python /tmp/ov-export-venv/bin/python \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch transformers openvino openvino-tokenizers sentencepiece protobuf
/tmp/ov-export-venv/bin/python contrib/offline-once/export_ovms_embeddings.py \
    --all --out /tmp/ov-ir --hf-out /tmp/ov-hf
```

Then flatten the model + tokenizer XML/BIN into `models/ov/<alias>/` in
the GenAI layout (`openvino_model.xml`, `openvino_tokenizer.xml`,
`tokenizer.json`). See `docs/intel-genai-embed.md`.
