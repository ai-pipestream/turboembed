# inferstream-intel: e2e parity goldens on GPU (Machine B)

Live **OpenVINO GenAI** Embed (`minilm` / `bge-small` / `mpnet`) on Intel
Battlemage, captured with `make e2e-parity-goldens TARGET=intel WRITE=1`
from Origin `main` at `2b8883c`. **No Python** on serve or capture.

Host: **Machine B**, GPU `Intel(R) Graphics [0xe223]`. Listen
`127.0.0.1:8473` (`config/intel.toml` / catalog façade). Unique commits
lived on leftover `ai-pipestream/intel-embed-parity-b22a`; dumps below
are the same blobs.

## GPU GenAI proof (capture host)

| | |
|---|---|
| binary | `target/release/inferstream-intel` (features `llamacpp-sycl,openvino-genai`) |
| cmdline | `--listen 127.0.0.1:8473` (no OVMS, no Python) |
| ldd | `libopenvino.so.2631`, `libopenvino_genai.so.2631`, `libsycl.so.8`. **No `libpython`.** |
| `ListModels minilm` | `backend=openvino` `platform=openvino_genai` `device=GPU` `dim=384` |

`scripts/prove-intel-genai.sh 127.0.0.1:8473 change-me <pid> minilm` —
Embed 384-d FP32, no `libpython`, GPU plugin mapped.

Live replay against today's TurboEmbed façade still needs **Machine B**
(`make test-turboembed-intel` + `make e2e-parity-goldens TARGET=intel`).

## Capture

```
INFERSTREAM_E2E_INTEL_ADDR=127.0.0.1:8473 \
  make e2e-parity-goldens TARGET=intel WRITE=1
```

| alias | dim | items | path |
|---|---|---|---|
| minilm | 384 | 213 | `testdata/e2e/goldens/intel/minilm.json` |
| bge-small | 384 | 213 | `testdata/e2e/goldens/intel/bge-small.json` |
| mpnet | 768 | 213 | `testdata/e2e/goldens/intel/mpnet.json` |

Pooling: MiniLM / MPNet **mean**, BGE-small **CLS**. `normalize=true`.

Same-arch live-vs-dump (`WRITE=0`) on the capture host: **min=1.0000
mean=1.0000** on all three aliases (213/213).

## nvidia ↔ intel (dump-vs-dump, no GPU)

Machine A nvidia dumps (`5eed879`, still the blobs on `main`) vs these
Intel dumps. Intersection is 213 ids (`FETCH_CORPUS=0` on Intel).

```
make e2e-parity \
  DUMP_NVIDIA=testdata/e2e/goldens/nvidia \
  DUMP_INTEL=testdata/e2e/goldens/intel
```

| pair | n | worst-pair cosine | worst id | mean |
|---|---|---|---|---|
| minilm | 213 | **0.999999310862** | `tiny-shakespeare-excerpt:p0003:s0001` (`resolved.`) | 0.999999857402 |
| bge-small | 213 | **0.999998043062** | `sts-0004:a` | 0.999998818175 |
| mpnet | — | skipped | nvidia dump absent | — |

`parity:short` (`hello world`) minilm: **0.999999724032**.
Committed ORT CUDA short golden vs intel `parity:short`: **0.999999750917**.

Public `make fetch-ov-genai` still omits Intel `bge-small` / `mpnet`
tokenizer IRs. Live intel cases skip those aliases until Machine B lists
them; the committed dumps are for dump-vs-dump only.
