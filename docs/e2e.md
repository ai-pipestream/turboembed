# Unified E2E harness

One client, one suite, any of the three arch gRPC servers. The harness is
`inferstream-e2e` (`crates/e2e`) — a Rust gRPC client. It does **not** start
GPU servers. Point it at a host that is already serving.

**Missing weights:** `make e2e-nvidia` (and intel / apple) default to
`FETCH=1`. The harness verifies SHA-256 against the committed manifests and
downloads only files that are missing or mismatched, then runs the suite.
`make e2e-mock` / CI do **not** fetch. No Python — NVIDIA/Intel go through
`inferstream-fetch`; Apple MLX through `cargo xtask fetch --mlx`. See
[fetching-models.md](fetching-models.md).

The same cases run against nvidia / intel / apple:

| case | what it asserts |
|---|---|
| `server-live` / `list-models` | `ServerLive.live`, logical names present, `minilm` ready |
| `embed:<alias>` | non-empty vectors; `minilm` dim 384 (required); others skip if not served |
| `tokenize:<alias>` | Tokenize + Detokenize on `minilm` and the first served LLM |
| `generate:<alias>` | short `ModelStreamInfer` for `default-llm` / `qwen-0.5b` / `qwen-7b` — non-empty tokens, `final=true` |

Skip vs fail:

- **Required** (`minilm`): missing from `ListModels` or a failed RPC is a hard fail.
- **Catalog `NotAvailableOnArch`**: alias has no `[models.<alias>.<arch>]` row (e.g. `mpnet` on apple) **and** the host is not serving it → soft skip with that reason.
- **Not served**: catalog supports it but this host did not put it on `serve` → soft skip.
- A served alias is always exercised, even if the catalog omits it for that arch.

Optional cosine check: if `testdata/e2e/goldens/<arch>/<alias>.json` exists
(same schema as `testdata/reference_embeddings/`), the first vector must
cosine-match at ≥ 0.99 (override with `--cosine-min`). Missing goldens are
not an error.

## Build / run

```bash
# Against a live worker (server already up). FETCH=1 is the default:
make e2e-nvidia          # default addr krick:8461; fetch minilm + 0.5B GGUF if missing
make e2e-intel           # default addr krick-1:8461; fetch ov-genai minilm + GGUF
make e2e-apple           # default addr krickert-mac:8461; cargo xtask fetch --mlx

# Skip download (weights already on disk, or CI):
make e2e-nvidia FETCH=0

# Widen the set (qwen-7b is ~5.1 GiB — not in the default set):
make e2e-nvidia FETCH=all              # every matrix alias on this arch
make e2e-intel FETCH=serve             # config/intel.toml serve list
make fetch-e2e-nvidia                  # ensure artifacts only; no gRPC

# Or the binary directly:
cargo run -p inferstream-e2e -- --target nvidia --addr krick:8461 --token "$KEY" --fetch
cargo run -p inferstream-e2e -- --target nvidia --fetch-only
scripts/ensure-models.sh nvidia        # thin wrapper: cargo + --fetch-only
```

Environment / flags (equivalent):

| flag | env | default |
|---|---|---|
| `--target` | `INFERSTREAM_E2E_TARGET` | inferred from `--addr` (`krick` / `krick-1` / `krickert-mac` / localhost) |
| `--addr` | `INFERSTREAM_E2E_ADDR` | per-target live worker |
| `--token` | `INFERSTREAM_E2E_TOKEN` | `change-me` (empty string = no auth) |
| `--matrix` | `INFERSTREAM_E2E_MATRIX` | built-in `testdata/e2e/matrix.json` |
| `--goldens` | `INFERSTREAM_E2E_GOLDENS` | `testdata/e2e/goldens/` if present |
| `--only minilm,mpnet` | | all matrix aliases |
| `--suite all\|list\|embed\|tokenize\|generate` | | `all` |
| `--fetch` | `INFERSTREAM_E2E_FETCH` | off (`make e2e-{nvidia,intel,apple}` passes it) |
| `--fetch-only` | | ensure artifacts, then exit |
| `--fetch-all` | | with `--fetch`: every matrix alias on this target |
| `--fetch-serve` | | with `--fetch`: `config/<arch>.toml` `serve` |

### What `--fetch` downloads (per target)

Idempotent: `inferstream-fetch` / `xtask` verify SHA-256 if the file is
present and only download on miss or mismatch.

| target | default (`FETCH=1`) | `FETCH=all` / `FETCH=serve` |
|---|---|---|
| **nvidia** | ONNX `minilm` + GGUF `default-llm` / `qwen-0.5b` | remaining ONNX embeds + `qwen-7b` |
| **intel** | `fetch --ov-genai minilm` + the same GGUFs (GenAI only; OVMS gRPC is out of scope) | remaining GenAI IR dirs that the manifest pins + `qwen-7b` |
| **apple** | `cargo xtask fetch --mlx minilm default-llm qwen-0.5b` + GGUF tokenizer for the 0.5B family | remaining MLX aliases + `qwen-7b` |
| **mock** | nothing | nothing |

`--only minilm,qwen-0.5b` restricts both the fetch set and the suite.
Aliases with no fetchable artifact on that arch (intel `bge-small` has no
public GenAI IR; apple `mpnet` has no MLX repo) are skipped with a reason,
not a hard error.

`make e2e-all` runs each arch whose `INFERSTREAM_E2E_<ARCH>_ADDR` is set.
With none set it prints a skip line and exits 0 — CI cloud must not start
remote GPUs.

Optional soak/STS corpus (`FETCH_CORPUS=1` / `--fetch-corpus`) is **off**
by default so CI stays on committed micro fixtures. See
[`testdata/corpus/README.md`](../testdata/corpus/README.md).

## Cross-arch embedding parity

Same alias + same texts → nearly identical vectors. Modes:

```bash
# Capture goldens on one arch, then replay:
make e2e-parity-goldens TARGET=nvidia WRITE=1
make e2e-parity-goldens TARGET=nvidia

# Pairwise cosine across live addrs and/or dumps:
INFERSTREAM_E2E_NVIDIA_ADDR=krick:8461 \
INFERSTREAM_E2E_INTEL_ADDR=krick-1:8461 \
INFERSTREAM_E2E_APPLE_ADDR=krickert-mac:8461 \
  make e2e-parity

cargo run -p inferstream-e2e -- --parity-cross \
  --peer nvidia=krick:8461 \
  --dump intel=testdata/e2e/goldens/intel
```

Thresholds: **0.99** same-arch and nvidia↔intel MiniLM FP; **0.97** for
any pair with apple (English MiniLM is ~1.000; min 0.9795 is CJK UNK
drift). Rationale: [`e2e-parity.md`](e2e-parity.md).

```bash
INFERSTREAM_E2E_NVIDIA_ADDR=krick:8461 \
INFERSTREAM_E2E_INTEL_ADDR=krick-1:8461 \
  make e2e-all
```

## Live hosts

Bearer token on all three example configs is `change-me` unless you overrode
`INFERSTREAM_API_KEYS`.

### krick (nvidia)

```bash
# on krick, or any box that can reach it:
scripts/run-nvidia.sh --config config/nvidia.toml   # already running is fine
make e2e-nvidia INFERSTREAM_E2E_ADDR=127.0.0.1:8461
# from another machine:
make e2e-nvidia INFERSTREAM_E2E_ADDR=krick:8461
```

Serves `minilm` + `default-llm` out of the box. Extra embed / `qwen-0.5b` /
`qwen-7b` aliases skip until they are on `serve` and fetched.
`make e2e-nvidia FETCH=1` (the default) pulls the smoke set onto this
machine first — useful on the worker itself, a no-op if the hashes already
match.

### krick-1 (intel)

```bash
scripts/run-intel.sh --config config/intel.toml
make e2e-intel INFERSTREAM_E2E_ADDR=127.0.0.1:8461
# remote:
make e2e-intel INFERSTREAM_E2E_ADDR=krick-1:8461
```

In-process SYCL currently serves `default-llm` / `qwen-0.5b` (and `qwen-7b`
when fetched). `qwen-0.5b` is **not** skipped: the catalog has an intel row.
Only skip it if you pass a matrix JSON that drops intel from that alias.

### krickert-mac (apple)

```bash
make apple
./swift/.build/release/inferstream-apple --config config/apple.toml
make e2e-apple INFERSTREAM_E2E_ADDR=127.0.0.1:8461
# or the bring-up wrapper (starts the Swift server, then the harness):
scripts/smoke-apple.sh
```

`mpnet` and `nomic-embed-text` skip with `NotAvailableOnArch` (no MLX path).

## Local mock (CI / no GPU)

`cargo test -p inferstream-e2e` starts an in-process mock registered as
`minilm` / `default-llm` / `qwen-0.5b` and runs the **same** suite with
`--target mock` (dims come from `ListModels`, not 384).

To drive the binary by hand:

```bash
cargo run -p inferstream-server -- --config config/e2e-mock.toml
cargo run -p inferstream-e2e -- --target mock --addr 127.0.0.1:8461 --token ""
# or: make e2e-mock INFERSTREAM_E2E_TOKEN=
```

## Ad-hoc smoke scripts

`scripts/smoke-embeddings.sh`, `scripts/smoke-llms.sh`, and the RPC half of
`scripts/smoke-apple.sh` are thin wrappers around this harness. Prefer
`inferstream-e2e` / `make e2e-*` as the canonical path.

## Matrix JSON

Override which aliases each arch must support:

```json
{
  "embeds": [
    {"alias": "minilm", "dim": 384, "required": true, "arches": ["nvidia", "intel", "apple"]}
  ],
  "llms": [
    {"alias": "default-llm", "arches": ["nvidia", "intel", "apple"]}
  ]
}
```

`required: true` → missing from `ListModels` is a fail. LLM aliases default
to not required so a host that has not fetched `qwen-7b` skips cleanly.
