# Test corpus

SHA-pinned text used for soak/chunking and embedding-quality / cross-arch
parity. **Default CI never downloads this.** Unit tests use the committed
micro fixtures only.

| artifact | role | how it lands |
|---|---|---|
| `fixtures/tiny-shakespeare.excerpt.txt` | first 12 speeches of Tiny Shakespeare | committed; unit tests / default parity texts |
| `fixtures/sts-micro.jsonl` | 12 STS-style pairs | committed; unit tests |
| `sts-pairs.jsonl` | 96 original STS-style pairs (scores 0–5) | committed; SHA-verified by the corpus manifest |
| `tiny-shakespeare.txt` | full Karpathy Tiny Shakespeare (~1.1 MiB) | **fetched** by `make fetch-corpus` |

## Fetch (optional)

```bash
make fetch-corpus                 # SHA-256-pinned; idempotent
make verify-corpus                # offline hash check
cargo run -p inferstream-fetch -- --corpus --list
# e2e optional path (CI stays FETCH_CORPUS=0):
make e2e-nvidia FETCH_CORPUS=1
```

Pins live in `models/manifests/corpus.json`. Tiny Shakespeare is downloaded
from `karpathy/char-rnn` at commit `370cbcd448eb7daf32f21a6be560b70e0b33c4e3`
(the last commit that touched `data/tinyshakespeare/input.txt`). The STS
list is a committed fixture: fetch verifies the hash and does not hit the
network when the file is already present.

## Chunker

`inferstream-e2e` splits a source into paragraphs (blank-line separated)
and sentences, and assigns **stable ids** that only depend on source name +
order in that file:

```
tiny-shakespeare:p0000          # first paragraph
tiny-shakespeare:p0000:s0000    # first sentence of that paragraph
```

Same pinned file → same ids → same Embed batches. See `docs/e2e-parity.md`.
