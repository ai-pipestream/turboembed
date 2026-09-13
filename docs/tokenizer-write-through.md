# Tokenizer write-through (SOLIDIFY 5)

Token ids land in rented `turbo_buffer` rows. After warmup there is no
heap token vector on the serve / score / embed hot path for MiniLM
WordPiece. Vocab is a frozen mmap / image loaded at create.

Hostnames stay out of this file (Machine A / B / C only).

## Done criteria

| # | Gate | Proof |
|---|---|---|
| 1 | Rerank pack writes into the rented i32 row | `turborerank_pack_text` calls `wordpiece_pack_pair` on `buffer->input_ids` / mask / type / pos. No `tok_q` / `tok_d` scratch. `turbo_buffer_arena_owns` on those pointers. |
| 2 | WordPiece hot path is heap-free | `native/wordpiece/encode.cpp` has no `std::vector` / `std::string` / `unordered_map`. `make wordpiece-tripwire` fails if they return. `wordpiece_hot_alloc_counter() == 0` after pack / encode. |
| 3 | Frozen vocab | `vocab.txt` is mmap'd. `tokenizer.json` WordPiece `vocab` becomes one blob + open-address table at load. Lookups are two-span FNV into that image. |
| 4 | Intel GenAI | `ov::genai::Tokenizer.encode` has no caller-buffer API. MiniLM uses WordPiece into rented ZE SHARED / HOST i32 USM. `copy_tokens_to_i32` is gone. Tripwire greps `tokenizer.encode`. |
| 5 | NVIDIA ORT | MiniLM-compatible models load WordPiece and write i64 ids into PINNED / HOST arena rows (`elem_width=8`). SentencePiece stays on `tokenizers` (we do not control that tokenizer). |
| 6 | Apple Metal | `libturbo_buffer_apple.a` ships encode + load. `MlxEngine.embedArena` writes i32 ids into SHARED slots when `vocab.txt` / WordPiece `tokenizer.json` is next to the MLX dir. |
| 7 | Goldens | Berlin CE still in band. Embed MiniLM cosine still in band (Machine A/B/C receipts). |

## Commands

```bash
make wordpiece-tests          # tiny fixture + arena owns() + tripwire
make turborerank-tests        # includes wordpiece-tests; Berlin when weights exist
make test-turboembed-intel    # Machine B: WordPiece → USM + MiniLM ≥0.99
make test-turboembed-nvidia   # Machine A: WordPiece → PINNED i64 + MiniLM goldens
make test-turboembed-apple    # Machine C: WordPiece → Metal SHARED i32
```

## What is not this item

gRPC packed bytes (6). Machine B final bench is
`docs/solidify-bench-machine-b.md`. SentencePiece / Unigram models we
do not own (BGE-M3 etc. keep the HF tokenizer).
