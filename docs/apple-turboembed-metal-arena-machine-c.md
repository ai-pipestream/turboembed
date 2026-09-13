# TurboEmbed Metal SHARED arena (Machine C) — SOLIDIFY (4)

Live proof that `libTurboEmbed.dylib` rents token / activation / result
rows from `include/turbo_buffer.h` Metal `SHARED`
(`MTLResourceStorageModeShared`). MLX wraps those contents pointers.
Lookup is `turbo_buffer_metal_lookup` only. No private MTL registry.
After load warmup, `turbo_buffer_alloc_counter() == 0` on embed.

| Field | Value |
|---|---|
| Host | Machine C, Apple M2, Metal |
| Tree | `c6f838c` + receipt refresh |
| Command | `make libturbo-buffer-apple` + `make test-turboembed-apple` |
| Receipt | `testdata/receipts/turboembed/apple-minilm.json` (`metal_owns_result=true`, `allocs_after_forward=0`) |
| vs nvidia | min `0.97945654`, mean `0.99973994`, floor `0.97` (`sts-0056:a`) |
| vs apple | min `0.9999999`, mean `1.0`, floor `0.99` |

## What was proven

1. **Engine create attaches a Metal arena.**
   `turbo_buffer_arena_create(METAL)` on `TURBOEMBED_DEVICE_METAL` /
   `AUTO`. Missing Metal is `UNAVAILABLE` — never CPU.
2. **Load rents work slots.** `input_ids` / `attention_mask` /
   `token_type_ids` are SHARED i32 `[32, 256]`. Last-hidden activations
   are SHARED f32 `[32, 256×384]`. A result slab of `[32, 384]` is
   warmed and returned so the next embed reuses it.
3. **Embed writes tokens into those slots.** `MlxEngine.embedArena`
   wraps the arena pointers as `MLXArray(rawPointer:)`. If MLX
   `make_buffer` copies onto a private buffer, embed fails loud.
4. **Result.values is arena-owned.** `turbo_buffer_arena_owns` and
   `turbo_buffer_metal_owns` are both true.
   `turbo_buffer_metal_lookup` returns the arena `id<MTLBuffer>`.
5. **allocs/forward == 0** after warmup. Reintroducing
   `UnsafeMutablePointer<Float>.allocate` for MiniLM results, or a
   private `newBufferWithLength` token path, fails the receipt test.
6. **Goldens stay in band.** Same mean+L2 MiniLM as before (not BERT
   NSP pooler, not 8-d FNV).

BERT layer intermediates still come from mlx-swift's allocator (same
as ORT/GenAI keeping their own device graphs). The ABI-visible token /
last-hidden / result path is the arena.

## Commands

```bash
make libturbo-buffer-apple       # native/turbo_buffer/build/libturbo_buffer_apple.a
make test-turboembed-apple       # Rust mlx-live + receipt
swift test --package-path swift --filter MetalArenaEmbedTests
```
