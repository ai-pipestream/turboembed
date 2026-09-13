# native/turbo_buffer

Implementation of [`include/turbo_buffer.h`](../../include/turbo_buffer.h).

```bash
make turbo-buffer-tests
```

CPU is always compiled. CUDA PINNED mapped + DEVICE is LIVE on
Machine A when built with `TURBO_BUFFER_CUDA` (token kernels read
mapped pointers; `h2d_bytes/forward == 0`). Level Zero / Metal are
compile-gated and fail loud when missing — never a silent CPU arena.

Machine B LIVE: `make turbo-buffer-intel-receipt` plus
[`docs/turbo-buffer-ze-machine-b.md`](../../docs/turbo-buffer-ze-machine-b.md).

Metal SHARED rent/return is LIVE on Machine C
(`docs/apple-turbo-buffer-metal-arena-machine-c.md`).
`libTurboEmbed.dylib` links `libturbo_buffer_apple.a` and rents
SHARED tokens / last-hidden / results
(`docs/apple-turboembed-metal-arena-machine-c.md`).
