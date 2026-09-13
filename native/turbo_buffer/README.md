# native/turbo_buffer

Implementation of [`include/turbo_buffer.h`](../../include/turbo_buffer.h).

```bash
make turbo-buffer-tests
```

CPU is always compiled. CUDA PINNED+DEVICE is LIVE on Machine A when
built with `TURBO_BUFFER_CUDA`. Level Zero / Metal are compile-gated
and fail loud when missing — never a silent CPU arena.
