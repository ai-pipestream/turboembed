# native/turbo_buffer

Implementation of [`include/turbo_buffer.h`](../../include/turbo_buffer.h).

```bash
make turbo-buffer-tests
```

CPU is always compiled. CUDA / Level Zero / Metal are compile-gated
(`TURBO_BUFFER_CUDA`, `TURBO_BUFFER_ZE`, `TURBO_BUFFER_METAL`) and fail
loud when missing — never a silent CPU arena.
