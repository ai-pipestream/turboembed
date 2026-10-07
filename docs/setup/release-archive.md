# The Linux release archive

A tag `v<version>` publishes `turboembed-v<version>-linux-x86_64.tar.gz`
on the repository's releases page: the library built with the cpu and
cuda backends, its header and its licences, with a SHA-256 beside it
([../release.md](../release.md)).

## Using it

```
sha256sum -c turboembed-v<version>-linux-x86_64.tar.gz.sha256
tar xzf turboembed-v<version>-linux-x86_64.tar.gz
cd turboembed-v<version>-linux-x86_64
cat BUILD                                   # commit, compilers, CUDA version, architectures
cc -std=c11 -I include <your program>.c -L lib -lturbo -Wl,-rpath,'$ORIGIN/lib' -o <your program>
```

[`embed.c`](embed.c) builds against the archive the same way it builds
against the tree.

- The CPU backend needs nothing.
- The CUDA backend needs the NVIDIA driver alone: the CUDA runtime is
  linked in, and cuBLAS is not linked. The driver must support the CUDA
  version `BUILD` names (a 12.x build needs driver 525 or newer). A
  machine without a driver or an NVIDIA GPU lists no CUDA device and runs
  the CPU as usual.
- The archive carries machine code for the architectures `BUILD` lists,
  one per NVIDIA generation from sm_75, and PTX for the newest.

`scripts/setup/cuda.sh` checks the driver and GPU side on its own; its
toolkit checks matter only for building.

## When to build instead

The archive is Linux x86_64 with cpu and cuda. Every other setup
(Jetson, Intel GPU and NPU, Metal, Hailo) builds from the tree, as its
page says, and so does a CUDA build with `cuda-cublas` or for a single
architecture.
