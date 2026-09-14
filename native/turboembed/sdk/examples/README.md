# TurboEmbed Prepared external C example

This example is a separate CMake project. It uses the installed package target
and the public `turboembed_prepared.h` header only; it does not include
OpenVINO headers or refer to a source checkout.

Configure it with the SDK installation prefix, then run it with a verified
bundle directory:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/turboembed-prefix
cmake --build build
./build/turboembed_prepared_embed /path/to/minilm-bundle
./build/turboembed_prepared_embed /path/to/minilm-bundle cpu
```

GPU is the default. The optional `cpu` argument is the only CPU selection; a
missing GPU returns the SDK's explicit error. The example creates a fixed
`[1, 32]` slot, writes UTF-8 `hello world`, performs synchronous execution,
copies 384 floats to host memory, prints the resolved device and vector norm,
then releases the result before its slot, model, and context.
