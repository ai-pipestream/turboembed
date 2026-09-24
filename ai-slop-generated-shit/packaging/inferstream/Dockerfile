# syntax=docker/dockerfile:1
#
# The Inferstream service image: the `inferstream` binary, libturbo, and
# the providers this image can build, on a slim Debian base. The default
# build is the CPU image (the ggml provider with llama.cpp's CPU backend;
# the mock and static providers are built into libturbo); the CUDA image is the same
# file with the NVIDIA bases and GGML_FEATURES=cuda:
#
#   scripts/inferstream-image.sh                 # turbo-inferstream:cpu
#   scripts/inferstream-image.sh cuda            # turbo-inferstream:cuda
#
# The image holds no model: a bundle directory is mounted (or, under
# KServe, fetched by the storage initializer) and named with --model. The
# server listens on 0.0.0.0:8000 (HTTP) and 0.0.0.0:8001 (gRPC).
#
# The ONNX providers (cuda, openvino) need their vendor SDKs and are not
# in this image today; see docs/packaging.md for what is left out and why.

ARG BUILD_BASE=debian:bookworm-slim
ARG RUNTIME_BASE=debian:bookworm-slim

FROM ${BUILD_BASE} AS build
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# protoc for the OIP proto; cmake, a C++ toolchain and libclang (bindgen) for llama.cpp.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl build-essential cmake git pkg-config protobuf-compiler libclang-dev \
    && rm -rf /var/lib/apt/lists/*

ARG RUST_TOOLCHAIN=stable
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain "${RUST_TOOLCHAIN}" --no-modify-path
ENV PATH=/root/.cargo/bin:${PATH}
ENV RUSTUP_TOOLCHAIN=${RUST_TOOLCHAIN}
ENV CARGO_NET_RETRY=5

# ggml backends compiled into llama.cpp: empty for CPU, `cuda` on the
# NVIDIA devel base (nvcc must be on the path there).
ARG GGML_FEATURES=""

WORKDIR /src
COPY . .

RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,target=/src/target,sharing=locked \
    set -eux; \
    features=""; \
    if [ -n "${GGML_FEATURES}" ]; then features="--features turbo-provider-ggml/${GGML_FEATURES}"; fi; \
    cargo build --release -p turbo-inferstream -p turbo-provider-ggml ${features}; \
    mkdir -p /out/bin /out/providers; \
    cp -L target/release/inferstream /out/bin/; \
    cp -L target/release/libturbo_provider_ggml.so /out/providers/; \
    for f in /out/bin/inferstream /out/providers/*.so; do \
        if ldd "$f" | grep -q 'not found'; then echo "unresolved libraries in $f:" >&2; ldd "$f" >&2; exit 1; fi; \
    done; \
    cp LICENSE /out/

FROM ${RUNTIME_BASE} AS runtime
# libstdc++ and OpenMP for llama.cpp; ca-certificates for clients that
# reach out from inside the container (none in the server itself).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libstdc++6 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 1000 --create-home --shell /usr/sbin/nologin turbo
COPY --from=build /out /opt/turbo
ENV PATH=/opt/turbo/bin:${PATH}
USER turbo
EXPOSE 8000 8001
# The listeners bind every interface by default (a --http or --grpc flag
# overrides the variable). The providers this image built are loaded at
# start; a model is named with --model, for example
#   --model name=qwen,bundle=/models/qwen05-gguf,provider=ggml,generations=2
ENV INFERSTREAM_HTTP=0.0.0.0:8000 INFERSTREAM_GRPC=0.0.0.0:8001
ENTRYPOINT ["inferstream", "--provider-lib", "/opt/turbo/providers/libturbo_provider_ggml.so"]
