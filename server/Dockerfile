# syntax=docker/dockerfile:1
#
# The turbo-kserve image (docs/release.md, Images): the gRPC server alone,
# built from the tree, on a slim Debian base. The build context is the
# repository root:
#
#   docker build -f server/Dockerfile -t turbo-kserve .                         # cpu
#   docker build -f server/Dockerfile --build-arg FEATURES=cuda \
#       --build-arg NVIDIA_VISIBLE_DEVICES=all -t turbo-kserve:cuda .           # cpu and cuda
#
# The CUDA build links the toolkit's runtime statically and cuBLAS not at
# all, as the release archive does, so the image runs on the plain Debian
# base and needs the host's NVIDIA driver alone, which the NVIDIA
# container toolkit mounts in.
#
# The image holds no model. Bundles are mounted under /bundles, one
# directory each, and every one found there is served on device 0 with one
# session; TURBO_KSERVE_MODELS or --model sets anything else
# (docs/grpc.md, Flags). The server listens on 0.0.0.0:8081.
#
# Both bases are pinned by the digest of their multi-arch index, so a
# rebuild of a tag is the same build; a new digest is a reviewed change.

ARG BUILD_BASE=rust:1.98-slim-bookworm@sha256:ff521445a372125ed4f76e1453a1f8098f2d05332d1601d30db1c1f62757e730
ARG RUNTIME_BASE=debian:bookworm-slim@sha256:7ff6ba31527b9e336162fa061746c366c4b2032f4d2cff2078bea43f78efe3f8

FROM ${BUILD_BASE} AS build
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Empty for the cpu backend alone, `cuda` to link the CUDA backend in too.
ARG FEATURES=""
# The toolkit and architectures the release archive is built with
# (.github/workflows/release.yml).
ARG CUDA_VERSION=12.8
ARG CUDA_PACKAGE=12-8
ARG TURBO_CUDA_ARCH=75,80,86,89,90,100,120

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl g++; \
    if [ "${FEATURES}" = cuda ]; then \
        curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb; \
        dpkg -i cuda-keyring_1.1-1_all.deb; \
        rm cuda-keyring_1.1-1_all.deb; \
        apt-get update; \
        apt-get install -y --no-install-recommends \
            "cuda-nvcc-${CUDA_PACKAGE}" "cuda-cudart-dev-${CUDA_PACKAGE}" "cuda-cccl-${CUDA_PACKAGE}"; \
    fi; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target,sharing=locked \
    set -eux; \
    export TURBO_CUDA_ROOT="/usr/local/cuda-${CUDA_VERSION}" TURBO_CUDA_ARCH; \
    cargo build --locked --release -p turbo-kserve ${FEATURES:+--features "${FEATURES}"}; \
    bin=target/release/turbo-kserve; \
    ldd "$bin"; \
    if ldd "$bin" | grep -qE 'not found|libcu(dart|blas)'; then \
        echo "turbo-kserve links a missing or CUDA library dynamically" >&2; exit 1; \
    fi; \
    mkdir -p /out/bin /out/licenses; \
    cp "$bin" /out/bin/; \
    cp LICENSE /out/licenses/LICENSE; \
    if [ "${FEATURES}" = cuda ]; then cp core/cuda/cutlass/LICENSE.txt /out/licenses/CUTLASS-LICENSE.txt; fi

FROM ${RUNTIME_BASE}
# libstdc++ for the backends' C++ host side; ca-certificates for the
# catalog's HTTPS fetches (--catalog-dir).
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates libstdc++6; \
    rm -rf /var/lib/apt/lists/*; \
    useradd --uid 10001 --user-group --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin turbo; \
    mkdir -p /bundles

COPY --from=build /out /opt/turbo

# What the NVIDIA container toolkit reads: `void` keeps the GPUs out of
# the cpu image; the cuda image is built with `all`.
ARG NVIDIA_VISIBLE_DEVICES=void
ENV PATH=/opt/turbo/bin:${PATH} \
    TURBO_KSERVE_LISTEN=0.0.0.0:8081 \
    TURBO_KSERVE_BUNDLE_DIR=/bundles \
    NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES} \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

LABEL org.opencontainers.image.title="turbo-kserve" \
      org.opencontainers.image.description="turboembed's gRPC server: the Open Inference Protocol and the analysis services" \
      org.opencontainers.image.source="https://github.com/ai-pipestream/turboembed" \
      org.opencontainers.image.licenses="Apache-2.0"

USER 10001:10001
VOLUME ["/bundles"]
EXPOSE 8081
STOPSIGNAL SIGTERM
# grpc.health.v1.Health/Check on the listen address: SERVING once every
# bundle is loaded, NOT_SERVING while loading and draining. The start
# period covers a large bundle's load.
HEALTHCHECK --interval=10s --timeout=5s --start-period=120s --retries=3 CMD ["turbo-kserve", "health"]
ENTRYPOINT ["turbo-kserve"]
