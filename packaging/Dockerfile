# syntax=docker/dockerfile:1
#
# Build the Linux distribution archive on the manylinux_2_28 floor
# (glibc 2.28, AlmaLinux 8), so the archive runs on any glibc >= 2.28
# distribution regardless of the machine that built it. The archive holds
# libturbo, the headers, turbo-bundle, turbo-bench, and the mock and static
# providers; the CUDA, OpenVINO, and Hailo providers need their vendor
# toolchains and the ggml provider compiles llama.cpp from source, so all
# four are built outside this image (see docs/packaging.md).
#
#   scripts/package-container.sh [x86_64|aarch64]
#
# The build runs scripts/package.sh inside the image, including its ldd
# gate, the C smoke test, and the provider load check, and the final stage
# exports only dist/.

ARG BASE=quay.io/pypa/manylinux_2_28_x86_64
FROM ${BASE} AS build

# RUN runs under /bin/sh -c, which has no pipefail: a curl that fails or is
# truncated would leave sh exiting 0 and the layer cached with a partial or
# missing Rust install, and the failure would surface one layer later as
# "cargo: command not found".
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG RUST_TOOLCHAIN=stable
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain "${RUST_TOOLCHAIN}" --no-modify-path
ENV PATH=/root/.cargo/bin:${PATH}
# rust-toolchain.toml in /src would otherwise override rustup's default and
# silently build with its channel instead of the pinned toolchain.
# RUSTUP_TOOLCHAIN takes precedence over the file.
ENV RUSTUP_TOOLCHAIN=${RUST_TOOLCHAIN}
ENV CARGO_NET_RETRY=5
# This image is the glibc floor the archive claims; package.sh gates every
# packaged library's newest GLIBC_* symbol version against it.
ENV TURBO_GLIBC_FLOOR=2.28

WORKDIR /src
COPY . .

# Cargo's registry and the compiled dependencies are cached between builds;
# the archive itself is built fresh every time.
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,target=/src/target,sharing=locked \
    scripts/package.sh --no-cuda --no-openvino --no-hailo --no-ggml \
    && cp -r /src/dist /out

FROM scratch AS export
COPY --from=build /out/ /
