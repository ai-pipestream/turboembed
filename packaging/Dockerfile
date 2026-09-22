# syntax=docker/dockerfile:1
#
# Build the Linux distribution archive on the manylinux_2_28 floor
# (glibc 2.28, AlmaLinux 8), so the archive runs on any glibc >= 2.28
# distribution regardless of the machine that built it. The archive holds
# libturbo, the headers, turbo-bundle, turbo-bench, and the mock and static
# providers; the CUDA, OpenVINO, and Hailo providers need their vendor
# toolchains and are built outside this image (see docs/packaging.md).
#
#   scripts/package-container.sh [x86_64|aarch64]
#
# The build runs scripts/package.sh inside the image, including its ldd
# gate, the C smoke test, and the provider load check, and the final stage
# exports only dist/.

ARG BASE=quay.io/pypa/manylinux_2_28_x86_64
FROM ${BASE} AS build

ARG RUST_TOOLCHAIN=stable
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain "${RUST_TOOLCHAIN}" --no-modify-path
ENV PATH=/root/.cargo/bin:${PATH}
ENV CARGO_NET_RETRY=5

WORKDIR /src
COPY . .

# Cargo's registry and the compiled dependencies are cached between builds;
# the archive itself is built fresh every time.
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,target=/src/target,sharing=locked \
    scripts/package.sh --no-cuda --no-openvino --no-hailo \
    && cp -r /src/dist /out

FROM scratch AS export
COPY --from=build /out/ /
