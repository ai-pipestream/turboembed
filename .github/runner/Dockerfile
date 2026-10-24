# The self-hosted runner for ci.yml's Linux lanes and the models-*
# training jobs, used when the repository variable TURBO_CI_LINUX names
# its labels. It carries the tools the workflows call that the hosted
# ubuntu-24.04 image has; rustup is installed by dtolnay/rust-toolchain at
# the start of each job. No docker socket is passed in, since the jobs run
# a pull request's code; a GPU is passed in when the runner is started
# (README.md).
FROM ubuntu:24.04@sha256:68ee22004a9109b44973313b02ff7b1de250f1de0f5ecb628d73f579aaff33ac

ARG RUNNER_VERSION=2.338.1
ARG RUNNER_SHA256=08e9ccb3604d2462c24b5e33604dc30810296ca15cc65f23cffd4317eb144c37

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl git jq pkg-config libssl-dev \
      unzip xz-utils zstd tini \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1001 runner
USER runner
WORKDIR /home/runner/actions-runner
RUN curl -fsSL -o runner.tar.gz \
      "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" \
 && echo "${RUNNER_SHA256}  runner.tar.gz" | sha256sum -c - \
 && tar xzf runner.tar.gz && rm runner.tar.gz
USER root
RUN ./bin/installdependencies.sh && rm -rf /var/lib/apt/lists/*
USER runner

COPY --chmod=755 entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
