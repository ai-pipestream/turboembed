#!/bin/sh
# Registers the runner on the first start, from RUNNER_TOKEN, then runs
# it. /home/runner is a volume, so the registration, the runner's updates
# and the toolchains are kept across restarts.
set -eu
cd /home/runner/actions-runner
if [ ! -f .runner ]; then
  : "${RUNNER_TOKEN:?a registration token is needed on the first start}"
  ./config.sh --unattended --replace \
    --url "${RUNNER_URL:-https://github.com/ai-pipestream/turboembed}" \
    --token "$RUNNER_TOKEN" \
    --name "$RUNNER_NAME" \
    --labels "$RUNNER_LABELS" \
    --work _work
fi
unset RUNNER_TOKEN
exec ./run.sh
