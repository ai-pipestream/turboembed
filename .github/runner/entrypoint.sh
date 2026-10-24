#!/bin/sh
# Runs one job and exits. The just-in-time configuration comes from
# supervise.sh, which asks GitHub for a new one and starts a new container
# for each job, so nothing one job leaves behind reaches the next.
set -eu
cd /home/runner/actions-runner
: "${RUNNER_JITCONFIG:?supervise.sh passes a just-in-time configuration}"
jit=$RUNNER_JITCONFIG
unset RUNNER_JITCONFIG
exec ./run.sh --jitconfig "$jit"
