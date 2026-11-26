#!/bin/sh
# supervise.sh NAME LABELS [docker run options...]
#
# Keeps one ephemeral runner going: asks GitHub for a just-in-time
# runner configuration that takes a single job, runs it in a new
# container that is removed afterwards, and repeats. LABELS is a comma
# list, to which self-hosted, Linux and X64 are added. DOCKER may name
# another docker command, such as "ssh storage-host docker", to run the
# container on a host without a GitHub login. The configuration goes to
# the container on standard input, never on a command line.
#
# WORK_DIR, when set, is a directory on the docker host where this
# runner's own size-capped filesystem is mounted (a loop-mounted ext4
# image, see /etc/fstab). It becomes the runner's _work directory, so the
# checkout, target/ and TMPDIR live there and a full disk stops at this
# runner instead of filling the host's root filesystem. The filesystem
# root holds a .turbo-ci-volume marker; without it (not mounted) no job
# is taken. The directory is wiped before and after every job.
set -u
name=$1 labels=$2
shift 2
repo=${RUNNER_REPO:-ai-pipestream/turboembed}
image=${RUNNER_IMAGE:-turbo-ci-runner}
docker=${DOCKER:-docker}
work_dir=${WORK_DIR:-}
work=/home/runner/actions-runner/_work

label_args=
for l in self-hosted Linux X64 $(echo "$labels" | tr , ' '); do
  label_args="$label_args -f labels[]=$l"
done

# Empties the runner's filesystem, keeping only the marker and
# lost+found, and recreates TMPDIR. Fails when the marker is missing.
# The script goes on standard input so that it survives DOCKER being ssh.
wipe() {
  [ -n "$work_dir" ] || return 0
  # shellcheck disable=SC2016
  printf '%s\n' \
    'set -e' \
    'test -f /w/.turbo-ci-volume || { echo "turbo-ci: /w is not the runner volume" >&2; exit 1; }' \
    'find /w -mindepth 1 -maxdepth 1 ! -name .turbo-ci-volume ! -name lost+found -exec rm -rf {} +' \
    'mkdir -p /w/_tmp && chown 1001:1001 /w /w/_tmp && chmod 1777 /w/_tmp' \
  | $docker run -i --rm --user root --network none -v "$work_dir:/w" \
      --entrypoint sh "$image" -s
}

# The container is named after the runner, so a stop of this script, or a
# container left from a crash, is cleaned up by name.
trap '$docker rm -f "$name" >/dev/null 2>&1; exit 0' TERM INT

work_args=
if [ -n "$work_dir" ]; then
  work_args="-v $work_dir:$work -e TMPDIR=$work/_tmp"
fi

while :; do
  $docker rm -f "$name" >/dev/null 2>&1
  wipe || { echo "turbo-ci: $name: work volume $work_dir unavailable, not taking jobs" >&2; sleep 60; continue; }
  # A runner killed mid-job stays registered until GitHub drops it, so
  # each configuration gets a name of its own.
  # shellcheck disable=SC2086
  jit=$(gh api -X POST "repos/$repo/actions/runners/generate-jitconfig" \
          -f name="$name-$(date +%s)" -F runner_group_id=1 $label_args \
          --jq .encoded_jit_config) || { sleep 30; continue; }
  # shellcheck disable=SC2086
  printf 'RUNNER_JITCONFIG=%s\n' "$jit" \
    | $docker run --rm --name "$name" --env-file /dev/stdin $work_args "$@" "$image"
  wipe
  sleep 2
done
