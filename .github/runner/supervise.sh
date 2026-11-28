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
set -u
name=$1 labels=$2
shift 2
repo=${RUNNER_REPO:-ai-pipestream/turboembed}
image=${RUNNER_IMAGE:-turbo-ci-runner}
docker=${DOCKER:-docker}

label_args=
for l in self-hosted Linux X64 $(echo "$labels" | tr , ' '); do
  label_args="$label_args -f labels[]=$l"
done

# The container is named after the runner, so a stop of this script, or a
# container left from a crash, is cleaned up by name.
trap '$docker rm -f "$name" >/dev/null 2>&1; exit 0' TERM INT

while :; do
  $docker rm -f "$name" >/dev/null 2>&1
  # A runner killed mid-job stays registered until GitHub drops it, so
  # each configuration gets a name of its own.
  # shellcheck disable=SC2086
  reply=$(gh api -X POST "repos/$repo/actions/runners/generate-jitconfig" \
          -f name="$name-$(date +%s)" -F runner_group_id=1 $label_args \
          --jq '"\(.runner.id) \(.encoded_jit_config)"') || { sleep 30; continue; }
  id=${reply%% *} jit=${reply#* }
  # shellcheck disable=SC2086
  printf 'RUNNER_JITCONFIG=%s\n' "$jit" \
    | $docker run --rm --name "$name" --env-file /dev/stdin "$@" "$image"
  status=$?
  # 125 is a failure of docker run itself: the container did not start,
  # so the runner took no job and GitHub still lists it. Each such
  # registration counts toward the repository's limit of runners, so it is
  # removed, and the next try waits a minute instead of registering
  # another at once.
  if [ "$status" -eq 125 ]; then
    echo "turbo-ci: $name: the container did not start; removing runner $id" >&2
    gh api -X DELETE "repos/$repo/actions/runners/$id" >/dev/null 2>&1 \
      || echo "turbo-ci: $name: runner $id could not be removed" >&2
    sleep 60
    continue
  fi
  sleep 2
done
