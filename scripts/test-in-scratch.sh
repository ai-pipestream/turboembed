#!/usr/bin/env bash
# Runs a cargo test command with the tests' temp directory set to
# $TEST_SCRATCH, so that a later step can see what they left there.
# The command is run once with --no-run first: the build then writes
# its own temporary files to the usual place, and only the tests write
# to $TEST_SCRATCH.
#
#   TEST_SCRATCH=<dir> scripts/test-in-scratch.sh cargo test -p <crate> ...
set -euo pipefail
: "${TEST_SCRATCH:?set TEST_SCRATCH to an existing directory}"
[ -d "$TEST_SCRATCH" ] || { echo "$TEST_SCRATCH is not a directory" >&2; exit 2; }
# --no-run is a cargo option: it goes before a `--`, which starts the
# test binaries' own arguments.
build=()
placed=0
for arg in "$@"; do
  if [ "$placed" = 0 ] && [ "$arg" = -- ]; then
    build+=(--no-run)
    placed=1
  fi
  build+=("$arg")
done
[ "$placed" = 1 ] || build+=(--no-run)
"${build[@]}"
TMPDIR="$TEST_SCRATCH" "$@"
