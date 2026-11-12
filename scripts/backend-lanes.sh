#!/usr/bin/env bash
# Says which accelerator lint lanes a CI run needs, as lines for
# $GITHUB_OUTPUT: `linux=true|false` (CUDA and Level Zero) and
# `macos=true|false` (Metal). A lane is needed when the run is not for a
# pull request or a merge group, or when the commits it tests change a
# path the lane builds from. When the changed paths cannot be read, both
# lanes are needed.
#
#   scripts/backend-lanes.sh <event> [<base> <head>]
#
# <base> and <head> are the two ends of the change: the files that differ
# between their merge base and <head> are the ones looked at.
set -uo pipefail

event="${1:?event name}"
both='^(core/|analysis/|include/|Cargo\.toml$|Cargo\.lock$|\.github/workflows/ci\.yml$|scripts/backend-lanes\.sh$)'
linux_only='^\.github/actions/apt-install/'

linux=true
macos=true
case "$event" in
  pull_request | merge_group)
    # Without rename detection: a file moved out of a watched directory
    # is then listed under its old path too.
    if paths="$(git diff --name-only --no-renames "${2:?base}...${3:?head}")"; then
      linux=false
      macos=false
      if grep -qE "$both" <<<"$paths"; then
        linux=true
        macos=true
      elif grep -qE "$linux_only" <<<"$paths"; then
        linux=true
      fi
    else
      echo "cannot read the changed paths between $2 and $3: both lanes run" >&2
    fi
    ;;
esac
echo "linux=$linux"
echo "macos=$macos"
