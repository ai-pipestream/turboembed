#!/usr/bin/env bash
# Checks scripts/backend-lanes.sh against a small repository made for
# the purpose: each case is one commit on top of the same base.
set -euo pipefail

lanes="$(cd "$(dirname "$0")" && pwd)/backend-lanes.sh"
repo="$(mktemp -d)"
trap 'rm -rf "$repo"' EXIT
cd "$repo"
git init -q -b main .
git config user.name test
git config user.email test@example.invalid
mkdir -p core/metal docs .github/actions/apt-install
echo kernel > core/metal/backend.mm
echo text > docs/page.md
echo action > .github/actions/apt-install/action.yml
git add -A
git commit -q -m base
base="$(git rev-parse HEAD)"

failed=0
# expect <name> <event> <linux> <macos>: compares the script's answer for
# base...HEAD, then goes back to the base.
expect() {
  local got want="linux=$3 macos=$4"
  got="$("$lanes" "$2" "$base" HEAD 2>/dev/null | tr '\n' ' ')"
  if [ "$got" != "$want " ]; then
    echo "FAIL $1: got '$got', want '$want'" >&2
    failed=1
  else
    echo "ok   $1"
  fi
  git reset -q --hard "$base"
}
change() {
  git add -A
  git commit -q -m change
}

echo more >> docs/page.md; change
expect "a docs change needs neither lane" merge_group false false

echo more >> core/metal/backend.mm; change
expect "a core change needs both lanes" merge_group true true

echo more >> .github/actions/apt-install/action.yml; change
expect "an apt-install change needs the Linux lane" pull_request true false

mkdir -p docs/archive; git mv core/metal/backend.mm docs/archive/backend.mm; change
expect "a file moved out of core needs both lanes" merge_group true true

git mv docs/page.md core/page.md; change
expect "a file moved into core needs both lanes" pull_request true true

git rm -q core/metal/backend.mm; change
expect "a file deleted from core needs both lanes" merge_group true true

echo more >> docs/page.md; change
expect "the daily run needs both lanes" schedule true true

got="$("$lanes" merge_group no-such-commit HEAD 2>/dev/null | tr '\n' ' ')"
if [ "$got" != "linux=true macos=true " ]; then
  echo "FAIL unreadable paths: got '$got'" >&2
  failed=1
else
  echo "ok   unreadable paths need both lanes"
fi

# A pull request or merge group without both ends is a usage error: it
# prints no lane lines, so a caller cannot take silence for "skip".
if got="$("$lanes" pull_request 2>/dev/null)" || [ -n "$got" ]; then
  echo "FAIL a missing base: got '$got' and no error" >&2
  failed=1
else
  echo "ok   a missing base is an error with no lane lines"
fi

exit "$failed"
