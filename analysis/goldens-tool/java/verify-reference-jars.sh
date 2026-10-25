#!/bin/sh
# Verify the installed artifacts and every packaged SNAPSHOT, classifiers
# included, against receipts captured just after the pinned source builds.
set -eu
lib=${1:?the generator library directory is required}
shift
test "$#" -gt 0
for receipt in "$@"; do
  sha256sum -c "$receipt"
done
for jar in "$lib"/*-SNAPSHOT*.jar; do
  test -f "$jar"
  name=${jar##*/}
  matched=false
  for receipt in "$@"; do
    while read -r digest source; do
      if [ "${source##*/}" = "$name" ]; then
        cmp "$source" "$jar"
        matched=true
      fi
    done < "$receipt"
  done
  if [ "$matched" != true ]; then
    echo "unrecorded SNAPSHOT in generator: $name" >&2
    exit 1
  fi
done
