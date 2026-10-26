#!/bin/sh
# Download only; never train or overwrite unverified existing files.
set -eu
recipe=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
destination=${1:?usage: prepare.sh OUTPUT_DIRECTORY}
mkdir -p "$destination"
cd "$destination"
base=https://raw.githubusercontent.com/UniversalDependencies/UD_Finnish-FTB/948615ba5289828a8d5cf7161d9cedf803e9efe6
while read -r hash name; do
    if test ! -e "$name"; then
        temporary=$(mktemp "./download.XXXXXX")
        trap 'rm -f "$temporary"' EXIT HUP INT TERM
        curl --fail --location --retry 2 "$base/$name" -o "$temporary"
        printf '%s  %s\n' "$hash" "$temporary" | sha256sum --check --status
        mv -n "$temporary" "$name"
        rm -f "$temporary"
        trap - EXIT HUP INT TERM
    fi
done < "$recipe/SHA256SUMS"
sha256sum --check "$recipe/SHA256SUMS"
