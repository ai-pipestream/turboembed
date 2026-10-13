#!/bin/sh
# Regenerates the golden files. See ../README.md.
#
# Usage: generate.sh [<snowball-data checkout>]
#
# OpenNLP must be built at commit PIN (below) and installed into the local
# Maven repository (mvn install -DskipTests), and javac and java must be a
# JDK 21. With a snowball-data checkout at commit DATA (below), the samples
# in inputs/ and snowball/ are first taken from it again.
set -eu

PIN=cdf33c653d5d61d5af7d37ffee189a83bd8392ea
DATA=da07690d20ce69ee16b5fb95796f92090cc9b4a1
# The languages whose snowball-data files are under the BSD license of the
# data's own COPYING. arabic (GPL 3), catalan, french, greek and indonesian
# (in part from Wikipedia, CC BY-SA) are left out; their words are only the
# generated ones.
SAMPLED="danish dutch english finnish german hungarian irish italian norwegian porter portuguese romanian russian spanish swedish turkish"
# Words per sampled language.
SAMPLE=1000

golden=$(cd "$(dirname "$0")" && pwd)
data=${1:-}

if [ -n "$data" ]; then
    head=$(git -C "$data" rev-parse HEAD)
    if [ "$head" != "$DATA" ]; then
        echo "snowball-data is at $head, not $DATA" >&2
        exit 1
    fi
    for lang in $SAMPLED; do
        n=$(wc -l < "$data/$lang/voc.txt")
        step=$(( (n + SAMPLE - 1) / SAMPLE ))
        # Every step-th line from the first, the same lines of both files.
        # No word in these files has a backslash or starts with #, so the
        # lines need no escapes.
        awk -v s="$step" '(NR - 1) % s == 0' "$data/$lang/voc.txt" > "$golden/inputs/$lang.txt"
        awk -v s="$step" '(NR - 1) % s == 0' "$data/$lang/output.txt" > "$golden/snowball/$lang.txt"
    done
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

m2=${M2_REPO:-$HOME/.m2/repository}
cp=""
for jar in opennlp-runtime opennlp-api; do
    cp="$cp$m2/org/apache/opennlp/$jar/3.0.0-SNAPSHOT/$jar-3.0.0-SNAPSHOT.jar:"
done
cp="$cp$m2/org/slf4j/slf4j-api/2.0.19/slf4j-api-2.0.19.jar"

javac -encoding UTF-8 -d "$work" -cp "$cp" "$golden/src/StemGolden.java"
java -Dfile.encoding=UTF-8 -cp "$work:$cp" StemGolden "$golden" "$PIN"
