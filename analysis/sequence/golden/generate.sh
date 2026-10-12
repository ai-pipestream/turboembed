#!/bin/sh
# Regenerates the files in expected/, and with TRAIN=1 the fixture models
# in models/ first (which changes their SHA-256, so every beam file
# changes too). See ../README.md.
#
# Usage: generate.sh <opennlp checkout> [<directory of the public models>]
#
# The checkout must be at commit PIN (below), built and installed into the
# local Maven repository (mvn install -DskipTests), with the
# ready/string-pattern-unicode commit fetched, for the code point
# StringPattern files.
set -eu

PIN=cdf33c653d5d61d5af7d37ffee189a83bd8392ea
BRANCH=b2473ca80d65ae2ef10a24b08b1d0ae6a9231109

checkout=$1
models=${2:-}
golden=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

m2=${M2_REPO:-$HOME/.m2/repository}
cp=""
for jar in opennlp-runtime opennlp-api opennlp-ml-commons opennlp-ml-maxent opennlp-ml-perceptron opennlp-ml-bayes; do
    cp="$cp$m2/org/apache/opennlp/$jar/3.0.0-SNAPSHOT/$jar-3.0.0-SNAPSHOT.jar:"
done
cp="$cp$m2/org/slf4j/slf4j-api/2.0.19/slf4j-api-2.0.19.jar"

mkdir -p "$work/classes"
javac -d "$work/classes" -cp "$cp" $(find "$golden/src" -name '*.java')
run() {
    java -cp "$work/classes:$cp" "$@"
}

if [ "${TRAIN:-0}" = 1 ]; then
    run FixtureModels "$golden"
fi

run PatternGolden "$golden" "$golden/expected/pattern-units.txt" "$PIN" utf16-units
run FeatureGolden "$golden" "$golden/expected" "$PIN"
run CodecGolden "$golden/expected" "$PIN"
if [ -n "$models" ]; then
    run BeamGolden "$golden" "$golden/expected" "$PIN" "$models"
else
    run BeamGolden "$golden" "$golden/expected" "$PIN"
fi

# The code point StringPattern: the branch's StringPattern and
# FeatureGeneratorUtil, with its ParamChecks (which commit $PIN does not
# have), ahead of the jars built from $PIN.
overlay="$work/overlay"
mkdir -p "$overlay/src/opennlp/tools/util/featuregen" "$overlay/classes"
git -C "$checkout" show "$BRANCH:opennlp-api/src/main/java/opennlp/tools/util/ParamChecks.java" \
    > "$overlay/src/opennlp/tools/util/ParamChecks.java"
for f in StringPattern FeatureGeneratorUtil; do
    git -C "$checkout" show \
        "$BRANCH:opennlp-core/opennlp-runtime/src/main/java/opennlp/tools/util/featuregen/$f.java" \
        > "$overlay/src/opennlp/tools/util/featuregen/$f.java"
done
javac -d "$overlay/classes" -cp "$cp" $(find "$overlay/src" -name '*.java')
java -cp "$overlay/classes:$work/classes:$cp" PatternGolden "$golden" \
    "$golden/expected/pattern-code-points.txt" "$BRANCH" code-points
