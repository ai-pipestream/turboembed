#!/bin/sh
# Regenerates the files in expected/. See ../README.md.
#
# Usage: generate.sh
#
# OpenNLP must be built at commit PIN (below) and installed into the local
# Maven repository (mvn install -DskipTests), and javac and java must be a
# JDK 21.
set -eu

PIN=cdf33c653d5d61d5af7d37ffee189a83bd8392ea

golden=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

m2=${M2_REPO:-$HOME/.m2/repository}
cp=""
for jar in opennlp-runtime opennlp-api opennlp-ml-commons opennlp-ml-maxent opennlp-ml-perceptron opennlp-ml-bayes; do
    cp="$cp$m2/org/apache/opennlp/$jar/3.0.0-SNAPSHOT/$jar-3.0.0-SNAPSHOT.jar:"
done
cp="$cp$m2/org/slf4j/slf4j-api/2.0.19/slf4j-api-2.0.19.jar"

javac -encoding UTF-8 -d "$work" -cp "$cp" "$golden/src/NameFindGolden.java"
java -Dfile.encoding=UTF-8 -cp "$work:$cp" NameFindGolden "$golden" "$PIN"
