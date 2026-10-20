#!/usr/bin/env bash
set -euo pipefail
: "${OPENNLP_SOURCE:?Set OPENNLP_SOURCE to the pinned OpenNLP checkout}"
: "${OPENNLP_CP:?Set OPENNLP_CP to compatible OpenNLP and slf4j jars}"
ref=cdf33c653d5d61d5af7d37ffee189a83bd8392ea
test "$(git -C "$OPENNLP_SOURCE" rev-parse HEAD)" = "$ref"
out=target/feedforward-reference
mkdir -p "$out/source" "$out/classes" "$out/data"
api=opennlp-api/src/main/java/opennlp/tools
runtime=opennlp-core/opennlp-runtime/src/main/java/opennlp/tools
for name in DependencyArc DependencyGraph DependencyParser DependencySample; do
  git -C "$OPENNLP_SOURCE" show "$ref:$api/depparse/$name.java" > "$out/source/$name.java"
done
for name in ArcStandardOracle ArcStandardState ParserInput Transition FeedforwardContext FeedforwardDependencyModel FeedforwardDependencyParser FeedforwardDependencyTrainer; do
  git -C "$OPENNLP_SOURCE" show "$ref:$runtime/depparse/$name.java" > "$out/source/$name.java"
done
for entry in util/StringUtil commons/ThreadSafe; do
  git -C "$OPENNLP_SOURCE" show "$ref:$api/$entry.java" > "$out/source/${entry##*/}.java"
done
javac -cp "$OPENNLP_CP" -d "$out/classes" "$out"/source/*.java analysis/depparse/tests/feedforward/data/NeuralReference.java
java -cp "$out/classes:$OPENNLP_CP" opennlp.tools.depparse.NeuralReference "$out/data"
for file in plain.bin dropout.bin plain.tsv dropout.tsv normalize.tsv; do
  cmp "analysis/depparse/tests/feedforward/data/$file" "$out/data/$file"
done
