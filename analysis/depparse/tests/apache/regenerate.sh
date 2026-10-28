#!/usr/bin/env bash
set -euo pipefail
: "${OPENNLP_SOURCE:?Set OPENNLP_SOURCE to a checkout containing the pinned Apache commit}"
: "${OPENNLP_CP:?Set OPENNLP_CP to compatible OpenNLP and slf4j jars}"
ref=95695633b580a445617a791f05a02b61b65b3b74
out=target/apache-dependency-reference
mkdir -p "$out/source" "$out/classes" "$out/data"
api=opennlp-api/src/main/java/opennlp/tools
runtime=opennlp-core/opennlp-runtime/src/main/java/opennlp/tools
formats=opennlp-core/opennlp-formats/src/main/java/opennlp/tools/formats
for name in DependencyArc DependencyParser DependencySample DependencyTree; do
  git -C "$OPENNLP_SOURCE" show "$ref:$api/depparse/$name.java" > "$out/source/$name.java"
done
for name in ParamChecks ResourceLimits; do
  git -C "$OPENNLP_SOURCE" show "$ref:$api/util/$name.java" > "$out/source/$name.java"
done
for name in ConlluTagset ConlluDependencySampleStream; do
  git -C "$OPENNLP_SOURCE" show "$ref:$formats/conllu/$name.java" > "$out/source/$name.java"
done
for name in ArcStandardOracle ArcStandardState DependencyContextGenerator DependencyEventStream DependencyModel DependencyParserME Transition FeedforwardContext FeedforwardDependencyModel FeedforwardDependencyParser FeedforwardDependencyTrainer FeedforwardNetwork ArcStandardBeam ContributionCache AttachmentScores; do
  git -C "$OPENNLP_SOURCE" show "$ref:$runtime/depparse/$name.java" > "$out/source/$name.java"
done
javac -cp "$OPENNLP_CP" -d "$out/classes" "$out"/source/*.java analysis/depparse/tests/apache/Generate.java
java -cp "$out/classes:$OPENNLP_CP" opennlp.tools.depparse.Generate "$out/data" analysis/depparse/tests/data/ud-en.conllu analysis/depparse/tests/data/ud-es.conllu
for name in apache apache-refined linear; do
  cmp "$out/data/$name.tsv" "analysis/depparse/tests/apache/$name.tsv"
  entry=feedforward.network
  if [[ $name == linear ]]; then entry=depparse.model; fi
  mkdir -p "$out/$name-actual" "$out/$name-expected"
  root=$PWD
  (cd "$out/$name-actual"; jar xf "$root/$out/data/$name.bin" "$entry")
  (cd "$out/$name-expected"; jar xf "$root/analysis/depparse/tests/apache/$name.bin" "$entry")
  cmp "$out/$name-actual/$entry" "$out/$name-expected/$entry"
done
