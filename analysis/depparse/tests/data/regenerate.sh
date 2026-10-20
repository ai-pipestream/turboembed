#!/usr/bin/env bash
set -euo pipefail
: "${OPENNLP_CP:?Set OPENNLP_CP to compatible OpenNLP and slf4j jars}"
ref=db39bfe35fb28f47a463a74d13e4721020caec15
out=target/depparse-reference
mkdir -p "$out/source" "$out/classes"
api=opennlp-api/src/main/java/opennlp/tools
runtime=opennlp-core/opennlp-runtime/src/main/java/opennlp/tools
for name in DependencyArc DependencyGraph DependencyParser DependencySample; do
  gh api "repos/ai-pipestream/opennlp/contents/$api/depparse/$name.java?ref=$ref" --jq .content | base64 -d > "$out/source/$name.java"
done
for name in ArcStandardOracle ArcStandardState DependencyAnnotator DependencyContextGenerator DependencyCrossValidator DependencyEvaluator DependencyEventStream DependencyModel DependencyParserME ParserInput Transition; do
  gh api "repos/ai-pipestream/opennlp/contents/$runtime/depparse/$name.java?ref=$ref" --jq .content | base64 -d > "$out/source/$name.java"
done
for entry in util/StringUtil commons/ThreadSafe; do
  gh api "repos/ai-pipestream/opennlp/contents/$api/$entry.java?ref=$ref" --jq .content | base64 -d > "$out/source/${entry##*/}.java"
done
javac -cp "$OPENNLP_CP" -d "$out/classes" "$out"/source/*.java analysis/depparse/tests/data/Reference.java
java -cp "$out/classes:$OPENNLP_CP" opennlp.tools.depparse.Reference analysis/depparse/tests/data/ud-en.conllu analysis/depparse/tests/data/ud-es.conllu > "$out/reference.tsv"
cmp analysis/depparse/tests/data/reference.tsv "$out/reference.tsv"
