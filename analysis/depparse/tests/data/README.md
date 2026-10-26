# Reference fixtures

`reference.tsv` was generated with `Reference.java` against OpenNLP commit
`db39bfe35fb28f47a463a74d13e4721020caec15` on
[OPENNLP-1920-relation-extraction](https://github.com/ai-pipestream/opennlp/tree/db39bfe35fb28f47a463a74d13e4721020caec15).
JDK: OpenJDK GraalVM CE 25.0.3, javac 25.0.3, x86_64.

Coverage: 7 inputs, 152 oracle contexts with 37 ordered features, Unicode,
valency/distance caps, non-projectivity and greedy prediction ties.
Java QNModel configurations: `s0t=NOUN` with SHIFT/LEFT_ARC:dep/RIGHT_ARC:dep
weights 0/1/2; equal-score RIGHT_ARC:dep/SHIFT. Rust loads matching QN data.
These are parity fixtures, not accuracy benchmarks.

## Regeneration

Set `OPENNLP_CP` to compatible OpenNLP API/runtime/ML-commons/ML-maxent and
slf4j jars. Generation used 3.0.0-SNAPSHOT and slf4j 2.0.17. Run
`regenerate.sh` from the repository root. Pinned sources compile ahead of
the jars. Output goes under `target/depparse-reference/` for comparison.
CI uses committed data without JVM or network access.

## Attribution

`ud-en.conllu`: first 2 blocks from
[English EWT dev](https://github.com/UniversalDependencies/UD_English-EWT/blob/4a4d77f599ea53cc405f85d0cec4b2f14f81d42b/en_ewt-ud-dev.conllu).
Annotations copyright 2013-2021 The Board of Trustees of The Leland Stanford
Junior University. Annotations/database rights: CC BY-SA 4.0. Underlying text
rights remain with original authors, as documented by the treebank README.

`ud-es.conllu`: first 2 MWT-containing blocks from
[Spanish GSD test](https://github.com/UniversalDependencies/UD_Spanish-GSD/blob/267f3530d4f122ee85d1891800211a06dfb79347/es_gsd-ud-test.conllu).
Credit: UD Spanish GSD contributors, including Miguel Ballesteros, Dan Zeman
and Héctor Martínez Alonso; converted from the legacy UD treebank. CC BY-SA 4.0.

Excerpt text and annotations are unchanged. Upstream licenses are included.

`ud-en-nonprojective.conllu`: the unchanged 16-word sentence
`weblog-juancole.com_juancole_20051126063000_ENG_20051126_063000-0022` from
[English EWT r2.14 train](https://github.com/UniversalDependencies/UD_English-EWT/blob/b70ecaec872a5807646e889334f9e224ec07c3b1/en_ewt-ud-train.conllu).
It tests explicit non-projective training exclusions. The English EWT
attribution, CC BY-SA 4.0 annotation/database license and underlying-text
rights above apply to this excerpt too.
UD-derived golden features/predictions retain CC BY-SA 4.0 and this attribution.
OpenNLP is Apache-2.0. The harness is project code.

`ud-fi-ftb-crossing.conllu` is unchanged sentence `pg1os-206` from
UD Finnish-FTB r2.14 train at commit
`948615ba5289828a8d5cf7161d9cedf803e9efe6`, distributed under the CC BY 4.0
alternative in that repository's LICENSE.txt. Copyright 2014 FIN-CLARIN;
text adapted from VISK grammar examples, annotations by FIN-CLARIN,
University of Helsinki. Source train SHA256:
`5d2be0424cb2e5e125505faf523e2d5539279f3fe7bb6ec9596fe98af475a731`.
The fixture tests crossing-tree handling, not held-out accuracy.
