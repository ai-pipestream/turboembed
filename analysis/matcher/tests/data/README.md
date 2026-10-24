# spaCy pattern oracle

`drugs_patterns.jsonl` is the complete, unmodified 118-rule file from
[Explosion projects](https://github.com/explosion/projects/blob/504e1e428238a08fb038c04af04ea4b3b34756af/tutorials/ner_drugs/assets/drugs_patterns.jsonl).
Its MIT licence is `projects-LICENSE`. SHA-256:
`59ef77c3cd0a0a64b9794346b9d83fc9e2bfcbb6fb4f68b09de92c828ed56457`.

`operators.jsonl` adds project-authored rules for the additional operators.
`spacy_oracle.rs` records spaCy 3.8.7 output on 31 real sentences, 485 tokens
and their gold annotations from
[UD English EWT r2.16](https://github.com/UniversalDependencies/UD_English-EWT/tree/r2.16).
The EWT licence is included as `ewt-LICENSE` (CC BY-SA 4.0); that licence
also applies to the extracted sentences and annotations in the fixture.
The train file SHA-256 is
`3fb78e2b55482c8ee00caa653af436c88c9e5bc9a73ce26e4ae3ddc79d2b7e7b`.

The selection, tokenization constraints and generator are in
`../../reference/spacy_patterns.py`. It selects every reconstructable
sentence matching a published drug rule, twenty short sentences, and ten
short sentences matching fuzzy operators. It compares all matches,
including non-matches, not only the published pattern hits. There are 2,580
matches (two from the published drug rules). The fixture preserves original
text and UTF-8 byte offsets. No model runs in place of the gold annotations.

Regeneration uses the repository's reference container image
`sha256:8d07bd26184d6efe28889ca6a78e3d672f4eaa31a14930f1472cad243476914d`
with spaCy 3.8.7 installed separately for the oracle; it adds no Rust
dependency. Run the generator with the EWT train file, this directory, and
an output Rust filename. Pin the image and spaCy version when repeating.
