# turbo-morfologik

Morfologik finite-state automata readers, morphological dictionary readers,
lemmatizer, POS tag dictionary, and dictionary builder, ported from OpenNLP 3.x
(`opennlp-formats` and `opennlp-morfologik`).

## Overview

- **`FsaReader`, `Fsa5Reader`, and `Cfsa2Reader`**:
  - Read and validate `FSA5` (version `0x05`) and `CFSA2` (version `0xc6`)
    finite-state automata without third-party dependencies.
  - Validate reachable nodes and arcs on load using a non-recursive traversal,
    rejecting truncated headers, out-of-range offsets, and cycles.
  - Support full sequence enumeration in stored lexicographic order as well as
    $O(|word|)$ prefix traversal for morphological lookups.
- **`BaseFormEncoding` (`decode_base_form`, `encode_base_form`)**:
  - Decode and encode relative base forms (`NONE`, `SUFFIX`, `PREFIX`, `INFIX`)
    using unsigned control bytes offset by `'A'` (`0x41`) modulo 256, with `255`
    reserved for full surface-form replacement (`length == 255` or
    `suffix == 255` in `INFIX`, while `position == 255` remains an ordinary byte
    offset).
- **`MorfologikMetadata` and `MorfologikInputConversion`**:
  - Parse UTF-8 `.info` Java properties files (`fsa.dict.separator`,
    `fsa.dict.encoding`, `fsa.dict.encoder`, and optional
    `fsa.dict.input-conversion`), validating single-byte separators in the
    target charset (`UTF-8`, `US-ASCII`, `ISO-8859-1`, `ISO-8859-2`,
    `windows-1250`, `windows-1251`, `windows-1252`).
- **`MorfologikDictionaryReader` and `PoliMorfDictionaryReader`**:
  - Expand Morfologik `.dict` automata or PoliMorf `surfaceForm\tlemma\ttag`
    tables into an `ExpandedDictionaryLemmatizer` with `Locale.ROOT` case
    folding (`turbo_unicode::java::to_lowercase`), first-seen distinct lemma
    ordering, and `StringUtil.isWhitespace` blank-line skipping (including
    `U+00A0` and `U+202F`).
- **`MorfologikDictionary`, `MorfologikLemmatizer`, `MorfologikTagDictionary`,
  and `MorfologikDictionaryBuilder`**:
  - Provide direct FSA-backed lemmatization (`Annotator` writing `opennlp:lemmas`),
    case-sensitive and case-insensitive POS tag dictionary lookups
    (`SequenceValidator<&str>`), and `.txt` + `.info` to `.dict` compilation.
