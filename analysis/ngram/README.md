# turbo-ngram

N-gram frequency models, character n-gram models, n-gram probability utilities,
and Stupid Backoff language modeling for `turboembed`, porting Apache OpenNLP's
`opennlp.tools.ngram` and `opennlp.tools.languagemodel` packages.

## Components

- **`NGramModel` (`opennlp.tools.ngram.NGramModel`)**:
  - Stores case-sensitive token n-grams (`StringList`) and their `i32` counts in
    insertion order (`LinkedHashMap` semantics).
  - Supports zero-allocation `&[&str]` lookups (`contains`, `get_count`,
    `set_count`, `remove`), contiguous sub-span extraction (`add_range`),
    UTF-16 character n-gram extraction (`add_chars`), frequency cutoffs
    (`cutoff`), conversion to `turbo_maxent::Dictionary` (`to_dictionary`), and
    OpenNLP `<dictionary>` XML serialization (`read` and `serialize`).
- **`NGramCharModel` (`opennlp.tools.ngram.NGramCharModel`)**:
  - Counts character n-grams lowercased with Java's `Character.toLowerCase(int)`
    (`StringUtil.toLowerCase` and `StringUtil.toLowerCaseCharBuffer`).
- **`NGramGenerator` and `NGramUtils` (`opennlp.tools.ngram`)**:
  - `NGramGenerator::generate` and `NGramGenerator::generate_chars` build joined
    n-gram strings with a separator.
  - `NGramUtils` computes Maximum Likelihood (unigram, bigram, trigram, n-gram),
    Laplace-smoothed, prior-smoothed, linearly interpolated, and missing-mass
    probabilities.
- **`LanguageModel` and `NGramLanguageModel` (`opennlp.tools.languagemodel`)**:
  - Scores token sequences (`calculate_probability`) and predicts continuation
    n-grams (`predict_next_tokens`) using Stupid Backoff (`0.4` backoff weight)
    with `turbo_jvm::strict_math::{log, exp}` for bit-for-bit JVM parity.
