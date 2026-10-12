# turbo-jvm

Helpers that reproduce JVM behaviour bit for bit, for the Rust port of
Apache OpenNLP 3.x in the analysis crates. OpenNLP runs on Java, and some
of its results depend on the platform rather than on OpenNLP:

- `chars`: Java `char` values are UTF-16 code units. Context generators,
  `StringPattern` and character n-grams work on them, and
  `String.compareTo` (BeamSearch tie breaks) orders strings by them.
  `Utf16Units` yields the UTF-16 units of a `&str` with the UTF-8 byte
  offset of each unit's character; `compare_to` and `compare` are
  `String.compareTo`; `is_java_whitespace` is `Character.isWhitespace`.
- `strict_math`: `exp` and `log` are ports of fdlibm (`e_exp.c`,
  `e_log.c`), which is what `StrictMath.exp` and `StrictMath.log` compute
  and what OpenNLP's maxent, perceptron and naive Bayes models call.

The crate has no dependencies.
