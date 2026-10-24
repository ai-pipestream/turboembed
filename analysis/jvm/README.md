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
- `random`: `JavaRandom` is `java.util.Random` from a seed (every method
  OpenNLP uses, `nextGaussian` included) and its `shuffle` is
  `Collections.shuffle(List, Random)`.
- `text`: `double_to_string` and `float_to_string` are `Double.toString`
  and `Float.toString` (event files, model manifests and training
  reports hold numbers in that form); `parse_double`, `parse_float`,
  `parse_int` and `parse_boolean` read numbers as `Double.parseDouble`,
  `Float.parseFloat`, `Integer.parseInt` and `Boolean.parseBoolean` do
  (training parameters, real valued features), hexadecimal floating
  point literals included, except that `parse_int` refuses the decimal
  digits of other scripts, which Java reads.
- `hash_map`: `JavaHash` (`String.hashCode`, `CharBuffer.hashCode`,
  record/pair and `List.hashCode`), `spread` (`HashMap.hash`),
  `HashMapOrder` (allocation-free bucket-sort model of `HashMap`
  iteration order for `put`, `merge` and `clear`, which hands a map with
  a tree bin to `JavaHashMap`), `put_order` and `JavaHashMap` (full
  `HashMap` container for `new HashMap<>()`, `new HashMap<>(n)`, `put`,
  `merge`, `putAll` and `clear` with the default load factor `0.75f`,
  red-black tree bins included). The one order no model can match is
  that of keys with equal hash codes in a tree bin whose class `HashMap`
  cannot compare, which the JVM orders by `System.identityHashCode`;
  `is_jvm_exact` says when that happened.

The only runtime dependency is the standard-library-only `turbo-unicode`,
which supplies the shared UTF-16 length helper.

## Test vectors

`tests/jvm_vectors.rs` compares every function with values a JVM wrote,
bit for bit (a NaN only as a NaN, since Java does not fix a NaN's bit
pattern). The vectors are in `tests/vectors/`:

| File | Contents |
|---|---|
| `strict_math_exp.tsv` | `StrictMath.exp` of about 7,600 inputs: both sides of every branch threshold in `e_exp.c`, the overflow and underflow thresholds, where results turn subnormal, a sweep over the finite range, tiny arguments, zeros, infinities, NaN and random bit patterns |
| `strict_math_log.tsv` | `StrictMath.log` of about 11,500 inputs: values near 1, both sides of every threshold in `e_log.c` at several exponents, every power of two and its neighbours, subnormals, negative values, zeros, infinities, NaN, a logarithmic sweep and random bit patterns |
| `random.tsv` | For nine seeds (0, 1, -1, 42, `Long.MIN_VALUE`, `Long.MAX_VALUE` and others): the first 64 values of each `java.util.Random` method from a fresh generator (`nextInt(bound)` for twelve bounds), a run mixing methods, and `Collections.shuffle` of the list 0..n for eight sizes |
| `compare.tsv` | `String.compareTo` over pairs of strings with ASCII, Latin-1, BMP and supplementary characters |
| `whitespace.tsv` | Every code point for which `Character.isWhitespace` is true |
| `text.tsv` | `Double.toString` and `Float.toString` of values on both sides of every rule of their specification (layout thresholds, subnormals, k x 10^n for every exponent, random bit patterns), and `parseDouble`, `parseFloat`, `parseInt` and `parseBoolean` of texts on both sides of their grammars |
| `hash_map.tsv` | `String.hashCode`, `CharBuffer.hashCode`, 2-component record `hashCode`, `concat` `hashCode` and `List.hashCode`, plus `HashMap` entry iteration order across resize thresholds for `put`, `merge` (with and without trailing duplicates at `threshold + 1`), `putAll` (`tableSizeFor` pre-sizing), `clear` (retained capacity) and 8/9-key colliding bins below `MIN_TREEIFY_CAPACITY` |

Each file's first line names the generator and the JVM that ran it
(`java.version`, `java.vm.version`); the JDK does not expose its Unicode
version at run time. Records are tab separated. A double is its 16 hex
digit IEEE 754 bit pattern, a float is widened to a double (exactly) and
written the same way, an integer is decimal, and a string has every
character outside printable ASCII, and the backslash, written as `\u`
and four hex digits.

The files were made with the JDK named in their headers, the one in the
golden generator's container (`analysis/goldens-tool`), by running the
JDK 21 source launcher in it from the repository root:

```sh
docker run --rm --entrypoint java --user "$(id -u):$(id -g)" \
    -v "$PWD/analysis/jvm:/jvm" <image> /jvm/java/Vectors.java /jvm/tests/vectors
```

`java/Vectors.java` is a single file for the source launcher and needs no
build tool; any JDK 21 gives the same records, and only the header line
names the JDK that ran it.

`tests/jdk_hash_map.rs` compares `JavaHashMap` and `HashMapOrder` with
`java.util.HashMap` on a running JDK 21 (`tests/jdk/HashMapProbe.java`)
over 4000 random scenarios of `String`, `CharBuffer`, `Integer` and
non-`Comparable` keys, `null` included, and of `put`, `merge`, `putAll`,
`clear` and initial capacities, most of them with keys drawn to collide
so that bins become red-black trees and split back into lists. It runs
only with `TURBO_TEST_JDK` set (`java` and `javac` on the PATH);
`TURBO_TEST_JDK_SEED` and `TURBO_TEST_JDK_COUNT` pick the seed and the
number of scenarios.

## What is not here

`Character.isLetter` and `Character.isDigit` are not provided. They are
defined by the Unicode general categories (L* and Nd) of the JDK's
Unicode version, which the standard library does not expose, and a
correct version needs those category tables in the crate.
`is_java_whitespace` is provided because the set it accepts is 25 code
points, listed in its documentation and checked against the JVM for
every code point.
