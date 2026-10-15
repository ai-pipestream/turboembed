//! OpenNLP's document categorizer feature generators: the strings a
//! document's tokens give, in the order `generateContext` collects them
//! (each generator's features in turn, in the order the generators are
//! listed).

use std::fmt;

/// A `FeatureGenerator` of `opennlp.tools.doccat`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FeatureGenerator {
    /// `BagOfWordsFeatureGenerator()`: `bow=` and the token, for every
    /// token.
    BagOfWords,
    /// `NGramFeatureGenerator(min, max)`: from each token, `ng=` and then
    /// `:` and a token for each of the next `min` to `max` tokens, one
    /// feature per length.
    NGram { min: u32, max: u32 },
}

/// The range `NGramFeatureGenerator` refuses: a bound below 1 or a minimum
/// above the maximum.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NGramRangeError {
    pub min: i64,
    pub max: i64,
}

impl fmt::Display for NGramRangeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.min > 0 && self.max > 0 {
            write!(f, "n-gram range {}..={}: the minimum is above the maximum", self.min, self.max)
        } else {
            write!(f, "n-gram range {}..={}: both bounds must be at least 1", self.min, self.max)
        }
    }
}

impl std::error::Error for NGramRangeError {}

impl FeatureGenerator {
    /// `new NGramFeatureGenerator(min, max)`, with its checks.
    pub fn ngram(min: i64, max: i64) -> Result<FeatureGenerator, NGramRangeError> {
        if min > 0 && max > 0 && min <= max && max <= i64::from(u32::MAX) {
            Ok(FeatureGenerator::NGram { min: min as u32, max: max as u32 })
        } else {
            Err(NGramRangeError { min, max })
        }
    }

    /// Calls `emit` with each feature of `tokens`, built in `buf`.
    pub fn generate(&self, tokens: &[&str], buf: &mut String, emit: &mut impl FnMut(&str)) {
        match *self {
            FeatureGenerator::BagOfWords => {
                for t in tokens {
                    buf.clear();
                    buf.push_str("bow=");
                    buf.push_str(t);
                    emit(buf);
                }
            }
            FeatureGenerator::NGram { min, max } => {
                let (min, max) = (min as usize, max as usize);
                if tokens.len() < min {
                    return;
                }
                for i in 0..=tokens.len() - min {
                    buf.clear();
                    buf.push_str("ng=");
                    for (y, t) in tokens[i..].iter().take(max).enumerate() {
                        buf.push(':');
                        buf.push_str(t);
                        if y + 1 >= min {
                            emit(buf);
                        }
                    }
                }
            }
        }
    }
}

/// Every feature of every generator, in turn.
pub fn for_each_feature(
    generators: &[FeatureGenerator],
    tokens: &[&str],
    buf: &mut String,
    mut emit: impl FnMut(&str),
) {
    for g in generators {
        g.generate(tokens, buf, &mut emit);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn features(generators: &[FeatureGenerator], tokens: &[&str]) -> Vec<String> {
        let mut out = Vec::new();
        for_each_feature(generators, tokens, &mut String::new(), |f| out.push(f.to_string()));
        out
    }

    const TOKENS: [&str; 7] = ["a", "b", "c", "d", "e", "f", "g"];

    // BagOfWordsFeatureGeneratorTest: testEmpty, testUseAllTokens.
    #[test]
    fn bag_of_words_prefixes_every_token() {
        let bow = [FeatureGenerator::BagOfWords];
        assert!(features(&bow, &[]).is_empty());
        assert_eq!(
            features(&bow, &["it", "is", "12.345", "feet", "long"]),
            ["bow=it", "bow=is", "bow=12.345", "bow=feet", "bow=long"]
        );
    }

    // NGramFeatureGeneratorTest: testEmpty and the five ranges.
    #[test]
    fn ngrams_join_tokens_after_a_colon_each() {
        let ngram = |min, max| [FeatureGenerator::ngram(min, max).unwrap()];
        assert!(features(&ngram(2, 2), &[]).is_empty());
        assert_eq!(features(&ngram(1, 1), &TOKENS), ["ng=:a", "ng=:b", "ng=:c", "ng=:d", "ng=:e", "ng=:f", "ng=:g"]);
        assert_eq!(features(&ngram(2, 2), &TOKENS), ["ng=:a:b", "ng=:b:c", "ng=:c:d", "ng=:d:e", "ng=:e:f", "ng=:f:g"]);
        assert_eq!(features(&ngram(3, 3), &TOKENS), ["ng=:a:b:c", "ng=:b:c:d", "ng=:c:d:e", "ng=:d:e:f", "ng=:e:f:g"]);
        assert_eq!(
            features(&ngram(1, 2), &TOKENS),
            [
                "ng=:a", "ng=:a:b", "ng=:b", "ng=:b:c", "ng=:c", "ng=:c:d", "ng=:d", "ng=:d:e", "ng=:e", "ng=:e:f",
                "ng=:f", "ng=:f:g", "ng=:g"
            ]
        );
        assert_eq!(
            features(&ngram(1, 3), &TOKENS),
            [
                "ng=:a",
                "ng=:a:b",
                "ng=:a:b:c",
                "ng=:b",
                "ng=:b:c",
                "ng=:b:c:d",
                "ng=:c",
                "ng=:c:d",
                "ng=:c:d:e",
                "ng=:d",
                "ng=:d:e",
                "ng=:d:e:f",
                "ng=:e",
                "ng=:e:f",
                "ng=:e:f:g",
                "ng=:f",
                "ng=:f:g",
                "ng=:g"
            ]
        );
        assert!(features(&ngram(2, 3), &["x"]).is_empty());
        assert_eq!(features(&ngram(2, 3), &["x", "y"]), ["ng=:x:y"]);
    }

    #[test]
    fn generators_run_in_turn() {
        let g = [FeatureGenerator::BagOfWords, FeatureGenerator::ngram(2, 2).unwrap()];
        assert_eq!(features(&g, &["x", "y"]), ["bow=x", "bow=y", "ng=:x:y"]);
    }

    // NGramFeatureGeneratorTest: testInvalidGramSize1 and 2.
    #[test]
    fn an_ngram_range_needs_positive_ordered_bounds() {
        assert_eq!(FeatureGenerator::ngram(0, 2), Err(NGramRangeError { min: 0, max: 2 }));
        assert_eq!(FeatureGenerator::ngram(2, -1), Err(NGramRangeError { min: 2, max: -1 }));
        assert_eq!(FeatureGenerator::ngram(3, 2), Err(NGramRangeError { min: 3, max: 2 }));
        assert!(FeatureGenerator::ngram(3, 2).unwrap_err().to_string().contains("above the maximum"));
        assert!(FeatureGenerator::ngram(0, 2).unwrap_err().to_string().contains("at least 1"));
    }
}
