//! The recipe subset `PREFIX\d+SUFFIX`, matched against the whole token.

#[derive(Debug)]
pub(super) struct SkipRule {
    prefix: String,
    suffix: String,
}

impl SkipRule {
    pub(super) fn parse(pattern: &str) -> Result<Self, String> {
        let bad =
            || format!("distill.skip pattern {pattern:?}: expected literal PREFIX\\d+SUFFIX with one ASCII digit run");
        let mut prefix = String::new();
        let mut suffix = String::new();
        let mut digits = false;
        let mut chars = pattern.chars();
        while let Some(c) = chars.next() {
            let literal = if c == '\\' {
                match chars.next() {
                    Some('d') if !digits && chars.next() == Some('+') => {
                        digits = true;
                        continue;
                    }
                    Some(c) if c.is_ascii_punctuation() => c,
                    _ => return Err(bad()),
                }
            } else {
                if ".*+?[](){}^$|".contains(c) {
                    return Err(bad());
                }
                c
            };
            if digits { &mut suffix } else { &mut prefix }.push(literal);
        }
        if !digits {
            return Err(bad());
        }
        Ok(Self { prefix, suffix })
    }

    pub(super) fn is_match(&self, text: &str) -> bool {
        text.strip_prefix(&self.prefix)
            .and_then(|rest| rest.strip_suffix(&self.suffix))
            .is_some_and(|middle| !middle.is_empty() && middle.bytes().all(|b| b.is_ascii_digit()))
    }
}

#[cfg(test)]
mod tests {
    use super::SkipRule;

    #[test]
    fn literal_edges_and_ascii_digits_match_the_whole_token() {
        let rule = SkipRule::parse(r"\[unused\d+\]").unwrap();
        for s in ["[unused0]", "[unused993]", "[unused001]"] {
            assert!(rule.is_match(s), "{s}");
        }
        for s in ["[UNK]", "[CLS]", "unused", "[unused١]", "[unused]", "x[unused1]", "[unused1]\n"] {
            assert!(!rule.is_match(s), "{s}");
        }
        assert!(SkipRule::parse(r"\d+").unwrap().is_match("123"));
        assert!(SkipRule::parse(r"é\.\d+\+中").unwrap().is_match("é.123+中"));
        assert!(SkipRule::parse(r"a\d+12").unwrap().is_match("a312"));
        assert!(!SkipRule::parse(r"a\d+12").unwrap().is_match("a12"));
        for p in b'!'..=b'~' {
            if p.is_ascii_punctuation() {
                let pattern = format!("\\{}\\d+\\{}", p as char, p as char);
                assert!(SkipRule::parse(&pattern).unwrap().is_match(&format!("{}7{}", p as char, p as char)));
            }
        }
    }

    #[test]
    fn unsupported_patterns_name_the_pattern() {
        for pattern in [r"a|b", r"[0-9]+", r"\w+", r"\d+\d+", r"\[unused\d*\]", "", r"abc", r"\d+.", r"\d+\"] {
            let err = SkipRule::parse(pattern).unwrap_err();
            assert!(err.contains(&format!("{pattern:?}")), "{err}");
        }
    }

    #[test]
    fn real_vocabulary_and_both_recipe_patterns_keep_the_same_ids() {
        let tokenizer: serde_json::Value =
            serde_json::from_str(include_str!("../../../testdata/all-minilm-l6-v2/tokenizer.json")).unwrap();
        let vocab = tokenizer["model"]["vocab"].as_object().unwrap();
        assert_eq!(vocab.len(), 30_522);
        for recipe in [
            include_str!("../../recipes/all-minilm-l6-v2-static.json"),
            include_str!("../../recipes/bge-base-en-v1.5-static.json"),
        ] {
            let recipe: serde_json::Value = serde_json::from_str(recipe).unwrap();
            let rule = SkipRule::parse(recipe["distill"]["skip"].as_str().unwrap()).unwrap();
            let skipped: std::collections::BTreeSet<_> = vocab.keys().filter(|t| rule.is_match(t)).cloned().collect();
            let expected = (0..994).map(|i| format!("[unused{i}]")).collect();
            assert_eq!(skipped, expected);
        }
    }
}
