//! `DefaultLemmatizerContextGenerator`: the features of a token for the
//! maxent lemmatizer.

use turbo_maxent::JavaString;
use turbo_sequence::{ContextGenerator, Features, TokenTag};
use turbo_unicode::utf16::utf16_len;

/// `PREFIX_LENGTH`: the prefix features are the first 2 to 5 units.
const PREFIX_LENGTH: usize = 5;
/// `SUFFIX_LENGTH`: the suffix features are the last 2 to 7 units.
const SUFFIX_LENGTH: usize = 7;

/// The features of `DefaultLemmatizerContextGenerator.getContext`, in its
/// order: the word (`w0=`), the tag (`t0=`), the previous lemma class
/// (`p_1=`, `bos` for the first token) alone and joined to each of the
/// two, the suffixes and prefixes of the word in UTF-16 units, `h` for a
/// hyphen, `c` for an ASCII capital letter and `d` for an ASCII digit.
/// The previous class is a [`JavaString`], written into the feature as
/// its modified UTF-8: a class holding half of a surrogate pair gives the
/// feature the JVM looks up.
///
/// Java fills the suffix and prefix arrays from index 1, so each array's
/// first entry is null and the features `suf=null` and `pre=null` are in
/// every context. The models were trained with them, so they are
/// written here as well.
#[derive(Clone, Copy, Debug, Default)]
pub struct LemmaContext;

impl LemmaContext {
    /// Appends the features of `word` tagged `tag` to `out`, given the
    /// class decided for the token before it.
    pub fn features(&self, out: &mut Features, word: &str, tag: &str, previous: Option<&JavaString>) {
        let w0 = |out: &mut Features| {
            out.push_str("w0=").push_str(word);
        };
        let t0 = |out: &mut Features| {
            out.push_str("t0=").push_str(tag);
        };
        let p1 = |out: &mut Features| {
            out.push_str("p_1=");
            match previous {
                Some(class) => out.push_modified_utf8(class.as_bytes()),
                None => out.push_str("bos"),
            };
        };
        w0(out);
        out.finish();
        t0(out);
        out.finish();
        p1(out);
        out.finish();
        p1(out);
        t0(out);
        out.finish();
        p1(out);
        w0(out);
        out.finish();
        let len = utf16_len(word);
        out.add("suf=null");
        for li in 1..SUFFIX_LENGTH {
            out.push_str("suf=").push_units(word, len.saturating_sub(li + 1), len).finish();
        }
        out.add("pre=null");
        for li in 1..PREFIX_LENGTH {
            out.push_str("pre=").push_units(word, 0, (li + 1).min(len)).finish();
        }
        if word.contains('-') {
            out.add("h");
        }
        if word.bytes().any(|b| b.is_ascii_uppercase()) {
            out.add("c");
        }
        if word.bytes().any(|b| b.is_ascii_digit()) {
            out.add("d");
        }
    }
}

impl<'a> ContextGenerator<TokenTag<'a>, JavaString> for LemmaContext {
    fn context(&mut self, index: usize, sequence: &[TokenTag<'a>], prior: &[&JavaString], out: &mut Features) {
        let Some(&item) = sequence.get(index) else { return };
        let previous = index.checked_sub(1).and_then(|i| prior.get(i).copied());
        self.features(out, item.token, item.tag, previous);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn features(word: &str, tag: &str, previous: Option<&str>) -> Vec<String> {
        let mut out = Features::new();
        LemmaContext.features(&mut out, word, tag, previous.map(JavaString::new).as_ref());
        (0..out.len()).map(|i| out.to_string_lossy(i)).collect()
    }

    #[test]
    fn the_features_of_a_first_token() {
        assert_eq!(
            features("Running", "VERB", None),
            [
                "w0=Running",
                "t0=VERB",
                "p_1=bos",
                "p_1=bost0=VERB",
                "p_1=bosw0=Running",
                "suf=null",
                "suf=ng",
                "suf=ing",
                "suf=ning",
                "suf=nning",
                "suf=unning",
                "suf=Running",
                "pre=null",
                "pre=Ru",
                "pre=Run",
                "pre=Runn",
                "pre=Runni",
                "c",
            ]
        );
    }

    #[test]
    fn the_previous_class_hyphen_and_digit() {
        let f = features("x-1", "NUM", Some("D0s"));
        assert_eq!(&f[2..5], ["p_1=D0s", "p_1=D0st0=NUM", "p_1=D0sw0=x-1"]);
        // A short word repeats itself in the longer suffixes and prefixes.
        assert_eq!(&f[5..12], ["suf=null", "suf=-1", "suf=x-1", "suf=x-1", "suf=x-1", "suf=x-1", "suf=x-1"]);
        assert_eq!(&f[12..17], ["pre=null", "pre=x-", "pre=x-1", "pre=x-1", "pre=x-1"]);
        assert_eq!(&f[17..], ["h", "d"]);
    }

    #[test]
    fn a_previous_class_with_half_a_surrogate_pair_is_the_jvms_feature() {
        // The class of "😀" with the lemma "_" (the goldens' conllu-whitespace
        // sample), followed by "x": p_1= holds both halves alone, as
        // modified UTF-8.
        let class = JavaString::from_utf16(&[0x52, 0x31, 0xDE00, 0x5F, 0x44, 0x30, 0xD83D]);
        let mut out = Features::new();
        LemmaContext.features(&mut out, "x", "_", Some(&class));
        assert_eq!(out.get(2), b"p_1=R1\xED\xB8\x80_D0\xED\xA0\xBD");
        assert_eq!(out.to_string_lossy(3), "p_1=R1\u{FFFD}_D0\u{FFFD}t0=_");
        assert_eq!(out.to_string_lossy(4), "p_1=R1\u{FFFD}_D0\u{FFFD}w0=x");
        let mut context = LemmaContext;
        let mut by_search = Features::new();
        context.context(
            1,
            &[TokenTag { token: "\u{1F600}", tag: "_" }, TokenTag { token: "x", tag: "_" }],
            &[&class],
            &mut by_search,
        );
        assert_eq!(by_search, out);
    }

    #[test]
    fn affixes_are_cut_at_utf16_units() {
        let f = features("\u{1F600}ab", "SYM", None);
        // The two-unit emoji counts as two: the suffix of two units is "ab",
        // of three is the low surrogate and "ab".
        assert_eq!(f[6], "suf=ab");
        assert_eq!(f[7], "suf=\u{FFFD}ab");
        assert_eq!(f[13], "pre=\u{1F600}");
        assert!(!f.contains(&"c".to_string()));
    }

    #[test]
    fn capitals_and_digits_outside_ascii_do_not_count() {
        let f = features("Ärger٣", "NOUN", None);
        assert!(!f.contains(&"c".to_string()) && !f.contains(&"d".to_string()));
    }

    #[test]
    fn context_guards_out_of_bounds_index_and_short_prior() {
        let mut context = LemmaContext;
        let mut out = Features::new();
        let seq = [TokenTag { token: "dogs", tag: "NNS" }, TokenTag { token: "bark", tag: "VBP" }];
        context.context(5, &seq, &[], &mut out);
        assert_eq!(out.len(), 0);
        context.context(1, &seq, &[], &mut out);
        assert_eq!(out.to_string_lossy(2), "p_1=bos");
    }
}
