use tokenizers::normalizers::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::tokenizer::{NormalizedString, PreTokenizedString, PreTokenizer as _};
use tokenizers::{Normalizer as _, OffsetReferential, OffsetType};
use turbo_unicode::normalize::Form;

fn clean(text: &str, strip_accents: bool) -> String {
    let cleaned: String = text
        .chars()
        .filter(|&c| {
            !(c == '\0' || c == '\u{fffd}' || (!matches!(c, '\t' | '\n' | '\r') && turbo_unicode::hf::is_other(c)))
        })
        .map(|c| if matches!(c, '\t' | '\n' | '\r') || c.is_whitespace() { ' ' } else { c })
        .collect();
    if strip_accents {
        turbo_unicode::hf::aligned_chars(Form::Nfd, cleaned.chars())
            .map(|(c, _)| c)
            .filter(|&c| !turbo_unicode::hf::is_mark_nonspacing(c))
            .collect()
    } else {
        cleaned
    }
}

fn upstream_clean(normalizer: &BertNormalizer, text: &str) -> String {
    let mut normalized = NormalizedString::from(text);
    normalizer.normalize(&mut normalized).unwrap();
    normalized.get().to_owned()
}

fn our_words(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut begin = 0;
    for (i, c) in text.char_indices() {
        if c.is_ascii_punctuation() || (!c.is_ascii() && turbo_unicode::hf::is_punctuation(c)) {
            if begin < i {
                out.push(text[begin..i].to_owned());
            }
            out.push(c.to_string());
            begin = i + c.len_utf8();
        }
    }
    if begin < text.len() {
        out.push(text[begin..].to_owned());
    }
    out
}

fn upstream_words(text: &str) -> Vec<String> {
    let mut pre = PreTokenizedString::from(text);
    BertPreTokenizer.pre_tokenize(&mut pre).unwrap();
    pre.get_splits(OffsetReferential::Original, OffsetType::Byte)
        .into_iter()
        .map(|(piece, _, _)| piece.to_owned())
        .collect()
}

#[test]
fn category_and_nfd_tables_match_upstream_on_every_scalar() {
    let plain = BertNormalizer::new(true, false, Some(false), false);
    let stripped = BertNormalizer::new(true, false, Some(true), false);
    for c in (0..=0x10ffffu32).filter_map(char::from_u32) {
        let text = format!("a{c}b");
        assert_eq!(clean(&text, false), upstream_clean(&plain, &text), "clean_text for {c:?}");
        assert_eq!(clean(&text, true), upstream_clean(&stripped, &text), "NFD and Mn for {c:?}");
        if !c.is_whitespace() {
            assert_eq!(our_words(&text), upstream_words(&text), "punctuation for {c:?}");
        }
        let mut upstream = NormalizedString::from(text.as_str());
        upstream.nfc();
        let ours: String = turbo_unicode::hf::aligned(Form::Nfc, &text).map(|(c, _)| c).collect();
        assert_eq!(ours, upstream.get(), "NFC for {c:?}");
    }
}
