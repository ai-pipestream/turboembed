//! Real jieba dictionary rows and its trained BMES model, without mocks.
use turbo_analysis::{Annotator, Document, Span, Spans, View};
use turbo_analysis_layers::Sentences;
use turbo_tokenize::{UnigramLexicon, UnigramMode, UnigramSegmenter, WhitespaceMode, unigram::Scratch};
use turbo_unicode::utf16::{byte_to_utf16, utf16_to_byte};

fn segmenter(mode: UnigramMode) -> UnigramSegmenter {
    let lexicon =
        UnigramLexicon::parse(include_bytes!("../data/jieba-hmm/test-dict.txt"), WhitespaceMode::Unicode).unwrap();
    UnigramSegmenter::new(lexicon, WhitespaceMode::Unicode).with_mode(mode)
}

fn words<'a>(s: &UnigramSegmenter, text: &'a str) -> Vec<&'a str> {
    let mut out = Vec::new();
    s.segment(text, &mut out);
    out.iter().map(|s| &text[s.range()]).collect()
}

#[test]
fn real_names_and_unknown_words_with_dictionary_anchors() {
    let s = segmenter(UnigramMode::Chinese);
    // Jieba examples with the 41-row source subset, checked with jieba-rs 0.11.0.
    assert_eq!(words(&s, "小明硕士毕业于中国科学院计算所"), ["小明", "硕士", "毕业", "于", "中国科学院", "计算", "所"]);
    assert_eq!(words(&s, "他来到了网易杭研大厦"), ["他来", "到", "了", "网易", "杭研", "大厦"]);
    let compatibility = s.clone().with_mode(UnigramMode::OpenNlp);
    assert_eq!(words(&compatibility, "杭研"), ["杭", "研"]);
    assert_eq!(words(&s, "𠀀杭研🙂"), ["𠀀", "杭研", "🙂"]);
}

#[test]
fn ascii_runs_are_atomic_and_other_scripts_do_not_enter_the_hmm() {
    for mode in [UnigramMode::Chinese, UnigramMode::Dictionary] {
        let s = segmenter(mode);
        assert_eq!(words(&s, "abc123北京2026年"), ["abc123", "北京", "2026", "年"]);
        assert_eq!(words(&s, "a12.5%é৯９ＡБ🙂"), ["a12", ".", "5", "%", "é", "৯", "９", "Ａ", "Б", "🙂"]);
        assert!(words(&s, "").is_empty());
    }
}

#[test]
fn scratch_appends_and_preserves_every_non_whitespace_byte() {
    let text = "\u{85}𠀀杭研🙂 abc123\u{a0}北京2026年!\n小明";
    let mut scratch = Scratch::new();
    let mut out = vec![Span::new(999, 1000)];
    for mode in [UnigramMode::Chinese, UnigramMode::Dictionary, UnigramMode::OpenNlp] {
        out.truncate(1);
        segmenter(mode).segment_with(text, &mut scratch, &mut out);
        assert_eq!(out[0], Span::new(999, 1000));
        let mut end = 0;
        for span in &out[1..] {
            assert!(span.start < span.end);
            assert!(text[end..span.start as usize].chars().all(|c| WhitespaceMode::Unicode.is_whitespace(c as u32)));
            for offset in [span.start, span.end] {
                let unit = byte_to_utf16(text, offset as usize).unwrap();
                assert_eq!(utf16_to_byte(text, unit), Some(offset as usize));
            }
            end = span.end as usize;
        }
        let reconstructed: String = out[1..].iter().map(|s| &text[s.range()]).collect();
        let expected: String = text.chars().filter(|&c| !WhitespaceMode::Unicode.is_whitespace(c as u32)).collect();
        assert_eq!(reconstructed, expected);
    }
}

#[test]
fn sentence_ownership_and_model_policy_provenance() {
    let text = "杭研。 abc123北京";
    let doc = Document::new(text);
    let view = doc.get::<View>().unwrap();
    let mut spans = Spans::new(view.id());
    spans.push(Span::new(0, 9));
    spans.push(Span::new(10, text.len() as u32));
    let sentences = Sentences { spans };
    let mut hashes = Vec::new();
    for mode in [UnigramMode::Chinese, UnigramMode::Dictionary, UnigramMode::OpenNlp] {
        let s = segmenter(mode);
        hashes.push(s.config_hash());
        let (tokens,) = s.annotate((view, Some(&sentences))).unwrap();
        let owners = tokens.sentence.as_ref().unwrap();
        for ((_, token), owner) in tokens.spans.iter().zip(owners) {
            let sentence = sentences.spans.get(*owner);
            assert!(sentence.start <= token.start && token.end <= sentence.end);
            assert!(!text[token.range()].is_empty());
        }
        assert_eq!(s.model().unwrap().sha256, s.lexicon().sha256().to_hex());
    }
    hashes.sort_unstable();
    hashes.dedup();
    assert_eq!(hashes.len(), 3);
    let lexicon = segmenter(UnigramMode::OpenNlp).lexicon().clone();
    assert_eq!(UnigramSegmenter::new(lexicon, WhitespaceMode::Unicode).mode(), UnigramMode::Dictionary);
}
