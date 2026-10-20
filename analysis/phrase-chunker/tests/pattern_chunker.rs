//! Integration and adversarial tests for `TagPatternRule`, `PatternChunker`,
//! and `PatternChunkComposer` (Issue #749).

use std::path::Path;

use turbo_analysis::{Analyzer, Annotator, Document, Ix, IxRange, Key, Provenance, Span, Spans, View, ViewId};
use turbo_analysis_layers::{PhraseChunks, PosTags, Sentences, Tag, TagSet, Tokens};
use turbo_phrase_chunker::{
    ChunkerModel, ComposeMode, PatternChunkComposer, PatternChunker, PatternError, PhraseChunker, TagPatternRule,
    TokenMatcher, TokenTag,
};

fn build_doc_with_tags(text: &str, tagset_name: &str, word_tags: &[(&str, &str, u32)]) -> Document {
    let mut doc = Document::new(text);
    let v0 = ViewId::SOURCE;
    let mut tok_spans = Spans::new(v0);
    let mut sent_col = Vec::with_capacity(word_tags.len());
    let mut labels: Vec<&str> = Vec::new();
    let mut cursor = 0usize;

    for &(word, tag, sent_idx) in word_tags {
        let rel = text[cursor..].find(word).expect("token in text");
        let start = cursor + rel;
        let end = start + word.len();
        cursor = end;
        tok_spans.push(Span::new(start as u32, end as u32));
        sent_col.push(Ix::<Sentences>::new(sent_idx));
        if !labels.contains(&tag) {
            labels.push(tag);
        }
    }

    let tagset = TagSet::new(tagset_name, &labels);
    let tags: Vec<Tag> = word_tags.iter().map(|&(_, t, _)| tagset.tag(t).unwrap()).collect();

    let mut sent_spans = Spans::new(v0);
    if !text.is_empty() {
        sent_spans.push(Span::new(0, text.len() as u32));
    }
    doc.add(&Key::<Sentences>::DEFAULT, Sentences { spans: sent_spans }, Provenance::by("test")).unwrap();
    doc.add(&Key::<Tokens>::DEFAULT, Tokens { spans: tok_spans, sentence: Some(sent_col) }, Provenance::by("test"))
        .unwrap();
    doc.add(&Key::<PosTags>::DEFAULT, PosTags { tagset, tags, prob: None }, Provenance::by("test")).unwrap();
    doc
}

fn summarize_chunks<'a>(text: &'a str, chunks: &'a PhraseChunks) -> Vec<(&'a str, &'a str, (usize, usize))> {
    let (starts, ends) = chunks.spans.columns();
    (0..chunks.spans.len())
        .map(|i| {
            let r = chunks.tokens[i];
            (
                &text[starts[i] as usize..ends[i] as usize],
                chunks.kinds.label(chunks.kind[i]),
                (r.start.get(), r.end.get()),
            )
        })
        .collect()
}

#[test]
fn penn_treebank_and_universal_dependencies_np_vp_pp_fixtures() {
    let text = "The quick brown fox rapidly jumped over the lazy dog .";

    // 1. Penn Treebank tagset fixture
    let penn_tokens = [
        ("The", "DT", 0),
        ("quick", "JJ", 0),
        ("brown", "JJ", 0),
        ("fox", "NN", 0),
        ("rapidly", "RB", 0),
        ("jumped", "VBD", 0),
        ("over", "IN", 0),
        ("the", "DT", 0),
        ("lazy", "JJ", 0),
        ("dog", "NN", 0),
        (".", ".", 0),
    ];
    let doc_penn = build_doc_with_tags(text, "penn", &penn_tokens);
    let view = doc_penn.get::<View>().unwrap();
    let toks = doc_penn.get::<Tokens>().unwrap();
    let pos = doc_penn.get::<PosTags>().unwrap();

    let penn_chunker = PatternChunker::penn_default();
    let (penn_chunks,) = penn_chunker.annotate((view, toks, pos)).unwrap();
    assert_eq!(
        summarize_chunks(text, &penn_chunks),
        vec![
            ("The quick brown fox", "NP", (0, 4)),
            ("rapidly jumped", "VP", (4, 6)),
            ("over", "PP", (6, 7)),
            ("the lazy dog", "NP", (7, 10)),
        ]
    );

    // Custom full-PP rule ("<IN><DT>?<JJ.*>*<NN.*>+") taking priority over NP
    let full_pp_chunker = PatternChunker::from_patterns([
        ("PP", "<IN><DT>?<JJ.*>*<NN.*>+"),
        ("NP", "<DT>?<JJ.*>*<NN.*>+"),
        ("VP", "<RB.*>*<VB.*>+"),
    ])
    .unwrap();
    let (full_pp_chunks,) = full_pp_chunker.annotate((view, toks, pos)).unwrap();
    assert_eq!(
        summarize_chunks(text, &full_pp_chunks),
        vec![
            ("The quick brown fox", "NP", (0, 4)),
            ("rapidly jumped", "VP", (4, 6)),
            ("over the lazy dog", "PP", (6, 10)),
        ]
    );

    // 2. Universal Dependencies tagset fixture
    let ud_tokens = [
        ("The", "DET", 0),
        ("quick", "ADJ", 0),
        ("brown", "ADJ", 0),
        ("fox", "NOUN", 0),
        ("rapidly", "ADV", 0),
        ("jumped", "VERB", 0),
        ("over", "ADP", 0),
        ("the", "DET", 0),
        ("lazy", "ADJ", 0),
        ("dog", "NOUN", 0),
        (".", "PUNCT", 0),
    ];
    let doc_ud = build_doc_with_tags(text, "ud", &ud_tokens);
    let view_ud = doc_ud.get::<View>().unwrap();
    let toks_ud = doc_ud.get::<Tokens>().unwrap();
    let pos_ud = doc_ud.get::<PosTags>().unwrap();

    let ud_chunker = PatternChunker::ud_default();
    let (ud_chunks,) = ud_chunker.annotate((view_ud, toks_ud, pos_ud)).unwrap();
    assert_eq!(
        summarize_chunks(text, &ud_chunks),
        vec![
            ("The quick brown fox", "NP", (0, 4)),
            ("rapidly jumped", "VP", (4, 6)),
            ("over", "PP", (6, 7)),
            ("the lazy dog", "NP", (7, 10)),
        ]
    );
}

#[test]
fn adversary_synthetic_pos_tags_with_angle_brackets_and_pipes_preserve_exact_offsets() {
    // Include multi-byte UTF-8 characters alongside synthetic POS tags containing '<', '>', and '|'
    let text = "The café <token> and 🌟 dog | waited > quietly .";
    let synthetic_tokens = [
        ("The", "DT", 0),
        ("café", "JJ|<SPECIAL>", 0),
        ("<token>", "<NN|NNS>", 0),
        ("and", ">|<", 0),
        ("🌟", "<DT>", 0),
        ("dog", "NN", 0),
        ("|", "|PIPE|", 0),
        ("waited", "VBD", 0),
        (">", "><", 0),
        ("quietly", "RB", 0),
        (".", ".", 0),
    ];
    let doc = build_doc_with_tags(text, "synthetic", &synthetic_tokens);
    let view = doc.get::<View>().unwrap();
    let toks = doc.get::<Tokens>().unwrap();
    let pos = doc.get::<PosTags>().unwrap();

    // 1. Standard rules + escaped synthetic tag rules
    let chunker = PatternChunker::from_patterns([
        ("SYN_NP", r"<DT><JJ\|<SPECIAL\>><\<NN\|NNS\>>"),
        ("PLAIN_NP", r"<\<DT\>>?<NN>+"),
        ("PIPE_CONJ", r"<\>\|\<>"),
        ("VP", "<VBD>"),
    ])
    .unwrap();

    let (chunks,) = chunker.annotate((view, toks, pos)).unwrap();
    assert_eq!(
        summarize_chunks(text, &chunks),
        vec![
            ("The café <token>", "SYN_NP", (0, 3)),
            ("and", "PIPE_CONJ", (3, 4)),
            ("🌟 dog", "PLAIN_NP", (4, 6)),
            ("waited", "VP", (7, 8)),
        ]
    );

    // 2. Programmatic builder with unescaped literal tags containing '<', '>', and '|'
    let prog_rule = TagPatternRule::from_steps(
        "LITERAL_SPECIAL",
        [
            (TokenMatcher::exact_tag(">|<"), 1, Some(1)),
            (TokenMatcher::exact_tag("<DT>"), 1, Some(1)),
            (TokenMatcher::exact_tag("NN"), 1, Some(2)),
        ],
    )
    .unwrap();
    let prog_chunker = PatternChunker::new(vec![prog_rule]);
    let (prog_chunks,) = prog_chunker.annotate((view, toks, pos)).unwrap();
    assert_eq!(summarize_chunks(text, &prog_chunks), vec![("and 🌟 dog", "LITERAL_SPECIAL", (3, 6))]);
}

#[test]
fn quantifiers_word_literals_negation_and_priority_resolution() {
    let sentence = [
        TokenTag { token: "Put", tag: "VB" },
        TokenTag { token: "up", tag: "RP" },
        TokenTag { token: "with", tag: "IN" },
        TokenTag { token: "very", tag: "RB" },
        TokenTag { token: "extremely", tag: "RB" },
        TokenTag { token: "high", tag: "JJ" },
        TokenTag { token: "latency", tag: "NN" },
        TokenTag { token: "spikes", tag: "NNS" },
    ];

    // Phrasal verb with case-insensitive word literals and bounded quantifier {1,2}
    let chunker = PatternChunker::from_patterns([
        ("PHRASAL_VP", r#"<VB.*:"put"i><RP:"up"i><IN:"with"i>"#),
        ("ADJP", "<RB>{1,2}<JJ>"),
        ("NP", "<JJ>?<NN.*>{1,2}"),
    ])
    .unwrap();

    let matches = chunker.chunk_sentence(&sentence);
    let summary: Vec<(&str, usize, usize)> = matches.iter().map(|m| (m.kind.as_str(), m.start, m.end)).collect();
    // ADJP has higher priority than NP and claims "very extremely high" (3..6);
    // NP then claims the remaining unclaimed longest match "latency spikes" (6..8) without overlapping token 5!
    assert_eq!(summary, vec![("PHRASAL_VP", 0, 3), ("ADJP", 3, 6), ("NP", 6, 8)]);

    // Standalone quoted literals and negated tags
    let neg_rule = TagPatternRule::compile("NON_PUNCT", r#""put"i <!IN|RP>{1,2}"#).unwrap();
    assert_eq!(neg_rule.longest_match_at(&sentence, 0), None);
}

#[test]
fn composition_combines_chunker_me_and_pattern_chunker_in_fallback_and_override_modes() {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../goldens/data/fixtures/chunker-fixture.bin");
    let me = PhraseChunker::new(ChunkerModel::read(model_path).unwrap());

    let text = "The cat sat on the mat .";
    let words = [
        ("The", "DT", 0),
        ("cat", "NN", 0),
        ("sat", "VBD", 0),
        ("on", "IN", 0),
        ("the", "DT", 0),
        ("mat", "NN", 0),
        (".", ".", 0),
    ];
    let mut doc = build_doc_with_tags(text, "penn", &words);

    // 1. Override mode: a pattern rule for full prepositional phrases ("on the mat" -> PP_FULL)
    // overrides ChunkerME's separate ("on", PP) and ("the mat", NP), while preserving ("The cat", NP) and ("sat", VP).
    let override_patterns = PatternChunker::from_patterns([("PP_FULL", "<IN><DT><NN>")]).unwrap();
    let override_composer =
        PatternChunkComposer::new(override_patterns, ComposeMode::Override).with_me(PhraseChunker::new(
            ChunkerModel::read(
                Path::new(env!("CARGO_MANIFEST_DIR")).join("../goldens/data/fixtures/chunker-fixture.bin"),
            )
            .unwrap(),
        ));

    let analyzer = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<PosTags>::DEFAULT)
        .add(override_composer.clone())
        .build()
        .unwrap();
    assert_eq!(analyzer.run(std::slice::from_mut(&mut doc)), [Ok(())]);

    let chunks = doc.get::<PhraseChunks>().unwrap();
    assert_eq!(
        summarize_chunks(text, chunks),
        vec![("The cat", "NP", (0, 2)), ("sat", "VP", (2, 3)), ("on the mat", "PP_FULL", (3, 6))]
    );

    // 2. Fallback mode: ChunkerME's chunks are all preserved, and a pattern rule matching "." ("PUNCT_CHUNK")
    // fills in the unclaimed final token (6..7) without overriding "on" + "the mat".
    let fallback_patterns =
        PatternChunker::from_patterns([("PP_FULL", "<IN><DT><NN>"), ("PUNCT_CHUNK", "<\\.>")]).unwrap();
    let fallback_composer = PatternChunkComposer::new(fallback_patterns, ComposeMode::Fallback).with_me(me);
    let view = doc.get::<View>().unwrap();
    let toks = doc.get::<Tokens>().unwrap();
    let pos = doc.get::<PosTags>().unwrap();
    let (fb_chunks,) = fallback_composer.annotate((view, toks, pos)).unwrap();
    assert_eq!(
        summarize_chunks(text, &fb_chunks),
        vec![
            ("The cat", "NP", (0, 2)),
            ("sat", "VP", (2, 3)),
            ("on", "PP", (3, 4)),
            ("the mat", "NP", (4, 6)),
            (".", "PUNCT_CHUNK", (6, 7)),
        ]
    );
    assert_ne!(override_composer.config_hash(), fallback_composer.config_hash());
}

#[test]
fn invalid_patterns_and_malformed_inputs_are_rejected() {
    assert_eq!(TagPatternRule::compile("  ", "<NN>").unwrap_err(), PatternError::EmptyKind);
    assert_eq!(TagPatternRule::compile("NP", "   ").unwrap_err(), PatternError::EmptyPattern);
    assert_eq!(TagPatternRule::compile("NP", "<DT>?<JJ>*").unwrap_err(), PatternError::ZeroLengthPattern);
    assert!(matches!(TagPatternRule::compile("NP", "<NN").unwrap_err(), PatternError::UnclosedTag { .. }));
    assert!(matches!(TagPatternRule::compile("NP", "<>").unwrap_err(), PatternError::EmptyTag { .. }));
    assert!(matches!(TagPatternRule::compile("NP", "\"unclosed").unwrap_err(), PatternError::UnclosedQuote { .. }));
    assert!(matches!(TagPatternRule::compile("NP", "(<DT><NN>").unwrap_err(), PatternError::UnclosedGroup { .. }));
    assert!(matches!(TagPatternRule::compile("NP", "<NN>{3,1}").unwrap_err(), PatternError::InvalidQuantifier { .. }));
    assert!(matches!(TagPatternRule::compile("NP", "<NN>{0,0}").unwrap_err(), PatternError::InvalidQuantifier { .. }));
    assert!(matches!(
        TagPatternRule::compile("NP", "<NN>{0,9999}").unwrap_err(),
        PatternError::InvalidQuantifier { .. }
    ));
    assert!(matches!(
        TagPatternRule::compile("NP", "(<NN>{256}){256}").unwrap_err(),
        PatternError::InvalidQuantifier { .. }
    ));
    assert!(matches!(
        TagPatternRule::compile("NP", "((<NN>{256}){256}){256}").unwrap_err(),
        PatternError::InvalidQuantifier { .. }
    ));

    // Provenance config_hash distinguishes different rule sets, including programmatic TagCond/WordCond variants
    let c1 = PatternChunker::from_patterns([("NP", "<DT>?<NN>+")]).unwrap();
    let c2 = PatternChunker::from_patterns([("NP", "<DT><NN>+")]).unwrap();
    assert_ne!(c1.config_hash(), c2.config_hash());

    let r_exact_tag = TagPatternRule::from_steps("NP", [(TokenMatcher::exact_tag("NN"), 1, Some(1))]).unwrap();
    let r_prefix_tag = TagPatternRule::from_steps("NP", [(TokenMatcher::prefix_tag("NN"), 1, Some(1))]).unwrap();
    assert_ne!(r_exact_tag.fingerprint(), r_prefix_tag.fingerprint());
    assert_ne!(
        PatternChunker::new(vec![r_exact_tag]).config_hash(),
        PatternChunker::new(vec![r_prefix_tag]).config_hash()
    );

    let r_exact_word = TagPatternRule::from_steps("NP", [(TokenMatcher::exact_word("dog"), 1, Some(1))]).unwrap();
    let r_icase_word = TagPatternRule::from_steps("NP", [(TokenMatcher::word_ignore_case("dog"), 1, Some(1))]).unwrap();
    assert_ne!(r_exact_word.fingerprint(), r_icase_word.fingerprint());
    assert_ne!(
        PatternChunker::new(vec![r_exact_word]).config_hash(),
        PatternChunker::new(vec![r_icase_word]).config_hash()
    );

    // Malformed base PhraseChunks in compose() are rejected
    let doc = build_doc_with_tags("The cat", "penn", &[("The", "DT", 0), ("cat", "NN", 0)]);
    let view = doc.get::<View>().unwrap();
    let toks = doc.get::<Tokens>().unwrap();
    let pos = doc.get::<PosTags>().unwrap();
    let composer = PatternChunkComposer::new(c1, ComposeMode::Fallback);

    let mut bad_spans = Spans::new(view.id());
    bad_spans.push(Span::new(0, 3));
    let bad_base = PhraseChunks {
        spans: bad_spans,
        tokens: vec![IxRange::new(0, 99)],
        kind: vec![Tag(0)],
        kinds: TagSet::new("chunks", &["NP"]),
    };
    assert!(composer.compose(view, toks, pos, &bad_base).is_err());
}
