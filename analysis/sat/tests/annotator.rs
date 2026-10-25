//! The SaT step on the small random model of testdata/tiny-sat-bundle:
//! batching, windows, texts without words, the options, and the provider.
//! What the scores are is analysis/sat/tests/parity.rs's.

use std::path::{Path, PathBuf};

use turbo_analysis::registry::{Registry, Spec};
use turbo_analysis::{Analyzer, Annotator, Document, Span};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::Sentences;
use turbo_sat::{SatAnnotator, SatOptions, SatProvider, cuts};

fn testdata() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata")
}

fn bundle() -> PathBuf {
    testdata().join("tiny-sat-bundle")
}

fn texts() -> Vec<String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/texts.jsonl");
    std::fs::read_to_string(path).unwrap().lines().map(|l| serde_json::from_str::<String>(l).unwrap()).collect()
}

fn sat(o: SatOptions) -> SatAnnotator {
    SatAnnotator::open(&bundle(), &o).unwrap()
}

fn sentences(a: &SatAnnotator, text: &str) -> Vec<String> {
    let (spans, scores) = a.scores(&[text]).remove(0).unwrap();
    let mut out = Vec::new();
    a.sentences(text, &spans, &scores, &mut out);
    out.iter().map(|s| text[s.start as usize..s.end as usize].to_owned()).collect()
}

/// A document's scores are the same alone, in one run with the others,
/// and spread over runs of three rows; its sentences are the same.
#[test]
fn a_document_reads_the_same_in_any_batch() {
    let texts = texts();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let o = SatOptions { stride: 8, block_size: 24, ..SatOptions::default() };
    let wide = sat(o.clone());
    let narrow = sat(SatOptions { encoder: EncoderOptions { max_batch: 3, ..EncoderOptions::default() }, ..o });
    let together = wide.scores(&refs);
    let small_runs = narrow.scores(&refs);
    let mut worst = 0f32;
    for (k, t) in refs.iter().enumerate() {
        let (spans, alone) = wide.scores(&[t]).remove(0).unwrap();
        for other in [&together[k], &small_runs[k]] {
            let (s, scores) = other.as_ref().unwrap();
            assert_eq!(s, &spans, "{t:?}");
            for (a, b) in alone.iter().zip(scores) {
                worst = worst.max((a - b).abs());
            }
        }
        assert_eq!(sentences(&wide, t), sentences(&narrow, t), "{t:?}");
    }
    assert!(worst < 1e-4, "scores differ by {worst} with the batch");
}

/// A text longer than a window is read in many, and its sentences are in
/// order, apart, on grapheme cluster boundaries and trimmed.
#[test]
fn a_long_text_is_read_in_windows() {
    let unit = "Der schnelle braune Fuchs springt. 東京は日本の首都です。 La phrase suivante est ici!\n";
    let text = unit.repeat(40);
    let a = sat(SatOptions::default());
    let (spans, scores) = a.scores(&[&text]).remove(0).unwrap();
    assert!(spans.len() > 10 * a.block_size() as usize, "{} tokens", spans.len());
    assert_eq!(spans.len(), scores.len());
    let mut out = Vec::new();
    a.sentences(&text, &spans, &scores, &mut out);
    let g = cuts::grapheme_index(&text);
    let mut last = 0;
    for s in &out {
        assert!(s.start >= last && s.end > s.start, "{s:?} after {last}");
        assert!(g.binary_search(&s.start).is_ok() && g.binary_search(&s.end).is_ok(), "{s:?}");
        let piece = &text[s.start as usize..s.end as usize];
        assert_eq!(piece, piece.trim_matches(cuts::is_python_space));
        assert!(!piece.contains('\n'), "a line feed ends a sentence: {piece:?}");
        last = s.end;
    }
    // Every line feed ends a sentence, so there are at least as many.
    assert!(out.len() >= 40, "{} sentences", out.len());
}

#[test]
fn a_text_without_words_has_no_sentences() {
    let a = sat(SatOptions::default());
    for t in ["", " ", " \t\n\r\u{a0}\u{2028}\u{1f}"] {
        let (spans, scores) = a.scores(&[t]).remove(0).unwrap();
        assert!(spans.is_empty() && scores.is_empty(), "{t:?}");
        assert!(sentences(&a, t).is_empty(), "{t:?}");
    }
    // A blank text in a batch leaves the others as they are.
    let r = a.scores(&["", "One. Two.", "  "]);
    assert!(r[0].as_ref().unwrap().0.is_empty() && r[2].as_ref().unwrap().0.is_empty());
    assert_eq!(r[1].as_ref().unwrap(), &a.scores(&["One. Two."]).remove(0).unwrap());
}

/// At threshold 1 no probability is over it, so only the line feeds cut;
/// at 0 every token ends a sentence.
#[test]
fn the_threshold_and_the_line_feeds() {
    let t = "first line here\nsecond line here";
    let never = SatOptions { threshold: 1.0, ..SatOptions::default() };
    assert_eq!(sentences(&sat(never.clone()), t), ["first line here", "second line here"]);
    let model = SatOptions { split_on_newlines: false, ..never };
    assert_eq!(sentences(&sat(model), t), [t]);
    let always = sat(SatOptions { threshold: 0.0, ..SatOptions::default() });
    let (spans, _) = always.scores(&[t]).remove(0).unwrap();
    let words = sentences(&always, t);
    // Every cut moves past whitespace, so tokens that end together inside
    // a word give one cut, and the sentences are the text's pieces.
    assert!(words.len() > 2 && words.len() <= spans.len(), "{words:?}");
    assert_eq!(words.concat(), t.split_whitespace().collect::<String>());
}

fn refused(o: SatOptions, dir: &Path, want: &str) {
    let e = SatAnnotator::open(dir, &o).err().unwrap_or_else(|| panic!("{o:?} is accepted"));
    assert!(e.to_string().contains(want), "{e}");
}

#[test]
fn bad_options_and_bundles_are_refused() {
    let d = SatOptions::default();
    let b = bundle();
    // A row of 64 tokens has room for 62 beside <s> and </s>.
    refused(SatOptions { block_size: 63, ..d.clone() }, &b, "block_size 63");
    refused(SatOptions { block_size: 10, stride: 11, ..d.clone() }, &b, "stride 11");
    refused(SatOptions { threshold: 1.5, ..d.clone() }, &b, "threshold 1.5");
    refused(SatOptions { threshold: f64::NAN, ..d.clone() }, &b, "threshold NaN");
    refused(SatOptions { label: 1, ..d.clone() }, &b, "label 1 of a classifier with 1 labels");
    refused(d.clone(), &testdata().join("tiny-static-bundle"), "a static model has no state per token");
    // tiny-bert's manifest declares no classifier.
    refused(d, &testdata().join("tiny-bert-bundle"), "the manifest declares no classifier");
}

/// The block is what a row has room for, 62 beside <s> and </s>, and the
/// stride wtpsplit's 64 or the block when that is smaller.
#[test]
fn the_defaults_fit_the_model() {
    let a = sat(SatOptions::default());
    assert_eq!((a.block_size(), a.stride()), (62, 62));
    let a = sat(SatOptions { block_size: 20, ..SatOptions::default() });
    assert_eq!((a.block_size(), a.stride()), (20, 20));
    let a = sat(SatOptions { block_size: 20, stride: 5, ..SatOptions::default() });
    assert_eq!((a.block_size(), a.stride()), (20, 5));
}

/// The configuration hash changes with every option that changes the
/// sentences, and the step names its model and where it runs.
#[test]
fn the_step_records_its_model_and_options() {
    let base = SatOptions { stride: 30, ..SatOptions::default() };
    let h = |o: SatOptions| sat(o).config_hash();
    let h0 = h(base.clone());
    assert_eq!(h0, h(base.clone()));
    for o in [
        SatOptions { threshold: 0.3, ..base.clone() },
        SatOptions { stride: 31, ..base.clone() },
        SatOptions { block_size: 40, ..base.clone() },
        SatOptions { split_on_newlines: false, ..base.clone() },
    ] {
        assert_ne!(h0, h(o.clone()), "{o:?}");
    }
    // A block of 0 is the room, 62, so the two are one configuration, as
    // are a stride of 0 and the stride it stands for.
    assert_eq!(h0, h(SatOptions { block_size: 62, ..base.clone() }));
    assert_eq!(h(SatOptions { stride: 0, ..base.clone() }), h(SatOptions { stride: 62, ..base.clone() }));
    let a = sat(base);
    let m = a.model().unwrap();
    assert_eq!(m.id, "turboembed/tiny-sat");
    assert_eq!(m.sha256.len(), 64);
    assert_eq!(a.execution().unwrap().backend, "cpu");
}

/// The layer, through an analyzer: the sentences of each document, in its
/// view, and a pipeline from configuration with the provider's options.
#[test]
fn the_provider_builds_the_step() {
    let mut r: Registry<dyn turbo_analysis::annotate::Step> = Registry::default();
    r.register(SatProvider);
    let spec = Spec { location: Some(bundle()), options: Vec::new() }
        .with("stride", "16")
        .with("threshold", "1")
        .with("newlines", "split");
    let a = r.pipeline(&[("sat", spec)], |b| b).unwrap();
    assert_eq!(a.plan(), vec![vec!["turbo:sat"]]);
    let mut docs = vec![Document::new("one line\ntwo line"), Document::new("")];
    for res in a.run(&mut docs) {
        res.unwrap();
    }
    let s = docs[0].get::<Sentences>().unwrap();
    let got: Vec<Span> = s.spans.iter().map(|(_, sp)| sp).collect();
    assert_eq!(got, [Span::new(0, 8), Span::new(9, 17)]);
    assert!(docs[1].get::<Sentences>().unwrap().spans.is_empty());

    let base = || Spec { location: Some(bundle()), options: Vec::new() };
    for (key, value, want) in [
        ("threshold", "high", "threshold \"high\" is not a number"),
        ("stride", "-1", "stride \"-1\" is not a number"),
        ("newlines", "keep", "newlines \"keep\": split or model"),
        ("precision", "half", "precision \"half\""),
        ("threshold", "2", "threshold 2"),
    ] {
        let e = r
            .pipeline(&[("sat", base().with(key, value))], |b| b)
            .err()
            .unwrap_or_else(|| panic!("{key}={value} is accepted"));
        assert!(e.to_string().contains(want), "{key}={value}: {e}");
    }
    assert!(r.pipeline(&[("sat", Spec::default())], |b| b).is_err(), "no location");
}

/// The same documents through the analyzer and through `scores` and
/// `sentences` give the same layer.
#[test]
fn the_analyzer_writes_what_the_step_finds() {
    let o = SatOptions { stride: 16, ..SatOptions::default() };
    let texts = texts();
    let direct = sat(o.clone());
    let analyzer = Analyzer::builder().add(sat(o)).build().unwrap();
    let mut docs: Vec<Document> = texts.iter().map(|t| Document::new(t.clone())).collect();
    for res in analyzer.run(&mut docs) {
        res.unwrap();
    }
    for (t, d) in texts.iter().zip(&docs) {
        let layer: Vec<String> = d
            .get::<Sentences>()
            .unwrap()
            .spans
            .iter()
            .map(|(_, s)| t[s.start as usize..s.end as usize].to_owned())
            .collect();
        assert_eq!(layer, sentences(&direct, t), "{t:?}");
    }
}

/// Texts made to break the cuts: characters the tokenizer drops or
/// expands, marks with no base, lone carriage returns, special-token
/// strings, a window's worth of one character, and whitespace of every
/// kind between them. Every sentence is a non-empty run on grapheme
/// cluster boundaries, trimmed, in order and apart, at every threshold.
#[test]
fn hostile_texts_give_well_formed_sentences() {
    let long_mark = format!("a{}", "\u{301}".repeat(300));
    let many = "\u{337f}".repeat(200);
    let texts = [
        "\u{1}",
        "\u{200b}",
        "\u{feff}x",
        "\u{301}",
        "\u{301}. \u{301}",
        "\r",
        "a\rb\r\r\nc",
        "\u{1c}\u{1d}a\u{1e}\u{1f}",
        "<s></s><pad><mask><unk>",
        " <pad> x <pad>",
        "\u{2028}\u{2029}\u{85}a\u{85}",
        "x\u{200d}\u{200d}\u{200d}.",
        "👨\u{200d}👩\u{200d}👧. 🇯🇵🇩🇪!",
        "\n\n\n.\n\n\n",
        "..........",
        &long_mark,
        &many,
    ];
    for threshold in [0.0, 0.25, 0.5, 1.0] {
        for block_size in [1, 3, 0] {
            let a = sat(SatOptions { threshold, block_size, stride: block_size.min(2), ..SatOptions::default() });
            for t in texts {
                let got = a.scores(&[t]).remove(0).unwrap();
                let mut out = Vec::new();
                a.sentences(t, &got.0, &got.1, &mut out);
                let g = cuts::grapheme_index(t);
                let mut last = 0;
                for s in &out {
                    assert!(s.start >= last && s.end > s.start, "{t:?}: {s:?} after {last}");
                    assert!(g.binary_search(&s.start).is_ok() && g.binary_search(&s.end).is_ok(), "{t:?}: {s:?}");
                    let piece = &t[s.start as usize..s.end as usize];
                    assert!(!piece.starts_with(cuts::is_python_space), "{t:?}: {piece:?}");
                    assert!(!piece.ends_with(cuts::is_python_space), "{t:?}: {piece:?}");
                    last = s.end;
                }
            }
        }
    }
}
