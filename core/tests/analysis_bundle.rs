//! The in-process analysis C ABI over a bundle with heads
//! (`turbo_analyzer_desc.bundle_path`): the sealed small BERT in testdata
//! with its five seeded heads writes the part-of-speech, morphology,
//! lemma, dependency and entity layers; the pre-tokenized tasks agree
//! with the graph; a bundle without heads leaves those layers
//! unsupported, by name.

use std::collections::BTreeSet;
use std::ffi::CStr;
use std::path::{Path, PathBuf};

use turbo::*;

fn t(s: &str) -> turbo_text {
    turbo_text { ptr: s.as_ptr().cast(), len: s.len() as u64 }
}

fn new_err() -> turbo_error {
    turbo_error {
        struct_size: size_of::<turbo_error>() as u32,
        code: -1,
        field: 0,
        message: [0; TURBO_ERROR_MESSAGE_LEN],
    }
}

fn message(err: &turbo_error) -> String {
    unsafe { CStr::from_ptr(err.message.as_ptr()) }.to_str().unwrap().to_owned()
}

fn testdata() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata")
}

/// The sealed tiny bundle with five heads, pinned.
fn heads_bundle() -> String {
    turbo_pins::require(testdata().join("tiny-bert-heads-bundle")).to_str().unwrap().to_owned()
}

/// A row with its label read from the pool.
#[derive(Clone, Debug, PartialEq)]
struct Row {
    span: turbo_span,
    label: String,
}

unsafe fn read_rows(r: *const turbo_analysis_result) -> Vec<Row> {
    let mut spans_ptr: *const turbo_span = std::ptr::null();
    let mut count: u32 = 0;
    let mut err = new_err();
    assert_eq!(unsafe { turbo_analysis_result_spans(r, &mut spans_ptr, &mut count, &mut err) }, status::OK);
    let mut pool_ptr: *const std::ffi::c_char = std::ptr::null();
    let mut pool_len: u64 = 0;
    assert_eq!(unsafe { turbo_analysis_result_strings(r, &mut pool_ptr, &mut pool_len, &mut err) }, status::OK);
    let spans = if count == 0 { &[][..] } else { unsafe { std::slice::from_raw_parts(spans_ptr, count as usize) } };
    let pool = if pool_len == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(pool_ptr.cast::<u8>(), pool_len as usize) }
    };
    spans
        .iter()
        .map(|&span| {
            let (a, b) = (span.label_offset as usize, (span.label_offset + span.label_len) as usize);
            Row { span, label: std::str::from_utf8(&pool[a..b]).unwrap().to_owned() }
        })
        .collect()
}

unsafe fn analyzer(bundle: &str) -> *mut turbo_analyzer {
    let mut desc: turbo_analyzer_desc = unsafe { std::mem::zeroed() };
    desc.struct_size = size_of::<turbo_analyzer_desc>() as u32;
    desc.bundle_path = t(bundle);
    let mut err = new_err();
    let mut a: *mut turbo_analyzer = std::ptr::null_mut();
    assert_eq!(unsafe { turbo_analyzer_create(&desc, &mut a, &mut err) }, status::OK, "{}", message(&err));
    a
}

unsafe fn graph(a: *mut turbo_analyzer, text: &str) -> (Vec<Row>, Vec<u8>) {
    let mut err = new_err();
    let mut res: *mut turbo_analysis_result = std::ptr::null_mut();
    assert_eq!(
        unsafe { turbo_analyzer_analyze_graph(a, t(text), &mut res, &mut err) },
        status::OK,
        "{}",
        message(&err)
    );
    let rows = unsafe { read_rows(res) };
    let (mut ptr, mut len) = (std::ptr::null(), 0u64);
    assert_eq!(unsafe { turbo_analysis_result_graph(res, &mut ptr, &mut len, &mut err) }, status::OK);
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len as usize) }.to_vec();
    unsafe { turbo_analysis_result_release(res) };
    (rows, bytes)
}

fn of(rows: &[Row], kind: u32) -> Vec<Row> {
    rows.iter().filter(|r| r.span.layer_kind == kind).cloned().collect()
}

const TEXT: &str = "Ann met Bob in Paris , and the quiet old dog slept by the river . Bob walked the old dog .";
const POS: [&str; 5] = ["NOUN", "VERB", "DET", "ADJ", "PUNCT"];
const FEATS: [&str; 4] = ["_", "Number=Sing", "Number=Plur", "Tense=Past|VerbForm=Fin"];
const RELATIONS: [&str; 6] = ["root", "nsubj", "obj", "det", "amod", "punct"];

#[test]
fn the_bundle_heads_write_every_layer_of_the_graph() {
    let bundle = heads_bundle();
    unsafe {
        let a = analyzer(&bundle);
        let (rows, bytes) = graph(a, TEXT);
        let kinds: BTreeSet<u32> = rows.iter().map(|r| r.span.layer_kind).collect();
        assert_eq!(
            kinds,
            BTreeSet::from([
                TURBO_ANALYSIS_LAYER_SENTENCES,
                TURBO_ANALYSIS_LAYER_TOKENS,
                TURBO_ANALYSIS_LAYER_POS,
                TURBO_ANALYSIS_LAYER_LEMMAS,
                TURBO_ANALYSIS_LAYER_ENTITIES,
                TURBO_ANALYSIS_LAYER_DEPENDENCIES,
                TURBO_ANALYSIS_LAYER_MORPHOLOGY,
            ])
        );
        let sentences = of(&rows, TURBO_ANALYSIS_LAYER_SENTENCES);
        let tokens = of(&rows, TURBO_ANALYSIS_LAYER_TOKENS);
        assert_eq!(sentences.len(), 2);
        assert_eq!(tokens.len(), TEXT.split(' ').count());

        // A tag a token, in the head's label set, with the softmax
        // probability of the tag, over the token's own text range.
        let pos = of(&rows, TURBO_ANALYSIS_LAYER_POS);
        assert_eq!(pos.len(), tokens.len());
        for (i, (p, tok)) in pos.iter().zip(&tokens).enumerate() {
            assert!(POS.contains(&p.label.as_str()), "{}", p.label);
            assert_eq!((p.span.token_start, p.span.token_end), (i as u32, i as u32 + 1));
            assert!(p.span.flags & TURBO_SPAN_PROBABILITY != 0 && p.span.prob > 0.0 && p.span.prob <= 1.0);
            assert_eq!(
                (p.span.source_byte_start, p.span.source_byte_end),
                (tok.span.source_byte_start, tok.span.source_byte_end)
            );
        }
        // A feature string a token, in the head's label set, with no
        // probability, over the token's own text range.
        let morph = of(&rows, TURBO_ANALYSIS_LAYER_MORPHOLOGY);
        assert_eq!(morph.len(), tokens.len());
        for (i, (m, tok)) in morph.iter().zip(&tokens).enumerate() {
            assert!(FEATS.contains(&m.label.as_str()), "{}", m.label);
            assert_eq!((m.span.token_start, m.span.token_end), (i as u32, i as u32 + 1));
            assert!(m.span.flags & TURBO_SPAN_LABEL != 0 && m.span.prob.is_nan());
            assert_eq!(
                (m.span.source_byte_start, m.span.source_byte_end),
                (tok.span.source_byte_start, tok.span.source_byte_end)
            );
        }
        // A lemma a token, with no probability: the head picks a tree.
        let lemmas = of(&rows, TURBO_ANALYSIS_LAYER_LEMMAS);
        assert_eq!(lemmas.len(), tokens.len());
        assert!(lemmas.iter().all(|l| l.span.flags & TURBO_SPAN_LABEL != 0 && l.span.prob.is_nan()));
        let forms: Vec<&str> =
            tokens.iter().map(|r| &TEXT[r.span.source_byte_start as usize..r.span.source_byte_end as usize]).collect();
        assert_eq!(forms[0], "Ann");
        // The trees keep the word, drop a final s or ed, or turn met into meet.
        for (l, f) in lemmas.iter().zip(&forms) {
            let ok = l.label == *f
                || f.strip_suffix('s') == Some(&l.label)
                || f.strip_suffix("ed") == Some(&l.label)
                || (*f == "met" && l.label == "meet");
            assert!(ok, "{f} -> {}", l.label);
        }
        // A head and a relation a token; one root a sentence under MST.
        let deps = of(&rows, TURBO_ANALYSIS_LAYER_DEPENDENCIES);
        assert_eq!(deps.len(), tokens.len());
        let mut roots = 0;
        for (i, d) in deps.iter().enumerate() {
            assert!(RELATIONS.contains(&d.label.as_str()), "{}", d.label);
            assert_eq!((d.span.token_start, d.span.token_end), (i as u32, i as u32 + 1));
            assert!(d.span.prob.is_nan());
            if d.span.classification == 0 {
                roots += 1;
            } else {
                let head = d.span.classification as usize - 1;
                assert!(head < tokens.len() && head != i, "token {i} under {head}");
                // The head is in the same sentence.
                let (s, h) = (&tokens[i].span, &tokens[head].span);
                let same = sentences.iter().any(|sent| {
                    let (a, b) = (sent.span.source_byte_start, sent.span.source_byte_end);
                    a <= s.source_byte_start
                        && s.source_byte_end <= b
                        && a <= h.source_byte_start
                        && h.source_byte_end <= b
                });
                assert!(same, "token {i} under {head} across sentences");
            }
        }
        assert_eq!(roots, 2);
        // Entities span whole tokens, with a kind and the mean probability.
        let entities = of(&rows, TURBO_ANALYSIS_LAYER_ENTITIES);
        assert!(!entities.is_empty(), "the seeded entity head marks nothing");
        for e in &entities {
            assert!(["PER", "LOC"].contains(&e.label.as_str()), "{}", e.label);
            assert!(e.span.token_start < e.span.token_end && e.span.token_end as usize <= tokens.len());
            let (a, b) = (&tokens[e.span.token_start as usize].span, &tokens[e.span.token_end as usize - 1].span);
            assert_eq!((e.span.source_byte_start, e.span.source_byte_end), (a.source_byte_start, b.source_byte_end));
            assert!(e.span.prob > 0.0 && e.span.prob <= 1.0);
        }

        // The graph carries the same rows back.
        let mut err = new_err();
        let mut res: *mut turbo_analysis_result = std::ptr::null_mut();
        assert_eq!(turbo_analyzer_inspect_graph(a, bytes.as_ptr(), bytes.len() as u64, &mut res, &mut err), status::OK);
        let back = read_rows(res);
        turbo_analysis_result_release(res);
        let key = |r: &Row| {
            (r.span.layer_kind, r.span.item_index, r.span.classification, r.span.token_start, r.label.clone())
        };
        let mut a_keys: Vec<_> = rows.iter().map(key).collect();
        let mut b_keys: Vec<_> = back.iter().map(key).collect();
        a_keys.sort();
        b_keys.sort();
        assert_eq!(a_keys, b_keys);
        turbo_analyzer_release(a);
    }
}

#[test]
fn the_pre_tokenized_tasks_agree_with_the_graph() {
    let bundle = heads_bundle();
    unsafe {
        let a = analyzer(&bundle);
        let (rows, _) = graph(a, TEXT);
        let tokens = of(&rows, TURBO_ANALYSIS_LAYER_TOKENS);
        let first = of(&rows, TURBO_ANALYSIS_LAYER_SENTENCES)[0].span;
        let in_first: Vec<usize> =
            (0..tokens.len()).filter(|&i| tokens[i].span.source_byte_end <= first.source_byte_end).collect();
        let forms: Vec<&str> = in_first
            .iter()
            .map(|&i| &TEXT[tokens[i].span.source_byte_start as usize..tokens[i].span.source_byte_end as usize])
            .collect();
        let words: Vec<turbo_text> = forms.iter().map(|f| t(f)).collect();
        let n = words.len() as u32;
        let mut err = new_err();

        // The first sentence's tokens, tagged on their own, get the tags
        // and probabilities the graph gave them.
        let mut res: *mut turbo_analysis_result = std::ptr::null_mut();
        assert_eq!(turbo_analyzer_pos_tag(a, words.as_ptr(), n, &mut res, &mut err), status::OK, "{}", message(&err));
        let pos = read_rows(res);
        turbo_analysis_result_release(res);
        let graph_pos = of(&rows, TURBO_ANALYSIS_LAYER_POS);
        assert_eq!(pos.len(), in_first.len());
        for (p, &i) in pos.iter().zip(&in_first) {
            assert_eq!((p.label.as_str(), p.span.prob), (graph_pos[i].label.as_str(), graph_pos[i].span.prob));
            assert_eq!(p.span.flags & (TURBO_SPAN_VIEW_RANGE | TURBO_SPAN_SOURCE_RANGE), 0, "token-only rows");
        }

        // The morphology rows of the sentence's tokens are the graph's.
        assert_eq!(
            turbo_analyzer_morphology(a, words.as_ptr(), n, &mut res, &mut err),
            status::OK,
            "{}",
            message(&err)
        );
        let morph = read_rows(res);
        turbo_analysis_result_release(res);
        let graph_morph = of(&rows, TURBO_ANALYSIS_LAYER_MORPHOLOGY);
        assert_eq!(morph.len(), in_first.len());
        for (m, &i) in morph.iter().zip(&in_first) {
            assert_eq!(m.label.as_str(), graph_morph[i].label.as_str());
            assert!(m.span.prob.is_nan());
            assert_eq!(m.span.flags & (TURBO_SPAN_VIEW_RANGE | TURBO_SPAN_SOURCE_RANGE), 0, "token-only rows");
        }

        // Lemmatized under those tags, the lemmas are the graph's.
        let tags: Vec<turbo_text> = pos.iter().map(|p| t(&p.label)).collect();
        assert_eq!(turbo_analyzer_lemmatize(a, words.as_ptr(), tags.as_ptr(), n, &mut res, &mut err), status::OK);
        let lemmas = read_rows(res);
        turbo_analysis_result_release(res);
        let graph_lemmas = of(&rows, TURBO_ANALYSIS_LAYER_LEMMAS);
        let got: Vec<&str> = lemmas.iter().map(|l| l.label.as_str()).collect();
        let want: Vec<&str> = in_first.iter().map(|&i| graph_lemmas[i].label.as_str()).collect();
        assert_eq!(got, want);
        // A tag outside the tag head's labels is refused by position.
        let bad: Vec<turbo_text> = (0..n).map(|i| if i == 2 { t("XYZ") } else { t("NOUN") }).collect();
        assert_eq!(
            turbo_analyzer_lemmatize(a, words.as_ptr(), bad.as_ptr(), n, &mut res, &mut err),
            status::INVALID_ARGUMENT
        );
        assert!(message(&err).contains("tags[2]"), "{}", message(&err));

        // The entities of the sentence's tokens, and of the text, are the
        // graph's.
        assert_eq!(turbo_analyzer_find_tokens(a, words.as_ptr(), n, &mut res, &mut err), status::OK);
        let ents = read_rows(res);
        turbo_analysis_result_release(res);
        let graph_ents = of(&rows, TURBO_ANALYSIS_LAYER_ENTITIES);
        let in_sentence: Vec<&Row> =
            graph_ents.iter().filter(|e| (e.span.token_end as usize) <= in_first.len()).collect();
        assert_eq!(ents.len(), in_sentence.len());
        for (e, g) in ents.iter().zip(&in_sentence) {
            assert_eq!(
                (e.span.token_start, e.span.token_end, &e.label),
                (g.span.token_start, g.span.token_end, &g.label)
            );
            assert_eq!(e.span.prob, g.span.prob);
            assert_eq!(e.span.flags & TURBO_SPAN_SOURCE_RANGE, 0);
        }
        assert_eq!(turbo_analyzer_find_text(a, t(TEXT), &mut res, &mut err), status::OK);
        let found = read_rows(res);
        turbo_analysis_result_release(res);
        let key = |r: &Row| {
            (
                r.span.source_byte_start,
                r.span.source_byte_end,
                r.span.token_start,
                r.span.token_end,
                r.label.clone(),
                r.span.prob.to_bits(),
            )
        };
        let got: Vec<_> = found.iter().map(key).collect();
        let want: Vec<_> = graph_ents.iter().map(key).collect();
        assert_eq!(got, want);
        turbo_analyzer_release(a);
    }
}

#[test]
fn a_bundle_without_heads_leaves_the_layers_unsupported_by_name() {
    let plain = turbo_pins::require(testdata().join("tiny-bert-bundle")).to_str().unwrap().to_owned();
    unsafe {
        let a = analyzer(&plain);
        let mut err = new_err();
        let words = [t("The"), t("river")];
        let mut res: *mut turbo_analysis_result = std::ptr::dangling_mut();
        assert_eq!(turbo_analyzer_pos_tag(a, words.as_ptr(), 2, &mut res, &mut err), status::UNSUPPORTED_TASK);
        assert!(res.is_null());
        assert!(message(&err).contains("no head writing pos"), "{}", message(&err));
        let mut res: *mut turbo_analysis_result = std::ptr::dangling_mut();
        assert_eq!(turbo_analyzer_morphology(a, words.as_ptr(), 2, &mut res, &mut err), status::UNSUPPORTED_TASK);
        assert!(res.is_null());
        assert!(message(&err).contains("no head writing morphology"), "{}", message(&err));
        assert_eq!(turbo_analyzer_find_text(a, t(TEXT), &mut res, &mut err), status::UNSUPPORTED_TASK);
        assert!(message(&err).contains("no head writing entities"), "{}", message(&err));
        let (rows, _) = graph(a, TEXT);
        let kinds: BTreeSet<u32> = rows.iter().map(|r| r.span.layer_kind).collect();
        assert_eq!(kinds, BTreeSet::from([TURBO_ANALYSIS_LAYER_SENTENCES, TURBO_ANALYSIS_LAYER_TOKENS]));
        // No input is no rows, head or not.
        let none: [turbo_text; 0] = [];
        assert_eq!(turbo_analyzer_pos_tag(a, none.as_ptr(), 0, &mut res, &mut err), status::OK);
        assert!(read_rows(res).is_empty());
        turbo_analysis_result_release(res);
        assert_eq!(turbo_analyzer_morphology(a, none.as_ptr(), 0, &mut res, &mut err), status::OK);
        assert!(read_rows(res).is_empty());
        turbo_analysis_result_release(res);
        assert_eq!(turbo_analyzer_find_text(a, t(""), &mut res, &mut err), status::OK);
        assert!(read_rows(res).is_empty());
        turbo_analysis_result_release(res);
        turbo_analyzer_release(a);

        // A bundle that is not there is refused as such.
        let mut desc: turbo_analyzer_desc = std::mem::zeroed();
        desc.struct_size = size_of::<turbo_analyzer_desc>() as u32;
        let missing = testdata().join("does-not-exist");
        let missing = missing.to_str().unwrap();
        desc.bundle_path = t(missing);
        let mut a: *mut turbo_analyzer = std::ptr::null_mut();
        assert_eq!(turbo_analyzer_create(&desc, &mut a, &mut err), status::BUNDLE_NOT_FOUND, "{}", message(&err));
        assert!(a.is_null());
        assert_eq!(err.field, 7, "the field number of bundle_path");
        assert!(message(&err).contains("bundle_path") && message(&err).contains(missing), "{}", message(&err));
    }
}
