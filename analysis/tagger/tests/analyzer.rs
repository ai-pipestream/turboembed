//! The in-process C analyzer (`turbo_analyzer_desc.bundle_path`) writes
//! what the annotators of this crate write: on the sealed small BERT in
//! testdata with its five seeded heads, the part-of-speech tags and
//! probabilities, morphological features, lemmas, dependency heads and
//! relations, and entities of the C graph are those of [`MultiTask`]
//! over the same tokens and sentences, field for field.

use std::ffi::CStr;
use std::path::{Path, PathBuf};

use turbo::{
    TURBO_ANALYSIS_LAYER_DEPENDENCIES, TURBO_ANALYSIS_LAYER_ENTITIES, TURBO_ANALYSIS_LAYER_LEMMAS,
    TURBO_ANALYSIS_LAYER_MORPHOLOGY, TURBO_ANALYSIS_LAYER_POS, TURBO_ANALYSIS_LAYER_SENTENCES,
    TURBO_ANALYSIS_LAYER_TOKENS, TURBO_ERROR_MESSAGE_LEN, status, turbo_analysis_result, turbo_analysis_result_release,
    turbo_analysis_result_spans, turbo_analysis_result_strings, turbo_analyzer, turbo_analyzer_analyze_graph,
    turbo_analyzer_create, turbo_analyzer_desc, turbo_analyzer_morphology, turbo_analyzer_release, turbo_error,
    turbo_span, turbo_text,
};
use turbo_analysis::{Annotator, Document, Key, Span, Spans, View, ViewId};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{Dependencies, Entities, Lemmas, MorphFeatures, PosTags, Sentences, Tokens};
use turbo_tagger::MultiTask;

type All = (PosTags, MorphFeatures<Tokens>, Entities, Dependencies<Tokens>, Lemmas);

const TEXT: &str = "Ann met Bob in Paris, and the quiet old dog slept by the river. Bob walked the old dog. \
                    Ann watched the river while the dog slept.";

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

fn heads_bundle() -> PathBuf {
    turbo_pins::require(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-heads-bundle"))
}

/// A row of the C graph with its label read from the pool.
#[derive(Clone, Debug)]
struct Row {
    span: turbo_span,
    label: String,
}

/// Every row of the C analyzer's graph of `text` over `bundle`.
fn c_rows(bundle: &Path, text: &str) -> Vec<Row> {
    let path = bundle.to_str().unwrap();
    let mut err = new_err();
    unsafe {
        let mut desc: turbo_analyzer_desc = std::mem::zeroed();
        desc.struct_size = size_of::<turbo_analyzer_desc>() as u32;
        desc.bundle_path = t(path);
        let mut a: *mut turbo_analyzer = std::ptr::null_mut();
        assert_eq!(turbo_analyzer_create(&desc, &mut a, &mut err), status::OK, "{}", message(&err));
        let mut res: *mut turbo_analysis_result = std::ptr::null_mut();
        assert_eq!(turbo_analyzer_analyze_graph(a, t(text), &mut res, &mut err), status::OK, "{}", message(&err));
        let mut spans_ptr: *const turbo_span = std::ptr::null();
        let mut count: u32 = 0;
        assert_eq!(turbo_analysis_result_spans(res, &mut spans_ptr, &mut count, &mut err), status::OK);
        let mut pool_ptr: *const std::ffi::c_char = std::ptr::null();
        let mut pool_len: u64 = 0;
        assert_eq!(turbo_analysis_result_strings(res, &mut pool_ptr, &mut pool_len, &mut err), status::OK);
        let spans = std::slice::from_raw_parts(spans_ptr, count as usize);
        let pool = std::slice::from_raw_parts(pool_ptr.cast::<u8>(), pool_len as usize);
        let rows = spans
            .iter()
            .map(|&span| {
                let (a, b) = (span.label_offset as usize, (span.label_offset + span.label_len) as usize);
                Row { span, label: std::str::from_utf8(&pool[a..b]).unwrap().to_owned() }
            })
            .collect();
        turbo_analysis_result_release(res);
        turbo_analyzer_release(a);
        rows
    }
}

fn of(rows: &[Row], kind: u32) -> Vec<Row> {
    let mut v: Vec<Row> = rows.iter().filter(|r| r.span.layer_kind == kind).cloned().collect();
    v.sort_by_key(|r| (r.span.token_start, r.span.source_byte_start));
    v
}

fn spans_of<L>(rows: &[Row]) -> Spans<L> {
    let mut spans = Spans::new(ViewId::SOURCE);
    for r in rows {
        spans.push(Span::new(r.span.source_byte_start, r.span.source_byte_end));
    }
    spans
}

#[test]
fn the_c_analyzer_writes_what_the_rust_annotators_write() {
    let bundle = heads_bundle();
    let rows = c_rows(&bundle, TEXT);
    let token_rows = of(&rows, TURBO_ANALYSIS_LAYER_TOKENS);
    let sentence_rows = of(&rows, TURBO_ANALYSIS_LAYER_SENTENCES);
    assert!(
        sentence_rows.len() >= 3 && token_rows.len() > 20,
        "{} sentences, {} tokens",
        sentence_rows.len(),
        token_rows.len()
    );

    // The Rust annotators over the C analyzer's own tokens and sentences.
    let doc = Document::new(TEXT);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let toks = Tokens { spans: spans_of(&token_rows), sentence: None };
    let sents = Sentences { spans: spans_of(&sentence_rows) };
    let multi = MultiTask::<All>::open(&bundle, &[None; 5], &EncoderOptions::default()).unwrap();
    let (pos, morph, ents, deps, lemmas) = multi.annotate((view, &toks, None, Some(&sents))).unwrap();
    let n = token_rows.len();
    assert_eq!((pos.tags.len(), morph.feats.len(), lemmas.lemma.len(), deps.head.len()), (n, n, n, n));

    // Part-of-speech tags with their probabilities.
    let c_pos = of(&rows, TURBO_ANALYSIS_LAYER_POS);
    assert_eq!(c_pos.len(), n);
    let prob = pos.prob.as_ref().expect("the tagger gives probabilities");
    for (i, r) in c_pos.iter().enumerate() {
        assert_eq!((r.span.token_start, r.span.token_end), (i as u32, i as u32 + 1));
        assert_eq!(r.label, pos.tagset.label(pos.tags[i]), "token {i}");
        assert_eq!(r.span.prob, prob[i], "token {i}");
    }

    // Morphological features: one row per token, the feature string as
    // the label, as MorphFeatures writes it.
    let c_morph = of(&rows, TURBO_ANALYSIS_LAYER_MORPHOLOGY);
    assert_eq!(c_morph.len(), n);
    for (i, r) in c_morph.iter().enumerate() {
        assert_eq!((r.span.token_start, r.span.token_end), (i as u32, i as u32 + 1));
        assert!(r.span.prob.is_nan());
        assert_eq!(r.label, morph.feats.get(i), "token {i}");
    }

    // And the pre-tokenized morphology task agrees on the first sentence.
    let first_end = sentence_rows[0].span.source_byte_end;
    let first_words: Vec<turbo_text> = token_rows
        .iter()
        .take_while(|r| r.span.source_byte_end <= first_end)
        .map(|r| t(&TEXT[r.span.source_byte_start as usize..r.span.source_byte_end as usize]))
        .collect();
    unsafe {
        let mut desc: turbo_analyzer_desc = std::mem::zeroed();
        desc.struct_size = size_of::<turbo_analyzer_desc>() as u32;
        desc.bundle_path = t(bundle.to_str().unwrap());
        let mut err = new_err();
        let mut a: *mut turbo_analyzer = std::ptr::null_mut();
        assert_eq!(turbo_analyzer_create(&desc, &mut a, &mut err), status::OK);
        let mut res: *mut turbo_analysis_result = std::ptr::null_mut();
        assert_eq!(
            turbo_analyzer_morphology(a, first_words.as_ptr(), first_words.len() as u32, &mut res, &mut err),
            status::OK,
            "{}",
            message(&err)
        );
        let mut spans_ptr: *const turbo_span = std::ptr::null();
        let mut count: u32 = 0;
        assert_eq!(turbo_analysis_result_spans(res, &mut spans_ptr, &mut count, &mut err), status::OK);
        let mut pool_ptr: *const std::ffi::c_char = std::ptr::null();
        let mut pool_len: u64 = 0;
        assert_eq!(turbo_analysis_result_strings(res, &mut pool_ptr, &mut pool_len, &mut err), status::OK);
        let spans = std::slice::from_raw_parts(spans_ptr, count as usize);
        let pool = std::slice::from_raw_parts(pool_ptr.cast::<u8>(), pool_len as usize);
        assert_eq!(spans.len(), first_words.len());
        for (i, sp) in spans.iter().enumerate() {
            assert_eq!(sp.layer_kind, TURBO_ANALYSIS_LAYER_MORPHOLOGY);
            assert_eq!((sp.token_start, sp.token_end), (i as u32, i as u32 + 1));
            assert!(sp.prob.is_nan());
            let (a, b) = (sp.label_offset as usize, (sp.label_offset + sp.label_len) as usize);
            assert_eq!(std::str::from_utf8(&pool[a..b]).unwrap(), morph.feats.get(i), "token {i}");
        }
        turbo_analysis_result_release(res);
        turbo_analyzer_release(a);
    }

    // Lemmas.
    let c_lemmas = of(&rows, TURBO_ANALYSIS_LAYER_LEMMAS);
    assert_eq!(c_lemmas.len(), n);
    for (i, r) in c_lemmas.iter().enumerate() {
        assert_eq!(r.span.token_start, i as u32);
        assert_eq!(r.label, lemmas.lemma.get(i), "token {i}");
    }

    // Dependencies: the relation as the label, the head token plus one
    // as the classification, zero for a root.
    let c_deps = of(&rows, TURBO_ANALYSIS_LAYER_DEPENDENCIES);
    assert_eq!(c_deps.len(), n);
    for (i, r) in c_deps.iter().enumerate() {
        assert_eq!(r.span.token_start, i as u32);
        assert_eq!(r.label, deps.relations.label(deps.relation[i]), "token {i}");
        assert_eq!(r.span.classification, deps.head[i].map_or(0, |h| h.get() as u32 + 1), "token {i}");
    }

    // Entities: the same token ranges, kinds and scores.
    let c_ents: Vec<(u32, u32, String, f64)> = of(&rows, TURBO_ANALYSIS_LAYER_ENTITIES)
        .iter()
        .map(|r| (r.span.token_start, r.span.token_end, r.label.clone(), r.span.prob))
        .collect();
    let rust_ents: Vec<(u32, u32, String, f64)> = (0..ents.kind.len())
        .map(|i| {
            let r = &ents.tokens[i];
            (r.start.get() as u32, r.end.get() as u32, ents.kinds.label(ents.kind[i]).to_owned(), ents.score[i])
        })
        .collect();
    assert!(!rust_ents.is_empty(), "the seeded entity head marks nothing");
    assert_eq!(c_ents, rust_ents);
}
