//! One encoder pass for many heads (#1517) on the small BERT in testdata
//! with its four seeded heads: each layer written as its single-head
//! annotator writes it, from one read of each window whatever the number
//! of heads or poolings, long sentences in windows, the bundle's licence
//! tier, and refusals.

mod common;

use common::*;
use serde_json::json;
use turbo::head::edit_tree::EditTree;
use turbo::manifest::LicenseClass;
use turbo_analysis::{Annotator, Document, Key, Layer, View};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{Dependencies, Entities, Lemmas, PosTags, Sentences, Tokens, Words};
use turbo_tagger::{DependencyParser, EntityTagger, Lemmatizer, MultiTask, PosTagger, SentenceTagger, WordTagger};

type All = (PosTags, Entities, Dependencies<Tokens>, Lemmas);

const LONG: &str = "Ann met Bob in Paris , and the quiet old dog slept by the river while Bob walked the dog \
                    and Ann watched the river and the old dog .";

/// Every layer of `text` from the four heads at once and from the four
/// single-head annotators, which must agree field for field.
fn agree(dir: &std::path::Path, o: &EncoderOptions, text: &str) {
    let doc = Document::new(text);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let toks = tokens(text);
    let multi = MultiTask::<All>::open(dir, &[None; 4], o).unwrap();
    let (pos, ents, deps, lemmas) = multi.annotate((view, &toks, None, None)).unwrap();

    let (p,) = PosTagger::open(dir, None, o).unwrap().annotate((view, &toks, None, None)).unwrap();
    assert_eq!(pos.tags, p.tags);
    assert_eq!(pos.prob, p.prob);
    assert_eq!(pos.tagset.label(turbo_analysis_layers::Tag(1)), p.tagset.label(turbo_analysis_layers::Tag(1)));
    let (e,) = EntityTagger::open(dir, None, o).unwrap().annotate((view, &toks, None, None)).unwrap();
    let spans = |e: &Entities| e.spans.iter().map(|(_, s)| (s.start, s.end)).collect::<Vec<_>>();
    assert_eq!((spans(&ents), &ents.tokens, &ents.kind, &ents.score), (spans(&e), &e.tokens, &e.kind, &e.score));
    let (d,) = DependencyParser::open(dir, None, o).unwrap().annotate((view, &toks, None, None)).unwrap();
    assert_eq!((&deps.head, &deps.relation), (&d.head, &d.relation));
    let (l,) = Lemmatizer::open(dir, None, o).unwrap().annotate((view, &toks, None, None)).unwrap();
    let words = |l: &Lemmas| l.lemma.iter().map(str::to_owned).collect::<Vec<_>>();
    assert_eq!(words(&lemmas), words(&l));
    assert_eq!(words(&lemmas).len(), toks.spans.len());
}

#[test]
fn each_layer_is_what_its_single_head_annotator_writes() {
    let dir = bundle("multi-agree", |_| {});
    agree(&dir, &EncoderOptions::default(), TEXT);
}

#[test]
fn long_sentences_read_in_windows_agree_too() {
    let dir = bundle("multi-windows", |_| {});
    // Rows of six pieces beside the special tokens: the long sentence is
    // read in many overlapping windows, several rows a batch.
    let o = EncoderOptions { max_seq: 8, max_batch: 3, ..EncoderOptions::default() };
    agree(&dir, &o, LONG);
    // One row a run holds fewer ranges than a row's words take under two
    // poolings, so such a row is read once, a range a piece.
    let o = EncoderOptions { max_seq: 8, max_batch: 1, ..EncoderOptions::default() };
    agree(&dir, &o, LONG);
}

#[test]
fn one_read_of_each_window_serves_every_head_and_pooling() {
    let dir = bundle("multi-rows", |_| {});
    let doc = Document::new(LONG);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let toks = tokens(LONG);
    // Many rows a run, and one: the pos head pools by the mean, the
    // others by the first piece.
    for max_batch in [0, 1] {
        let o = EncoderOptions { max_seq: 8, max_batch, ..EncoderOptions::default() };
        let one = MultiTask::<(PosTags,)>::open(&dir, &[None], &o).unwrap();
        let all = MultiTask::<All>::open(&dir, &[None; 4], &o).unwrap();
        let first = MultiTask::<(Entities,)>::open(&dir, &[None], &o).unwrap();
        for m in [&one as &dyn Rows, &all, &first] {
            m.run(view, &toks);
        }
        assert!(one.rows() > 1, "the long sentence is read in windows");
        assert_eq!(all.rows(), one.rows(), "max_batch {max_batch}");
        assert_eq!(first.rows(), one.rows(), "max_batch {max_batch}");
    }
}

/// A multi-task annotator run once, whatever its outputs.
trait Rows {
    fn run(&self, view: &View, toks: &Tokens);
}

impl<W: turbo_tagger::multi::Outputs> Rows for MultiTask<W> {
    fn run(&self, view: &View, toks: &Tokens) {
        self.layers(view, toks, None, None).map(|_| ()).unwrap();
    }
}

#[test]
fn two_outputs_of_one_layer_each_read_their_named_head() {
    let dir = bundle("multi-two-pos", |m| {
        let mut upos = m["heads"][0].clone();
        upos["name"] = json!("upos");
        upos["tensor_prefix"] = json!("heads.upos");
        m["heads"].as_array_mut().unwrap().push(upos);
    });
    let doc = Document::new(TEXT);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let toks = tokens(TEXT);
    let o = EncoderOptions::default();
    let m = MultiTask::<(PosTags, PosTags)>::open(&dir, &[Some("pos"), Some("upos")], &o).unwrap();
    let names: Vec<&str> = m.heads().iter().map(|h| h.spec().name.as_str()).collect();
    assert_eq!(names, ["pos", "upos"]);
    let (a, b) = m.annotate((view, &toks, None, None)).unwrap();
    let (p,) = PosTagger::open(&dir, Some("pos"), &o).unwrap().annotate((view, &toks, None, None)).unwrap();
    let (u,) = PosTagger::open(&dir, Some("upos"), &o).unwrap().annotate((view, &toks, None, None)).unwrap();
    assert_eq!((&a.tags, &b.tags), (&p.tags, &u.tags));
    // Two heads of seeded weights tag differently, so the outputs are not
    // one head's twice.
    assert_ne!(a.prob, b.prob);
    // With two heads on the layer, an output names its head.
    let e = MultiTask::<(PosTags,)>::open(&dir, &[None], &o).err().unwrap();
    assert!(e.message.contains("several heads"), "{e:?}");
}

#[test]
fn the_tier_is_the_strictest_of_the_encoder_and_the_heads() {
    let o = EncoderOptions::default();
    let dir = bundle("multi-tier", |m| {
        m["model"]["license_class"] = json!("CLASS_PERMISSIVE");
        m["heads"][1]["license_class"] = json!("CLASS_SHARE_ALIKE");
        m["heads"][2]["license_class"] = json!("CLASS_LGPL");
    });
    let m = MultiTask::<(PosTags,)>::open(&dir, &[None], &o).unwrap();
    // The bundle's tier, though this annotator runs only the pos head.
    assert_eq!(m.license_tier(), Some(LicenseClass::ShareAlike));
    let plain = bundle("multi-tier-plain", |m| m["model"]["license_class"] = json!("CLASS_PERMISSIVE"));
    let m = MultiTask::<(PosTags,)>::open(&plain, &[None], &o).unwrap();
    assert_eq!(m.license_tier(), Some(LicenseClass::Permissive));
    let unknown = bundle("multi-tier-unknown", |_| {});
    assert_eq!(MultiTask::<(PosTags,)>::open(&unknown, &[None], &o).unwrap().license_tier(), None);

    // A head's class without the encoder's would leave the encoder out.
    let dir = bundle("multi-tier-no-model", |m| m["heads"][1]["license_class"] = json!("CLASS_SHARE_ALIKE"));
    let e = MultiTask::<(PosTags,)>::open(&dir, &[None], &o).err().unwrap();
    assert!(e.message.contains("heads[1].license_class"), "{e:?}");
}

#[test]
fn refusals_name_the_problem() {
    let dir = bundle("multi-refuse", |_| {});
    let o = EncoderOptions::default();
    let msg = |r: Result<MultiTask<(PosTags, Lemmas)>, turbo::status::Error>| r.err().unwrap().message;
    assert!(msg(MultiTask::open(&dir, &[None], &o)).contains("1 head names for 2 outputs"));
    // A tags head cannot write entities, nor a parser part of speech.
    let e = MultiTask::<(Entities,)>::open(&dir, &[Some("pos")], &o).err().unwrap();
    assert!(e.message.contains("SCHEME_BIO"), "{e:?}");
    let e = MultiTask::<(PosTags,)>::open(&dir, &[Some("deps")], &o).err().unwrap();
    assert!(e.message.contains("is not a TokenTagger head"), "{e:?}");
    // One head is not two outputs.
    let e = MultiTask::<(PosTags, PosTags)>::open(&dir, &[None, Some("pos")], &o).err().unwrap();
    assert!(e.message.contains("named for two outputs"), "{e:?}");
    let e = MultiTask::<(Lemmas,)>::open(&dir, &[Some("nope")], &o).err().unwrap();
    assert!(e.message.contains("no head \"nope\""), "{e:?}");
    assert_eq!(<MultiTask<All> as Annotator>::ID, "turbo-tagger-multitask");
}

#[test]
fn sentence_and_word_split_heads_run_in_the_one_pass() {
    let and_split = ["an", "d"].iter().map(|w| EditTree::mine("and", w).label()).collect::<Vec<_>>().join(" ");
    let dir = bundle("multi-segment", |m| {
        let arr = m["heads"].as_array_mut().unwrap();
        arr.push(json!({
            "name": "sents", "layer": "turboembed:sentences", "kind": "HEAD_SENTENCE_BOUNDARY",
            "labels": ["inside", "last"], "scheme": "SCHEME_TAGS", "word_pooling": "WORD_POOLING_FIRST",
            "decode": "DECODE_ARGMAX", "artifact": "weights-f32", "tensor_prefix": "heads.sents"
        }));
        arr.push(json!({
            "name": "words", "layer": "turboembed:words", "kind": "HEAD_TOKEN_SPLIT",
            "labels": ["_", and_split], "scheme": "SCHEME_TAGS", "word_pooling": "WORD_POOLING_FIRST",
            "decode": "DECODE_EDIT_TREE", "artifact": "weights-f32", "tensor_prefix": "heads.words"
        }));
    });
    let o = EncoderOptions::default();
    let doc = Document::new(LONG);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let toks = tokens(LONG);

    // Without the sentence head, Words in MultiTask matches WordTagger.
    let mw = MultiTask::<(Words, PosTags)>::open(&dir, &[None, None], &o).unwrap();
    let (w_multi, p_multi) = mw.annotate((view, &toks, None, None)).unwrap();
    let (w_single,) = WordTagger::open(&dir, None, &o).unwrap().annotate((view, &toks, None, None)).unwrap();
    let (p_single,) = PosTagger::open(&dir, None, &o).unwrap().annotate((view, &toks, None, None)).unwrap();
    let word_triples =
        |l: &Words| l.iter(LONG, &toks).map(|w| (w.tokens, w.subspan, w.form.to_owned())).collect::<Vec<_>>();
    assert_eq!(word_triples(&w_multi), word_triples(&w_single));
    assert_eq!(p_multi.tags, p_single.tags);

    // With Sentences in MultiTask and no prior sentences, the encoder runs
    // once over the whole view, Sentences matches SentenceTagger, and the
    // other heads' sequences are cut by the decoded sentences.
    type Six = (Sentences, Words, PosTags, Entities, Dependencies<Tokens>, Lemmas);
    let six = MultiTask::<Six>::open(&dir, &[None; 6], &o).unwrap();
    let st = SentenceTagger::open(&dir, None, &o).unwrap();
    let (sents, words, pos, ents, deps, lemmas) = six.annotate((view, &toks, None, None)).unwrap();
    let (want_sents,) = st.annotate((view, &toks, None)).unwrap();
    assert_eq!(six.rows(), st.rows(), "one whole-view encoder pass serves all six heads");
    let sent_spans = |s: &Sentences| s.spans.iter().map(|(_, sp)| (sp.start, sp.end)).collect::<Vec<_>>();
    assert_eq!(sent_spans(&sents), sent_spans(&want_sents));
    assert!(sents.spans.len() > 1, "the seeded boundary head splits LONG into multiple sentences");
    words.check_against(LONG, &toks).unwrap();
    assert_eq!(pos.tags.len(), toks.spans.len());
    assert_eq!(lemmas.lemma.len(), toks.spans.len());
    // Each decoded sentence gets its own dependency tree with one root,
    // and no arc or entity crosses a decoded sentence boundary.
    let roots = deps.head.iter().filter(|h| h.is_none()).count();
    assert_eq!(roots, sents.spans.len());
    for (_, sp) in ents.spans.iter() {
        assert!(sents.spans.iter().any(|(_, s)| sp.start >= s.start && sp.end <= s.end));
    }

    // Empty document through the six-head MultiTask.
    let empty = Document::new("");
    let ev = empty.require(&Key::<View>::DEFAULT).unwrap();
    let none = Tokens { spans: turbo_analysis::Spans::new(ev.id()), sentence: None };
    let (es, ew, ep, ee, ed, el) = six.annotate((ev, &none, None, None)).unwrap();
    assert!(
        es.spans.is_empty()
            && ew.is_empty()
            && ep.tags.is_empty()
            && ee.spans.is_empty()
            && ed.head.is_empty()
            && el.lemma.is_empty()
    );
}
