//! One encoder pass for many heads (#1517) on the small BERT in testdata
//! with its four seeded heads: each layer written as its single-head
//! annotator writes it, from one read of each window whatever the number
//! of heads or poolings, long sentences in windows, the bundle's licence
//! tier, and refusals.

mod common;

use common::*;
use serde_json::json;
use turbo::manifest::LicenseClass;
use turbo_analysis::{Annotator, Document, Key, View};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{Dependencies, Entities, Lemmas, PosTags, Tokens};
use turbo_tagger::{DependencyParser, EntityTagger, Lemmatizer, MultiTask, PosTagger};

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
    // poolings, so such a row is read once a pooling.
    let o = EncoderOptions { max_seq: 8, max_batch: 1, ..EncoderOptions::default() };
    agree(&dir, &o, LONG);
}

#[test]
fn one_read_of_each_window_serves_every_head_and_pooling() {
    let dir = bundle("multi-rows", |_| {});
    let doc = Document::new(LONG);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let toks = tokens(LONG);
    let o = EncoderOptions { max_seq: 8, ..EncoderOptions::default() };
    // The pos head pools by the mean, the others by the first piece.
    let one = MultiTask::<(PosTags,)>::open(&dir, &[None], &o).unwrap();
    let all = MultiTask::<All>::open(&dir, &[None; 4], &o).unwrap();
    let first = MultiTask::<(Entities,)>::open(&dir, &[None], &o).unwrap();
    for m in [&one as &dyn Rows, &all, &first] {
        m.run(view, &toks);
    }
    assert!(one.rows() > 1, "the long sentence is read in windows");
    assert_eq!(all.rows(), one.rows());
    assert_eq!(first.rows(), one.rows());
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
