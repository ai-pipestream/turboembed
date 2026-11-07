//! The capability-driven `annotate-v2` profile and the head steps added
//! by #1832 on sealed tiny bundles: partial bundles, morphology, BIO and
//! span entities side by side, relations, learned sentences, syntactic
//! word splits and coreference, on Unicode text read in several windows.
//! Each layer is compared field for field with the tagger crate's own
//! annotators (or, where the whole-view read slices the sentences, with
//! the same steps run in process), and every head layer's offsets are
//! checked in UTF-8, UTF-16 and code points.

#[path = "progressive_heads/bundle.rs"]
mod bundle;
mod common;
#[path = "progressive_heads/harness.rs"]
mod harness;

use bundle::*;
use harness::*;
use serde_json::Value;
use tonic::Code;
use turbo_analysis::{Analyzer, Bound, Document, Edges, Groups, Key, View};
use turbo_analysis_layers::{Dependencies, Entities, Lemmas, MorphFeatures, PosTags, Sentences, Tokens, Words};
use turbo_kserve::progressive::ANNOTATE_V2;
use turbo_kserve::progressive::heads::{
    COREF_INSTANCE, CorefStep, GroupStep, HeadGroup, HeadStep, SPANS_INSTANCE, SentenceStep, SpanStep,
};
use turbo_kserve::proto::result::*;
use turbo_kserve::results::typed::inspect_document_graph;
use turbo_relation::Relations;
use turbo_tagger::{CorefTagger, MultiTask, RelationTagger, SpanTagger};
use turbo_tokenize::uax29::UntypedWordTokenizer;

/// Unicode in names and a fusional `del`, long enough to read each
/// sentence in several windows of `MAX_SEQ`.
const DOC: &str = "Zoë met Ann in São Paulo, and the quiet old dog slept by the river del Río. Ann walked \
                   the dog while Zoë watched the river and the old dog. Then Zoë and Ann left São Paulo.";

/// Every field of a layer, as text: two layers with the same fields are
/// the same bit for bit (`f64` prints its shortest round trip).
trait Fields {
    fn fields(&self) -> String;
}

fn spans_of(s: &turbo_analysis::Spans<impl Sized>) -> Vec<(u32, u32)> {
    s.iter().map(|(_, s)| (s.start, s.end)).collect()
}

impl Fields for Entities {
    fn fields(&self) -> String {
        format!("{:?}", (spans_of(&self.spans), &self.tokens, &self.kind, &self.score, self.kinds.labels()))
    }
}

impl Fields for PosTags {
    fn fields(&self) -> String {
        format!("{:?}", (&self.tags, &self.prob, self.tagset.labels()))
    }
}

impl Fields for Lemmas {
    fn fields(&self) -> String {
        format!("{:?}", self.lemma.iter().collect::<Vec<_>>())
    }
}

impl Fields for MorphFeatures<Tokens> {
    fn fields(&self) -> String {
        format!("{:?}", self.feats.iter().collect::<Vec<_>>())
    }
}

impl Fields for Dependencies<Tokens> {
    fn fields(&self) -> String {
        format!("{:?}", (&self.head, &self.relation, self.relations.labels()))
    }
}

impl Fields for Sentences {
    fn fields(&self) -> String {
        format!("{:?}", spans_of(&self.spans))
    }
}

impl Fields for Words {
    fn fields(&self) -> String {
        format!("{self:?}")
    }
}

impl Fields for Relations {
    fn fields(&self) -> String {
        let r = self;
        format!("{:?}", (spans_of(&r.spans), &r.subject, &r.object, &r.kind, &r.score, r.kinds.labels()))
    }
}

impl Fields for Groups<Entities> {
    fn fields(&self) -> String {
        format!("{:?}", (&self.members, &self.end))
    }
}

impl Fields for Edges<Entities, Entities> {
    fn fields(&self) -> String {
        format!("{:?}", (&self.from, &self.to, &self.kind, &self.score, self.kinds.labels()))
    }
}

fn same<T: Fields>(got: &T, want: &T, what: &str) {
    assert_eq!(got.fields(), want.fields(), "{what}");
}

fn push(m: &mut Value, heads: &[Value]) {
    m["heads"].as_array_mut().unwrap().extend(heads.iter().cloned());
}

fn typed(complete: &DocumentGraphResponse, offsets: OffsetMode) -> TypedDocumentResult {
    inspect_document_graph(&InspectDocumentGraphRequest {
        doc_id: "heads".into(),
        graph: complete.graph.to_vec(),
        view: 0,
        offsets: offsets as i32,
        pii_output: None,
    })
    .unwrap()
}

/// Every annotation offset of every layer in UTF-16 and code points is
/// the UTF-8 one converted over the text.
fn offsets_agree(complete: &DocumentGraphResponse, text: &str) {
    let (u8s, u16s, cps) =
        (typed(complete, OffsetMode::Utf8), typed(complete, OffsetMode::Utf16), typed(complete, OffsetMode::CodePoint));
    let utf16 = |b: u32| text[..b as usize].encode_utf16().count() as u32;
    let cp = |b: u32| text[..b as usize].chars().count() as u32;
    let mut checked = 0;
    for ((a, b), c) in u8s.layers.iter().zip(&u16s.layers).zip(&cps.layers) {
        for ((x, y), z) in a.annotations.iter().zip(&b.annotations).zip(&c.annotations) {
            for ((f, g), h) in x.spans.iter().zip(&y.spans).zip(&z.spans) {
                assert!(text.is_char_boundary(f.start as usize) && text.is_char_boundary(f.end as usize), "{}", a.id);
                assert_eq!((g.start, g.end), (utf16(f.start), utf16(f.end)), "UTF-16 in {}", a.id);
                assert_eq!((h.start, h.end), (cp(f.start), cp(f.end)), "code points in {}", a.id);
                checked += 1;
            }
        }
    }
    assert!(checked > 0);
}

#[tokio::test]
async fn annotate_v2_runs_a_span_only_entity_bundle_that_annotate_v1_refuses() {
    let dir = bundle("v2-partial", |m| m["heads"][1] = span_head());
    let (server, mut client) = serve_models(vec![model(&dir, "spans", MAX_SEQ)]).await;
    let err = client.analyze_document_progressive(request(DOC, "annotate-v1", &[])).await.unwrap_err();
    assert!(err.message().contains("head.entities"), "{}", err.message());

    let events = run(&mut client, request(DOC, "annotate-v2", &[])).await;
    let mut want = vec![
        "turbo-structure:blocks",
        "turbo-sentdetect:rules",
        "turbo-tokenize:uax29-untyped",
        "turbo-tagger:pos",
        "turbo-tagger:dependencies",
        "turbo-tagger:lemmas",
        "turbo-tagger:spans",
    ];
    want.sort_unstable();
    assert_eq!(started(&events), want);
    let complete = complete(&events);
    // With no other entity writer the spans are the default entity layer.
    assert!(layers(complete).contains(&("turbo:entities".into(), String::new())));
    let doc = document(complete);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let (tokens, sentences) = (doc.get::<Tokens>().unwrap(), doc.get::<Sentences>().unwrap());
    let o = encoder_options(MAX_SEQ);
    let spans = SpanTagger::open(&dir, None, &o).unwrap().entities(view, tokens, Some(sentences), None).unwrap();
    same(doc.get::<Entities>().unwrap(), &spans, "span entities");
    let (pos, deps, lemmas) = MultiTask::<(PosTags, Dependencies<Tokens>, Lemmas)>::open(&dir, &[None; 3], &o)
        .unwrap()
        .layers(view, tokens, Some(sentences), None)
        .unwrap();
    same(doc.get::<PosTags>().unwrap(), &pos, "pos");
    same(doc.get::<Dependencies<Tokens>>().unwrap(), &deps, "dependencies");
    same(doc.get::<Lemmas>().unwrap(), &lemmas, "lemmas");
    offsets_agree(complete, DOC);
    server.stop().await;
}

#[tokio::test]
async fn annotate_v2_writes_morphology_both_entity_heads_and_relations() {
    let dir = bundle("v2-sentences", |m| push(m, &[morph_head(), span_head(), relation_head(&["PER", "LOC"])]));
    let (server, mut client) = serve_models(vec![model(&dir, "heads", MAX_SEQ)]).await;
    let caps = client.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    let profile = caps.profiles.iter().find(|p| p.name == "annotate-v2").unwrap();
    assert_eq!(profile.required_bundles, ANNOTATE_V2.required_bundles);
    assert_eq!(profile.steps, ANNOTATE_V2.steps);

    let complete_events = run(&mut client, request(DOC, "annotate-v2", &[])).await;
    let complete = complete(&complete_events);
    let listed = layers(complete);
    for (layer, instance) in
        [("turbo:feats", ""), ("turbo:entities", ""), ("turbo:entities", SPANS_INSTANCE), ("turbo:relations", "")]
    {
        assert!(listed.contains(&(layer.into(), instance.into())), "{layer}#{instance} in {listed:?}");
    }
    let doc = document(complete);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let (tokens, sentences) = (doc.get::<Tokens>().unwrap(), doc.get::<Sentences>().unwrap());
    let o = encoder_options(MAX_SEQ);
    type Five = (PosTags, MorphFeatures<Tokens>, Entities, Dependencies<Tokens>, Lemmas);
    let (pos, feats, ents, deps, lemmas) = MultiTask::<Five>::open(&dir, &[None, None, Some("ner"), None, None], &o)
        .unwrap()
        .layers(view, tokens, Some(sentences), None)
        .unwrap();
    same(doc.get::<PosTags>().unwrap(), &pos, "pos");
    same(doc.get::<MorphFeatures<Tokens>>().unwrap(), &feats, "morphology");
    same(doc.get::<Entities>().unwrap(), &ents, "BIO entities");
    same(doc.get::<Dependencies<Tokens>>().unwrap(), &deps, "dependencies");
    same(doc.get::<Lemmas>().unwrap(), &lemmas, "lemmas");
    let spans =
        SpanTagger::open(&dir, Some("mentions"), &o).unwrap().entities(view, tokens, Some(sentences), None).unwrap();
    same(doc.require(&Key::<Entities>::named(SPANS_INSTANCE)).unwrap(), &spans, "span entities");
    // Relations read the default (BIO) entities.
    let (relations, edges) =
        RelationTagger::open(&dir, None, &o).unwrap().relations(view, tokens, &ents, Some(sentences), None).unwrap();
    same(doc.get::<Relations>().unwrap(), &relations, "relations");
    same(doc.get::<Edges<Entities, Entities>>().unwrap(), &edges, "relation edges");
    offsets_agree(complete, DOC);
    server.stop().await;
}

fn whole_bundle(name: &str) -> std::path::PathBuf {
    bundle(name, |m| {
        m["heads"] = serde_json::json!([heads()[0].clone(), span_head(), sentence_head(), split_head(), coref_head()])
    })
}

#[tokio::test]
async fn learned_sentences_word_splits_and_coreference_share_the_whole_view_read() {
    let dir = whole_bundle("v2-whole");
    let (server, mut client) = serve_models(vec![model(&dir, "whole", MAX_SEQ)]).await;
    let caps = client.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    assert_eq!(
        caps.bundles[0].head_steps,
        [
            "turbo-tagger:pos",
            "turbo-tagger:spans",
            "turbo-tagger:sentences",
            "turbo-tagger:words",
            "turbo-tagger:coref"
        ]
    );
    let events = run(&mut client, request(DOC, "annotate-v2", &[])).await;
    let mut want = vec![
        "turbo-structure:blocks",
        "turbo-tokenize:uax29-untyped",
        "turbo-tagger:pos",
        "turbo-tagger:spans",
        "turbo-tagger:sentences",
        "turbo-tagger:words",
        "turbo-tagger:coref",
    ];
    want.sort_unstable();
    // The learned sentence head replaces the rule sentence step.
    assert_eq!(started(&events), want);
    let complete = complete(&events);
    assert!(layers(complete).iter().any(|(_, i)| i == COREF_INSTANCE), "the links under their own instance");
    let doc = document(complete);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let tokens = doc.get::<Tokens>().unwrap();
    assert!(tokens.sentence.is_none(), "the tokens do not read the learned sentences");
    let o = encoder_options(MAX_SEQ);

    // The tagger crate's one-pass annotator under a sentence head.
    let (sents, pos, words) = MultiTask::<(Sentences, PosTags, Words)>::open(&dir, &[None; 3], &o)
        .unwrap()
        .layers(view, tokens, None, None)
        .unwrap();
    same(doc.get::<Sentences>().unwrap(), &sents, "learned sentences");
    same(doc.get::<PosTags>().unwrap(), &pos, "pos over the learned sentences");
    same(doc.get::<Words>().unwrap(), &words, "syntactic words");
    // The coreference annotator reads the view as the group does.
    let ents = doc.get::<Entities>().unwrap();
    let (groups, links) = CorefTagger::open(&dir, None, &o).unwrap().resolve(view, tokens, ents, None).unwrap();
    same(doc.get::<Groups<Entities>>().unwrap(), &groups, "coreference chains");
    same(doc.require(&Key::<Edges<Entities, Entities>>::named(COREF_INSTANCE)).unwrap(), &links, "coreference links");

    // The same steps in process, from one group.
    let steps = [HeadStep::Pos, HeadStep::Spans, HeadStep::Sentences, HeadStep::Words, HeadStep::Coref];
    let group = HeadGroup::open(&dir, &steps, &o).unwrap();
    let analyzer = Analyzer::builder()
        .step(Bound::new(UntypedWordTokenizer::default()).read(1, "unread"))
        .add(SentenceStep(group.clone()))
        .add(GroupStep::<PosTags>::new(group.clone()))
        .add(GroupStep::<Words>::new(group.clone()))
        .add(SpanStep { group: group.clone() })
        .step(Bound::new(CorefStep { group: group.clone() }).write(1, COREF_INSTANCE))
        .build()
        .unwrap();
    let mut docs = vec![Document::new(DOC)];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let local = &docs[0];
    same(doc.get::<Entities>().unwrap(), local.get::<Entities>().unwrap(), "span entities");
    same(doc.get::<Groups<Entities>>().unwrap(), local.get::<Groups<Entities>>().unwrap(), "chains in process");
    assert!(ents.spans.len() > 1, "the seeded span head marks mentions");
    offsets_agree(complete, DOC);
    server.stop().await;
}

#[tokio::test]
async fn competing_and_missing_head_selections_are_refused_by_name() {
    let second_pos = {
        let mut h = heads()[0].clone();
        h["name"] = "pos2".into();
        h["tensor_prefix"] = "heads.pos2".into();
        h
    };
    let dir = bundle("select", |m| push(m, &[second_pos]));
    let (server, mut client) = serve_models(vec![model(&dir, "two-pos", 0)]).await;
    let base = ["turbo-sentdetect:rules", "turbo-tokenize:uax29-untyped"];
    let refused = |steps: &[&str]| {
        let mut all = base.to_vec();
        all.extend_from_slice(steps);
        request(DOC, "", &all)
    };
    for (steps, code, named) in [
        (&["turbo-tagger:pos"][..], Code::InvalidArgument, "ambiguous head"),
        (&["turbo-tagger:pos@ner"][..], Code::Unimplemented, "unsupported head"),
        (&["turbo-tagger:pos@nope"][..], Code::NotFound, "missing head"),
        (&["turbo-tagger:sentences"][..], Code::NotFound, "head.sentences"),
        (&["turbo-tagger:entities", "turbo-geo:names"][..], Code::InvalidArgument, "ambiguous output"),
        (&["turbo-tagger:pos@pos", "turbo-tagger:pos@pos2"][..], Code::InvalidArgument, "ambiguous output"),
    ] {
        let err = client.analyze_document_progressive(refused(steps)).await.unwrap_err();
        assert_eq!(err.code(), code, "{steps:?}: {}", err.message());
        assert!(err.message().contains(named), "{steps:?}: {}", err.message());
    }
    let err = client.analyze_document_progressive(request(DOC, "annotate-v2", &[])).await.unwrap_err();
    assert!(err.message().contains("ambiguous head"), "{}", err.message());

    // A selection picks one of the two.
    let events = run(&mut client, refused(&["turbo-tagger:pos@pos2", "turbo-tagger:lemmas"])).await;
    let complete = complete(&events);
    let doc = document(complete);
    assert_eq!(doc.get::<PosTags>().unwrap().tagset.name, "pos2");
    server.stop().await;

    // A learned sentence step beside the rule one is ambiguous too.
    let dir = whole_bundle("select-sentences");
    let (server, mut client) = serve_models(vec![model(&dir, "whole", 0)]).await;
    let err = client.analyze_document_progressive(refused(&["turbo-tagger:sentences"])).await.unwrap_err();
    assert!(err.message().contains("ambiguous output"), "{}", err.message());
    server.stop().await;
}
