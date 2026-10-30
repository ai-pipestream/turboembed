//! Regression tests for partial-batch `ViewId` allocation when multiple steps in the
//! same topological stage produce derived [`View`] layers (`!all_full_batch`).

use turbo_analysis::export::{Codecs, export};
use turbo_analysis::{
    Analyzer, Annotator, Bound, Document, Error, Ix, Key, Refs, Span, Spans, StageCacheStatus, View, ViewBuilder,
};
use turbo_analysis_layers::export::register_codecs;
use turbo_analysis_layers::{Sentences, Tokens};

fn codecs() -> Codecs {
    let mut c = Codecs::new();
    register_codecs(&mut c);
    c
}

struct PrefixSentenceStep;

impl Annotator for PrefixSentenceStep {
    const ID: &'static str = "test:prefix-sentence";
    type Reads = (View,);
    type Writes = (Sentences,);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let mut spans = Spans::new(view.id());
        if !view.text().is_empty() {
            spans.push(Span::new(0, view.text().len() as u32));
        }
        Ok((Sentences { spans },))
    }
}

struct UpperViewStep {
    tag: u64,
}

impl Annotator for UpperViewStep {
    const ID: &'static str = "test:upper-view";
    type Reads = (View,);
    type Writes = (View,);

    fn config_hash(&self) -> u64 {
        self.tag
    }

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let text = view.text();
        let mut vb = ViewBuilder::new(text);
        if !text.is_empty() {
            vb.replace(text.len(), &text.to_ascii_uppercase());
        }
        let (out, align) = vb.finish();
        Ok((View::derived(view, out, align),))
    }
}

struct BracketViewStep {
    tag: u64,
}

impl Annotator for BracketViewStep {
    const ID: &'static str = "test:bracket-view";
    type Reads = (View,);
    type Writes = (View,);

    fn config_hash(&self) -> u64 {
        self.tag
    }

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        // Produce a derived view of different byte length so span bounds on each view are distinct.
        let text = view.text();
        let mut vb = ViewBuilder::new(text);
        if !text.is_empty() {
            vb.replace(text.len(), &format!("[{text}]"));
        }
        let (out, align) = vb.finish();
        Ok((View::derived(view, out, align),))
    }
}

struct WholeViewTokenizer;

impl Annotator for WholeViewTokenizer {
    const ID: &'static str = "test:whole-view-tok";
    type Reads = (View,);
    type Writes = (Tokens,);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let mut spans = Spans::new(view.id());
        if !view.text().is_empty() {
            spans.push(Span::new(0, view.text().len() as u32));
        }
        Ok((Tokens { spans, sentence: None },))
    }
}

#[test]
fn partial_batch_same_stage_view_producers_do_not_collide_view_ids() {
    let norm_a = Key::<View>::named("norm_a");
    let norm_b = Key::<View>::named("norm_b");
    let tok_a = Key::<Tokens>::named("tok_a");
    let tok_b = Key::<Tokens>::named("tok_b");

    // Stage 0 has 3 independent steps: PrefixSentenceStep, UpperViewStep -> norm_a, BracketViewStep -> norm_b.
    // Stage 1 has 2 steps: WholeViewTokenizer(norm_a -> tok_a), WholeViewTokenizer(norm_b -> tok_b).
    let warmup_sentences_only = Analyzer::builder().add(PrefixSentenceStep).build().unwrap();

    let full_pipeline = Analyzer::builder()
        .add(PrefixSentenceStep)
        .step(Bound::new(UpperViewStep { tag: 1 }).write(0, "norm_a"))
        .step(Bound::new(BracketViewStep { tag: 1 }).write(0, "norm_b"))
        .step(Bound::new(WholeViewTokenizer).read(0, "norm_a").write(0, "tok_a"))
        .step(Bound::new(WholeViewTokenizer).read(0, "norm_b").write(0, "tok_b"))
        .build()
        .unwrap();

    let mut d0 = Document::new("alpha beta");
    let d1 = Document::new("gamma delta epsilon");

    // Pre-populate Sentences on d0 only so Stage 0 takes the `!all_full_batch` branch:
    // PrefixSentenceStep skips on d0 and runs on d1, while both norm_a and norm_b run on [d0, d1].
    warmup_sentences_only.analyze_incremental(&mut d0).unwrap();

    let mut batch = vec![d0, d1];
    let reports = full_pipeline.run_incremental(&mut batch);
    let r0 = reports[0].as_ref().expect("d0 incremental succeeded");
    let r1 = reports[1].as_ref().expect("d1 incremental succeeded");

    assert_eq!(r0.stages[0].cache_status, StageCacheStatus::Skipped);
    assert_eq!(r0.stages[1].cache_status, StageCacheStatus::Executed);
    assert_eq!(r0.stages[2].cache_status, StageCacheStatus::Executed);
    assert_eq!(r1.stages_run, 5);

    let c = codecs();
    for (doc, raw) in batch.iter().zip(["alpha beta", "gamma delta epsilon"]) {
        let va = doc.get_key(&norm_a).expect("norm_a present");
        let vb = doc.get_key(&norm_b).expect("norm_b present");
        assert_ne!(va.id(), vb.id(), "derived views in the same stage must receive distinct ViewIds");
        assert_eq!(va.text(), raw.to_ascii_uppercase());
        assert_eq!(vb.text(), format!("[{raw}]"));

        let ta = doc.get_key(&tok_a).expect("tok_a present");
        let tb = doc.get_key(&tok_b).expect("tok_b present");
        assert_eq!(ta.spans.view(), va.id());
        assert_eq!(tb.spans.view(), vb.id());
        assert_eq!(ta.spans.get(Ix::new(0)), Span::new(0, va.text().len() as u32));
        assert_eq!(tb.spans.get(Ix::new(0)), Span::new(0, vb.text().len() as u32));

        let mut seq_doc = Document::new(raw);
        full_pipeline.run_sequential(std::slice::from_mut(&mut seq_doc)).pop().unwrap().unwrap();
        assert_eq!(
            export(doc, &c).unwrap().to_bytes().unwrap(),
            export(&seq_doc, &c).unwrap().to_bytes().unwrap(),
            "partial-batch incremental DocumentGraph must match sequential execution"
        );
    }

    // Now mutate only norm_b's config_hash and run on [batch[0], fresh_doc] so norm_a skips on batch[0]
    // and runs on fresh_doc, while norm_b runs on both.
    let updated_pipeline = Analyzer::builder()
        .add(PrefixSentenceStep)
        .step(Bound::new(UpperViewStep { tag: 1 }).write(0, "norm_a"))
        .step(Bound::new(BracketViewStep { tag: 2 }).write(0, "norm_b"))
        .step(Bound::new(WholeViewTokenizer).read(0, "norm_a").write(0, "tok_a"))
        .step(Bound::new(WholeViewTokenizer).read(0, "norm_b").write(0, "tok_b"))
        .build()
        .unwrap();

    let mut mixed_batch = vec![batch.remove(0), Document::new("zeta eta")];
    let mixed_reports = updated_pipeline.run_incremental(&mut mixed_batch);
    let mr0 = mixed_reports[0].as_ref().expect("mixed_batch[0] succeeded");
    let mr1 = mixed_reports[1].as_ref().expect("mixed_batch[1] succeeded");

    assert_eq!(mr0.stages[0].cache_status, StageCacheStatus::Skipped);
    assert_eq!(mr0.stages[1].cache_status, StageCacheStatus::Skipped);
    assert_eq!(mr0.stages[2].cache_status, StageCacheStatus::Executed);
    assert_eq!(mr0.stages[3].cache_status, StageCacheStatus::Skipped);
    assert_eq!(mr0.stages[4].cache_status, StageCacheStatus::Executed);
    assert_eq!(mr1.stages_run, 5);

    for (doc, raw) in mixed_batch.iter().zip(["alpha beta", "zeta eta"]) {
        let mut seq_doc = Document::new(raw);
        updated_pipeline.run_sequential(std::slice::from_mut(&mut seq_doc)).pop().unwrap().unwrap();
        assert_eq!(export(doc, &c).unwrap().to_bytes().unwrap(), export(&seq_doc, &c).unwrap().to_bytes().unwrap());
    }
}
