//! Real text and the production term analyzer, with source projections.
use std::num::NonZeroU32;

use turbo_analysis::*;
use turbo_analysis_layers::{Boundaries, Cut, Sentences, Terms, Tokens};
use turbo_chunker::{
    DiscourseBoundaries, DiscourseLanguage, Drift, LexicalBoundaries, LexicalForm, Score, WeighTextBoundaries, Weight,
};
use turbo_normalize::{CaseFold, normalize};
use turbo_stem::{TermAnalyzer, TermAnnotator};

fn layers(view: &View) -> (Sentences, Tokens, Terms) {
    let mut sentences = Spans::new(view.id());
    let mut tokens = Spans::new(view.id());
    let mut sentence = Vec::new();
    let tokenizer = TermAnalyzer::builder().build();
    let mut at = 0;
    for line in view.text().split_inclusive('\n') {
        let end = at + line.trim_end_matches('\n').len();
        if end > at {
            let si = Ix::new(sentences.len() as u32);
            sentences.push(Span::new(at as u32, end as u32));
            for term in tokenizer.analyze(&view.text()[at..end]).unwrap() {
                let s = term.span().unwrap();
                tokens.push(Span::new(at as u32 + s.start, at as u32 + s.end));
                sentence.push(si);
            }
        }
        at += line.len();
    }
    let tokens = Tokens { spans: tokens, sentence: Some(sentence) };
    let (terms,) = TermAnnotator::for_language("en").unwrap().annotate((view, &tokens, None, None, None)).unwrap();
    (Sentences { spans: sentences }, tokens, terms)
}

fn lexical() -> LexicalBoundaries {
    LexicalBoundaries {
        drift: Drift { window: NonZeroU32::new(1).unwrap(), smoothing: 0, ..Drift::default() },
        form: LexicalForm::Folded,
    }
}

#[test]
fn real_term_frequencies_form_a_depth_valley() {
    let doc = Document::new("Cats purr.\nCats purr.\nRust compiles.\nRust compiles.");
    let view = doc.get::<View>().unwrap();
    let (s, t, forms) = layers(view);
    let (out,) = lexical().annotate((view, &s, &t, &forms)).unwrap();
    assert_eq!(out.cost, [0, -4000, 0, 0]);
    assert_eq!(out.reason, [Cut::Sentence, Cut::TopicShift, Cut::Sentence, Cut::Sentence]);
    let mut wide = lexical();
    wide.drift.window = NonZeroU32::new(u32::MAX).unwrap();
    assert_eq!(wide.annotate((view, &s, &t, &forms)).unwrap().0.cost, [0; 4]);
    let mut stems = lexical();
    stems.form = LexicalForm::Stem;
    assert!(stems.annotate((view, &s, &t, &forms)).is_ok());
    stems.form = LexicalForm::Lemma;
    assert!(stems.annotate((view, &s, &t, &forms)).is_err());
}

#[test]
fn rolling_windows_match_a_fresh_term_count_for_each_gap() {
    let text = "Cats purr softly.\nThe cats sleep.\nCats sleep softly.\nRust compiles code.\nRust checks code.\nCode compiles quickly.";
    let doc = Document::new(text);
    let view = doc.get::<View>().unwrap();
    let (s, t, forms) = layers(view);
    let mut step = lexical();
    step.drift.window = NonZeroU32::new(2).unwrap();
    step.drift.score = Score::Distance;
    step.drift.floor = 0;
    step.drift.scale = 1_000_000;
    let (out,) = step.annotate((view, &s, &t, &forms)).unwrap();
    let counts = |a: usize, b: usize| {
        let mut map = std::collections::BTreeMap::<&str, f64>::new();
        for i in 0..forms.folded.len() {
            let sentence = t.sentence.as_ref().unwrap()[i].get();
            let word = forms.folded.get(i);
            if (a..b).contains(&sentence) && word != "." {
                *map.entry(word).or_default() += 1.0;
            }
        }
        map
    };
    for split in 2..=4 {
        let (a, b) = (counts(split - 2, split), counts(split, split + 2));
        let dot: f64 = a.iter().map(|(key, v)| v * b.get(key).unwrap_or(&0.0)).sum();
        let norm = |m: &std::collections::BTreeMap<&str, f64>| m.values().map(|x| x * x).sum::<f64>().sqrt();
        let expected = -(1_000_000 - (dot / (norm(&a) * norm(&b)) * 1_000_000.0).round() as i32);
        assert_eq!(out.cost[split - 1], expected);
    }
    assert_eq!(out.cost[0], 0);
    assert_eq!(out.cost[4..], [0, 0]);
}

#[test]
fn markers_match_whole_prefixes_in_each_language() {
    for (lang, marker) in [
        (DiscourseLanguage::English, "IN\u{a0}  CONTRAST"),
        (DiscourseLanguage::Spanish, "EN CONCLUSIÓN"),
        (DiscourseLanguage::Dutch, "TEN SLOTTE"),
    ] {
        let doc = Document::new(format!(
            "A previous observation.\n{marker}, another topic.\n{marker}x is no marker.\nQuoted {marker} is no prefix."
        ));
        let view = doc.get::<View>().unwrap();
        let (s, _, _) = layers(view);
        let step = DiscourseBoundaries { language: lang, discount: 1000 };
        assert_eq!(step.annotate((view, &s)).unwrap().0.cost, [-1000, 0, 0, 0]);
        assert_eq!(
            DiscourseBoundaries { discount: 0, ..step }.annotate((view, &s)).unwrap().0.reason,
            [Cut::Sentence; 4]
        );
    }
    let doc = Document::new(
        "Start.\nHowever_quoted is not a marker.\nHowever\u{301} is not a marker.\nHowever2 is not a marker.",
    );
    let view = doc.get::<View>().unwrap();
    let (s, _, _) = layers(view);
    assert_eq!(
        DiscourseBoundaries { language: DiscourseLanguage::English, discount: 1000 }
            .annotate((view, &s))
            .unwrap()
            .0
            .cost,
        [0; 4]
    );
}

#[test]
fn empty_text_and_punctuation_have_no_false_shift() {
    for text in ["", "!!!\n...", "\n", "Only one sentence."] {
        let doc = Document::new(text);
        let view = doc.get::<View>().unwrap();
        let (s, t, terms) = layers(view);
        assert!(lexical().annotate((view, &s, &t, &terms)).unwrap().0.cost.iter().all(|&x| x == 0));
    }
}

#[test]
fn malformed_layers_return_errors() {
    let doc = Document::new("Café grows.\nHowever, prices rise.");
    let view = doc.get::<View>().unwrap();
    let (s, mut t, mut terms) = layers(view);
    terms.folded.push("extra");
    assert!(lexical().annotate((view, &s, &t, &terms)).is_err());
    let (_, _, terms) = layers(view);
    t.sentence.as_mut().unwrap()[0] = Ix::new(1);
    assert!(lexical().annotate((view, &s, &t, &terms)).is_err());
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 4)); // Inside the UTF-8 encoding of é.
    let bad = Sentences { spans };
    assert!(DiscourseBoundaries { language: DiscourseLanguage::English, discount: 1 }.annotate((view, &bad)).is_err());
    let (_, t, _) = layers(view);
    assert!(lexical().annotate((view, &bad, &t, &terms)).is_err());
}

#[test]
fn derived_view_boundaries_keep_bidirectional_source_anchors() {
    let mut doc = Document::new("Straße bleibt.\nHowever, CAFÉ changes.");
    let parent = doc.get::<View>().unwrap();
    let (text, alignment) = normalize(&CaseFold::full(), parent.text());
    let child = View::derived(parent, text, alignment);
    doc.add(&Key::named("folded"), child, Provenance::by("case-fold")).unwrap();
    let view = doc.get_key(&Key::<View>::named("folded")).unwrap();
    let (s, t, terms) = layers(view);
    let (out,) =
        DiscourseBoundaries { language: DiscourseLanguage::English, discount: 1000 }.annotate((view, &s)).unwrap();
    assert_eq!(out.cost, [-1000, 0]);
    assert_eq!(lexical().annotate((view, &s, &t, &terms)).unwrap().0.cost.len(), s.spans.len());
    let source = doc.project_spans(&s.spans, ViewId::SOURCE).unwrap();
    let back = doc.project_spans(&source, view.id()).unwrap();
    assert_eq!(back.columns(), s.spans.columns());
    assert_eq!(&doc.source()[source.get(Ix::new(0)).range()], "Straße bleibt.");
    assert!(lexical().annotate((doc.get::<View>().unwrap(), &s, &t, &terms)).is_err());
}

#[test]
fn weights_validate_optional_layers_and_preserve_structural_reasons() {
    let doc = Document::new("A sentence.\nHowever, another one.");
    let (s, _, _) = layers(doc.get::<View>().unwrap());
    let base = Boundaries { cost: vec![1000; 2], reason: vec![Cut::Paragraph, Cut::Sentence] };
    let signal = Boundaries { cost: vec![-1500, 0], reason: vec![Cut::TopicShift, Cut::Sentence] };
    let step = WeighTextBoundaries { lexical: Weight::ONE, discourse: Weight::ONE, ..Default::default() };
    assert!(step.annotate((&s, &base, None, Some(&signal))).is_err());
    let (out,) = step.annotate((&s, &base, Some(&signal), Some(&signal))).unwrap();
    assert_eq!(out.cost, [-2000, 1000]);
    assert_eq!(out.reason, base.reason);
    assert_eq!(WeighTextBoundaries::default().annotate((&s, &base, None, None)).unwrap().0.cost, base.cost);
    let bad = Boundaries { cost: vec![1], reason: vec![] };
    assert!(step.annotate((&s, &base, Some(&bad), Some(&signal))).is_err());
}

#[test]
fn every_setting_affects_provenance() {
    let a = lexical();
    assert_ne!(a.config_hash(), LexicalBoundaries { form: LexicalForm::Stem, ..a }.config_hash());
    let mut b = a;
    b.drift.smoothing += 1;
    assert_ne!(a.config_hash(), b.config_hash());
    let d = DiscourseBoundaries { language: DiscourseLanguage::English, discount: 1000 };
    assert_ne!(d.config_hash(), DiscourseBoundaries { discount: 1, ..d }.config_hash());
    assert_ne!(d.config_hash(), DiscourseBoundaries { language: DiscourseLanguage::Dutch, ..d }.config_hash());
}

#[test]
fn named_steps_compose_in_a_real_pipeline() {
    let mut doc = Document::new("Coffee prices are rising.\nHowever, tea remains affordable.");
    let (s, t, _) = layers(doc.get::<View>().unwrap());
    doc.add(&Key::DEFAULT, s, Provenance::by("line-sentences")).unwrap();
    doc.add(&Key::DEFAULT, t, Provenance::by("term-tokenizer")).unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .add(TermAnnotator::for_language("en").unwrap())
        .step(lexical().bound())
        .step(DiscourseBoundaries { language: DiscourseLanguage::English, discount: 1000 }.bound())
        .step(
            Bound::new(turbo_chunker::StructureBoundaries { costs: Default::default() })
                .write(0, turbo_chunker::STRUCTURE),
        )
        .step(WeighTextBoundaries { lexical: Weight::ONE, discourse: Weight::ONE, ..Default::default() }.bound())
        .build()
        .unwrap();
    let mut docs = [doc];
    assert_eq!(analyzer.run(&mut docs)[0], Ok(()));
    assert_eq!(docs[0].get::<Boundaries>().unwrap().cost, [0, 1000]);
    let reads = &docs[0].provenance(Key::<Boundaries>::DEFAULT.id()).unwrap().reads;
    for name in [turbo_chunker::LEXICAL, turbo_chunker::DISCOURSE, turbo_chunker::STRUCTURE] {
        assert!(reads.contains(&Key::<Boundaries>::named(name).id()));
    }
}
