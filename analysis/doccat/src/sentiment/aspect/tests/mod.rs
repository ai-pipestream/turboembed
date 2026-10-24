use super::*;
use turbo_analysis::{Analyzer, Document, IxRange, Key, Provenance, Span, Spans, ViewId};

fn tokenize_simple(text: &str) -> Tokens {
    let mut spans = Spans::<Tokens>::new(ViewId(0));
    let mut sentence = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let start = i;
        if bytes[i].is_ascii_alphanumeric() {
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'\'') {
                i += 1;
            }
        } else {
            i += 1;
        }
        spans.push(Span::new(start as u32, i as u32));
        sentence.push(Ix::new(0));
    }
    Tokens { spans, sentence: Some(sentence) }
}

fn run_rule_absa(text: &str) -> Document {
    let mut doc = Document::new(text);
    let mut sents = Spans::<Sentences>::new(ViewId(0));
    sents.push(Span::new(0, text.len() as u32));
    doc.add(&Key::DEFAULT, Sentences { spans: sents }, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, tokenize_simple(text), Provenance::by("test")).unwrap();

    let pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .add(RuleAspectSentiment::new())
        .build()
        .unwrap();
    let mut docs = [doc];
    pipe.run(&mut docs).remove(0).unwrap();
    docs.into_iter().next().unwrap()
}

#[test]
fn battery_life_is_great_but_screen_scratches_gives_opposite_polarity_aspects() {
    // Exact example from #322 summary:
    // "battery life is great but the screen scratches"
    let doc = run_rule_absa("battery life is great but the screen scratches");
    let tokens = doc.get::<Tokens>().unwrap();
    let opinions = doc.get::<Opinions>().unwrap();
    let aspects = doc.get::<Aspects>().unwrap();

    assert_eq!(aspects.len(), 2);
    assert_eq!(opinions.len(), 2);

    let a0 = aspects.get(Ix::new(0));
    let a1 = aspects.get(Ix::new(1));
    assert_eq!(a0.target, AspectTarget::Term { tokens: IxRange::new(0, 2), sentence: Some(Ix::new(0)) });
    assert_eq!(aspects.category_label(Ix::new(0)), Some("battery"));
    assert_eq!(a0.polarity.label(), Polarity::Positive);
    assert_eq!(a0.opinions, &[Ix::<Opinions>::new(0)]);
    assert_eq!(opinions.get(Ix::new(0)).tokens, IxRange::new(3, 4)); // "great"

    assert_eq!(a1.target, AspectTarget::Term { tokens: IxRange::new(6, 7), sentence: Some(Ix::new(0)) });
    assert_eq!(aspects.category_label(Ix::new(1)), Some("display"));
    assert_eq!(a1.polarity.label(), Polarity::Negative);
    assert_eq!(a1.opinions, &[Ix::<Opinions>::new(1)]);
    assert_eq!(opinions.get(Ix::new(1)).tokens, IxRange::new(7, 8)); // "scratches"
    let _ = tokens;
}

#[test]
fn negation_with_a_curly_apostrophe_still_negates() {
    // A tokenizer that keeps contractions whole hands the extractor
    // tokens like "isn’t" (U+2019): the negation cues, written with
    // ASCII apostrophes, must still fire.
    let text = "The screen isn\u{2019}t great.";
    let mut doc = Document::new(text);
    let mut sents = Spans::<Sentences>::new(ViewId(0));
    sents.push(Span::new(0, text.len() as u32));
    doc.add(&Key::DEFAULT, Sentences { spans: sents }, Provenance::by("test")).unwrap();
    let mut spans = Spans::<Tokens>::new(ViewId(0));
    for (a, b) in [(0, 3), (4, 10), (11, 18), (19, 24), (24, 25)] {
        spans.push(Span::new(a, b));
    }
    let tokens = Tokens { spans, sentence: Some(vec![Ix::new(0); 5]) };
    doc.add(&Key::DEFAULT, tokens, Provenance::by("test")).unwrap();

    let pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .add(RuleAspectSentiment::new())
        .build()
        .unwrap();
    let mut docs = [doc];
    pipe.run(&mut docs).remove(0).unwrap();
    let doc = &docs[0];
    let opinions = doc.get::<Opinions>().unwrap();
    let aspects = doc.get::<Aspects>().unwrap();
    assert_eq!(opinions.len(), 1);
    assert!(opinions.get(Ix::new(0)).negated, "the curly-apostrophe negation must reach the opinion");
    assert_eq!(aspects.len(), 1);
    assert_eq!(aspects.get(Ix::new(0)).polarity.label(), Polarity::Negative);
}

#[test]
fn double_propagation_over_dependencies_extracts_new_targets_and_conjoined_aspects() {
    // "The sushi and tiramisu were delicious."
    // Notice "tiramisu" is not in SINGLE_WORD_ASPECT_TERMS, so R3 (`conj` from "sushi" or R12 from "delicious")
    // extracts it via Dependencies!
    let text = "The sushi and tiramisu were delicious.";
    let mut doc = Document::new(text);
    let mut sents = Spans::<Sentences>::new(ViewId(0));
    sents.push(Span::new(0, text.len() as u32));
    doc.add(&Key::DEFAULT, Sentences { spans: sents }, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, tokenize_simple(text), Provenance::by("test")).unwrap();

    // Tokens: 0:"The", 1:"sushi", 2:"and", 3:"tiramisu", 4:"were", 5:"delicious", 6:"."
    let rels = TagSet::new("ud", &["root", "det", "nsubj", "cc", "conj", "cop", "punct"]);
    let deps = Dependencies::<Tokens> {
        head: vec![
            Some(Ix::new(1)), // The -> sushi
            Some(Ix::new(5)), // sushi -> delicious (nsubj)
            Some(Ix::new(3)), // and -> tiramisu
            Some(Ix::new(1)), // tiramisu -> sushi (conj)
            Some(Ix::new(5)), // were -> delicious (cop)
            None,             // delicious (root)
            Some(Ix::new(5)), // . -> delicious
        ],
        relation: vec![
            rels.tag("det").unwrap(),
            rels.tag("nsubj").unwrap(),
            rels.tag("cc").unwrap(),
            rels.tag("conj").unwrap(),
            rels.tag("cop").unwrap(),
            rels.tag("root").unwrap(),
            rels.tag("punct").unwrap(),
        ],
        relations: rels,
    };
    doc.add(&Key::DEFAULT, deps, Provenance::by("test")).unwrap();

    let absa_pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Dependencies>::DEFAULT)
        .add(RuleAspectSentiment::new())
        .build()
        .unwrap();
    let mut docs = [doc];
    absa_pipe.run(&mut docs).remove(0).unwrap();
    let doc = &docs[0];

    let aspects = doc.get::<Aspects>().unwrap();
    assert_eq!(aspects.len(), 2);
    assert_eq!(aspects.get(Ix::new(0)).target.tokens(), Some(IxRange::new(1, 2))); // "sushi"
    assert_eq!(aspects.get(Ix::new(1)).target.tokens(), Some(IxRange::new(3, 4))); // "tiramisu" extracted by R3/R1!
    assert_eq!(aspects.get(Ix::new(0)).polarity.label(), Polarity::Positive);
    assert_eq!(aspects.get(Ix::new(1)).polarity.label(), Polarity::Positive);
}

#[test]
fn aspect_and_opinion_under_the_same_governor_link_without_panicking() {
    // "The battery works well": `nsubj(works, battery)` and `advmod(works, well)` share
    // the governor "works". Linking used to index an empty slice on this shape.
    let text = "The battery works well";
    let mut doc = Document::new(text);
    let mut sents = Spans::<Sentences>::new(ViewId(0));
    sents.push(Span::new(0, text.len() as u32));
    doc.add(&Key::DEFAULT, Sentences { spans: sents }, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, tokenize_simple(text), Provenance::by("test")).unwrap();
    let rels = TagSet::new("ud", &["root", "det", "nsubj", "advmod"]);
    let deps = Dependencies::<Tokens> {
        head: vec![Some(Ix::new(1)), Some(Ix::new(2)), None, Some(Ix::new(2))],
        relation: vec![
            rels.tag("det").unwrap(),
            rels.tag("nsubj").unwrap(),
            rels.tag("root").unwrap(),
            rels.tag("advmod").unwrap(),
        ],
        relations: rels,
    };
    doc.add(&Key::DEFAULT, deps, Provenance::by("test")).unwrap();
    let pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Dependencies>::DEFAULT)
        .add(RuleAspectSentiment::new())
        .build()
        .unwrap();
    let mut docs = [doc];
    pipe.run(&mut docs).remove(0).unwrap();
    let aspects = docs[0].get::<Aspects>().unwrap();
    let opinions = docs[0].get::<Opinions>().unwrap();
    assert_eq!(opinions.get(Ix::new(0)).tokens, IxRange::new(3, 4)); // "well"
    let battery = (0..aspects.len())
        .map(|i| Ix::<Aspects>::new(i as u32))
        .find(|&ix| aspects.get(ix).target.tokens() == Some(IxRange::new(1, 2)))
        .expect("battery is an aspect");
    assert_eq!(aspects.get(battery).polarity.label(), Polarity::Positive);
    assert_eq!(aspects.get(battery).opinions, &[Ix::<Opinions>::new(0)]);
}

#[test]
fn implicit_aspect_anchors_to_sentence_and_scores_category() {
    let doc = run_rule_absa("Way too expensive and completely overpriced.");
    let aspects = doc.get::<Aspects>().unwrap();
    let opinions = doc.get::<Opinions>().unwrap();
    assert_eq!(aspects.len(), 1);
    assert_eq!(opinions.len(), 2);
    let a0 = aspects.get(Ix::new(0));
    assert_eq!(a0.target, AspectTarget::Implicit { sentence: Ix::new(0) });
    assert_eq!(aspects.category_label(Ix::new(0)), Some("price"));
    assert_eq!(a0.polarity.label(), Polarity::Negative);
    assert_eq!(a0.opinions.len(), 2);
}

#[test]
fn phrase_chunks_extract_non_lexicon_noun_phrases_and_trim_leading_modifiers() {
    // Neither "calamari" nor "rubber gasket" is in SINGLE_WORD_ASPECT_TERMS or MULTIWORD_ASPECT_TERMS.
    let text = "The crisp calamari was delicious, but the rubber gasket was broken.";
    let without_chunks = run_rule_absa(text);
    let without_aspects = without_chunks.get::<Aspects>().unwrap();
    assert!(
        (0..without_aspects.len()).all(|i| without_aspects.get(Ix::new(i as u32)).target.tokens().is_none()),
        "without PhraseChunks, non-lexicon noun phrases are not extracted as explicit aspect terms"
    );

    let mut doc = Document::new(text);
    let mut sents = Spans::<Sentences>::new(ViewId(0));
    sents.push(Span::new(0, text.len() as u32));
    doc.add(&Key::DEFAULT, Sentences { spans: sents }, Provenance::by("test")).unwrap();
    let tokens = tokenize_simple(text);
    // Tokens:
    // 0:"The", 1:"crisp", 2:"calamari", 3:"was", 4:"delicious", 5:",",
    // 6:"but", 7:"the", 8:"rubber", 9:"gasket", 10:"was", 11:"broken", 12:"."
    let kinds = TagSet::new("chunk", &["NP", "VP"]);
    let np = kinds.tag("NP").unwrap();
    let vp = kinds.tag("VP").unwrap();
    let mut chunk_spans = Spans::<PhraseChunks>::new(ViewId(0));
    chunk_spans.push(Span::new(tokens.spans.get(Ix::new(0)).start, tokens.spans.get(Ix::new(2)).end));
    chunk_spans.push(Span::new(tokens.spans.get(Ix::new(3)).start, tokens.spans.get(Ix::new(3)).end));
    chunk_spans.push(Span::new(tokens.spans.get(Ix::new(7)).start, tokens.spans.get(Ix::new(9)).end));
    let chunks = PhraseChunks {
        spans: chunk_spans,
        tokens: vec![IxRange::new(0, 3), IxRange::new(3, 4), IxRange::new(7, 10)],
        kind: vec![np, vp, np],
        kinds,
    };
    doc.add(&Key::DEFAULT, tokens, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, chunks, Provenance::by("test")).unwrap();

    let pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<PhraseChunks>::DEFAULT)
        .add(RuleAspectSentiment::new())
        .build()
        .unwrap();
    let mut docs = [doc];
    pipe.run(&mut docs).remove(0).unwrap();
    let aspects = docs[0].get::<Aspects>().unwrap();
    assert_eq!(aspects.len(), 2);

    // "The crisp calamari" trims leading determiner "The" and leading opinion word "crisp" -> "calamari" (2..3)
    let a0 = aspects.get(Ix::new(0));
    assert_eq!(a0.target, AspectTarget::Term { tokens: IxRange::new(2, 3), sentence: Some(Ix::new(0)) });
    assert_eq!(a0.polarity.label(), Polarity::Positive);

    // "the rubber gasket" trims leading determiner "the" -> "rubber gasket" (8..10)
    let a1 = aspects.get(Ix::new(1));
    assert_eq!(a1.target, AspectTarget::Term { tokens: IxRange::new(8, 10), sentence: Some(Ix::new(0)) });
    assert_eq!(a1.polarity.label(), Polarity::Negative);
}

#[test]
fn pos_tags_extract_noun_sequences_and_expand_lexicon_compounds() {
    // "screen" is in SINGLE_WORD_ASPECT_TERMS, while "OLED", "risotto", and "terrace heater" are not.
    let text = "The OLED screen is bright, and the risotto was delicious, but the terrace heater was broken.";
    let without_pos = run_rule_absa(text);
    let without_aspects = without_pos.get::<Aspects>().unwrap();
    assert_eq!(without_aspects.len(), 1);
    assert_eq!(
        without_aspects.get(Ix::new(0)).target.tokens(),
        Some(IxRange::new(2, 3)),
        "without PosTags, only the bare lexicon noun 'screen' (2..3) is extracted"
    );

    let mut doc = Document::new(text);
    let mut sents = Spans::<Sentences>::new(ViewId(0));
    sents.push(Span::new(0, text.len() as u32));
    doc.add(&Key::DEFAULT, Sentences { spans: sents }, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, tokenize_simple(text), Provenance::by("test")).unwrap();

    // Tokens:
    // 0:"The"(DT), 1:"OLED"(NNP), 2:"screen"(NN), 3:"is"(VBZ), 4:"bright"(JJ), 5:","(,),
    // 6:"and"(CC), 7:"the"(DT), 8:"risotto"(NN), 9:"was"(VBD), 10:"delicious"(JJ), 11:","(,),
    // 12:"but"(CC), 13:"the"(DT), 14:"terrace"(NN), 15:"heater"(NN), 16:"was"(VBD), 17:"broken"(JJ), 18:"."(.)
    let tagset = TagSet::new("ptb", &["DT", "NNP", "NN", "VBZ", "VBD", "JJ", "CC", ",", "."]);
    let pos_labels = [
        "DT", "NNP", "NN", "VBZ", "JJ", ",", "CC", "DT", "NN", "VBD", "JJ", ",", "CC", "DT", "NN", "NN", "VBD", "JJ",
        ".",
    ];
    let pos = PosTags { tags: pos_labels.iter().map(|&l| tagset.tag(l).unwrap()).collect(), tagset, prob: None };
    doc.add(&Key::DEFAULT, pos, Provenance::by("test")).unwrap();

    let pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<PosTags>::DEFAULT)
        .add(RuleAspectSentiment::new())
        .build()
        .unwrap();
    let mut docs = [doc];
    pipe.run(&mut docs).remove(0).unwrap();
    let aspects = docs[0].get::<Aspects>().unwrap();
    assert_eq!(aspects.len(), 3);

    // 1) "screen" expands leftward over NNP "OLED" -> "OLED screen" (1..3)
    assert_eq!(aspects.get(Ix::new(0)).target.tokens(), Some(IxRange::new(1, 3)));
    assert_eq!(aspects.category_label(Ix::new(0)), Some("display"));
    assert_eq!(aspects.get(Ix::new(0)).polarity.label(), Polarity::Positive);

    // 2) non-lexicon NN "risotto" (8..9) extracted via PosTags
    assert_eq!(aspects.get(Ix::new(1)).target.tokens(), Some(IxRange::new(8, 9)));
    assert_eq!(aspects.get(Ix::new(1)).polarity.label(), Polarity::Positive);

    // 3) non-lexicon NN NN compound "terrace heater" (14..16) extracted via PosTags
    assert_eq!(aspects.get(Ix::new(2)).target.tokens(), Some(IxRange::new(14, 16)));
    assert_eq!(aspects.get(Ix::new(2)).polarity.label(), Polarity::Negative);
}

#[derive(Debug)]
struct HospitalityCategoryScorer {
    categories: Arc<TagSet>,
}

impl HospitalityCategoryScorer {
    fn new() -> Self {
        Self { categories: TagSet::new("hospitality", &["cuisine", "cellar", "general"]) }
    }
}

impl AspectCategoryScorer for HospitalityCategoryScorer {
    fn categories(&self) -> &Arc<TagSet> {
        &self.categories
    }

    fn score_category(&self, aspect_text: &str, opinion_text: &str, sentence_text: &str) -> Option<Tag> {
        let a = aspect_text.to_ascii_lowercase();
        let o = opinion_text.to_ascii_lowercase();
        if a.contains("wine") || a.contains("cocktails") {
            return self.categories.tag("cellar");
        }
        if a.contains("sushi") || (a.is_empty() && o.contains("delicious") && sentence_text.contains("omakase")) {
            return self.categories.tag("cuisine");
        }
        None
    }

    fn config_hash(&self) -> u64 {
        0x5150_cafe_d00d_beef
    }
}

#[test]
fn aspect_category_scorer_hook_scores_explicit_and_implicit_aspects_with_fallback() {
    let scorer: Arc<dyn AspectCategoryScorer> = Arc::new(HospitalityCategoryScorer::new());
    let rule = RuleAspectSentiment::new().with_category_scorer(Arc::clone(&scorer));
    assert_ne!(rule.config_hash(), RuleAspectSentiment::new().config_hash());
    assert_eq!(rule.categories().name, "hospitality");

    let make_doc = |text: &str| {
        let mut doc = Document::new(text);
        let mut sents = Spans::<Sentences>::new(ViewId(0));
        sents.push(Span::new(0, text.len() as u32));
        doc.add(&Key::DEFAULT, Sentences { spans: sents }, Provenance::by("test")).unwrap();
        doc.add(&Key::DEFAULT, tokenize_simple(text), Provenance::by("test")).unwrap();
        doc
    };

    let pipe =
        Analyzer::builder().given(Key::<Sentences>::DEFAULT).given(Key::<Tokens>::DEFAULT).add(rule).build().unwrap();

    let mut docs = [
        make_doc("The sushi was delicious, and the wine was fresh."),
        make_doc("The waiter was courteous."),
        make_doc("Absolutely delicious omakase!"),
    ];
    for res in pipe.run(&mut docs) {
        res.unwrap();
    }

    // Doc 0: explicit aspects "sushi" -> "cuisine", "wine" -> "cellar" via scorer
    let aspects0 = docs[0].get::<Aspects>().unwrap();
    assert_eq!(aspects0.categories.name, "hospitality");
    assert_eq!(aspects0.len(), 2);
    assert_eq!(aspects0.category_label(Ix::new(0)), Some("cuisine"));
    assert_eq!(aspects0.category_label(Ix::new(1)), Some("cellar"));

    // Doc 1: scorer returns None for "waiter" -> falls back to "general" in custom TagSet
    let aspects1 = docs[1].get::<Aspects>().unwrap();
    assert_eq!(aspects1.len(), 1);
    assert_eq!(aspects1.category_label(Ix::new(0)), Some("general"));

    // Doc 2: implicit aspect (aspect_text == "") -> scored as "cuisine" via opinion + sentence text
    let aspects2 = docs[2].get::<Aspects>().unwrap();
    assert_eq!(aspects2.len(), 1);
    assert_eq!(aspects2.get(Ix::new(0)).target, AspectTarget::Implicit { sentence: Ix::new(0) });
    assert_eq!(aspects2.category_label(Ix::new(0)), Some("cuisine"));
}

mod benchmark;
