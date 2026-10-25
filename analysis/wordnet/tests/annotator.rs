//! The WordNet annotator on a sentence: senses and expansions per token,
//! picked by POS tag and lemma when the document has them.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{Analyzer, Bound, Document, Key, Provenance, Span, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::export::codecs;
use turbo_analysis_layers::{Lemmas, LexicalExpansions, PosTags, SAME_SYNSET, Strs, Tokens, WordNetSenses};
use turbo_wordnet::{
    InMemoryWordNetLexicon, LemmaKey, MorphyExceptions, Synset, WordNetAnnotator, WordNetOptions, WordNetPos,
    WordNetRelation,
};

const TEXT: &str = "Big dogs bark at cats.";
const TOKENS: [(u32, u32); 6] = [(0, 3), (4, 8), (9, 13), (14, 16), (17, 21), (21, 22)];

fn synset(id: &str, pos: WordNetPos, lemmas: &[&str], gloss: &str, hypernym: Option<&str>) -> (String, Synset) {
    let mut relations = BTreeMap::new();
    if let Some(h) = hypernym {
        relations.insert(WordNetRelation::Hypernym, vec![h.to_owned()]);
    }
    let lemmas = lemmas.iter().map(|l| l.to_string()).collect();
    (id.to_owned(), Synset::new(id, pos, lemmas, gloss, relations).unwrap())
}

fn lexicon() -> Arc<InMemoryWordNetLexicon> {
    use WordNetPos::{Noun, Verb};
    let synsets: HashMap<String, Synset> = [
        synset("s-dog", Noun, &["dog", "domestic_dog"], "a domesticated canine", Some("s-canine")),
        synset("s-dog-v", Verb, &["dog", "tail"], "follow closely", None),
        synset("s-canine", Noun, &["canine", "canid"], "a carnivore of the dog family", Some("s-carnivore")),
        synset("s-carnivore", Noun, &["carnivore"], "a flesh-eating mammal", None),
        synset("s-bark-n", Noun, &["bark"], "the outer layer of a tree", None),
        synset("s-bark-v", Verb, &["bark"], "make the sound of a dog", None),
        synset("s-cat", Noun, &["cat", "true_cat"], "a feline mammal", None),
    ]
    .into_iter()
    .collect();
    let order =
        |lemma: &str, pos, ids: &[&str]| (LemmaKey::of(lemma, pos), ids.iter().map(|s| s.to_string()).collect());
    let sense_order: HashMap<LemmaKey, Vec<String>> = [
        order("dog", Noun, &["s-dog"]),
        order("domestic dog", Noun, &["s-dog"]),
        order("dog", Verb, &["s-dog-v"]),
        order("tail", Verb, &["s-dog-v"]),
        order("canine", Noun, &["s-canine"]),
        order("canid", Noun, &["s-canine"]),
        order("carnivore", Noun, &["s-carnivore"]),
        order("bark", Noun, &["s-bark-n"]),
        order("bark", Verb, &["s-bark-v"]),
        order("cat", Noun, &["s-cat"]),
        order("true cat", Noun, &["s-cat"]),
    ]
    .into_iter()
    .collect();
    Arc::new(InMemoryWordNetLexicon::new(synsets, sense_order).unwrap())
}

fn document(tagged: bool) -> Document {
    let mut doc = Document::new(TEXT);
    let mut spans = Spans::new(ViewId::SOURCE);
    for (a, b) in TOKENS {
        spans.push(Span::new(a, b));
    }
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
    if tagged {
        let labels = ["JJ", "NNS", "VBP", "IN", "NNS", "."];
        let tagset = TagSet::new("penn", &labels);
        let tags = (0..labels.len()).map(|i| Tag(i as u16)).collect();
        doc.add(&Key::DEFAULT, PosTags { tagset, tags, prob: None }, Provenance::by("test")).unwrap();
        let mut lemma = Strs::default();
        for l in ["big", "dog", "bark", "at", "cat", "."] {
            lemma.push(l);
        }
        doc.add(&Key::DEFAULT, Lemmas { lemma }, Provenance::by("test")).unwrap();
    }
    doc
}

fn run(annotator: WordNetAnnotator<InMemoryWordNetLexicon>, tagged: bool) -> Document {
    let mut builder = Analyzer::builder().given(Key::<Tokens>::DEFAULT);
    if tagged {
        builder = builder.given(Key::<PosTags>::DEFAULT).given(Key::<Lemmas>::DEFAULT);
    }
    let analyzer = builder.step(Bound::new(annotator)).build().unwrap();
    let mut docs = vec![document(tagged)];
    let results = analyzer.run(&mut docs);
    assert!(results.iter().all(Result::is_ok), "{results:?}");
    docs.pop().unwrap()
}

/// (token, synset, part of speech) for every sense row.
fn sense_rows(s: &WordNetSenses) -> Vec<(usize, &str, char)> {
    (0..s.token.len()).map(|i| (s.token[i].get(), s.synset_id.get(i), s.pos[i] as char)).collect()
}

#[test]
fn tags_and_lemmas_pick_one_part_of_speech_per_token() {
    let annotator = WordNetAnnotator::new(lexicon(), "fixture:1", WordNetOptions::default()).unwrap();
    let doc = run(annotator, true);
    let senses = doc.get::<WordNetSenses>().unwrap();
    // "dogs" is a plural noun and "bark" a verb; "Big" has no synset and
    // "at" and "." have no WordNet part of speech.
    assert_eq!(sense_rows(senses), [(1, "s-dog", 'n'), (2, "s-bark-v", 'v'), (4, "s-cat", 'n')]);
    assert_eq!(senses.spans.iter().map(|(_, s)| (s.start, s.end)).collect::<Vec<_>>(), [(4, 8), (9, 13), (17, 21)]);
    assert_eq!(senses.lemma.iter().collect::<Vec<_>>(), ["dog", "bark", "cat"]);
    assert_eq!(senses.members(0).collect::<Vec<_>>(), ["dog", "domestic_dog"]);
    assert_eq!(senses.gloss.get(1), "make the sound of a dog");
    assert_eq!(&*senses.lexicon_id, "fixture:1");

    let ex = doc.get::<LexicalExpansions>().unwrap();
    let rows: Vec<(usize, &str, u8, u8, &str)> = (0..ex.token.len())
        .map(|i| (ex.token[i].get(), ex.term.get(i), ex.relation[i], ex.depth[i], ex.synset_id.get(i)))
        .collect();
    let hypernym = WordNetRelation::Hypernym.layer_code();
    assert_eq!(
        rows,
        [
            (1, "domestic_dog", SAME_SYNSET, 0, "s-dog"),
            (1, "canid", hypernym, 1, "s-canine"),
            (1, "canine", hypernym, 1, "s-canine"),
            (4, "true_cat", SAME_SYNSET, 0, "s-cat"),
        ]
    );
    assert_eq!(ex.weight, [1.0, 0.5, 0.5, 1.0]);
    assert_eq!(WordNetRelation::from_layer_code(hypernym), Some(WordNetRelation::Hypernym));
    assert_eq!(WordNetRelation::from_layer_code(SAME_SYNSET), None);

    let bytes = export(&doc, &codecs()).unwrap().to_bytes().unwrap();
    let back = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs()).unwrap();
    assert_eq!(sense_rows(back.get::<WordNetSenses>().unwrap()), sense_rows(senses));
    assert_eq!(
        back.get::<LexicalExpansions>().unwrap().term.iter().collect::<Vec<_>>(),
        ex.term.iter().collect::<Vec<_>>()
    );
}

#[test]
fn without_tags_morphy_lemmatizes_and_every_part_of_speech_is_tried() {
    let options = WordNetOptions { hypernym_depth: 2, ..WordNetOptions::default() };
    let annotator =
        WordNetAnnotator::new(lexicon(), "fixture:1", options).unwrap().with_morphy(MorphyExceptions::empty());
    let doc = run(annotator, false);
    let senses = doc.get::<WordNetSenses>().unwrap();
    assert_eq!(
        sense_rows(senses),
        [(1, "s-dog", 'n'), (1, "s-dog-v", 'v'), (2, "s-bark-n", 'n'), (2, "s-bark-v", 'v'), (4, "s-cat", 'n')]
    );
    let ex = doc.get::<LexicalExpansions>().unwrap();
    let carnivore = ex.term.iter().position(|t| t == "carnivore").expect("two hypernym steps from dog");
    assert_eq!((ex.depth[carnivore], ex.weight[carnivore], ex.synset_id.get(carnivore)), (2, 0.25, "s-carnivore"));
    let tail = ex.term.iter().position(|t| t == "tail").expect("the verb sense of dog");
    assert_eq!((ex.relation[tail], ex.token[tail].get()), (SAME_SYNSET, 1));
}

#[test]
fn options_past_what_the_layer_holds_are_refused() {
    let deep = WordNetOptions { hypernym_depth: 256, ..WordNetOptions::default() };
    assert!(WordNetAnnotator::new(lexicon(), "fixture:1", deep).is_err());
    let none = WordNetOptions { max_senses: 0, ..WordNetOptions::default() };
    assert!(WordNetAnnotator::new(lexicon(), "fixture:1", none).is_err());
}

#[test]
fn hash_compat_options_identity_and_morphy() {
    use turbo_analysis::Annotator;
    let mut hashes = Vec::new();
    for id in ["fixture:1", "fixture:2", "é:词"] {
        for options in [
            WordNetOptions::default(),
            WordNetOptions { max_senses: 1, hypernym_depth: 0, include_hyponyms: true, max_expansions: 2 },
        ] {
            let plain = WordNetAnnotator::new(lexicon(), id, options).unwrap();
            let morphy = WordNetAnnotator::new(lexicon(), id, options).unwrap().with_morphy(MorphyExceptions::empty());
            hashes.extend([plain.config_hash(), morphy.config_hash()]);
        }
    }
    // Plain values retain the pre-migration byte stream. Morphy values
    // include the resource tag introduced by the identity repair in #1303,
    // pinned before replacing its local FNV loop with the canonical hasher.
    assert_eq!(
        hashes,
        [
            10151469796173529181,
            3680983188007610824,
            1611157893381296233,
            758218860574209860,
            9154899751243379368,
            1934318102296041461,
            10306967913311042588,
            16914308485395794521,
            17548468750267158981,
            11483478524313537840,
            6655273575599580657,
            5617332801517433580
        ]
    );
}
