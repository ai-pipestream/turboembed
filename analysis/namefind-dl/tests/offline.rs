//! The finder end to end on the small BERT in testdata, given a token
//! classification head whose weights are zero and whose bias favours
//! `B-PER`: every piece is labelled `B-PER` with the same probability, so
//! every piece is its own entity. What the head decides is then known
//! without a reference, and the tests check what the finder does around
//! it: chunks and their overlap, rows split at the row length, special
//! tokens, batches, sentences and the token ranges.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;

use turbo::bundle::Bundle;
use turbo::manifest::{PromptRole, Truncation};
use turbo::tokenizer::{Encode, Tokenizer};
use turbo_analysis::{Analyzer, Annotator, Bound, Document, Key, Provenance, Span, Spans, View, ViewId};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{Entities, Sentences, Subwords, Tokens};
use turbo_namefind_dl::decode::{Decoding, label_probability, name_words};
use turbo_namefind_dl::{NameFinderDl, NameFinderDlOptions};

const BIAS: [f32; 3] = [0.0, 2.0, 0.0];

/// The tiny bundle with a `[O, B-PER, I-PER]` token head appended.
fn bundle() -> PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| write_bundle("per", &["O", "B-PER", "I-PER"])).clone()
}

include!("offline/support.rs");

/// Each piece's bytes in `text`, the tokenizer read once over the text.
fn pieces(text: &str) -> Vec<(usize, usize)> {
    let t = Tokenizer::load(&Bundle::open(&bundle()).unwrap()).unwrap();
    let bare = Encode {
        add_special_tokens: false,
        truncation: Truncation::None,
        max_tokens: u32::MAX,
        prompt: PromptRole::None,
    };
    let (mut ids, mut spans) = (Vec::new(), Vec::new());
    t.encode_spans(text, bare, &mut ids, &mut spans).unwrap();
    spans.iter().filter(|s| s[0] < s[1]).map(|s| (s[0] as usize, s[1] as usize)).collect()
}

/// A finder decoding every piece, as `NameFinderDL` does.
fn finder(split: usize, overlap: usize, max_seq: u32, max_batch: u32) -> NameFinderDl {
    finder_by(Decoding::Piece, split, overlap, max_seq, max_batch)
}

fn finder_by(decoding: Decoding, split: usize, overlap: usize, max_seq: u32, max_batch: u32) -> NameFinderDl {
    let encoder = EncoderOptions { max_seq, max_batch, ..EncoderOptions::default() };
    NameFinderDl::open(&bundle(), &NameFinderDlOptions { encoder, split, overlap, decoding }).unwrap()
}

const TEXT: &str = "Hello world, this is an unbelievable test of the finder's chunks.";

#[test]
fn every_piece_once_however_the_text_is_cut() {
    let want = pieces(TEXT);
    let p = label_probability(&BIAS, 1);
    // Whole chunks in one row; chunks of 3 words sharing 1; rows of 2
    // pieces; and one row per run.
    for (split, overlap, max_seq, max_batch) in [(250, 50, 0, 0), (3, 1, 0, 0), (3, 2, 4, 0), (4, 1, 5, 1)] {
        let f = finder(split, overlap, max_seq, max_batch);
        let got = f.find(&[(TEXT, vec![(0, TEXT.len())]), ("", vec![(0, 0)]), (TEXT, vec![(6, 11)])]).unwrap();
        let spans: Vec<(usize, usize)> = got[0].iter().map(|c| (c.start, c.end)).collect();
        assert_eq!(spans, want, "split {split}, overlap {overlap}, max_seq {max_seq}");
        assert!(got[0].iter().all(|c| c.kind == 0 && (c.probability - p).abs() < 1e-12));
        assert!(got[1].is_empty());
        assert_eq!(got[2].iter().map(|c| (c.start, c.end)).collect::<Vec<_>>(), [(6, 11)]);
    }
    assert_eq!(finder(250, 50, 0, 0).kinds().label(turbo_analysis_layers::Tag(0)), "PER");
}

#[test]
fn options_that_cannot_cut_are_errors() {
    let open = |split, overlap, max_seq| {
        let encoder = EncoderOptions { max_seq, ..EncoderOptions::default() };
        NameFinderDl::open(&bundle(), &NameFinderDlOptions { encoder, split, overlap, ..Default::default() })
    };
    assert!(open(5, 5, 0).is_err());
    assert!(open(0, 0, 0).is_err());
    // A row of only `[CLS] [SEP]` has no room for a piece.
    assert!(open(5, 1, 2).is_err());
    // A head with no `B-<type>` label finds no entity type.
    let pos = write_bundle("pos", &["NOUN", "VERB", "I-"]);
    assert!(NameFinderDl::open(&pos, &NameFinderDlOptions::default()).is_err());
    let f = finder(250, 50, 0, 0);
    assert!(f.find(&[(TEXT, vec![(0, TEXT.len() + 1)])]).is_err());
    assert!(f.find(&[("é", vec![(0, 1)])]).is_err());
}

#[test]
fn as_a_step_it_reads_sentences_and_maps_tokens() {
    let text = "Ann met Bob . Carl stayed";
    let mut doc = Document::new(text);
    let mut tokens = Spans::new(ViewId::SOURCE);
    let mut start = 0;
    for word in text.split(' ') {
        tokens.push(Span::new(start as u32, (start + word.len()) as u32));
        start += word.len() + 1;
    }
    let mut sentences = Spans::new(ViewId::SOURCE);
    sentences.push(Span::new(0, 12));
    doc.add(&Key::DEFAULT, Tokens { spans: tokens, sentence: None }, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, Sentences { spans: sentences }, Provenance::by("test")).unwrap();
    let analyzer =
        Analyzer::builder().given(Key::<Tokens>::DEFAULT).step(Bound::new(finder(250, 50, 0, 0))).build().unwrap();
    let mut docs = vec![doc];
    assert!(analyzer.run(&mut docs).iter().all(Result::is_ok));
    let e = docs[0].get::<Entities>().unwrap();
    // Only the first sentence is read; each piece is an entity, and each
    // maps to the token holding it.
    let want = pieces(&text[..12]);
    let got: Vec<(usize, usize, usize, usize)> = e
        .spans
        .iter()
        .map(|(i, s)| (s.start as usize, s.end as usize, e.tokens[i.get()].start.get(), e.tokens[i.get()].end.get()))
        .collect();
    let word = |b: usize| text[..b].matches(" ").count();
    let expect: Vec<(usize, usize, usize, usize)> = want.iter().map(|&(a, b)| (a, b, word(a), word(a) + 1)).collect();
    assert_eq!(got, expect);
}

/// Each entity's bytes and token range on "Ann met Bob" with the token
/// rows `tokens`, each a list of fragments, or the step's failure.
fn ranges(tokens: &[&[(u32, u32)]]) -> Result<Vec<(u32, u32, usize, usize)>, String> {
    let mut doc = Document::new("Ann met Bob");
    let mut spans = Spans::new(ViewId::SOURCE);
    for t in tokens {
        let f: Vec<Span> = t.iter().map(|&(a, b)| Span::new(a, b)).collect();
        spans.push_fragments(&f);
    }
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
    let analyzer =
        Analyzer::builder().given(Key::<Tokens>::DEFAULT).step(Bound::new(finder(250, 50, 0, 0))).build().unwrap();
    let mut docs = vec![doc];
    if let Some(Err(e)) = analyzer.run(&mut docs).into_iter().next() {
        return Err(format!("{e:?}"));
    }
    let e = docs[0].get::<Entities>().unwrap();
    Ok(e.spans
        .iter()
        .map(|(i, s)| (s.start, s.end, e.tokens[i.get()].start.get(), e.tokens[i.get()].end.get()))
        .collect())
}

#[test]
fn tokens_out_of_order_or_in_pieces_map_by_overlap() {
    // Every word here is one piece of the tiny model's vocabulary.
    assert_eq!(pieces("Ann met Bob"), [(0, 3), (4, 7), (8, 11)]);
    assert_eq!(ranges(&[&[(8, 11)], &[(0, 3)], &[(4, 7)]]).unwrap(), [(0, 3, 1, 2), (4, 7, 2, 3), (8, 11, 0, 1)]);
    // A token of two fragments holds Ann and Bob, not met between them.
    assert_eq!(ranges(&[&[(0, 3), (8, 11)], &[(4, 7)]]).unwrap(), [(0, 3, 0, 1), (4, 7, 1, 2), (8, 11, 0, 1)]);
    // Ann overlaps rows 0 and 2 but not 1: no one range names it.
    assert!(ranges(&[&[(0, 2)], &[(4, 7)], &[(2, 3)], &[(8, 11)]]).is_err());
    // Bob overlaps no token, and the tokens are out of order.
    assert!(ranges(&[&[(4, 7)], &[(0, 3)]]).is_err());
    // In order, a name between tokens is the empty range where it sits.
    assert_eq!(ranges(&[&[(0, 3)], &[(8, 11)]]).unwrap(), [(0, 3, 0, 1), (4, 7, 1, 1), (8, 11, 1, 2)]);
}

#[test]
fn the_configuration_hash_follows_the_row_length() {
    use turbo_analysis::Annotator;
    let hash = |split, overlap, max_seq| finder(split, overlap, max_seq, 0).config_hash();
    assert_ne!(hash(250, 50, 4), hash(250, 50, 16));
    assert_ne!(hash(250, 50, 0), hash(250, 49, 0));
    assert_ne!(hash(250, 50, 0), hash(249, 50, 0));
    assert_eq!(hash(250, 50, 16), hash(250, 50, 16));
    let word = finder_by(Decoding::Word, 250, 50, 0, 0).config_hash();
    assert_ne!(word, hash(250, 50, 0));
}

#[test]
fn by_default_each_word_is_labelled_by_its_first_piece() {
    // Every piece is `B-PER`: by words, each word is one entity, whole,
    // where by pieces each piece is. Its words part punctuation from a
    // name: `world` and `,`, `finder` and `'s`, `chunks` and `.`.
    let want = name_words(TEXT);
    assert!(want.contains(&(6, 11)) && want.contains(&(11, 12)));
    assert!(want.contains(&(49, 55)) && want.contains(&(55, 57)));
    assert!(pieces(TEXT).len() > want.len());
    let p = label_probability(&BIAS, 1);
    let f = NameFinderDl::open(&bundle(), &NameFinderDlOptions::default()).unwrap();
    let got = f.find(&[(TEXT, vec![(0, TEXT.len())])]).unwrap();
    assert_eq!(got[0].iter().map(|c| (c.start, c.end)).collect::<Vec<_>>(), want);
    assert!(got[0].iter().all(|c| c.kind == 0 && (c.probability - p).abs() < 1e-12));
}

#[test]
fn a_word_across_a_row_boundary_is_one_entity() {
    // The word of most pieces is more than a row of 2 pieces holds, so
    // its pieces fall in two or more rows; it is still one entity.
    let text = "Hello world, a zqxjkvbw test of the finder's chunks.";
    let all = pieces(text);
    let count = |&(a, b): &(usize, usize)| all.iter().filter(|&&(s, e)| a <= s && e <= b).count();
    let (a, b) = name_words(text).into_iter().max_by_key(count).unwrap();
    assert!(count(&(a, b)) > 2, "{:?} is {} pieces", &text[a..b], count(&(a, b)));
    let want = name_words(text);
    for (split, overlap, max_seq, max_batch) in [(250, 50, 4, 0), (250, 50, 5, 1), (3, 1, 4, 2), (4, 2, 5, 0)] {
        let f = finder_by(Decoding::Word, split, overlap, max_seq, max_batch);
        let got = f.find(&[(text, vec![(0, text.len())]), (text, vec![(a, b)])]).unwrap();
        let spans = |i: usize| got[i].iter().map(|c| (c.start, c.end)).collect::<Vec<_>>();
        assert_eq!(spans(0), want, "split {split}, overlap {overlap}, max_seq {max_seq}");
        assert_eq!(spans(1), [(a, b)]);
    }
}

#[test]
fn a_name_keeps_apart_from_the_punctuation_and_ideographs_around_it() {
    // The words are CoNLL's, not the whitespace's: a name behind a
    // bracket or a quote is still the name's first piece, a name before a
    // full stop or comma ends before it, an abbreviation keeps its full
    // stops, and ideographs unspaced are a word each.
    let f = NameFinderDl::open(&bundle(), &NameFinderDlOptions::default()).unwrap();
    for (t, name) in [
        ("(Paris)", (1, 6)),
        ("\"Paris\" said", (1, 6)),
        ("Macron.", (0, 6)),
        ("Merkel, too", (0, 6)),
        ("Merkel's", (0, 6)),
        ("\u{6771}\u{4eac}", (3, 6)),
        ("U.N. troops", (0, 4)),
        ("1. Jesper Ronnback", (0, 2)),
        ("C. Spearman", (0, 2)),
        ("Mr. Smith.", (0, 3)),
    ] {
        let got = f.find(&[(t, vec![(0, t.len())])]).unwrap();
        let spans: Vec<_> = got[0].iter().map(|c| (c.start, c.end)).collect();
        assert!(spans.contains(&name), "{t:?}: {spans:?}");
        assert_eq!(spans, name_words(t), "{t:?}");
    }
}

#[test]
fn words_are_read_as_opennlp_rejoins_them() {
    // Form feed, line tabulation and next line are Unicode whitespace, so
    // they part words, but WordPiece drops them as control characters: the
    // chunk must reach the tokenizer with its words rejoined by spaces.
    let f = finder(250, 50, 0, 0);
    let spans = |t: &str| -> Vec<(usize, usize)> {
        f.find(&[(t, Vec::from([(0, t.len())]))]).unwrap()[0].iter().map(|c| (c.start, c.end)).collect()
    };
    for t in ["John\u{c}Smith", "John\u{b}Smith", "a  b\u{c}\u{c}c"] {
        assert_eq!(spans(t), pieces(&t.replace(['\u{b}', '\u{c}'], " ")), "{t:?}");
    }
    let t = "Ann\u{85}Lee";
    assert_eq!(spans(t), [(0, 3), (5, 8)]);
}

// Batches across documents: every document's chunks are read in shared
// encoder runs, each document's entities are what it gets alone, and a
// document whose inputs fail their checks fails alone.

/// Word tokens of `text`, split on spaces.
fn tokens(text: &str) -> Tokens {
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut start = 0;
    for word in text.split(' ') {
        spans.push(Span::new(start as u32, (start + word.len()) as u32));
        start += word.len() + 1;
    }
    Tokens { spans, sentence: None }
}

fn rows(e: &Entities) -> Vec<(u32, u32, usize, usize, u16, u64)> {
    e.spans
        .iter()
        .map(|(i, s)| {
            let r = e.tokens[i.get()];
            (s.start, s.end, r.start.get(), r.end.get(), e.kind[i.get()].0, e.score[i.get()].to_bits())
        })
        .collect()
}

const TEXTS: [&str; 4] =
    ["Ann met Bob in Paris", "Hello world , this is an unbelievable test", "", "Carl stayed . Dana left early"];

#[test]
fn a_batch_reads_documents_together_and_each_gets_what_it_gets_alone() {
    // Small rows and batches, so runs hold pieces of several documents.
    for (split, overlap, max_seq, max_batch) in [(250, 50, 0, 0), (3, 1, 5, 2), (4, 1, 6, 1)] {
        let f = finder(split, overlap, max_seq, max_batch);
        let docs: Vec<Document> = TEXTS.iter().map(|t| Document::new(*t)).collect();
        let toks: Vec<Tokens> = TEXTS.iter().map(|t| tokens(t)).collect();
        let mut sentences = Spans::new(ViewId::SOURCE);
        sentences.push(Span::new(0, 13));
        sentences.push(Span::new(14, 29));
        let sentences = Sentences { spans: sentences };
        let refs: Vec<_> = docs
            .iter()
            .zip(&toks)
            .enumerate()
            .map(|(i, (d, t))| (d.get::<View>().unwrap(), t, None, (i == 3).then_some(&sentences)))
            .collect();
        let alone: Vec<_> = refs.iter().map(|r| rows(&f.annotate(*r).unwrap().0)).collect();
        let together: Vec<_> = f.annotate_batch(refs).into_iter().map(|r| rows(&r.unwrap().0)).collect();
        assert_eq!(together, alone, "split {split}, overlap {overlap}, max_seq {max_seq}, max_batch {max_batch}");
        assert!(together[0].len() >= 5 && together[2].is_empty());
    }
}

#[test]
fn a_document_with_invalid_inputs_fails_alone() {
    let f = finder(250, 50, 0, 0);
    let docs: Vec<Document> = TEXTS.iter().map(|t| Document::new(*t)).collect();
    let good = tokens(TEXTS[0]);
    // An inverted token: a lone one too, which no window of two catches.
    let mut inverted = Spans::new(ViewId::SOURCE);
    inverted.push(Span { start: 5, end: 3 });
    let inverted = Tokens { spans: inverted, sentence: None };
    // Sentences that overlap.
    let mut overlapping = Spans::new(ViewId::SOURCE);
    overlapping.push(Span::new(0, 20));
    overlapping.push(Span::new(5, 29));
    let overlapping = Sentences { spans: overlapping };
    let other = tokens(TEXTS[3]);
    let view = |i: usize| docs[i].get::<View>().unwrap();
    let alone = rows(&f.annotate((view(0), &good, None, None)).unwrap().0);
    let results = f.annotate_batch(vec![
        (view(0), &good, None, None),
        (view(1), &inverted, None, None),
        (view(3), &other, None, Some(&overlapping)),
        (view(0), &good, None, None),
    ]);
    assert_eq!(rows(&results[0].as_ref().unwrap().0), alone);
    assert_eq!(rows(&results[3].as_ref().unwrap().0), alone);
    for r in &results[1..3] {
        assert!(matches!(r, Err(turbo_analysis::Error::Annotator { .. })), "{:?}", r.as_ref().err());
    }
    // A sentence off a character boundary fails its own read, not the batch's.
    let mut cut = Spans::new(ViewId::SOURCE);
    cut.push(Span::new(0, 3));
    let cut = Sentences { spans: cut };
    let zoe = Document::new("Zoë");
    let zoe_tokens = tokens("Zoë");
    let results = f.annotate_batch(vec![
        (view(0), &good, None, None),
        (zoe.get::<View>().unwrap(), &zoe_tokens, None, Some(&cut)),
    ]);
    assert_eq!(rows(&results[0].as_ref().unwrap().0), alone);
    assert!(results[1].is_err());
}

/// Throughput of 256 documents of about 40 words, one at a time against
/// one batch. `cargo test --release -p turbo-namefind-dl --test offline --
/// --ignored --nocapture throughput`
#[test]
#[ignore = "timing"]
fn throughput() {
    let f = finder(250, 50, 0, 0);
    let text = "Ann met Bob in Paris and they walked along the river to see an unbelievable exhibition of paintings . \
                Carl stayed home while Dana left early for the station , where the trains were late again today";
    let docs: Vec<Document> = (0..256).map(|_| Document::new(text)).collect();
    let toks = tokens(text);
    let refs = || docs.iter().map(|d| (d.get::<View>().unwrap(), &toks, None, None)).collect::<Vec<_>>();
    let best = |run: &dyn Fn() -> usize| {
        let mut best = f64::MAX;
        let mut n = 0;
        for _ in 0..5 {
            let start = Instant::now();
            n = run();
            best = best.min(start.elapsed().as_secs_f64());
        }
        (docs.len() as f64 / best, n)
    };
    let (one, a) = best(&|| refs().into_iter().map(|r| f.annotate(r).unwrap().0.spans.len()).sum());
    let (batch, b) = best(&|| f.annotate_batch(refs()).into_iter().map(|r| r.unwrap().0.spans.len()).sum());
    assert_eq!(a, b);
    println!("one at a time: {one:.0} docs/s; annotate_batch: {batch:.0} docs/s ({a} entities)");
}

#[test]
fn incoming_decoding_configuration_hashes_are_preserved() {
    use turbo_analysis::Annotator;
    let actual = [
        finder_by(Decoding::Word, 250, 50, 0, 0).config_hash(),
        finder_by(Decoding::Piece, 250, 50, 0, 0).config_hash(),
        finder_by(Decoding::Word, 250, 50, 4, 0).config_hash(),
        finder_by(Decoding::Piece, 250, 50, 16, 0).config_hash(),
        finder_by(Decoding::Word, 249, 50, 16, 0).config_hash(),
        finder_by(Decoding::Word, 250, 49, 16, 0).config_hash(),
    ];
    // Captured from the incoming four-setting implementation on real bundle opens.
    assert_eq!(
        actual,
        [
            6658229060589688275,
            4425913653622098866,
            13838384022699247,
            3264368134685183938,
            4958085615164733728,
            3960764871557555008
        ]
    );
}

fn subword_layer(f: &NameFinderDl, text: &str) -> Subwords {
    let tokenizer = Tokenizer::load(&Bundle::open(&bundle()).unwrap()).unwrap();
    let bare = Encode {
        add_special_tokens: false,
        truncation: Truncation::None,
        max_tokens: u32::MAX,
        prompt: PromptRole::None,
    };
    let (mut ids, mut raw) = (Vec::new(), Vec::new());
    tokenizer.encode_spans(text, bare, &mut ids, &mut raw).unwrap();
    let mut spans = Spans::new(ViewId::SOURCE);
    for &[start, end] in &raw {
        spans.push(Span::new(start, end));
    }
    Subwords { spans, ids, model: f.model().unwrap(), token: None }
}

#[test]
fn the_step_reuses_document_tokens_and_subwords() {
    let f = finder_by(Decoding::Piece, 250, 50, 0, 0);
    let source = "Ann met Bob";
    let changed = "Xxx xxx Xxx";
    assert_eq!(source.len(), changed.len());
    let mut doc = Document::new(changed);
    let mut tokens = Spans::new(ViewId::SOURCE);
    for (start, word) in [(0, "Xxx"), (4, "xxx"), (8, "Xxx")] {
        tokens.push(Span::new(start, start + word.len() as u32));
    }
    doc.add(&Key::DEFAULT, Tokens { spans: tokens, sentence: None }, Provenance::by("test:tokens")).unwrap();
    doc.add(&Key::DEFAULT, subword_layer(&f, source), Provenance::by("test:subwords")).unwrap();
    let analyzer = Analyzer::builder().given(Key::<Tokens>::DEFAULT).step(Bound::new(f)).build().unwrap();
    let mut docs = vec![doc];
    assert!(analyzer.run(&mut docs).iter().all(Result::is_ok));
    let entities = docs[0].get::<Entities>().unwrap();
    let got: Vec<(usize, usize)> = entities.spans.iter().map(|(_, s)| (s.start as usize, s.end as usize)).collect();
    assert_eq!(got, pieces(source));
}

#[test]
fn the_step_rejects_subwords_from_another_model() {
    let f = finder_by(Decoding::Piece, 250, 50, 0, 0);
    let mut layer = subword_layer(&f, "Ann met Bob");
    layer.model.id = "another-model".into();
    let mut doc = Document::new("Ann met Bob");
    let mut tokens = Spans::new(ViewId::SOURCE);
    tokens.push(Span::new(0, 11));
    doc.add(&Key::DEFAULT, Tokens { spans: tokens, sentence: None }, Provenance::by("test:tokens")).unwrap();
    doc.add(&Key::DEFAULT, layer, Provenance::by("test:subwords")).unwrap();
    let analyzer = Analyzer::builder().given(Key::<Tokens>::DEFAULT).step(Bound::new(f)).build().unwrap();
    let mut docs = vec![doc];
    let result = analyzer.run(&mut docs).remove(0);
    assert!(result.as_ref().is_err_and(|e| e.to_string().contains("another model")), "{result:?}");
}

#[path = "offline/word_slice_reuse.rs"]
mod word_slice_reuse;
