//! The subwords annotator on real bundles: the small BERT's WordPiece,
//! a byte-level BPE, BGE-M3's Unigram and SentencePiece `.model` files,
//! through the analyzer and into a document, which checks every span.
//! The WordPiece pieces are compared with upstream `tokenizers`, and the
//! SentencePiece pieces with Google's sentencepiece library, id for id
//! and span for span.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use turbo::bundle::{Bundle, sha256_hex};
use turbo::manifest::{
    Bpe, Manifest, PromptRole, SentencePiece, SentencePieceAlgorithm, SpecialRole, SpecialToken, Truncation, Unigram,
};
use turbo::tokenizer::{Encode, Tokenizer};
use turbo_analysis::registry::{Registry, Spec};
use turbo_analysis::*;
use turbo_analysis_embed::{SubwordAnnotator, SubwordsProvider};
use turbo_analysis_layers::{Subwords, Tokens};

#[path = "subwords/sentencepiece.rs"]
mod sentencepiece;

fn testdata() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata")
}

fn wordpiece_dir() -> PathBuf {
    testdata().join("tiny-bert-bundle")
}

fn wordpiece() -> SubwordAnnotator {
    SubwordAnnotator::open(&wordpiece_dir()).expect("the small BERT's bundle loads")
}

/// The texts the core's tokenizer is compared with upstream on, and a few
/// more: multibyte characters, emoji, a character no piece covers.
fn texts() -> Vec<String> {
    let lines = std::fs::read_to_string(testdata().join("tokenizer-texts.jsonl")).unwrap();
    let mut t: Vec<String> = lines
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap()["text"].as_str().unwrap().to_owned())
        .collect();
    t.extend(
        [
            "",
            "   ",
            "naïve café 東京 🙂 🥺x 𓀀𓀁 e\u{301}",
            "[CLS] a special token written in the text [SEP]",
            "it's (see below.) done",
        ]
        .map(str::to_owned),
    );
    t
}

/// A directory of its own for a bundle a test writes, removed on drop.
struct TempBundle(PathBuf);

impl Drop for TempBundle {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The small BERT's manifest with another tokenizer file, as a bundle in
/// a directory of its own. The manifest is edited as the typed `Manifest`
/// and written back; the tokenizer is built unchecked, since the
/// bundle's reference ids are the WordPiece ones.
fn bundle_with(name: &str, tokenizer_json: &Path, edit: impl FnOnce(&mut Manifest)) -> (TempBundle, Tokenizer, Bundle) {
    bundle_with_file(name, tokenizer_json, "tokenizer.json", edit)
}

/// The same with the tokenizer file under `file` in the bundle.
fn bundle_with_file(
    name: &str,
    tokenizer: &Path,
    file: &str,
    edit: impl FnOnce(&mut Manifest),
) -> (TempBundle, Tokenizer, Bundle) {
    // Tests run side by side, and two may write the same kind of bundle.
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("turbo-embed-test-{}-{name}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let bytes = std::fs::read(tokenizer).unwrap();
    std::fs::write(dir.join(file), &bytes).unwrap();
    let mut m = Manifest::parse(&std::fs::read(wordpiece_dir().join("manifest.json")).unwrap()).unwrap();
    let entry = m.files.iter_mut().find(|f| f.path == "tokenizer.json").unwrap();
    entry.path = file.into();
    m.tokenizer.file = file.into();
    entry.size = bytes.len() as u64;
    entry.sha256 = sha256_hex(&bytes);
    m.tokenizer.normalizer = None;
    m.tokenizer.wordpiece = None;
    edit(&mut m);
    std::fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    let bundle = Bundle::open(&dir).unwrap();
    let tok = Tokenizer::unchecked(&bundle).unwrap();
    (TempBundle(dir), tok, bundle)
}

fn special(role: SpecialRole, content: &str, id: u32, lstrip: bool) -> SpecialToken {
    SpecialToken { role, content: content.into(), id, lstrip, normalized: false, single_word: false, rstrip: false }
}

/// RoBERTa's special tokens and template.
fn roberta_specials(m: &mut Manifest, mask: u32) {
    m.tokenizer.special_tokens = vec![
        special(SpecialRole::Bos, "<s>", 0, false),
        special(SpecialRole::Pad, "<pad>", 1, false),
        special(SpecialRole::Eos, "</s>", 2, false),
        special(SpecialRole::Unk, "<unk>", 3, false),
        special(SpecialRole::Mask, "<mask>", mask, true),
    ];
    m.tokenizer.template = vec!["<s>".into(), "$TEXT".into(), "</s>".into()];
}

/// The byte-level BPE of testdata/tiny-bpe in a bundle.
fn bpe() -> (TempBundle, SubwordAnnotator) {
    let (dir, tok, bundle) = bundle_with("bpe", &testdata().join("tiny-bpe/tokenizer.json"), |m| {
        m.model.id = "turbo-testdata/tiny-bpe".into();
        m.tokenizer.bpe = Some(Bpe { add_prefix_space: false, ignore_merges: false });
        roberta_specials(m, 4);
    });
    assert_eq!(tok.kind_name(), "bpe");
    (dir, SubwordAnnotator::new(tok, &bundle))
}

/// BGE-M3's Unigram, cut down (testdata/README.md), in a bundle.
fn unigram() -> (TempBundle, SubwordAnnotator) {
    let (dir, tok, bundle) = bundle_with("unigram", &testdata().join("bge-m3/tokenizer.json"), |m| {
        m.model.id = "BAAI/bge-m3".into();
        m.tokenizer.unigram = Some(Unigram {
            precompiled_charsmap: true,
            collapse_spaces: true,
            metaspace: "\u{2581}".into(),
            add_prefix_space: true,
            space_punctuation: false,
            collapse_whitespace: false,
            strip: false,
            whole_text: false,
            whitespace_split: false,
        });
        roberta_specials(m, 28232);
        let a = m.architecture.as_mut().unwrap();
        a.vocab_size = 28233;
    });
    assert_eq!(tok.kind_name(), "unigram");
    (dir, SubwordAnnotator::new(tok, &bundle))
}

fn sentencepiece_dir() -> PathBuf {
    testdata().join("sentencepiece")
}

/// A model of testdata/sentencepiece in a bundle, with its control and
/// unknown pieces as the special tokens and `template`.
fn sentencepiece(
    model: &str,
    algorithm: SentencePieceAlgorithm,
    specials: &[(SpecialRole, &str, u32)],
    template: &[&str],
) -> (TempBundle, SubwordAnnotator) {
    let path = sentencepiece_dir().join(format!("{model}.model"));
    let (dir, tok, bundle) = bundle_with_file(model, &path, "tokenizer.model", |m| {
        m.model.id = format!("turbo-testdata/{model}");
        m.tokenizer.sentencepiece = Some(SentencePiece { algorithm });
        m.tokenizer.special_tokens = specials.iter().map(|&(r, c, id)| special(r, c, id, false)).collect();
        m.tokenizer.template = template.iter().map(|&t| t.to_owned()).collect();
        m.architecture.as_mut().unwrap().vocab_size = 32100;
    });
    assert_eq!(tok.kind_name(), "sentencepiece");
    (dir, SubwordAnnotator::new(tok, &bundle))
}

/// T5's model: unigram with the nmt_nfkc character map.
fn t5() -> (TempBundle, SubwordAnnotator) {
    let specials = [(SpecialRole::Pad, "<pad>", 0), (SpecialRole::Eos, "</s>", 1), (SpecialRole::Unk, "<unk>", 2)];
    sentencepiece("t5", SentencePieceAlgorithm::Unigram, &specials, &["$TEXT", "</s>"])
}

/// A small BPE with byte fallback, spaces kept as Llama's model keeps
/// them.
fn bpe_bytes() -> (TempBundle, SubwordAnnotator) {
    let specials = [(SpecialRole::Unk, "<unk>", 0), (SpecialRole::Bos, "<s>", 1), (SpecialRole::Eos, "</s>", 2)];
    sentencepiece("bpe-bytes", SentencePieceAlgorithm::Bpe, &specials, &["<s>", "$TEXT"])
}

/// A small unigram model with byte fallback.
fn unigram_bytes() -> (TempBundle, SubwordAnnotator) {
    let specials = [(SpecialRole::Unk, "<unk>", 0), (SpecialRole::Bos, "<s>", 1), (SpecialRole::Eos, "</s>", 2)];
    sentencepiece("unigram-bytes", SentencePieceAlgorithm::Unigram, &specials, &["<s>", "$TEXT", "</s>"])
}

/// The whole text, no special tokens, no prefix: what the layer holds.
fn bare() -> Encode {
    Encode { add_special_tokens: false, truncation: Truncation::None, max_tokens: u32::MAX, prompt: PromptRole::None }
}

/// Runs the annotator over each document, reading the tokens when they
/// are given.
fn run(a: SubwordAnnotator, docs: &mut [Document], with_tokens: bool) -> Vec<Result<(), Error>> {
    let b = Analyzer::builder();
    let b = if with_tokens { b.given(Key::<Tokens>::DEFAULT) } else { b };
    b.add(a).build().unwrap().run(docs)
}

/// The layer's pieces as (id, span) pairs.
fn pairs(sub: &Subwords) -> Vec<(i32, [u32; 2])> {
    sub.ids.iter().zip(sub.spans.iter()).map(|(&id, (_, s))| (id, [s.start, s.end])).collect()
}

/// Every annotator's layer on every text is taken by the document (spans
/// in bounds, on character boundaries, in order), holds the ids the
/// tokenizer gives the text with no special tokens and no truncation,
/// and the model in its provenance.
fn every_text_is_taken(a: SubwordAnnotator) {
    let model = Annotator::model(&a).unwrap();
    let ours: Vec<Vec<i32>> = texts().iter().map(|t| a.tokenizer().encode(t, bare()).unwrap()).collect();
    let mut docs: Vec<Document> = texts().into_iter().map(Document::new).collect();
    let status = run(a, &mut docs, false);
    for ((doc, s), ids) in docs.iter().zip(status).zip(ours) {
        s.unwrap_or_else(|e| panic!("{:?}: {e}", doc.source()));
        let sub = doc.get::<Subwords>().unwrap();
        assert_eq!(sub.ids, ids, "{:?}", doc.source());
        assert_eq!(sub.model, model);
        assert!(sub.token.is_none());
        let p = doc.provenance(Key::<Subwords>::DEFAULT.id()).unwrap();
        assert_eq!(p.annotator, "turbo:subwords");
        assert_eq!(p.model.as_ref(), Some(&model));
    }
}

#[test]
fn wordpiece_pieces_are_taken_on_every_text() {
    every_text_is_taken(wordpiece());
}

#[test]
fn bpe_pieces_are_taken_on_every_text() {
    let (_dir, a) = bpe();
    every_text_is_taken(a);
}

#[test]
fn unigram_pieces_are_taken_on_every_text() {
    let (_dir, a) = unigram();
    every_text_is_taken(a);
}

#[test]
fn sentencepiece_pieces_are_taken_on_every_text() {
    for (_dir, a) in [t5(), bpe_bytes(), unigram_bytes()] {
        every_text_is_taken(a);
    }
}

/// The bundle's id and revision, and the SHA-256 of its manifest, name
/// the model.
#[test]
fn the_model_is_the_bundles() {
    let a = wordpiece();
    let manifest = std::fs::read(wordpiece_dir().join("manifest.json")).unwrap();
    let m = Annotator::model(&a).unwrap();
    assert_eq!(m.id, "sentence-transformers/all-MiniLM-L6-v2");
    assert_eq!(m.revision, "1");
    assert_eq!(m.sha256, sha256_hex(&manifest));
}

/// `span` without whitespace at either end, unless that is all, as the
/// core trims its spans.
fn trimmed(text: &str, [a, b]: [u32; 2]) -> [u32; 2] {
    let piece = &text[a as usize..b as usize];
    let inner = piece.trim();
    if inner.is_empty() {
        return [a, b];
    }
    let start = a + (inner.as_ptr() as usize - piece.as_ptr() as usize) as u32;
    [start, start + inner.len() as u32]
}

/// The WordPiece pieces are upstream's, id for id, and their spans are
/// upstream's offsets, on every text.
#[test]
fn wordpiece_pieces_match_upstream() {
    let mut up = tokenizers::Tokenizer::from_file(wordpiece_dir().join("tokenizer.json")).unwrap();
    up.with_padding(None);
    up.with_truncation(None).unwrap();
    let mut docs: Vec<Document> = texts().into_iter().map(Document::new).collect();
    for s in run(wordpiece(), &mut docs, false) {
        s.unwrap();
    }
    for doc in &docs {
        let text = doc.source();
        let e = up.encode(text, false).unwrap();
        let theirs: Vec<(i32, [u32; 2])> = e
            .get_ids()
            .iter()
            .zip(e.get_offsets())
            .map(|(&id, &(a, b))| (id as i32, trimmed(text, [a as u32, b as u32])))
            .collect();
        assert_eq!(pairs(doc.get::<Subwords>().unwrap()), theirs, "{text:?}");
    }
}

/// A character no merge covers is four byte-level pieces, each spanning
/// the whole character; the ids still decode to the text.
#[test]
fn byte_level_pieces_of_one_character_share_its_span() {
    let (_dir, a) = bpe();
    let text = "a 𓀀 b 🥺";
    let ids = a.tokenizer().encode(text, bare()).unwrap();
    assert_eq!(a.tokenizer().decode(&ids, false).unwrap(), text);
    let mut docs = vec![Document::new(text)];
    run(a, &mut docs, false).remove(0).unwrap();
    let sub = docs[0].get::<Subwords>().unwrap();
    let glyph = text.find('𓀀').unwrap() as u32;
    let on_glyph: Vec<Span> = sub.spans.iter().map(|(_, s)| s).filter(|s| s.start == glyph).collect();
    assert!(on_glyph.len() >= 2, "{:?}", sub.spans);
    assert!(on_glyph.iter().all(|s| s.end == glyph + 4), "{:?}", sub.spans);
    for (_, s) in sub.spans.iter() {
        assert!(text.is_char_boundary(s.start as usize) && text.is_char_boundary(s.end as usize));
    }
}

/// A text far longer than the model's max_seq is covered to its end: the
/// layer is the untruncated row.
#[test]
fn a_long_text_is_covered_to_its_end() {
    let a = wordpiece();
    let max_seq = a.tokenizer().max_seq as usize;
    let text = "Embedding models turn text into vectors whose distances track meaning. ".repeat(40);
    let mut docs = vec![Document::new(text.as_str())];
    run(a, &mut docs, false).remove(0).unwrap();
    let sub = docs[0].get::<Subwords>().unwrap();
    assert!(sub.ids.len() > 8 * max_seq, "{} pieces", sub.ids.len());
    let last = sub.spans.get(Ix::new(sub.spans.len() as u32 - 1));
    assert_eq!(last.end as usize, text.trim_end().len());
    assert_eq!(&text[last.range()], ".");
}

/// The overhead is the template's special tokens, and the prefix's tokens
/// for a role that has one ("query: " is two in the small BERT).
#[test]
fn the_overhead_is_the_specials_and_the_prefix() {
    let a = wordpiece();
    assert_eq!(a.overhead(PromptRole::None), 2);
    assert_eq!(a.overhead(PromptRole::Document), 2);
    assert_eq!(a.overhead(PromptRole::Query), 4);
    // The text's pieces and the overhead are the model's whole row.
    let text = "how do I reset a password";
    let row = a.tokenizer().encode(text, Encode { add_special_tokens: true, prompt: PromptRole::Query, ..bare() });
    let pieces = a.tokenizer().encode(text, bare()).unwrap();
    assert_eq!(row.unwrap().len(), pieces.len() + a.overhead(PromptRole::Query) as usize);
}

/// Word tokens split at whitespace, with the trailing punctuation left
/// out, as some word tokenizers leave it.
fn word_tokens(doc: &mut Document) {
    let text = doc.source().to_owned();
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut at = 0;
    for w in text.split(' ') {
        let core = w.trim_end_matches(['.', ',', ')']);
        if !core.is_empty() {
            spans.push(Span::new(at as u32, (at + core.len()) as u32));
        }
        at += w.len() + 1;
    }
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
}

/// With word tokens, each piece gets its token: a token's pieces are a
/// range, and punctuation the word tokens leave out goes with the token
/// before it.
#[test]
fn pieces_get_their_word_token() {
    let text = "unaffable tokenizers, here.";
    let mut docs = vec![Document::new(text)];
    word_tokens(&mut docs[0]);
    run(wordpiece(), &mut docs, true).remove(0).unwrap();
    let doc = &docs[0];
    let sub = doc.get::<Subwords>().unwrap();
    let tokens = doc.get::<Tokens>().unwrap();
    let pieces = |t: u32| -> Vec<&str> { sub.of(Ix::new(t)).iter().map(|i| &text[sub.spans.get(i).range()]).collect() };
    assert_eq!(tokens.spans.len(), 3);
    let all: Vec<&str> = (0..3).flat_map(pieces).collect();
    assert_eq!(all.concat(), "unaffabletokenizers,here.");
    assert!(pieces(1).ends_with(&[","]), "{:?}", pieces(1));
    assert_eq!(pieces(2), ["here", "."]);
    // Each token's pieces lie in it, but for the punctuation it left out.
    for (t, span) in tokens.spans.iter() {
        for i in sub.of(t).iter() {
            let p = sub.spans.get(i);
            assert!(span.contains(p) || &text[p.range()] == "," || &text[p.range()] == ".", "{p:?} in {span:?}");
        }
    }
    let p = doc.provenance(Key::<Subwords>::DEFAULT.id()).unwrap();
    assert_eq!(p.reads, [Key::<View>::DEFAULT.id(), Key::<Tokens>::DEFAULT.id()]);
}

/// A byte-level piece over two word tokens goes with the first.
#[test]
fn a_bpe_piece_over_two_tokens_goes_with_the_first() {
    let (_dir, a) = bpe();
    let text = "it's";
    // The word tokenizer splits "it" and "'s"; check what the BPE makes.
    let pieces = {
        let (mut ids, mut spans) = (Vec::new(), Vec::new());
        a.tokenizer().encode_spans(text, bare(), &mut ids, &mut spans).unwrap();
        spans
    };
    let mut doc = Document::new(text);
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 1));
    spans.push(Span::new(1, 4));
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
    let mut docs = vec![doc];
    run(a, &mut docs, true).remove(0).unwrap();
    let sub = docs[0].get::<Subwords>().unwrap();
    let token: Vec<usize> = sub.token.as_ref().unwrap().iter().map(|t| t.get()).collect();
    // Each piece goes with the token its first byte is in.
    let want: Vec<usize> = pieces.iter().map(|s| usize::from(s[0] >= 1)).collect();
    assert_eq!(token, want, "{pieces:?}");
    assert!(pieces.iter().any(|s| s[0] < 1 && s[1] > 1), "a piece over both tokens: {pieces:?}");
}

/// Tokens in another view than the text, or out of order, fail the
/// document with the reason.
#[test]
fn tokens_that_cannot_be_aligned_fail_the_document() {
    let mut doc = Document::new("b a");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(2, 3));
    spans.push(Span::new(0, 1));
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
    let mut docs = vec![doc];
    let e = run(wordpiece(), &mut docs, true).remove(0).unwrap_err();
    assert!(e.to_string().contains("not in text order"), "{e}");
    assert!(docs[0].get::<Subwords>().is_none());

    // The text read from a normalized view, the tokens in the source.
    let mut doc = Document::new("ab");
    let mut b = ViewBuilder::new(doc.source());
    b.replace(1, "A");
    b.keep(2);
    let (text, alignment) = b.finish();
    let view = View::derived(doc.get::<View>().unwrap(), text, alignment);
    doc.add(&Key::named("upper"), view, Provenance::by("test")).unwrap();
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 2));
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
    let a = Analyzer::builder()
        .given(Key::<View>::named("upper"))
        .given(Key::<Tokens>::DEFAULT)
        .step(Bound::new(wordpiece()).read(0, "upper"))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    let e = a.run(&mut docs).remove(0).unwrap_err();
    assert!(e.to_string().contains("the tokens are in view 0, the text in view 1"), "{e}");
}

/// A pipeline from configuration finds the annotator by name, with the
/// bundle as the spec's location.
#[test]
fn the_provider_opens_the_bundle_it_is_given() {
    let mut r: Registry<dyn turbo_analysis::annotate::Step> = Registry::default();
    r.register(SubwordsProvider);
    let spec = Spec { location: Some(wordpiece_dir()), options: Vec::new() };
    let a = r.pipeline(&[("subwords", spec)], |b| b).unwrap();
    assert_eq!(a.plan(), vec![vec!["turbo:subwords"]]);
    let mut docs = vec![Document::new("hello world")];
    a.run(&mut docs).remove(0).unwrap();
    assert_eq!(docs[0].get::<Subwords>().unwrap().ids.len(), 2);

    assert!(r.pipeline(&[("subwords", Spec::default())], |b| b).is_err(), "no location");
    let missing = Spec { location: Some(testdata().join("no-such-bundle")), options: Vec::new() };
    let e = r.pipeline(&[("subwords", missing)], |b| b).err().unwrap();
    assert!(e.to_string().contains("no-such-bundle"), "{e}");
}
