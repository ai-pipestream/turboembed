//! The WordPiece subwords against OpenNLP's `WordpieceEncoder` on the
//! JVM, from the goldens under `analysis/goldens/data/subwords`: the
//! small BERT's bundle and the JVM encoder share one vocabulary, so the
//! ids and the spans are compared piece by piece.
//!
//! The Rust tokenizer follows upstream `tokenizers`, which differs from
//! OpenNLP in four ways (docs/analysis.md, "Against OpenNLP's WordPiece
//! encoder"):
//!
//! - A nonspacing mark that accent stripping removes is inside the span
//!   of the piece before it on the JVM, and outside it here.
//! - A capital sigma at the end of a word is lowercased to final sigma
//!   on the JVM, and to sigma here.
//! - Marks out of canonical order make every piece of the word span the
//!   whole word on the JVM.
//! - The ideographs U+2B820 to U+2B91F are words of their own on the
//!   JVM, as in the reference BERT; here they are not, as upstream.
//!
//! And special token strings written in the text, such as `[CLS]`, are
//! the special token here and ordinary text on the JVM.
//!
//! Every golden either matches exactly, matches once the stripped marks
//! are left out of the JVM's spans, or is one of the four inputs that
//! have a test of their own; which inputs fall where is listed, so a new
//! difference fails.

use std::path::{Path, PathBuf};

use turbo_analysis::{Analyzer, Document};
use turbo_analysis_embed::SubwordAnnotator;
use turbo_analysis_layers::Subwords;
use turbo_goldens::expect::without_trailing_marks;
use turbo_goldens::{Golden, Tolerance, assert_matches_golden, data_dir, matches_golden, sha512};

/// The run's directory: the vocabulary fixture's name without `.txt`.
const VOCABULARY: &str = "all-minilm-l6-v2-vocab";

fn bundle_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

fn vocabulary() -> Vec<String> {
    let text = std::fs::read_to_string(data_dir().join("fixtures").join(format!("{VOCABULARY}.txt"))).unwrap();
    text.strip_suffix('\n').expect("a line feed after the last piece").split('\n').map(str::to_owned).collect()
}

/// The id of a piece in the shared vocabulary: its line number.
fn id_of(piece: &str) -> u32 {
    vocabulary().iter().position(|p| p == piece).unwrap_or_else(|| panic!("{piece} is in the vocabulary")) as u32
}

/// One golden, its input and the document the annotator ran on.
struct Case {
    name: String,
    golden: Golden,
    doc: Document,
}

impl Case {
    fn text(&self) -> &str {
        self.doc.source()
    }

    /// The golden's pieces: id and byte span.
    fn jvm(&self) -> Vec<(u32, [u32; 2])> {
        let l = self.golden.layer("opennlp:subwords").expect("a subwords layer");
        let (start, end, id) = (l.u32s("start").unwrap(), l.u32s("end").unwrap(), l.u32s("id").unwrap());
        (0..l.len()).map(|i| (id[i], [start[i], end[i]])).collect()
    }

    /// The Rust pieces: id and byte span.
    fn rust(&self) -> Vec<(u32, [u32; 2])> {
        let sub = self.doc.get::<Subwords>().expect("the subwords layer");
        sub.ids.iter().zip(sub.spans.iter()).map(|(&id, (_, s))| (id as u32, [s.start, s.end])).collect()
    }
}

/// Every golden of the run, in path order, with the annotator run on its
/// input.
fn cases() -> Vec<Case> {
    let dir = data_dir().join("subwords").join(VOCABULARY);
    let mut paths = Vec::new();
    for set in std::fs::read_dir(&dir).unwrap() {
        for file in std::fs::read_dir(set.unwrap().path()).unwrap() {
            paths.push(file.unwrap().path());
        }
    }
    paths.sort();
    let analyzer = Analyzer::builder().add(SubwordAnnotator::open(&bundle_dir()).unwrap()).build().unwrap();
    let mut out = Vec::new();
    for path in paths {
        let golden = Golden::read(&path).unwrap_or_else(|e| panic!("{e}"));
        let mut docs = vec![Document::new(golden.input_text(data_dir()).unwrap())];
        analyzer.run(&mut docs).remove(0).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let i = &golden.header.input;
        out.push(Case { name: format!("{}/{}", i.set, i.id), golden, doc: docs.remove(0) });
    }
    out
}

/// Our pieces for `text`, which is not a golden's input.
fn rust_pieces(text: &str) -> Vec<(u32, [u32; 2])> {
    let analyzer = Analyzer::builder().add(SubwordAnnotator::open(&bundle_dir()).unwrap()).build().unwrap();
    let mut docs = vec![Document::new(text)];
    analyzer.run(&mut docs).remove(0).unwrap();
    let sub = docs[0].get::<Subwords>().expect("the subwords layer");
    sub.ids.iter().zip(sub.spans.iter()).map(|(&id, (_, s))| (id as u32, [s.start, s.end])).collect()
}

fn case(name: &str) -> Case {
    cases().into_iter().find(|c| c.name == name).unwrap_or_else(|| panic!("no golden for {name}"))
}

/// The inputs that have a test of their own.
const OWN_TEST: [&str; 4] =
    ["subwords/accents-reordered", "subwords/cjk-extension", "subwords/final-sigma", "subwords/special-tokens"];

/// The inputs where a stripped mark is in a JVM span.
const STRIPPED_MARKS: [&str; 7] = [
    "edge/combining-marks",
    "edge/emoji-keycap",
    "edge/emoji-single",
    "en/emoji",
    "multi/ar",
    "subwords/accents-combining",
    "subwords/spacing-marks",
];

/// The vocabulary fixture the JVM read is the bundle's: the same pieces
/// with the same ids, and every golden names it with its SHA-512 and the
/// encoder's options, which are the bundle's.
#[test]
fn the_jvm_read_the_bundles_vocabulary() {
    let up = tokenizers::Tokenizer::from_file(bundle_dir().join("tokenizer.json")).unwrap();
    let vocab = vocabulary();
    assert_eq!(vocab.len(), up.get_vocab_size(true));
    for (id, piece) in vocab.iter().enumerate() {
        assert_eq!(up.token_to_id(piece), Some(id as u32), "{piece}");
    }
    let bytes = std::fs::read(data_dir().join("fixtures").join(format!("{VOCABULARY}.txt"))).unwrap();
    let cases = cases();
    assert_eq!(cases.len(), 76);
    for c in &cases {
        let h = &c.golden.header;
        assert_eq!(h.annotator.class, "opennlp.tools.tokenize.WordpieceEncoder", "{}", c.name);
        assert_eq!(h.annotator.method, "encode");
        assert_eq!(h.models.len(), 1);
        assert_eq!(h.models[0].file, format!("{VOCABULARY}.txt"));
        assert_eq!(h.models[0].sha512, sha512(&bytes));
        let config: Vec<(&str, &str)> = h.config.iter().map(|e| (e.key.as_str(), e.value.as_str())).collect();
        assert_eq!(config, [("lowerCase", "true"), ("maxWordCodePoints", "100")]);
    }
    let manifest = std::fs::read_to_string(bundle_dir().join("manifest.json")).unwrap();
    assert!(manifest.contains("\"lowercase\": true") && manifest.contains("\"max_chars_per_word\": 100"));
}

/// Every golden matches exactly, or once the stripped marks are left out
/// of the JVM's spans, but for the four inputs with tests of their own;
/// the inputs the marks touch are exactly the ones listed.
#[test]
fn every_golden_matches_but_for_the_listed_differences() {
    let (mut exact, mut pieces) = (0, 0);
    let mut marks = Vec::new();
    for c in cases() {
        pieces += c.jvm().len();
        if matches_golden::<Subwords>(&c.doc, &c.golden, Tolerance::exact()).is_ok() {
            exact += 1;
            continue;
        }
        if OWN_TEST.contains(&c.name.as_str()) {
            continue;
        }
        let text = c.text();
        let jvm = c.jvm();
        let trimmed: Vec<_> = jvm.iter().map(|&(id, s)| (id, without_trailing_marks(text, s))).collect();
        assert_ne!(trimmed, jvm, "{}: differs otherwise", c.name);
        marks.push(c.name.clone());
        assert_eq!(c.rust(), trimmed, "{}", c.name);
    }
    assert_eq!(marks, STRIPPED_MARKS);
    assert_eq!(exact, 76 - STRIPPED_MARKS.len() - OWN_TEST.len());
    assert_eq!(pieces, 2682);
}

/// The inputs that match exactly go through `assert_matches_golden`, so
/// its report is what a regression shows.
#[test]
fn the_exact_inputs_match_through_the_golden_comparison() {
    let differ: Vec<&str> = STRIPPED_MARKS.iter().chain(&OWN_TEST).copied().collect();
    for c in cases().iter().filter(|c| !differ.contains(&c.name.as_str())) {
        assert_matches_golden::<Subwords>(&c.doc, &c.golden, Tolerance::exact());
    }
}

/// A nonspacing mark that accent stripping removes ends inside the
/// JVM's span of the piece before it: `WordpieceEncoder` gives the
/// character before the mark the mark's end. Upstream drops the mark
/// with its source, so the piece ends before it.
#[test]
fn a_stripped_mark_is_in_the_jvm_span_and_not_in_ours() {
    let c = case("subwords/accents-combining");
    let text = c.text();
    // "Cafe\u{301}": the piece "cafe" is 4 bytes here and 6 on the JVM.
    assert!(text.starts_with("Cafe\u{301} "));
    let (jvm, rust) = (c.jvm(), c.rust());
    assert_eq!(jvm[0], (id_of("cafe"), [0, 6]));
    assert_eq!(rust[0], (id_of("cafe"), [0, 4]));
    // The ids are the same throughout; only the spans differ.
    assert_eq!(jvm.iter().map(|p| p.0).collect::<Vec<_>>(), rust.iter().map(|p| p.0).collect::<Vec<_>>());

    // A variation selector is a nonspacing mark too: the unknown piece of
    // a heart with U+FE0F runs over the selector on the JVM.
    let c = case("edge/emoji-single");
    let heart = c.text().find("\u{2764}\u{fe0f}").unwrap() as u32;
    let unk = id_of("[UNK]");
    assert!(c.jvm().contains(&(unk, [heart, heart + 6])));
    assert!(c.rust().contains(&(unk, [heart, heart + 3])));
}

/// A capital sigma ending a word after a cased letter is lowercased to
/// final sigma (U+03C2) on the JVM, as the reference BERT's Python
/// `lower()` does; upstream lowercases each character alone, to sigma.
/// Final sigma has pieces of its own, so the words segment differently:
/// the JVM's pieces are exactly ours for the text with those capitals
/// written as final sigma, which is the same length. A sigma alone, or
/// written in lower case, is the same on both.
#[test]
fn a_word_final_capital_sigma_is_final_sigma_on_the_jvm_only() {
    let c = case("subwords/final-sigma");
    let text = c.text();
    assert_eq!(text, "ΟΔΥΣΣΕΥΣ ΣΟΦΟΣ ΣΑΣ Σ ΑΣ. λόγος ΛΌΓΟΣ\n");
    let as_jvm = "ΟΔΥΣΣΕΥς ΣΟΦΟς ΣΑς Σ Ας. λόγος ΛΌΓΟς\n";
    assert_eq!(as_jvm.len(), text.len());
    assert_eq!(rust_pieces(as_jvm), c.jvm());
    assert_ne!(c.rust(), c.jvm());
    // The end of ΣΟΦΟΣ is one piece "##ος" on the JVM, and "##ο", "##σ" here.
    let end = text.find(" ΣΑΣ").unwrap() as u32;
    assert!(c.jvm().contains(&(id_of("##ος"), [end - 4, end])));
    assert!(c.rust().contains(&(id_of("##ο"), [end - 4, end - 2])));
    assert!(c.rust().contains(&(id_of("##σ"), [end - 2, end])));
}

/// Marks out of canonical order: NFD reorders them, so the run's NFD
/// is not each character's NFD in turn, and `WordpieceEncoder` gives
/// every piece of the whitespace-delimited run the whole run's span,
/// punctuation included, as it strips accents before it splits off
/// punctuation. Every word of the input has marks out of order, so the
/// JVM's pieces are exactly ours with each span widened to its run.
#[test]
fn marks_out_of_order_give_every_jvm_piece_the_whole_run() {
    let c = case("subwords/accents-reordered");
    let text = c.text();
    assert_eq!(text, "Vie\u{302}\u{323}t xa\u{301}\u{323}y o\u{308}\u{331}k.\n");
    let runs: Vec<[u32; 2]> = text
        .split([' ', '\n'])
        .filter(|r| !r.is_empty())
        .map(|r| {
            let start = r.as_ptr() as usize - text.as_ptr() as usize;
            [start as u32, (start + r.len()) as u32]
        })
        .collect();
    assert_eq!(runs, [[0, 8], [9, 16], [17, 24]]);
    let rust = c.rust();
    let widened: Vec<_> = rust
        .iter()
        .map(|&(id, s)| (id, *runs.iter().find(|r| r[0] <= s[0] && s[1] <= r[1]).expect("a piece in a run")))
        .collect();
    assert_ne!(rust, widened);
    assert_eq!(c.jvm(), widened);
    // "xa\u{301}\u{323}y" is two pieces, "x" and "##ay", both over all of it.
    assert_eq!(c.jvm()[1..3], [(id_of("x"), [9, 16]), (id_of("##ay"), [9, 16])]);
}

/// The ideographs U+2B820 to U+2B91F are CJK on the JVM, which uses the
/// reference BERT's range, so each is a word of its own; upstream's
/// range starts at U+2B920, so here they are one word with what follows.
/// U+2B81D and U+2B920, inside both ranges, are the same on both.
#[test]
fn the_ideographs_from_u2b820_are_words_on_the_jvm_only() {
    let c = case("subwords/cjk-extension");
    let text = c.text();
    let at = text.find('\u{2B820}').unwrap() as u32;
    let unk = id_of("[UNK]");
    let (jvm, rust) = (c.jvm(), c.rust());
    assert!(rust.contains(&(unk, [at, at + 8])));
    // The JVM's pieces are ours with that one unknown piece split in two.
    let split: Vec<_> = rust
        .iter()
        .flat_map(
            |&p| if p == (unk, [at, at + 8]) { vec![(unk, [at, at + 4]), (unk, [at + 4, at + 8])] } else { vec![p] },
        )
        .collect();
    assert_eq!(jvm, split);
}

/// A special token's string in the text is that special token here, as
/// upstream matches added tokens before normalizing, and ordinary text on
/// the JVM: `[CLS]` is one piece, id 101, here and four on the JVM.
#[test]
fn special_token_strings_are_special_tokens_here_only() {
    let c = case("subwords/special-tokens");
    let (jvm, rust) = (c.jvm(), c.rust());
    assert_eq!(rust[0], (id_of("[CLS]"), [0, 5]));
    let cls = [id_of("["), id_of("cl"), id_of("##s"), id_of("]")];
    assert_eq!(jvm[..4].iter().map(|p| p.0).collect::<Vec<_>>(), cls);
    assert_eq!(jvm[..4].iter().map(|p| p.1).collect::<Vec<_>>(), [[0, 1], [1, 3], [3, 4], [4, 5]]);
    // The words between the special tokens are the same on both.
    let first = c.text().find("first").unwrap() as u32;
    assert!(rust.contains(&(id_of("first"), [first, first + 5])));
    assert!(jvm.contains(&(id_of("first"), [first, first + 5])));
    // Written in lower case the strings are no special tokens here either,
    // as upstream matches them case-sensitively, and every piece is the JVM's.
    let lower = c.text().to_ascii_lowercase();
    assert_ne!(lower, c.text());
    assert_eq!(rust_pieces(&lower), jvm);
}
