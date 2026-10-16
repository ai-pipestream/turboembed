//! Short-text accuracy on lingua's test sets: single words, word pairs
//! and sentences, 1000 of each per language. The test sets are not part
//! of this repository; give the `language-models` directory of a lingua-rs
//! checkout.
//!
//! cargo run --release -p turbo-langspan --example short_text -- \
//!     <langdetect-183.bin> <lingua-rs/language-models>
//!
//! Prints one line per language and set: the language's directory name,
//! the set, how many lines, and how many each detector got right:
//! OpenNLP's document detector (the best of the compared languages) and
//! the span detector's summary (its best language). Both answer only the
//! 68 languages lingua and the model share; a code in the second column
//! of LANGUAGES that is not the language's first counts as right too (the
//! model has two codes for Estonian, Latvian and Persian).

use std::path::Path;

use turbo_langdetect::{LanguageDetector, LanguageModel, Scratch};
use turbo_langspan::{Run, Settings, SpanDetector, SpanScratch, summary};
use turbo_maxent::ModelFile;

/// Lingua's directory name for a language and the model's codes for it.
const LANGUAGES: &[(&str, &[&str])] = &[
    ("af", &["afr"]),
    ("ar", &["ara"]),
    ("az", &["aze"]),
    ("be", &["bel"]),
    ("bg", &["bul"]),
    ("bn", &["ben"]),
    ("bs", &["bos"]),
    ("ca", &["cat"]),
    ("cs", &["ces"]),
    ("cy", &["cym"]),
    ("da", &["dan"]),
    ("de", &["deu"]),
    ("el", &["ell"]),
    ("en", &["eng"]),
    ("eo", &["epo"]),
    ("es", &["spa"]),
    ("et", &["est", "ekk"]),
    ("eu", &["eus"]),
    ("fa", &["fas", "pes"]),
    ("fi", &["fin"]),
    ("fr", &["fra"]),
    ("ga", &["gle"]),
    ("gu", &["guj"]),
    ("he", &["heb"]),
    ("hi", &["hin"]),
    ("hr", &["hrv"]),
    ("hu", &["hun"]),
    ("hy", &["hye"]),
    ("id", &["ind"]),
    ("is", &["isl"]),
    ("it", &["ita"]),
    ("ja", &["jpn"]),
    ("ka", &["kat"]),
    ("kk", &["kaz"]),
    ("ko", &["kor"]),
    ("la", &["lat"]),
    ("lt", &["lit"]),
    ("lv", &["lav", "lvs"]),
    ("mi", &["mri"]),
    ("mk", &["mkd"]),
    ("mn", &["mon"]),
    ("mr", &["mar"]),
    ("ms", &["msa"]),
    ("nb", &["nob"]),
    ("nl", &["nld"]),
    ("nn", &["nno"]),
    ("pa", &["pan"]),
    ("pl", &["pol"]),
    ("pt", &["por"]),
    ("ro", &["ron"]),
    ("ru", &["rus"]),
    ("sk", &["slk"]),
    ("sl", &["slv"]),
    ("so", &["som"]),
    ("sq", &["sqi"]),
    ("sr", &["srp"]),
    ("sv", &["swe"]),
    ("sw", &["swa"]),
    ("ta", &["tam"]),
    ("te", &["tel"]),
    ("th", &["tha"]),
    ("tl", &["tgl"]),
    ("tr", &["tur"]),
    ("uk", &["ukr"]),
    ("ur", &["urd"]),
    ("vi", &["vie"]),
    ("zh", &["cmn"]),
    ("zu", &["zul"]),
];

const SETS: [&str; 3] = ["single-words", "word-pairs", "sentences"];

fn main() {
    let mut args = std::env::args().skip(1);
    let model = args.next().expect("a model path");
    let root = args.next().expect("lingua's language-models directory");
    let open = || LanguageDetector::new(LanguageModel::open(ModelFile::read(&model).unwrap()).unwrap());
    let compared = |code: &str| LANGUAGES.iter().any(|(_, codes)| codes.contains(&code));
    let document = open();
    let allowed: Vec<usize> =
        document.model().supported_languages().enumerate().filter(|(_, c)| compared(c)).map(|(i, _)| i).collect();
    let spans = SpanDetector::restricted(open(), Settings::default(), compared);
    let mut scratch = Scratch::new();
    let mut span_scratch = SpanScratch::new();
    let mut runs: Vec<Run> = Vec::new();
    for (dir, codes) in LANGUAGES {
        for set in SETS {
            let path = Path::new(&root).join(dir).join("testdata").join(format!("{set}.txt"));
            let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let (mut n, mut opennlp, mut by_span) = (0, 0, 0);
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                n += 1;
                document.predict_language(line, &mut scratch);
                let p = scratch.probabilities();
                let best = allowed.iter().copied().fold(allowed[0], |b, i| if p[i] > p[b] { i } else { b });
                opennlp += usize::from(codes.contains(&document.model().code(best)));
                spans.detect(line, None, None, &mut span_scratch, &mut runs);
                let languages = summary(&spans, &runs);
                by_span += usize::from(languages.best().is_some_and(|(c, _)| codes.contains(&c)));
            }
            println!("{dir}\t{set}\t{n}\t{opennlp}\t{by_span}");
        }
    }
}
