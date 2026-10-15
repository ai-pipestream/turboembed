//! The dictionary segmenter against gold word segmentations: the Thai
//! PUD treebank of Universal Dependencies and the myPOS Burmese corpus.
//!
//! Only the words inside complex script runs are scored, since that is
//! all the dictionaries decide: in each maximal run of Thai (or Myanmar)
//! characters that the segmenter hands to a dictionary, the gold word
//! boundaries strictly inside the run and the run's two ends cut the gold
//! words, the segmenter's boundaries cut its words, and a word is correct
//! when both have a word over exactly the same characters.
//!
//! Neither corpus is in the repository (Thai PUD is CC BY-SA 4.0, myPOS
//! CC BY-NC-SA 4.0), so the test is ignored and reads them from the
//! environment:
//!
//! ```text
//! TURBO_THAI_PUD=<UD_Thai-PUD>/th_pud-ud-test.conllu \
//! TURBO_MYPOS=<myPOS>/corpus-ver-3.0/corpus/mypos-ver.3.0.txt \
//!   cargo test --release -p turbo-tokenize --test complex_eval -- --ignored --nocapture
//! ```
//!
//! With `TURBO_COMPLEX_EVAL_OUT=<dir>` it also writes the gold and the
//! segmenter's segmentation of every sentence there (`<corpus>.gold` and
//! `<corpus>.dict`, segments separated by `|`), for comparisons outside
//! the repository. docs/segmentation.md has the numbers and the versions
//! of the data.

use std::fmt::Write as _;

use turbo_tokenize::complex::{bundled, complex_script};

/// Counts over the complex script runs of a corpus.
#[derive(Debug, Default, PartialEq, Eq)]
struct Score {
    sentences: usize,
    gold: usize,
    returned: usize,
    correct: usize,
}

impl Score {
    fn precision(&self) -> f64 {
        self.correct as f64 / self.returned as f64
    }
    fn recall(&self) -> f64 {
        self.correct as f64 / self.gold as f64
    }
    fn f1(&self) -> f64 {
        2.0 * self.precision() * self.recall() / (self.precision() + self.recall())
    }
}

/// The maximal runs of characters of one complex script, as byte ranges.
fn runs(text: &str) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut current = None;
    for (i, c) in text.char_indices() {
        let script = complex_script(c);
        match (script, current) {
            (Some(s), Some(t)) if s == t => out.last_mut().unwrap().1 = i + c.len_utf8(),
            (Some(s), _) => {
                out.push((i, i + c.len_utf8()));
                current = Some(s);
            }
            (None, _) => current = None,
        }
    }
    out
}

/// The words a boundary set cuts a run into.
fn words(run: (usize, usize), boundaries: &[usize]) -> Vec<(usize, usize)> {
    let mut cuts = vec![run.0];
    cuts.extend(boundaries.iter().copied().filter(|&b| run.0 < b && b < run.1));
    cuts.push(run.1);
    cuts.windows(2).map(|w| (w[0], w[1])).collect()
}

/// Scores the segmenter on `sentences`, each the list of its gold words
/// (spaces included as words of their own, so the words make the text).
fn score(name: &str, sentences: &[Vec<String>]) -> Score {
    let out = std::env::var("TURBO_COMPLEX_EVAL_OUT").ok();
    let (mut gold_file, mut dict_file) = (String::new(), String::new());
    let mut score = Score::default();
    for words_of in sentences {
        let text: String = words_of.concat();
        let mut gold = Vec::new();
        let mut at = 0;
        for w in words_of {
            at += w.len();
            gold.push(at);
        }
        let mut predicted = Vec::new();
        bundled().for_each_segment(&text, |_, end| predicted.push(end as usize));
        for run in runs(&text) {
            let g = words(run, &gold);
            let p = words(run, &predicted);
            score.gold += g.len();
            score.returned += p.len();
            score.correct += p.iter().filter(|w| g.binary_search(w).is_ok()).count();
        }
        score.sentences += 1;
        if out.is_some() {
            assert!(!text.contains('|'), "{text}");
            writeln!(gold_file, "{}", words_of.join("|")).unwrap();
            let mut start = 0;
            let segments: Vec<&str> = predicted
                .iter()
                .map(|&end| {
                    let s = &text[start..end];
                    start = end;
                    s
                })
                .collect();
            writeln!(dict_file, "{}", segments.join("|")).unwrap();
        }
    }
    if let Some(dir) = out {
        std::fs::write(format!("{dir}/{name}.gold"), gold_file).unwrap();
        std::fs::write(format!("{dir}/{name}.dict"), dict_file).unwrap();
    }
    println!(
        "{name}: {} sentences, gold {} returned {} correct {}: P {:.3} R {:.3} F1 {:.3}",
        score.sentences,
        score.gold,
        score.returned,
        score.correct,
        score.precision(),
        score.recall(),
        score.f1()
    );
    score
}

/// The sentences of a CoNLL-U file as their word forms, with a space
/// word after every word without `SpaceAfter=No`. Multiword token lines
/// and empty nodes are skipped (Thai PUD has none). Each sentence's words
/// must give back its `# text` line.
fn conllu(text: &str) -> Vec<Vec<String>> {
    let mut sentences = Vec::new();
    let mut words: Vec<String> = Vec::new();
    let mut expected = String::new();
    for line in text.lines() {
        if let Some(t) = line.strip_prefix("# text = ") {
            expected = t.to_owned();
        } else if line.is_empty() {
            if !words.is_empty() {
                if words.last().is_some_and(|w| w == " ") {
                    words.pop();
                }
                assert_eq!(words.concat(), expected);
                sentences.push(std::mem::take(&mut words));
            }
        } else if !line.starts_with('#') {
            let fields: Vec<&str> = line.split('\t').collect();
            if fields[0].contains(['-', '.']) {
                continue;
            }
            words.push(fields[1].to_owned());
            if !fields[9].split('|').any(|f| f == "SpaceAfter=No") {
                words.push(" ".to_owned());
            }
        }
    }
    sentences
}

/// The sentences of myPOS: space-separated `word/tag` items, where a
/// compound joins its parts with `|` (`နတ်/n|ဆရာ/n`) and is one word.
fn mypos(text: &str) -> Vec<Vec<String>> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            line.split_whitespace()
                .map(|item| {
                    item.split('|').map(|part| part.rsplit_once('/').map_or(part, |(w, _)| w)).collect::<String>()
                })
                .collect()
        })
        .collect()
}

#[test]
#[ignore = "needs the corpora: TURBO_THAI_PUD and TURBO_MYPOS"]
fn dictionary_against_gold() {
    let thai = std::fs::read_to_string(std::env::var("TURBO_THAI_PUD").expect("TURBO_THAI_PUD")).unwrap();
    let thai = score("th_pud", &conllu(&thai));
    let burmese = std::fs::read_to_string(std::env::var("TURBO_MYPOS").expect("TURBO_MYPOS")).unwrap();
    let burmese = score("mypos", &mypos(&burmese));
    // UD_Thai-PUD commit 670083cd185a2ae215ce7aef1127c2b63640217d and
    // myPOS commit f1a620ba271cae3d8658441eb626450fca09cdfa: the numbers in
    // docs/segmentation.md.
    assert_eq!(thai, Score { sentences: 1000, gold: 21640, returned: 24761, correct: 17095 });
    assert_eq!(burmese, Score { sentences: 43196, gold: 468105, returned: 558677, correct: 358106 });
}
