//! The dictionary and LSTM segmenters against gold word segmentations across
//! all four complex scripts:
//! - Thai: the Thai PUD treebank of Universal Dependencies (`TURBO_THAI_PUD`)
//! - Burmese: the myPOS 3.0 corpus (`TURBO_MYPOS`)
//! - Khmer: the khPOS 1.0 corpus (`TURBO_KHPOS`, optional)
//! - Lao: the SeqLabeling Lao POS corpus (`TURBO_LAO_POS`, optional)
//!
//! Only the words inside complex script runs are scored, since that is
//! all the complex segmenters decide: in each maximal run of characters of
//! one complex script, the gold word boundaries strictly inside the run and
//! the run's two ends cut the gold words, the segmenter's boundaries cut its
//! words, and a word is correct when both have a word over exactly the same
//! characters.
//!
//! None of the corpora is in the repository, so the test is ignored and reads
//! them from the environment:
//!
//! ```text
//! TURBO_THAI_PUD=<UD_Thai-PUD>/th_pud-ud-test.conllu \
//! TURBO_MYPOS=<myPOS>/corpus-ver-3.0/corpus/mypos-ver.3.0.txt \
//! TURBO_KHPOS=<khPOS>/corpus-draft-ver-1.0/data/after-replace/train.all \
//! TURBO_LAO_POS=<SeqLabeling>/corpus/Lao_train.txt \
//!   cargo test --release -p turbo-tokenize --test complex_eval -- --ignored --nocapture
//! ```
//!
//! With `TURBO_COMPLEX_EVAL_OUT=<dir>` it also writes the gold and each
//! segmenter's segmentation of every sentence there (`<corpus>.gold`,
//! `<corpus>.dict`, and `<corpus>.lstm`, segments separated by `|`).
//! docs/segmentation.md has the numbers and the versions of the data.

use std::fmt::Write as _;
use std::hint::black_box;
use std::time::Instant;

use turbo_tokenize::complex::{
    ComplexScript, ComplexSegmenter, bundled, bundled_dictionaries, bundled_lstms, complex_script,
};

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

/// The maximal runs of characters of one complex script, as byte ranges
/// paired with their script.
fn runs(text: &str) -> Vec<(ComplexScript, (usize, usize))> {
    let mut out: Vec<(ComplexScript, (usize, usize))> = Vec::new();
    let mut current = None;
    for (i, c) in text.char_indices() {
        let script = complex_script(c);
        match (script, current) {
            (Some(s), Some(t)) if s == t => out.last_mut().unwrap().1.1 = i + c.len_utf8(),
            (Some(s), _) => {
                out.push((s, (i, i + c.len_utf8())));
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

/// Scores `segmenter` on `sentences`, each the list of its gold words
/// (spaces included as words of their own, so the words make the text),
/// optionally filtering to runs of `only_script`.
fn score_with(
    segmenter: &ComplexSegmenter<'_>,
    name: &str,
    kind: &str,
    sentences: &[Vec<String>],
    only_script: Option<ComplexScript>,
) -> Score {
    let out = std::env::var("TURBO_COMPLEX_EVAL_OUT").ok();
    let (mut gold_file, mut seg_file) = (String::new(), String::new());
    let mut score = Score::default();
    let mut texts = Vec::with_capacity(sentences.len());
    for words_of in sentences {
        let text: String = words_of.concat();
        let mut gold = Vec::new();
        let mut at = 0;
        for w in words_of {
            at += w.len();
            gold.push(at);
        }
        let mut predicted = Vec::new();
        segmenter.for_each_segment(&text, |_, end| predicted.push(end as usize));
        for (script, run) in runs(&text) {
            if only_script.is_some_and(|s| s != script) {
                continue;
            }
            let g = words(run, &gold);
            let p = words(run, &predicted);
            score.gold += g.len();
            score.returned += p.len();
            score.correct += p.iter().filter(|w| g.binary_search(w).is_ok()).count();
        }
        score.sentences += 1;
        if out.is_some() && !text.contains('|') {
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
            writeln!(seg_file, "{}", segments.join("|")).unwrap();
        }
        texts.push(text);
    }
    if let Some(dir) = out {
        std::fs::write(format!("{dir}/{name}.gold"), gold_file).unwrap();
        std::fs::write(format!("{dir}/{name}.{kind}"), seg_file).unwrap();
    }

    // Measure CPU segmentation throughput (best of 5 after 1 warm pass).
    let total_bytes: usize = texts.iter().map(String::len).sum();
    let mut warm_segments = 0usize;
    for t in &texts {
        segmenter.for_each_segment(black_box(t.as_str()), |_, _| warm_segments += 1);
    }
    black_box(warm_segments);
    let mut best_secs = f64::INFINITY;
    let mut seg_count = 0usize;
    for _ in 0..5 {
        let mut n = 0usize;
        let t0 = Instant::now();
        for t in &texts {
            segmenter.for_each_segment(black_box(t.as_str()), |_, _| n += 1);
        }
        let dt = t0.elapsed().as_secs_f64();
        if dt < best_secs {
            best_secs = dt;
        }
        seg_count = n;
    }
    println!(
        "{name} ({kind}): {} sentences, gold {} returned {} correct {}: P {:.3} R {:.3} F1 {:.3} | {total_bytes} B, {seg_count} segs, best of 5: {:.4}s ({:.2} MB/s, {:.0} segs/s)",
        score.sentences,
        score.gold,
        score.returned,
        score.correct,
        score.precision(),
        score.recall(),
        score.f1(),
        best_secs,
        (total_bytes as f64 / 1e6) / best_secs,
        seg_count as f64 / best_secs,
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

/// The sentences of khPOS: space-separated `word/tag` items, where a
/// compound joins its parts with `_`, `~` or `^` (`ក្រៅ_ពី/IN`,
/// `ការ~កិបកេង/NN`, `កសិ^កម្ម/NN`) and is one word.
fn khpos(text: &str) -> Vec<Vec<String>> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            line.split_whitespace()
                .map(|item| {
                    let w = item.rsplit_once('/').map_or(item, |(w, _)| w);
                    w.chars().filter(|&c| !matches!(c, '_' | '~' | '^')).collect::<String>()
                })
                .collect()
        })
        .collect()
}

/// The sentences of SeqLabeling Lao (`Lao_train.txt`): tab-separated
/// `word\u{200B}TAG` items.
fn lao_pos(text: &str) -> Vec<Vec<String>> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            line.split('\t')
                .map(|item| item.split_once('\u{200B}').map_or(item, |(w, _)| w).to_owned())
                .filter(|w| !w.is_empty())
                .collect()
        })
        .collect()
}

#[test]
#[ignore = "needs the corpora: TURBO_THAI_PUD and TURBO_MYPOS (and optional TURBO_KHPOS, TURBO_LAO_POS)"]
fn dictionary_and_lstm_against_gold() {
    let thai = std::fs::read_to_string(std::env::var("TURBO_THAI_PUD").expect("TURBO_THAI_PUD")).unwrap();
    let thai_sents = conllu(&thai);
    let thai_dict = score_with(bundled_dictionaries(), "th_pud", "dict", &thai_sents, None);
    let thai_lstm = score_with(bundled_lstms(), "th_pud", "lstm", &thai_sents, None);
    let thai_def = score_with(bundled(), "th_pud", "default", &thai_sents, Some(ComplexScript::Thai));
    assert!(thai_lstm.f1() > thai_dict.f1());
    assert_eq!(thai_def, thai_lstm);

    let burmese = std::fs::read_to_string(std::env::var("TURBO_MYPOS").expect("TURBO_MYPOS")).unwrap();
    let burmese_sents = mypos(&burmese);
    let burmese_dict = score_with(bundled_dictionaries(), "mypos", "dict", &burmese_sents, None);
    let burmese_lstm = score_with(bundled_lstms(), "mypos", "lstm", &burmese_sents, None);
    let burmese_def = score_with(bundled(), "mypos", "default", &burmese_sents, Some(ComplexScript::Myanmar));
    let burmese_dict_my =
        score_with(bundled_dictionaries(), "mypos_my", "dict", &burmese_sents, Some(ComplexScript::Myanmar));
    assert!(burmese_dict.f1() > burmese_lstm.f1());
    assert_eq!(burmese_def, burmese_dict_my);

    // UD_Thai-PUD commit 670083cd185a2ae215ce7aef1127c2b63640217d and
    // myPOS commit f1a620ba271cae3d8658441eb626450fca09cdfa: the numbers in
    // docs/segmentation.md.
    assert_eq!(thai_dict, Score { sentences: 1000, gold: 21640, returned: 24761, correct: 17095 });
    assert_eq!(thai_lstm, Score { sentences: 1000, gold: 21640, returned: 23372, correct: 17847 });
    assert_eq!(burmese_dict, Score { sentences: 43196, gold: 468105, returned: 558677, correct: 358106 });
    assert_eq!(burmese_lstm, Score { sentences: 43196, gold: 468105, returned: 558493, correct: 297365 });

    if let Ok(kh_path) = std::env::var("TURBO_KHPOS") {
        let khmer = std::fs::read_to_string(kh_path).unwrap();
        let kh_sents = khpos(&khmer);
        let kh_dict = score_with(bundled_dictionaries(), "khpos", "dict", &kh_sents, None);
        let kh_lstm = score_with(bundled_lstms(), "khpos", "lstm", &kh_sents, None);
        let kh_def = score_with(bundled(), "khpos", "default", &kh_sents, Some(ComplexScript::Khmer));
        assert!(kh_dict.f1() > kh_lstm.f1());
        assert_eq!(kh_def, kh_dict);
        // ye-kyaw-thu/khPOS commit 0211405c145b0a340ad558569bd06c946b2296b4
        assert_eq!(kh_dict, Score { sentences: 12000, gold: 121294, returned: 125258, correct: 92253 });
        assert_eq!(kh_lstm, Score { sentences: 12000, gold: 121294, returned: 122853, correct: 85097 });
    }

    if let Ok(lao_path) = std::env::var("TURBO_LAO_POS") {
        let lao = std::fs::read_to_string(lao_path).unwrap();
        let lao_sents = lao_pos(&lao);
        let lao_dict = score_with(bundled_dictionaries(), "lao_pos", "dict", &lao_sents, None);
        let lao_lstm = score_with(bundled_lstms(), "lao_pos", "lstm", &lao_sents, None);
        let lao_def = score_with(bundled(), "lao_pos", "default", &lao_sents, Some(ComplexScript::Lao));
        let lao_dict_lo =
            score_with(bundled_dictionaries(), "lao_pos_lo", "dict", &lao_sents, Some(ComplexScript::Lao));
        assert!(lao_dict.f1() > lao_lstm.f1());
        assert_eq!(lao_def, lao_dict_lo);
        // ye-kyaw-thu/SeqLabeling commit 8025a122e95422901e0907f0fc8ca4b5c1c34235
        assert_eq!(lao_dict, Score { sentences: 4613, gold: 64882, returned: 73125, correct: 45471 });
        assert_eq!(lao_lstm, Score { sentences: 4613, gold: 64882, returned: 74433, correct: 44019 });
    }
}
