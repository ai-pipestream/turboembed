//! The unigram segmenter with jieba's dictionary on the PKU and MSR test
//! sets of the SIGHAN 2005 bakeoff, scored as the bakeoff's `score` script
//! scores them: a word counts as correct when the gold file has a word
//! over exactly the same characters, and recall on words outside the
//! training word list is the out-of-vocabulary recall.
//!
//! Neither the dictionary nor the bakeoff data is in the repository (the
//! bakeoff data is for research use only), so the test is ignored and
//! needs both:
//!
//! ```text
//! TURBO_JIEBA_DICT=<jieba>/jieba/dict.txt TURBO_SIGHAN_DIR=<icwb2-data> \
//!   cargo test --release -p turbo-tokenize --test unigram_sighan -- --ignored --nocapture
//! ```
//!
//! docs/segmentation.md has the numbers and the dictionary and data
//! versions they were made with.

use std::collections::HashSet;
use std::path::PathBuf;

use turbo_tokenize::unigram::Scratch;
use turbo_tokenize::{UnigramLexicon, UnigramMode, UnigramSegmenter, WhitespaceMode};

/// Counts of one test set: gold words, words returned, correct words, and
/// the out-of-vocabulary gold words and how many of them were found.
#[derive(Debug, PartialEq, Eq)]
struct Score {
    gold: usize,
    returned: usize,
    correct: usize,
    oov: usize,
    oov_correct: usize,
}

impl Score {
    fn f1(&self) -> f64 {
        let p = self.correct as f64 / self.returned as f64;
        let r = self.correct as f64 / self.gold as f64;
        2.0 * p * r / (p + r)
    }
}

/// Spans of `words` over the characters of their concatenation.
fn spans<'a>(words: impl Iterator<Item = &'a str>) -> Vec<(usize, usize, &'a str)> {
    let mut at = 0;
    words
        .map(|w| {
            let n = w.chars().count();
            at += n;
            (at - n, at, w)
        })
        .collect()
}

fn score(segmenter: &UnigramSegmenter, dir: &std::path::Path, corpus: &str) -> Score {
    let read = |p: PathBuf| std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let test = read(dir.join(format!("testing/{corpus}_test.utf8")));
    let gold = read(dir.join(format!("gold/{corpus}_test_gold.utf8")));
    let training = read(dir.join(format!("gold/{corpus}_training_words.utf8")));
    let known: HashSet<&str> = training.lines().map(str::trim).collect();

    let mut words: Vec<&str> = Vec::new();
    let mut scratch = Scratch::new();
    let mut out = Vec::new();
    for line in test.lines() {
        out.clear();
        segmenter.segment_with(line, &mut scratch, &mut out);
        words.extend(out.iter().map(|s| &line[s.range()]));
    }
    // Whole-file spans: the gold file of MSR breaks a few lines elsewhere
    // than the test file, which the characters do not see.
    let returned = spans(words.iter().copied());
    let expected = spans(gold.split_whitespace());
    assert_eq!(words.concat(), gold.split_whitespace().collect::<String>(), "{corpus}: identical text");
    let found: HashSet<(usize, usize)> = returned.iter().map(|&(s, e, _)| (s, e)).collect();
    let mut s = Score { gold: expected.len(), returned: returned.len(), correct: 0, oov: 0, oov_correct: 0 };
    for &(start, end, w) in &expected {
        let hit = found.contains(&(start, end));
        s.correct += usize::from(hit);
        if !known.contains(w) {
            s.oov += 1;
            s.oov_correct += usize::from(hit);
        }
    }
    s
}

#[test]
#[ignore = "needs jieba's dict.txt and the SIGHAN 2005 data, which are not in the repository"]
fn pku_and_msr() {
    let (Some(dict), Some(dir)) = (std::env::var_os("TURBO_JIEBA_DICT"), std::env::var_os("TURBO_SIGHAN_DIR")) else {
        panic!("set TURBO_JIEBA_DICT and TURBO_SIGHAN_DIR");
    };
    let lexicon = UnigramLexicon::read(dict, WhitespaceMode::Unicode).unwrap();
    let dir = PathBuf::from(dir);
    // All modes use the same unmodified pinned dictionary and test text.
    let counts = [
        (UnigramMode::OpenNlp, [(102198, 84241, 2519), (108642, 89087, 955)]),
        (UnigramMode::Dictionary, [(98986, 84985, 3262), (108642, 89087, 955)]),
        (UnigramMode::Chinese, [(96562, 82137, 3481), (106224, 86799, 1271)]),
    ];
    for (mode, expected) in counts {
        let segmenter = UnigramSegmenter::new(lexicon.clone(), WhitespaceMode::Unicode).with_mode(mode);
        for ((corpus, gold, oov), (returned, correct, oov_correct)) in
            [("pku", 104372, 6006), ("msr", 106873, 2829)].into_iter().zip(expected)
        {
            let s = score(&segmenter, &dir, corpus);
            println!(
                "{corpus} {mode:?}: P {:.9} R {:.9} F1 {:.9} R_oov {:.9} ({s:?})",
                s.correct as f64 / s.returned as f64,
                s.correct as f64 / s.gold as f64,
                s.f1(),
                s.oov_correct as f64 / s.oov as f64
            );
            assert_eq!(s, Score { gold, returned, correct, oov, oov_correct }, "{corpus} {mode:?}");
        }
    }
}
