use std::{collections::HashSet, env, fs, hint::black_box, io::Cursor, time::Instant};
#[cfg(feature = "policies")]
use turbo_tokenize::UnigramMode;
use turbo_tokenize::{UnigramLexicon, UnigramSegmenter, WhitespaceMode, unigram::Scratch};

mod output_hash;

fn main() {
    let args: Vec<_> = env::args().collect();
    let dict = fs::read(&args[1]).unwrap();
    let jieba = jieba_rs::Jieba::with_dict(&mut Cursor::new(&dict)).unwrap();
    let lex = UnigramLexicon::parse(&dict, WhitespaceMode::Unicode).unwrap();
    #[cfg(feature = "policies")]
    let models: Vec<_> = [UnigramMode::OpenNlp, UnigramMode::Dictionary, UnigramMode::Chinese]
        .into_iter()
        .map(|m| UnigramSegmenter::new(lex.clone(), WhitespaceMode::Unicode).with_mode(m))
        .collect();
    #[cfg(not(feature = "policies"))]
    let models = vec![UnigramSegmenter::new(lex, WhitespaceMode::Unicode)];
    #[cfg(feature = "main-baseline")]
    let baseline = turbo_tokenize_main::UnigramSegmenter::new(
        turbo_tokenize_main::UnigramLexicon::parse(&dict, WhitespaceMode::Unicode).unwrap(),
        WhitespaceMode::Unicode,
    );
    #[cfg(feature = "main-baseline")]
    let mut baseline_scratch = turbo_tokenize_main::unigram::Scratch::new();
    #[cfg(feature = "main-baseline")]
    let names = ["opennlp", "dictionary", "chinese", "main-source", "jieba-dictionary", "jieba-hmm"];
    #[cfg(all(feature = "policies", not(feature = "main-baseline")))]
    let names = ["opennlp", "dictionary", "chinese", "jieba-dictionary", "jieba-hmm"];
    #[cfg(not(feature = "policies"))]
    let names = ["main", "jieba-dictionary", "jieba-hmm"];
    let passes: usize = args[3].parse().unwrap();
    let reps: usize = args[4].parse().unwrap();
    for corpus in ["pku", "msr"] {
        let raw = fs::read_to_string(format!("{}/testing/{corpus}_test.utf8", args[2])).unwrap();
        let gold = fs::read_to_string(format!("{}/gold/{corpus}_test_gold.utf8", args[2])).unwrap();
        let train = fs::read_to_string(format!("{}/gold/{corpus}_training_words.utf8", args[2])).unwrap();
        let known: HashSet<_> = train.lines().map(str::trim).collect();
        let expected = word_spans(gold.split_whitespace());
        let mut scratch = Scratch::new();
        let mut spans = Vec::new();
        for (which, name) in names.iter().enumerate() {
            let mut words = Vec::new();
            let mut hash = turbo_analysis::Fnv1a::new().finish();
            for line in raw.lines() {
                let tokens: Vec<_> = if let Some(model) = models.get(which) {
                    spans.clear();
                    model.segment_with(line, &mut scratch, &mut spans);
                    spans.iter().map(|s| &line[s.range()]).collect()
                } else {
                    #[cfg(feature = "main-baseline")]
                    {
                        if which == models.len() {
                            spans.clear();
                            baseline.segment_with(line, &mut baseline_scratch, &mut spans);
                            spans.iter().map(|s| &line[s.range()]).collect()
                        } else {
                            jieba_words(&jieba, line, which == names.len() - 1)
                        }
                    }
                    #[cfg(not(feature = "main-baseline"))]
                    {
                        jieba_words(&jieba, line, which == names.len() - 1)
                    }
                };
                hash = output_hash::feed_line(hash, &tokens);
                words.extend(tokens);
            }
            assert_eq!(words.concat(), gold.split_whitespace().collect::<String>());
            let found = word_spans(words.iter().copied());
            let positions: HashSet<_> = found.iter().map(|&(a, b, _)| (a, b)).collect();
            let correct = expected.iter().filter(|&&(a, b, _)| positions.contains(&(a, b))).count();
            let oov = expected.iter().filter(|&&(_, _, w)| !known.contains(w)).count();
            let oov_correct =
                expected.iter().filter(|&&(a, b, w)| !known.contains(w) && positions.contains(&(a, b))).count();
            println!(
                "score {corpus} {name} gold={} returned={} correct={correct} oov={oov} oov_correct={oov_correct} output_fnv64={hash:016x}",
                expected.len(),
                found.len()
            );
        }
        for pass in 0..=passes {
            for n in 0..names.len() {
                let which = if pass % 2 == 0 { n } else { names.len() - 1 - n };
                let start = Instant::now();
                let mut count = 0;
                for _ in 0..reps {
                    for line in raw.lines() {
                        if let Some(model) = models.get(which) {
                            spans.clear();
                            model.segment_with(black_box(line), &mut scratch, &mut spans);
                            count += black_box(&spans).len();
                        } else {
                            #[cfg(feature = "main-baseline")]
                            if which == models.len() {
                                spans.clear();
                                baseline.segment_with(black_box(line), &mut baseline_scratch, &mut spans);
                                count += black_box(&spans).len();
                                continue;
                            }
                            count += black_box(jieba.cut(black_box(line), which == names.len() - 1)).len();
                        }
                    }
                }
                black_box(count);
                if pass > 0 {
                    println!(
                        "timing {corpus} {} pass={pass} ms={:.9} count={count}",
                        names[which],
                        start.elapsed().as_secs_f64() * 1000.0 / reps as f64
                    );
                }
            }
        }
    }
}

fn jieba_words<'a>(jieba: &'a jieba_rs::Jieba, line: &'a str, hmm: bool) -> Vec<&'a str> {
    jieba
        .cut(line, hmm)
        .into_iter()
        .map(|t| t.word)
        .filter(|w| !w.chars().all(|c| WhitespaceMode::Unicode.is_whitespace(c as u32)))
        .collect()
}

fn word_spans<'a>(words: impl Iterator<Item = &'a str>) -> Vec<(usize, usize, &'a str)> {
    let mut at = 0;
    words
        .map(|w| {
            let a = at;
            at += w.chars().count();
            (a, at, w)
        })
        .collect()
}
