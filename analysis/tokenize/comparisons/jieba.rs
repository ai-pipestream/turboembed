use std::{collections::HashSet, env, fs, hint::black_box, io::Cursor, time::Instant};
use turbo_tokenize::{UnigramLexicon, UnigramMode, UnigramSegmenter, WhitespaceMode, unigram::Scratch};
fn main() {
    let args: Vec<_> = env::args().collect();
    let dict = fs::read(&args[1]).unwrap();
    let jieba = jieba_rs::Jieba::with_dict(&mut Cursor::new(&dict)).unwrap();
    // Keep one loaded model for all passes; a new dictionary is not needed per line.
    let lex = UnigramLexicon::parse(&dict, WhitespaceMode::Unicode).unwrap();
    let modes = [UnigramMode::OpenNlp, UnigramMode::Dictionary, UnigramMode::Chinese];
    let models: Vec<_> =
        modes.iter().map(|&m| UnigramSegmenter::new(lex.clone(), WhitespaceMode::Unicode).with_mode(m)).collect();
    for sample in ["小明硕士毕业于中国科学院计算所", "他来到了网易杭研大厦", "abc123北京2026年", "𠀀杭研🙂"]
    {
        println!("sample {sample} jieba {:?}", jieba.cut(sample, true));
        for s in &models {
            let mut out = Vec::new();
            s.segment(sample, &mut out);
            println!("  {:?} {:?}", s.mode(), out.iter().map(|p| &sample[p.range()]).collect::<Vec<_>>());
        }
    }
    if args.get(3).is_some_and(|a| a == "samples") {
        return;
    }
    for corpus in ["pku", "msr"] {
        let raw = fs::read_to_string(format!("{}/testing/{corpus}_test.utf8", args[2])).unwrap();
        let gold = fs::read_to_string(format!("{}/gold/{corpus}_test_gold.utf8", args[2])).unwrap();
        let train = fs::read_to_string(format!("{}/gold/{corpus}_training_words.utf8", args[2])).unwrap();
        let known: HashSet<_> = train.lines().map(str::trim).collect();
        let expected = word_spans(gold.split_whitespace());
        let mut scratch = Scratch::new();
        let mut spans = Vec::new();
        for which in 0..5 {
            let mut words = Vec::new();
            for line in raw.lines() {
                if let Some(model) = models.get(which) {
                    spans.clear();
                    model.segment_with(line, &mut scratch, &mut spans);
                    words.extend(spans.iter().map(|s| &line[s.range()]));
                } else {
                    words.extend(
                        jieba
                            .cut(line, which == 4)
                            .into_iter()
                            .map(|t| t.word)
                            .filter(|w| !w.chars().all(|c| WhitespaceMode::Unicode.is_whitespace(c as u32))),
                    );
                }
            }
            let found = word_spans(words.iter().copied());
            assert_eq!(words.concat(), gold.split_whitespace().collect::<String>());
            let positions: HashSet<_> = found.iter().map(|&(a, b, _)| (a, b)).collect();
            let correct = expected.iter().filter(|&&(a, b, _)| positions.contains(&(a, b))).count();
            let oov = expected.iter().filter(|&&(_, _, w)| !known.contains(w)).count();
            let oov_correct =
                expected.iter().filter(|&&(a, b, w)| !known.contains(w) && positions.contains(&(a, b))).count();
            println!(
                "{corpus} {which} gold={} returned={} correct={correct} oov={oov} oov_correct={oov_correct} P={:.9} R={:.9} F1={:.9}",
                expected.len(),
                found.len(),
                correct as f64 / found.len() as f64,
                correct as f64 / expected.len() as f64,
                2.0 * correct as f64 / (expected.len() + found.len()) as f64
            );
        }
        // One warmup then eight alternating passes, excluding model/file load and scoring.
        let mut times: Vec<Vec<f64>> = vec![Vec::new(); 5];
        for pass in 0..9 {
            for n in 0..5 {
                let which = if pass % 2 == 0 { n } else { 4 - n };
                let start = Instant::now();
                let mut count = 0;
                for _ in 0..50 {
                    for line in raw.lines() {
                        if let Some(model) = models.get(which) {
                            spans.clear();
                            model.segment_with(black_box(line), &mut scratch, &mut spans);
                            count += black_box(&spans).len();
                        } else {
                            count += black_box(jieba.cut(black_box(line), which == 4)).len();
                        }
                    }
                }
                black_box(count);
                if pass > 0 {
                    times[which].push(start.elapsed().as_secs_f64() * 1000.0 / 50.0);
                }
            }
        }
        for (which, values) in times.iter().enumerate() {
            println!("timing {corpus} {which} {values:?}");
        }
    }
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
