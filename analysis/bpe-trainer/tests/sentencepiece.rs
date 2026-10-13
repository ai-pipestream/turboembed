//! The SentencePiece trainer against the sentencepiece library 0.2.2,
//! whose models for the same corpora and settings are committed as
//! testdata/sentencepiece.jsonl (testdata/sentencepiece.py wrote it),
//! and the trained `.model` through the core's SentencePiece BPE.

mod common;

use common::*;
use turbo::sentencepiece::{Algorithm, PieceType, Scratch, SentencePiece};
use turbo_bpe_trainer::sentencepiece::{PieceKind, normalize};
use turbo_bpe_trainer::{SentencePieceBpe, SentencePieceConfig, TrainError};

fn flat(t: &str) -> String {
    t.replace(['\r', '\n'], " ")
}

/// The corpora by the names sentencepiece.py gives them, read as it
/// reads them.
fn corpus(name: &str) -> Vec<String> {
    match name {
        "tokenizer-texts" => tokenizer_texts().iter().map(|t| flat(t)).collect(),
        "tiny-bpe" => jsonl_strings(&testdata().join("tiny-bpe/corpus.jsonl")).iter().map(|t| flat(t)).collect(),
        "edge" => jsonl_strings(&repo().join("analysis/bpe-trainer/testdata/edge.jsonl")),
        "bpe-inputs" => {
            bpe_inputs().iter().flat_map(|(_, t)| t.split('\n').map(str::to_owned).collect::<Vec<_>>()).collect()
        }
        _ => panic!("no corpus {name}"),
    }
}

/// One run of the library.
struct Run {
    corpus: String,
    config: SentencePieceConfig,
    outcome: Result<Vec<(String, f32, u64)>, TrainError>,
}

/// The library's errors, as the trainer's.
fn error(message: &str) -> TrainError {
    let numbers: Vec<usize> =
        message.split(|c: char| !c.is_ascii_digit()).filter(|s| !s.is_empty()).map(|s| s.parse().unwrap()).collect();
    // After the source line: "... smaller than required_chars. 120 vs 421."
    // or "... too high (100000). Please set it to a value <= 1006."
    let n = numbers.len();
    if message.contains("smaller than required_chars") {
        TrainError::VocabTooSmall { asked: numbers[n - 2], least: numbers[n - 1] }
    } else if message.contains("too high") {
        TrainError::VocabTooLarge { asked: numbers[n - 2], most: numbers[n - 1] }
    } else {
        panic!("{message}")
    }
}

fn runs() -> Vec<Run> {
    let file = std::fs::read_to_string(repo().join("analysis/bpe-trainer/testdata/sentencepiece.jsonl")).unwrap();
    file.lines()
        .map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).unwrap();
            let config = SentencePieceConfig {
                vocab_size: v["vocab_size"].as_u64().unwrap() as usize,
                character_coverage: v["character_coverage"].as_f64().unwrap() as f32,
            };
            let outcome = match v.get("pieces") {
                Some(p) => Ok(p
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| (p[0].as_str().unwrap().to_owned(), p[1].as_f64().unwrap() as f32, p[2].as_u64().unwrap()))
                    .collect()),
                None => Err(error(v["error"].as_str().unwrap())),
            };
            Run { corpus: v["corpus"].as_str().unwrap().to_owned(), config, outcome }
        })
        .collect()
}

fn ours(run: &Run) -> Result<Vec<(String, f32, u64)>, TrainError> {
    let lines = corpus(&run.corpus);
    let m = SentencePieceBpe::train(lines.iter().map(String::as_str), run.config.clone())?;
    Ok(m.pieces().iter().map(|p| (p.piece.clone(), p.score, p.kind as u64)).collect())
}

/// Every run of the library: the same pieces in the same order with the
/// same scores and types, or the same error with the same numbers.
#[test]
fn every_run_gives_the_librarys_pieces() {
    let runs = runs();
    let mut models = 0;
    let mut pieces = 0;
    for run in &runs {
        let got = ours(run);
        match (&got, &run.outcome) {
            (Ok(a), Ok(b)) => {
                let at = a.iter().zip(b).position(|(x, y)| x != y);
                assert!(
                    a == b,
                    "{} {:?}: {} pieces, the library {}; first difference at {at:?}: {:?} {:?}",
                    run.corpus,
                    run.config,
                    a.len(),
                    b.len(),
                    at.map(|i| &a[i]),
                    at.map(|i| &b[i]),
                );
                models += 1;
                pieces += a.len();
            }
            (Err(a), Err(b)) => assert_eq!(a, b, "{} {:?}", run.corpus, run.config),
            _ => panic!(
                "{} {:?}: {:?}, the library {:?}",
                run.corpus,
                run.config,
                got.as_ref().err(),
                run.outcome.as_ref().err()
            ),
        }
    }
    assert_eq!(runs.len(), 60);
    assert!(models >= 25 && pieces > 40_000, "{models} models, {pieces} pieces");
}

/// The trained file loads in the core as a BPE model with the trained
/// pieces, and each line of the corpus that holds no boundary decodes from its
/// ids to the text as the library normalizes it, the `▁` in front taken
/// off and the others spaces again.
#[test]
fn the_trained_model_loads_in_the_core_and_round_trips() {
    for (name, size) in [("tokenizer-texts", 3000), ("tiny-bpe", 1100), ("bpe-inputs", 900)] {
        let lines = corpus(name);
        let config = SentencePieceConfig { vocab_size: size, character_coverage: 1.0 };
        let m = SentencePieceBpe::train(lines.iter().map(String::as_str), config).unwrap();
        let sp = SentencePiece::parse(&m.to_model_proto()).unwrap();
        assert_eq!(sp.algorithm(), Algorithm::Bpe);
        assert_eq!(sp.len(), size);
        assert_eq!(sp.unk_id(), 0);
        assert!(sp.pieces().eq(m.pieces().iter().map(|p| p.piece.as_str())));
        for (id, p) in m.pieces().iter().enumerate() {
            let want = match p.kind {
                PieceKind::Normal => PieceType::Normal,
                PieceKind::Unknown => PieceType::Unknown,
                PieceKind::Control => PieceType::Control,
            };
            assert_eq!(sp.kind(id as u32), Some(want));
        }

        let mut scratch = Scratch::default();
        let mut normalized = turbo::sentencepiece::Normalized::default();
        let mut ids = Vec::new();
        let mut checked = 0;
        for line in &lines {
            // U+2585 and the meta pieces leave a line out of training, a
            // tab or a meta piece is a boundary, and NUL is never a piece.
            if line.is_empty()
                || line.contains(['\u{2585}', '\t', '\0'])
                || ["<unk>", "<s>", "</s>"].iter().any(|m| line.contains(m))
            {
                continue;
            }
            sp.normalize(line, &mut normalized);
            let n = normalize(line);
            assert_eq!(String::from_utf8(normalized.text.clone()).unwrap(), n, "{line:?}");
            ids.clear();
            sp.encode_ids(line, &mut scratch, &mut ids);
            assert!(!ids.contains(&0), "{name}: an unknown piece in {line:?}");
            let back = sp.decode(&ids).unwrap();
            assert_eq!(back, n.trim_start_matches('\u{2581}').replace('\u{2581}', " "), "{name}: {line:?}");
            checked += 1;
        }
        assert!(checked > 20, "{name}: {checked}");
    }
}

/// Characters left out by the coverage are unknown: they are no piece,
/// are in no merged piece, and encode to `<unk>`.
#[test]
fn characters_the_coverage_leaves_out_are_unknown() {
    let lines = corpus("tokenizer-texts");
    let config = SentencePieceConfig { vocab_size: 1000, character_coverage: 0.98 };
    let m = SentencePieceBpe::train(lines.iter().map(String::as_str), config).unwrap();
    let sp = SentencePiece::parse(&m.to_model_proto()).unwrap();
    let single: std::collections::HashSet<char> = m
        .pieces()
        .iter()
        .skip(3)
        .filter_map(|p| p.piece.chars().nth(1).is_none().then(|| p.piece.chars().next().unwrap()))
        .collect();
    let mut left_out = 0;
    for line in &lines {
        for c in line.chars().filter(|c| !c.is_whitespace() && !single.contains(c)) {
            assert!(m.pieces().iter().all(|p| !p.piece.contains(c)), "{c:?}");
            let mut ids = Vec::new();
            sp.encode_ids(&c.to_string(), &mut Scratch::default(), &mut ids);
            assert!(ids.contains(&0), "{c:?}: {ids:?}");
            left_out += 1;
        }
    }
    assert!(left_out > 10, "{left_out}");
}

/// Training twice gives the same file, and so does the corpus in another
/// order: ties are broken by the pieces, not by the order the words were
/// seen in.
#[test]
fn training_is_deterministic_and_ignores_the_order_of_the_texts() {
    let mut lines = corpus("tokenizer-texts");
    lines.extend(corpus("tiny-bpe"));
    let config = SentencePieceConfig { vocab_size: 4000, character_coverage: 0.9995 };
    let a = SentencePieceBpe::train(lines.iter().map(String::as_str), config.clone()).unwrap();
    let b = SentencePieceBpe::train(lines.iter().map(String::as_str), config.clone()).unwrap();
    let c = SentencePieceBpe::train(lines.iter().rev().map(String::as_str), config).unwrap();
    assert_eq!(a.to_model_proto(), b.to_model_proto());
    assert_eq!(a.to_model_proto(), c.to_model_proto());
}

/// Small random corpora at the largest size the library allows for each
/// (testdata/sentencepiece-random.jsonl): the same pieces, so the same
/// merges and, once no pair occurs, the same pairs at count 0.
#[test]
fn random_corpora_at_the_largest_size_give_the_librarys_pieces() {
    let file =
        std::fs::read_to_string(repo().join("analysis/bpe-trainer/testdata/sentencepiece-random.jsonl")).unwrap();
    let mut runs = 0;
    for line in file.lines() {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        let strings = |k: &str| -> Vec<String> {
            v[k].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_owned()).collect()
        };
        let (lines, want) = (strings("lines"), strings("pieces"));
        let config = SentencePieceConfig { vocab_size: want.len(), character_coverage: 1.0 };
        let m = SentencePieceBpe::train(lines.iter().map(String::as_str), config).unwrap();
        let got: Vec<&str> = m.pieces().iter().map(|p| p.piece.as_str()).collect();
        assert_eq!(got, want, "{lines:?}");
        // One more piece is more than the corpus allows.
        let config = SentencePieceConfig { vocab_size: want.len() + 1, character_coverage: 1.0 };
        assert!(matches!(
            SentencePieceBpe::train(lines.iter().map(String::as_str), config),
            Err(TrainError::VocabTooLarge { most, .. }) if most == want.len()
        ));
        runs += 1;
    }
    assert_eq!(runs, 300);
}

/// The trainer reads scripts from current Unicode, the library from an
/// older table in which a character assigned since is Common. The
/// library's pieces for each corpus are written out here, from the
/// Python package 0.2.2 at the largest size it allows.
#[test]
fn scripts_are_current_unicodes() {
    let train = |text: &str, size| {
        let c = SentencePieceConfig { vocab_size: size, character_coverage: 1.0 };
        let m = SentencePieceBpe::train([text], c);
        m.map(|m| m.pieces().iter().skip(3).map(|p| p.piece.clone()).collect::<Vec<_>>())
    };
    // U+A7C0, a Latin letter since Unicode 14: Common to the library, so
    // "aꟀ" mixes two scripts there and is one script here. The library
    // allows 7 pieces: ["▁a", "a", "▁", "Ꟁ"] after the meta pieces.
    assert_eq!(train("aꟀ aꟀ", 9).unwrap(), ["aꟀ", "▁aꟀ", "▁a", "a", "▁", "Ꟁ"]);
    // U+10570, Vithkuqi since Unicode 14: Common to the library, which so
    // joins it with a comma and allows 9 pieces: ["▁𐕰", "▁𐕰,", "𐕰,",
    // ",", "▁", "𐕰"].
    assert!(matches!(train("𐕰, 𐕰,", 9), Err(TrainError::VocabTooLarge { asked: 9, most: 7 })));
    assert_eq!(train("𐕰, 𐕰,", 7).unwrap(), ["▁𐕰", ",", "▁", "𐕰"]);
}
