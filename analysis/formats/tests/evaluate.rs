//! The evaluators against the JVM: every `evaluate` golden of
//! `analysis/goldens` is what one of OpenNLP's evaluators made of a file
//! of sample pairs (a reference, then the prediction for it) from the
//! `formats` input set, with a stand-in component answering each
//! reference with its prediction. Each test reads the same pairs with the
//! readers here, gives each evaluator the same predictions, and rebuilds
//! the verdicts, the scores after every pair (compared bit for bit), the
//! static precision and recall of each pair's spans, and the folds of
//! cross-validation.

use std::collections::BTreeMap;
use std::path::PathBuf;

use turbo_analysis::Span;
use turbo_formats::evaluate::{
    ChunkerEvaluator, CrossValidation, DocumentCategorizerEvaluator, EvalError, FMeasure, LabeledSpan,
    LanguageDetectorEvaluator, LemmatizerEvaluator, NameFinderEvaluator, PosEvaluator, SentenceDetectorEvaluator,
    TokenizerEvaluator, Verdict, precision, recall,
};
use turbo_formats::native::{
    ChunkReader, DocumentReader, LanguageReader, LemmaReader, NameReader, SentenceReader, TokenReader, WordTagReader,
};
use turbo_formats::{Field, FormatError};
use turbo_goldens::{ColumnData, ColumnSpec, ColumnType, Golden, GoldenAnchor, GoldenLayer, Shape};
use turbo_unicode::WhitespaceMode;

const U32: ColumnType = ColumnType::U32;
const F64: ColumnType = ColumnType::F64;
const STR: ColumnType = ColumnType::Str;

fn shapes() -> Vec<Shape> {
    const VERDICTS: &[ColumnSpec] = &[
        ColumnSpec::new("sample", U32),
        ColumnSpec::new("verdict", STR),
        ColumnSpec::new("class", STR),
        ColumnSpec::new("message", STR),
    ];
    const FMEASURE: &[ColumnSpec] = &[
        ColumnSpec::new("after", U32),
        ColumnSpec::new("precision", F64),
        ColumnSpec::new("recall", F64),
        ColumnSpec::new("fmeasure", F64),
    ];
    const ACCURACY: &[ColumnSpec] =
        &[ColumnSpec::new("after", U32), ColumnSpec::new("accuracy", F64), ColumnSpec::new("count", U32)];
    const SPAN_SCORES: &[ColumnSpec] =
        &[ColumnSpec::new("sample", U32), ColumnSpec::new("precision", F64), ColumnSpec::new("recall", F64)];
    const FOLDS: &[ColumnSpec] = &[
        ColumnSpec::new("partitions", U32),
        ColumnSpec::new("fold", U32),
        ColumnSpec::new("part", STR),
        ColumnSpec::new("sample", U32),
    ];
    let doc = |id: &str, columns| Shape { id: id.into(), anchor: GoldenAnchor::Document, columns };
    vec![
        doc("turbo:verdicts", VERDICTS),
        doc("turbo:fmeasure", FMEASURE),
        doc("turbo:accuracy", ACCURACY),
        doc("turbo:span-scores", SPAN_SCORES),
        doc("turbo:folds", FOLDS),
    ]
}

/// A cell written as text; an `f64` as its bits, so the comparison is
/// exact (NaN included).
fn cells(layer: &GoldenLayer) -> Vec<Vec<String>> {
    (0..layer.len())
        .map(|i| {
            layer
                .columns
                .iter()
                .map(|c| match &c.data {
                    ColumnData::U32(v) => v[i].to_string(),
                    ColumnData::Str(v) => v.get(i).to_owned(),
                    ColumnData::F64(v) => format!("{:016x}", v[i].to_bits()),
                })
                .collect()
        })
        .collect()
}

fn bits(x: f64) -> String {
    format!("{:016x}", x.to_bits())
}

#[derive(Default)]
struct Rebuilt(BTreeMap<&'static str, Vec<Vec<String>>>);

impl Rebuilt {
    fn row(&mut self, id: &'static str, cells: Vec<String>) {
        self.0.entry(id).or_default().push(cells);
    }

    fn verdict(&mut self, pair: usize, verdict: Result<Verdict, EvalError>) {
        let cells = match verdict {
            Ok(Verdict::Correct) => vec!["correct".into(), String::new(), String::new()],
            Ok(Verdict::Misclassified) => vec!["misclassified".into(), String::new(), String::new()],
            Err(e) => vec!["error".into(), e.java_class().into(), e.to_string()],
        };
        let mut row = vec![pair.to_string()];
        row.extend(cells);
        self.row("turbo:verdicts", row);
    }

    fn fmeasure(&mut self, pair: usize, f: &FMeasure) {
        self.row("turbo:fmeasure", vec![pair.to_string(), bits(f.precision()), bits(f.recall()), bits(f.f_measure())]);
    }

    fn accuracy(&mut self, pair: usize, accuracy: f64, count: u64) {
        self.row("turbo:accuracy", vec![pair.to_string(), bits(accuracy), count.to_string()]);
    }

    fn span_scores<T: PartialEq>(&mut self, pair: usize, references: &[T], predictions: &[T]) {
        let (p, r) = (precision(references, predictions), recall(references, predictions));
        self.row("turbo:span-scores", vec![pair.to_string(), bits(p), bits(r)]);
    }
}

/// The samples of a reader, in pairs.
fn pairs<S>(items: impl Iterator<Item = Result<S, FormatError>>) -> Vec<(S, S)> {
    let samples: Vec<S> = items.map(|s| s.expect("the pairs read")).collect();
    assert_eq!(samples.len() % 2, 0, "an odd number of samples");
    let mut it = samples.into_iter();
    std::iter::from_fn(|| Some((it.next()?, it.next()?))).collect()
}

fn texts<'f>(fields: &'f [Field<'_>]) -> Vec<&'f str> {
    fields.iter().map(Field::as_str).collect()
}

/// A byte span of `from` as Java applies it to `to`: the same UTF-16
/// offsets, past `to`'s end by as many units as it is.
fn remap(span: Span, from: &str, to: &str) -> Span {
    let unit = |b: u32| from[..b as usize].encode_utf16().count();
    let byte = |u: usize| {
        let mut seen = 0;
        for (i, c) in to.char_indices() {
            if seen >= u {
                return i;
            }
            seen += c.len_utf16();
        }
        to.len() + (u - seen.min(u))
    };
    Span::new(byte(unit(span.start)) as u32, byte(unit(span.end)) as u32)
}

fn rebuild(run: &str, input: &str) -> Rebuilt {
    let (kind, mode) = match run.strip_suffix("+legacy") {
        Some(k) => (k, WhitespaceMode::Legacy),
        None => (run, WhitespaceMode::Unicode),
    };
    let unicode = WhitespaceMode::Unicode;
    let mut out = Rebuilt::default();
    match kind {
        "pos" => {
            let mut e = PosEvaluator::default();
            for (i, (r, p)) in pairs(WordTagReader::new(input, unicode).unwrap()).iter().enumerate() {
                out.verdict(i, e.evaluate(r, &texts(p.tags())));
                out.accuracy(i, e.word_accuracy(), e.word_count());
            }
        }
        "lemma" => {
            let mut e = LemmatizerEvaluator::default();
            for (i, (r, p)) in pairs(LemmaReader::new(input).unwrap()).iter().enumerate() {
                out.verdict(i, e.evaluate(r, &texts(p.lemmas())));
                out.accuracy(i, e.word_accuracy(), e.word_count());
            }
        }
        "name" => {
            let mut e = NameFinderEvaluator::default();
            for (i, (r, p)) in pairs(NameReader::new(input, unicode).unwrap()).iter().enumerate() {
                out.verdict(i, e.evaluate(r, p.names()));
                out.fmeasure(i, e.fmeasure());
                let refs: Vec<LabeledSpan<'_>> = r.names().iter().map(LabeledSpan::from).collect();
                let preds: Vec<LabeledSpan<'_>> = p.names().iter().map(LabeledSpan::from).collect();
                out.span_scores(i, &refs, &preds);
            }
        }
        "chunk" => {
            let mut e = ChunkerEvaluator::default();
            for (i, (r, p)) in pairs(ChunkReader::new(input).unwrap()).iter().enumerate() {
                out.verdict(i, e.evaluate(r, &texts(p.chunks())));
                out.fmeasure(i, e.fmeasure());
            }
        }
        "token" => {
            let mut e = TokenizerEvaluator::default();
            for (i, (r, p)) in pairs(TokenReader::new(input, unicode).unwrap()).iter().enumerate() {
                let predicted: Vec<Span> = p.tokens().iter().map(|&s| remap(s, p.text(), r.text())).collect();
                out.verdict(i, e.evaluate(r, &predicted));
                out.fmeasure(i, e.fmeasure());
                out.span_scores(i, r.tokens(), &predicted);
            }
        }
        "sentence" => {
            let mut e = SentenceDetectorEvaluator::new(mode);
            for (i, (r, p)) in pairs(SentenceReader::new(input).unwrap()).iter().enumerate() {
                let predicted: Vec<Span> =
                    p.sentences().iter().map(|&s| remap(s, p.document(), r.document())).collect();
                out.verdict(i, e.evaluate(r, &predicted));
                out.fmeasure(i, e.fmeasure());
            }
        }
        "doccat" => {
            let mut e = DocumentCategorizerEvaluator::default();
            for (i, (r, p)) in pairs(DocumentReader::new(input, unicode).unwrap()).iter().enumerate() {
                out.verdict(i, Ok(e.evaluate(r, p.category.as_str())));
                out.accuracy(i, e.accuracy(), e.document_count());
            }
        }
        "language" => {
            let mut e = LanguageDetectorEvaluator::default();
            for (i, (r, p)) in pairs(LanguageReader::new(input).unwrap()).iter().enumerate() {
                out.verdict(i, Ok(e.evaluate(r, p.language.as_str())));
                out.accuracy(i, e.accuracy(), e.document_count());
            }
        }
        "cross-validation" => {
            let count = WordTagReader::new(input, unicode).unwrap().count();
            for partitions in 1..=4 {
                for fold in CrossValidation::new(count, partitions) {
                    for (part, places) in [("train", &fold.training), ("test", &fold.test)] {
                        for place in places {
                            out.row(
                                "turbo:folds",
                                vec![partitions.to_string(), fold.index.to_string(), part.into(), place.to_string()],
                            );
                        }
                    }
                }
            }
        }
        other => panic!("no evaluator for the run {other}"),
    }
    out
}

fn goldens() -> Vec<(String, PathBuf)> {
    let root = turbo_goldens::data_dir().join("evaluate");
    let mut out = Vec::new();
    for run in std::fs::read_dir(&root).unwrap() {
        let run = run.unwrap().path();
        let name = run.file_name().unwrap().to_str().unwrap().to_owned();
        for file in std::fs::read_dir(run.join("formats")).unwrap() {
            out.push((name.clone(), file.unwrap().path()));
        }
    }
    out.sort();
    out
}

#[test]
fn every_evaluator_golden_matches() {
    let data = turbo_goldens::data_dir();
    let shapes = shapes();
    let all = goldens();
    assert!(all.len() >= 10, "only {} evaluate goldens", all.len());
    for (run, path) in all {
        let golden = Golden::read_with(&path, &shapes).unwrap();
        let input = golden.input_text(&data).unwrap();
        let mut rebuilt = rebuild(&run, &input).0;
        for layer in &golden.layers {
            let ours = rebuilt.remove(layer.id.as_str()).unwrap_or_default();
            let theirs = cells(layer);
            for (i, (a, b)) in ours.iter().zip(&theirs).enumerate() {
                assert_eq!(a, b, "{}: {} row {i}", path.display(), layer.id);
            }
            assert_eq!(ours.len(), theirs.len(), "{}: {} rows", path.display(), layer.id);
        }
        assert!(rebuilt.is_empty(), "{}: layers the golden lacks: {:?}", path.display(), rebuilt.keys());
    }
}
