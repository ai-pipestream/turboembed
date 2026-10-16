//! Parity with the neural part-of-speech taggers of the OpenNLP addons
//! (`neural-postag`) on a JVM: the goldens of `PostagNeuralGolden.java` in
//! `analysis/goldens-tool`, under `analysis/goldens/data/postag-neural/<run>`.
//!
//! Each run trained one model on the JVM from `inputs/tagged/wordtag.txt`
//! and wrote two goldens: `tagged/wordtag.golden` (the training sentences,
//! with the model file) and `sequence/sentences.golden` (a fixed token
//! list). Every golden of every run is read and compared:
//!
//! - The model file loads, writes back to the same bytes, and has the
//!   recorded magic, length, SHA-256 and tags.
//! - Feedforward runs: every context symbol, embedding row, vector row and
//!   emission of the left to right decoding, the tags, the probabilities
//!   and the sequence scores are exact.
//! - BiLSTM runs: the tags and the CRF Viterbi paths are exact. The
//!   emissions may differ in the last bits: the LSTM gate sigmoids call
//!   `Math.exp`, which HotSpot computes with an intrinsic routine that is
//!   not ported (see the sigmoid in `src/neural/lstm.rs`), and so do the
//!   CRF's normalizations (`Math.exp` and `Math.log`). The emissions, the
//!   CRF marginals (from the Rust emissions and from the JVM's), the tag
//!   probabilities and the sequence scores are within [`BILSTM_TOLERANCE`],
//!   and the test reports how many are bit for bit.
//! - Training: the Rust trainers are run with each run's settings on the
//!   same corpus. The feedforward model file is the JVM's, byte for byte,
//!   which also checks the `java.util.Random` stream the trainers draw
//!   from. A BiLSTM model trained in Rust scores the training sentences
//!   within [`BILSTM_TRAINED_TOLERANCE`] of the JVM's model.

use std::path::{Path, PathBuf};

use turbo_analysis::Annotator;
use turbo_goldens::{ColumnSpec, ColumnType, Golden, GoldenAnchor, GoldenLayer, Shape, known_shapes, sha256};
use turbo_postag::neural::feedforward::ABSENT;
use turbo_postag::neural::{
    BilstmPosModel, BilstmPosTagger, BilstmPosTrainer, BilstmSettings, CrfScorer, FeedforwardPosContext,
    FeedforwardPosModel, FeedforwardPosTagger, FeedforwardPosTrainer, FeedforwardSettings, NeuralSequence, PosSample,
};

/// The largest difference allowed between a BiLSTM value computed in Rust
/// from the JVM's model and the JVM's value.
const BILSTM_TOLERANCE: f64 = 1e-13;

/// The largest difference allowed between an emission of a BiLSTM model
/// trained in Rust and one of the model the JVM trained.
const BILSTM_TRAINED_TOLERANCE: f64 = 1e-12;

/// The command that writes the goldens.
const GENERATE: &str = "cargo run -p turbo-goldens-tool -- generate";

const TOKENS: &[ColumnSpec] =
    &[ColumnSpec::new("sentence", ColumnType::U32), ColumnSpec::new("token", ColumnType::Str)];
const MODEL: &[ColumnSpec] = &[
    ColumnSpec::new("magic", ColumnType::Str),
    ColumnSpec::new("length", ColumnType::U32),
    ColumnSpec::new("sha256", ColumnType::Str),
];
const MODEL_BYTES: &[ColumnSpec] =
    &[ColumnSpec::new("offset", ColumnType::U32), ColumnSpec::new("hex", ColumnType::Str)];
const TAGS: &[ColumnSpec] = &[ColumnSpec::new("tag", ColumnType::Str)];
const SCORES: &[ColumnSpec] = &[
    ColumnSpec::new("sentence", ColumnType::U32),
    ColumnSpec::new("index", ColumnType::U32),
    ColumnSpec::new("tag", ColumnType::U32),
    ColumnSpec::new("score", ColumnType::F64),
];
const VITERBI: &[ColumnSpec] = &[
    ColumnSpec::new("sentence", ColumnType::U32),
    ColumnSpec::new("index", ColumnType::U32),
    ColumnSpec::new("tag", ColumnType::U32),
];
const MARGINALS: &[ColumnSpec] = &[
    ColumnSpec::new("sentence", ColumnType::U32),
    ColumnSpec::new("index", ColumnType::U32),
    ColumnSpec::new("tag", ColumnType::U32),
    ColumnSpec::new("probability", ColumnType::F64),
];
const CONTEXT: &[ColumnSpec] = &[
    ColumnSpec::new("sentence", ColumnType::U32),
    ColumnSpec::new("index", ColumnType::U32),
    ColumnSpec::new("slot", ColumnType::U32),
    ColumnSpec::new("present", ColumnType::U32),
    ColumnSpec::new("symbol", ColumnType::Str),
    ColumnSpec::new("row", ColumnType::U32),
];
const VECTOR_ROWS: &[ColumnSpec] = &[
    ColumnSpec::new("sentence", ColumnType::U32),
    ColumnSpec::new("index", ColumnType::U32),
    ColumnSpec::new("slot", ColumnType::U32),
    ColumnSpec::new("present", ColumnType::U32),
    ColumnSpec::new("row", ColumnType::U32),
];
const TAGGED: &[ColumnSpec] = &[
    ColumnSpec::new("sentence", ColumnType::U32),
    ColumnSpec::new("index", ColumnType::U32),
    ColumnSpec::new("tag", ColumnType::Str),
    ColumnSpec::new("probability", ColumnType::F64),
];
const SEQUENCES: &[ColumnSpec] =
    &[ColumnSpec::new("sentence", ColumnType::U32), ColumnSpec::new("score", ColumnType::F64)];

fn shapes() -> Vec<Shape> {
    let mut shapes = known_shapes();
    for (id, columns) in [
        ("turbo:sentence-tokens", TOKENS),
        ("turbo:neural-model", MODEL),
        ("turbo:neural-model-bytes", MODEL_BYTES),
        ("turbo:neural-tags", TAGS),
        ("turbo:neural-emissions", SCORES),
        ("turbo:crf-viterbi", VITERBI),
        ("turbo:crf-marginals", MARGINALS),
        ("turbo:neural-context", CONTEXT),
        ("turbo:neural-vector-rows", VECTOR_ROWS),
        ("turbo:neural-tagged", TAGGED),
        ("turbo:neural-sequences", SEQUENCES),
    ] {
        shapes.push(Shape { id: id.into(), anchor: GoldenAnchor::Document, columns });
    }
    shapes
}

fn layer<'g>(g: &'g Golden, id: &str) -> &'g GoldenLayer {
    g.layer(id).unwrap_or_else(|| panic!("no layer {id}"))
}

fn u32s<'g>(l: &'g GoldenLayer, name: &str) -> &'g [u32] {
    l.u32s(name).unwrap_or_else(|| panic!("{}: no u32 column {name}", l.id))
}

fn f64s<'g>(l: &'g GoldenLayer, name: &str) -> &'g [f64] {
    l.f64s(name).unwrap_or_else(|| panic!("{}: no f64 column {name}", l.id))
}

fn strs(l: &GoldenLayer, name: &str) -> Vec<String> {
    l.strs(name).unwrap_or_else(|| panic!("{}: no str column {name}", l.id)).iter().map(str::to_owned).collect()
}

/// The directories in `dir`, sorted.
fn sorted(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.is_dir())
        .collect();
    out.sort();
    out
}

/// The sentences of `turbo:sentence-tokens`, in order.
fn sentences(g: &Golden) -> Vec<Vec<String>> {
    let l = layer(g, "turbo:sentence-tokens");
    let mut out: Vec<Vec<String>> = Vec::new();
    for (&s, token) in u32s(l, "sentence").iter().zip(strs(l, "token")) {
        if s as usize == out.len() {
            out.push(Vec::new());
        }
        assert_eq!(s as usize + 1, out.len(), "sentence numbers are consecutive");
        out.last_mut().unwrap().push(token);
    }
    out
}

/// A `[sentence][index][tag]` table of a layer with those three columns
/// and one `f64` column, checking that the rows come in that order and
/// cover every token and tag.
fn table(g: &Golden, id: &str, value: &str, sentences: &[Vec<String>], tags: usize) -> Vec<Vec<Vec<f64>>> {
    let l = layer(g, id);
    let (s, i, t, v) = (u32s(l, "sentence"), u32s(l, "index"), u32s(l, "tag"), f64s(l, value));
    let mut out = Vec::new();
    let mut row = 0;
    for (k, sentence) in sentences.iter().enumerate() {
        let mut rows = Vec::new();
        for index in 0..sentence.len() {
            let mut scores = Vec::new();
            for tag in 0..tags {
                assert_eq!((s[row], i[row], t[row]), (k as u32, index as u32, tag as u32), "{id}: row {row}");
                scores.push(v[row]);
                row += 1;
            }
            rows.push(scores);
        }
        out.push(rows);
    }
    assert_eq!(row, l.len(), "{id}: rows past the last token");
    out
}

/// The SHA-256 of `bytes` in lowercase hex.
fn sha256_hex(bytes: &[u8]) -> String {
    turbo_goldens::hex::to_hex(&sha256(bytes))
}

/// The model file of the training golden, checked against its recorded
/// magic, length and SHA-256.
fn model_bytes(g: &Golden) -> Vec<u8> {
    let rows = layer(g, "turbo:neural-model-bytes");
    let mut bytes = Vec::new();
    for (&offset, hex) in u32s(rows, "offset").iter().zip(strs(rows, "hex")) {
        assert_eq!(offset as usize, bytes.len(), "model bytes offset");
        assert!(hex.len().is_multiple_of(2), "model bytes at {offset}");
        for pair in hex.as_bytes().chunks(2) {
            bytes.push(u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).expect("hex"));
        }
    }
    let model = layer(g, "turbo:neural-model");
    let magic_length = usize::from(u16::from_be_bytes([bytes[0], bytes[1]]));
    assert_eq!(std::str::from_utf8(&bytes[2..2 + magic_length]).unwrap(), strs(model, "magic")[0]);
    assert_eq!(bytes.len(), u32s(model, "length")[0] as usize);
    assert_eq!(sha256_hex(&bytes), strs(model, "sha256")[0]);
    bytes
}

/// The training corpus as `WordTagSampleStream` reads it: one sample per
/// line, tokens split at whitespace and each at its last `_`, and a line
/// with a token without `_` read as a sample with no tokens.
fn corpus(g: &Golden) -> Vec<PosSample> {
    let text = g.input_text(turbo_goldens::data_dir()).unwrap_or_else(|e| panic!("{e}"));
    text.lines()
        .map(|line| {
            let pairs: Option<Vec<(&str, &str)>> = line.split_whitespace().map(|t| t.rsplit_once('_')).collect();
            let (words, tags): (Vec<&str>, Vec<&str>) = pairs.unwrap_or_default().into_iter().unzip();
            PosSample::new(words, tags).unwrap()
        })
        .collect()
}

/// The word vectors of the runs with vectors: the word's length in UTF-16
/// units times `0.1f`, and `0.5f`.
fn vectors(word: &str) -> Option<Vec<f32>> {
    Some(vec![word.encode_utf16().count() as f32 * 0.1f32, 0.5f32])
}

/// How many values were compared, how many were bit for bit, and the
/// largest difference.
#[derive(Default)]
struct Count {
    values: usize,
    exact: usize,
    max_diff: f64,
}

impl Count {
    /// Compares `rust` with `jvm`, within `tolerance` (0 for exact).
    fn compare(&mut self, rust: f64, jvm: f64, tolerance: f64, what: &dyn Fn() -> String) {
        let diff = (rust - jvm).abs();
        self.values += 1;
        self.exact += usize::from(rust.to_bits() == jvm.to_bits());
        self.max_diff = self.max_diff.max(diff);
        if tolerance == 0.0 {
            assert_eq!(rust.to_bits(), jvm.to_bits(), "{}: {rust} against {jvm}", what());
        } else {
            assert!(diff <= tolerance, "{}: {rust} against {jvm}", what());
        }
    }
}

impl std::ops::AddAssign for Count {
    fn add_assign(&mut self, other: Count) {
        self.values += other.values;
        self.exact += other.exact;
        self.max_diff = self.max_diff.max(other.max_diff);
    }
}

impl std::fmt::Display for Count {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} of {} bit for bit, largest difference {:e}", self.exact, self.values, self.max_diff)
    }
}

/// Counts over every golden of every run.
#[derive(Default)]
struct Totals {
    runs: usize,
    goldens: usize,
    sentences: usize,
    tags: usize,
    contexts: usize,
    ff_emissions: Count,
    bilstm_emissions: Count,
    marginals_of_jvm_emissions: Count,
    marginals: Count,
    probabilities: Count,
    sequence_scores: Count,
}

impl std::ops::AddAssign for Totals {
    fn add_assign(&mut self, other: Totals) {
        self.runs += other.runs;
        self.goldens += other.goldens;
        self.sentences += other.sentences;
        self.tags += other.tags;
        self.contexts += other.contexts;
        self.ff_emissions += other.ff_emissions;
        self.bilstm_emissions += other.bilstm_emissions;
        self.marginals_of_jvm_emissions += other.marginals_of_jvm_emissions;
        self.marginals += other.marginals;
        self.probabilities += other.probabilities;
        self.sequence_scores += other.sequence_scores;
    }
}

/// Runs `check` on every run, one thread each (the runs are independent),
/// and returns the results in run order. A panic in a thread is raised
/// again here.
fn each_run<T: Send>(runs: &[Run], check: impl Fn(&Run) -> T + Sync) -> Vec<T> {
    std::thread::scope(|scope| {
        let handles: Vec<_> = runs.iter().map(|run| scope.spawn(|| check(run))).collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e))).collect()
    })
}

/// The tagger's two entry points, as the recipe calls them.
struct Tagging<'a> {
    tag: &'a dyn Fn(&[String]) -> Vec<String>,
    best: &'a dyn Fn(&[String]) -> Vec<NeuralSequence>,
}

/// Compares the tagger's best sequence with `turbo:neural-tagged` and
/// `turbo:neural-sequences`.
fn check_sequences(
    what: &str,
    g: &Golden,
    sentences: &[Vec<String>],
    tagging: Tagging<'_>,
    tolerance: f64,
    totals: &mut Totals,
) {
    let tagged = layer(g, "turbo:neural-tagged");
    let (s, i, t, p) =
        (u32s(tagged, "sentence"), u32s(tagged, "index"), strs(tagged, "tag"), f64s(tagged, "probability"));
    let scores = f64s(layer(g, "turbo:neural-sequences"), "score");
    assert_eq!(scores.len(), sentences.len(), "{what}: sequence scores");
    let mut row = 0;
    for (k, sentence) in sentences.iter().enumerate() {
        let sequences = (tagging.best)(sentence);
        assert_eq!(sequences.len(), 1, "{what}: sentence {k}: sequences");
        let sequence = &sequences[0];
        assert_eq!((tagging.tag)(sentence), sequence.outcomes(), "{what}: sentence {k}: tag and topKSequences");
        for index in 0..sentence.len() {
            assert_eq!((s[row], i[row]), (k as u32, index as u32), "{what}: turbo:neural-tagged row {row}");
            assert_eq!(sequence.outcomes()[index], t[row], "{what}: sentence {k} token {index}: tag");
            totals.probabilities.compare(sequence.probs()[index], p[row], tolerance, &|| {
                format!("{what}: sentence {k} token {index}: probability")
            });
            totals.tags += 1;
            row += 1;
        }
        totals
            .sequence_scores
            .compare(sequence.score(), scores[k], tolerance, &|| format!("{what}: sentence {k}: sequence score"));
    }
    assert_eq!(row, tagged.len(), "{what}: turbo:neural-tagged rows past the last token");
    totals.sentences += sentences.len();
}

fn check_bilstm(what: &str, g: &Golden, model: &BilstmPosModel, tagger: &BilstmPosTagger, totals: &mut Totals) {
    let sentences = sentences(g);
    let tags = model.tags().len();
    let emissions = table(g, "turbo:neural-emissions", "score", &sentences, tags);
    let crf = model.is_crf();
    let viterbi = crf.then(|| u32s(layer(g, "turbo:crf-viterbi"), "tag").to_vec());
    let marginals = crf.then(|| table(g, "turbo:crf-marginals", "probability", &sentences, tags));
    let mut path_row = 0;
    for (k, sentence) in sentences.iter().enumerate() {
        let rust = model.score(sentence).unwrap_or_else(|e| panic!("{what}: sentence {k}: {e}"));
        assert!(rust.iter().all(|row| row.len() == tags), "{what}: sentence {k}: scores per token");
        for (index, row) in rust.iter().enumerate() {
            for (tag, &score) in row.iter().enumerate() {
                totals.bilstm_emissions.compare(score, emissions[k][index][tag], BILSTM_TOLERANCE, &|| {
                    format!("{what}: sentence {k} token {index} tag {tag}: emission")
                });
            }
        }
        if let (Some(viterbi), Some(marginals)) = (&viterbi, &marginals) {
            let (transitions, start, end) =
                (model.transition_weights().unwrap(), model.start_weights().unwrap(), model.end_weights().unwrap());
            let path = CrfScorer::viterbi(&rust, transitions, start, end).unwrap();
            for (index, &tag) in path.iter().enumerate() {
                assert_eq!(tag as u32, viterbi[path_row], "{what}: sentence {k} token {index}: Viterbi tag");
                path_row += 1;
            }
            let of_jvm = CrfScorer::marginals(&emissions[k], transitions, start, end).unwrap();
            let of_rust = CrfScorer::marginals(&rust, transitions, start, end).unwrap();
            for index in 0..sentence.len() {
                for tag in 0..tags {
                    let jvm = marginals[k][index][tag];
                    let at = || format!("{what}: sentence {k} token {index} tag {tag}: marginal");
                    totals.marginals_of_jvm_emissions.compare(of_jvm[index][tag], jvm, BILSTM_TOLERANCE, &at);
                    totals.marginals.compare(of_rust[index][tag], jvm, BILSTM_TOLERANCE, &at);
                }
            }
        }
    }
    if let Some(viterbi) = &viterbi {
        assert_eq!(path_row, viterbi.len(), "{what}: Viterbi rows past the last token");
    } else {
        assert!(g.layer("turbo:crf-viterbi").is_none() && g.layer("turbo:crf-marginals").is_none(), "{what}");
    }
    let tagging = Tagging { tag: &|s| tagger.tag(s).unwrap(), best: &|s| tagger.top_k_sequences(s).unwrap() };
    check_sequences(what, g, &sentences, tagging, BILSTM_TOLERANCE, totals);
}

fn check_feedforward(what: &str, g: &Golden, model: &FeedforwardPosModel, totals: &mut Totals) {
    let sentences = sentences(g);
    let tags = model.tags();
    let emissions = table(g, "turbo:neural-emissions", "score", &sentences, tags.len());
    let context = layer(g, "turbo:neural-context");
    let (c_sentence, c_index, c_slot) = (u32s(context, "sentence"), u32s(context, "index"), u32s(context, "slot"));
    let (c_present, c_symbol, c_row) = (u32s(context, "present"), strs(context, "symbol"), u32s(context, "row"));
    let vector_rows = g.layer("turbo:neural-vector-rows");
    assert_eq!(vector_rows.is_some(), model.uses_pretrained_vectors(), "{what}: turbo:neural-vector-rows");
    let mut row = 0;
    let mut vector_row = 0;
    for (k, (sentence, jvm_emissions)) in sentences.iter().zip(&emissions).enumerate() {
        let mut assigned: Vec<&str> = Vec::new();
        for (index, jvm_scores) in jvm_emissions.iter().enumerate() {
            let previous = index.checked_sub(1).map(|i| assigned[i]);
            let before_previous = index.checked_sub(2).map(|i| assigned[i]);
            let symbols = FeedforwardPosContext::extract(sentence, index, previous, before_previous);
            let ids = model.feature_ids(&symbols);
            for (slot, (symbol, &id)) in symbols.iter().zip(&ids).enumerate() {
                let at = format!("{what}: sentence {k} token {index} slot {slot}");
                assert_eq!((c_sentence[row], c_index[row], c_slot[row]), (k as u32, index as u32, slot as u32), "{at}");
                // Java has null where the Rust context has the padding symbol.
                let (present, jvm_symbol) = (c_present[row] == 1, c_symbol[row].as_str());
                assert_eq!(if present { jvm_symbol } else { ABSENT }, symbol, "{at}: symbol");
                assert!(present || jvm_symbol.is_empty(), "{at}: an absent symbol is empty");
                assert_eq!(id as u32, c_row[row], "{at}: embedding row");
                totals.contexts += 1;
                row += 1;
            }
            let rows = model.pretrained_rows(sentence, index);
            if let Some(l) = vector_rows {
                let rows = rows.expect("vector rows");
                let (s, i, slot, present, r) =
                    (u32s(l, "sentence"), u32s(l, "index"), u32s(l, "slot"), u32s(l, "present"), u32s(l, "row"));
                for (p, &rust) in rows.iter().enumerate() {
                    let at = format!("{what}: sentence {k} token {index} vector slot {p}");
                    let key = (s[vector_row], i[vector_row], slot[vector_row]);
                    assert_eq!(key, (k as u32, index as u32, p as u32), "{at}");
                    let jvm = if present[vector_row] == 1 { r[vector_row] as isize } else { -1 };
                    assert_eq!(rust, jvm, "{at}");
                    vector_row += 1;
                }
            }
            let scores = model
                .score_with_pretrained(&ids, rows.as_ref().map(|r| &r[..]))
                .unwrap_or_else(|e| panic!("{what}: sentence {k} token {index}: {e}"));
            assert_eq!(scores.len(), jvm_scores.len(), "{what}: sentence {k} token {index}: scores");
            for (tag, (&score, &jvm)) in scores.iter().zip(jvm_scores).enumerate() {
                totals
                    .ff_emissions
                    .compare(score, jvm, 0.0, &|| format!("{what}: sentence {k} token {index} tag {tag}: emission"));
            }
            let mut best = 0;
            for o in 1..scores.len() {
                if scores[o] > scores[best] {
                    best = o;
                }
            }
            assigned.push(&tags[best]);
        }
    }
    assert_eq!(row, context.len(), "{what}: context rows past the last token");
    if let Some(l) = vector_rows {
        assert_eq!(vector_row, l.len(), "{what}: vector rows past the last token");
    }
    let tagger = FeedforwardPosTagger::new(model.clone());
    let tagging = Tagging { tag: &|s| tagger.tag(s).unwrap(), best: &|s| tagger.top_k_sequences(s).unwrap() };
    check_sequences(what, g, &sentences, tagging, 0.0, totals);
}

/// The value of a golden's `config` record.
fn config<'g>(g: &'g Golden, key: &str) -> Option<&'g str> {
    g.header.config.iter().find(|c| c.key == key).map(|c| c.value.as_str())
}

/// The value of `name=` in a Java record's `toString`.
fn setting<T: std::str::FromStr>(settings: &str, name: &str) -> T {
    let at = settings.find(&format!("[{name}=")).or_else(|| settings.find(&format!(" {name}=")));
    let start = at.unwrap_or_else(|| panic!("no {name} in {settings}")) + name.len() + 2;
    let end = settings[start..].find([',', ']']).map_or(settings.len(), |e| start + e);
    settings[start..end].parse().unwrap_or_else(|_| panic!("{name} in {settings}"))
}

fn bilstm_settings(s: &str) -> BilstmSettings {
    BilstmSettings::new(
        setting(s, "wordEmbeddingSize"),
        setting(s, "charEmbeddingSize"),
        setting(s, "charHiddenSize"),
        setting(s, "hiddenSize"),
        setting(s, "epochs"),
        setting(s, "batchSize"),
        setting(s, "learningRate"),
        setting(s, "clipNorm"),
        setting(s, "dropout"),
        setting(s, "wordCutoff"),
        setting(s, "maxWordLength"),
        setting(s, "seed"),
        setting(s, "threads"),
        setting(s, "wordDropout"),
        setting(s, "learningRateHalfLife"),
        setting(s, "crf"),
        setting(s, "encoderLayers"),
        setting(s, "pretrainedDropout"),
        setting(s, "encoderDropout"),
        setting(s, "auxLossWeight"),
        setting(s, "pretrainedTuning"),
        setting(s, "pretrainedAdapter"),
    )
    .unwrap()
}

fn feedforward_settings(s: &str) -> FeedforwardSettings {
    FeedforwardSettings::new(
        setting(s, "embeddingSize"),
        setting(s, "hiddenSize"),
        setting(s, "epochs"),
        setting(s, "batchSize"),
        setting(s, "learningRate"),
        setting(s, "l2"),
        setting(s, "dropout"),
        setting(s, "wordCutoff"),
        setting(s, "suffixCutoff"),
        setting(s, "seed"),
    )
    .unwrap()
}

/// One run's two goldens, its model file and its training corpus.
struct Run {
    name: String,
    training: Golden,
    tokens: Golden,
    bytes: Vec<u8>,
    samples: Vec<PosSample>,
    /// The non-empty samples' sentences, which the training golden tags.
    train_sentences: Vec<Vec<String>>,
}

impl Run {
    fn read(dir: &Path, shapes: &[Shape]) -> Run {
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let read = |file: &str| {
            let path = dir.join(file);
            let g = Golden::read_with(&path, shapes).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(config(&g, "run"), Some(name.as_str()), "{}", path.display());
            g
        };
        let training = read("tagged/wordtag.golden");
        let tokens = read("sequence/sentences.golden");
        // The fixed token list is the checked in input.
        tokens.input_text(turbo_goldens::data_dir()).unwrap_or_else(|e| panic!("{e}"));
        if let Some(v) = config(&training, "vectors") {
            assert_eq!(v, "length*0.1f,0.5f", "{name}: the vectors this test gives the trainers");
        }
        let bytes = model_bytes(&training);
        let samples = corpus(&training);
        let train_sentences: Vec<Vec<String>> =
            samples.iter().filter(|s| !s.sentence.is_empty()).map(|s| s.sentence.clone()).collect();
        assert_eq!(train_sentences, sentences(&training), "{name}: training sentences");
        Run { name, training, tokens, bytes, samples, train_sentences }
    }

    fn is_bilstm(&self) -> bool {
        let annotator = self.training.header.annotator.class.as_str();
        let bilstm = annotator.ends_with(".BilstmPOSTagger");
        assert!(bilstm || annotator.ends_with(".FeedforwardPOSTagger"), "{}: annotator {annotator}", self.name);
        bilstm
    }

    fn settings(&self) -> &str {
        config(&self.training, "settings").expect("the settings record")
    }

    fn with_vectors(&self) -> bool {
        config(&self.training, "vectors").is_some()
    }

    fn goldens(&self) -> [(String, &Golden); 2] {
        [
            (format!("{}/tagged/wordtag", self.name), &self.training),
            (format!("{}/sequence/sentences", self.name), &self.tokens),
        ]
    }
}

/// The runs under `postag-neural`, read and checked against their inputs.
fn runs() -> Vec<Run> {
    let root = turbo_goldens::data_dir().join("postag-neural");
    assert!(root.is_dir(), "{} is absent; write the goldens with `{GENERATE}`", root.display());
    let shapes = shapes();
    let runs: Vec<Run> = sorted(&root).iter().map(|dir| Run::read(dir, &shapes)).collect();
    assert_eq!(runs.len(), 8, "{}: runs", root.display());
    runs
}

/// Loads the JVM's model of `run`, checks that it writes back to the same
/// bytes, and compares both goldens.
fn check_inference(run: &Run, totals: &mut Totals) {
    let (name, bytes) = (&run.name, &run.bytes);
    let mut written = Vec::new();
    if run.is_bilstm() {
        let model = BilstmPosModel::load(bytes.as_slice()).unwrap_or_else(|e| panic!("{name}: {e}"));
        model.serialize(&mut written).unwrap();
        assert!(&written == bytes, "{name}: the model writes back to other bytes");
        let tagger = BilstmPosTagger::new(model.clone());
        let model_ref = Annotator::model(&tagger).expect("a ModelRef");
        assert_eq!(model_ref.sha256, sha256_hex(bytes), "{name}: ModelRef");
        for (what, g) in run.goldens() {
            assert_eq!(strs(layer(g, "turbo:neural-tags"), "tag"), model.tags(), "{what}: tags");
            check_bilstm(&what, g, &model, &tagger, totals);
            totals.goldens += 1;
        }
    } else {
        let model = FeedforwardPosModel::load(bytes.as_slice()).unwrap_or_else(|e| panic!("{name}: {e}"));
        model.serialize(&mut written).unwrap();
        assert!(&written == bytes, "{name}: the model writes back to other bytes");
        assert_eq!(model.model_ref().sha256, sha256_hex(bytes), "{name}: ModelRef");
        for (what, g) in run.goldens() {
            assert_eq!(strs(layer(g, "turbo:neural-tags"), "tag"), model.tags(), "{what}: tags");
            check_feedforward(&what, g, &model, totals);
            totals.goldens += 1;
        }
    }
    totals.runs += 1;
}

/// Trains a model in Rust with the run's settings and corpus and compares
/// it with the JVM's: the feedforward file byte for byte, the BiLSTM
/// emissions on the training sentences within [`BILSTM_TRAINED_TOLERANCE`].
/// Returns the count of compared emissions, empty for a feedforward run.
fn check_training(run: &Run) -> Count {
    let name = &run.name;
    let mut count = Count::default();
    if run.is_bilstm() {
        let settings = bilstm_settings(run.settings());
        let trained = if run.with_vectors() {
            BilstmPosTrainer::train_with_vectors(&run.samples, settings, Some(vectors), None, None)
        } else {
            BilstmPosTrainer::train(&run.samples, settings)
        }
        .unwrap_or_else(|e| panic!("{name}: training: {e}"));
        let tags = strs(layer(&run.training, "turbo:neural-tags"), "tag");
        assert_eq!(trained.tags(), tags, "{name}: trained tags");
        let emissions = table(&run.training, "turbo:neural-emissions", "score", &run.train_sentences, tags.len());
        for (k, (sentence, jvm_emissions)) in run.train_sentences.iter().zip(&emissions).enumerate() {
            for (index, (row, jvm_row)) in trained.score(sentence).unwrap().iter().zip(jvm_emissions).enumerate() {
                assert_eq!(row.len(), jvm_row.len(), "{name}: sentence {k} token {index}: scores");
                for (tag, (&score, &jvm)) in row.iter().zip(jvm_row).enumerate() {
                    count.compare(score, jvm, BILSTM_TRAINED_TOLERANCE, &|| {
                        format!("{name}: trained in Rust: sentence {k} token {index} tag {tag}: emission")
                    });
                }
            }
        }
    } else {
        let settings = feedforward_settings(run.settings());
        let trained = if run.with_vectors() {
            FeedforwardPosTrainer::train_with_vectors(&run.samples, settings, vectors)
        } else {
            FeedforwardPosTrainer::train(&run.samples, settings)
        }
        .unwrap_or_else(|e| panic!("{name}: training: {e}"));
        let mut trained_bytes = Vec::new();
        trained.serialize(&mut trained_bytes).unwrap();
        assert!(trained_bytes == run.bytes, "{name}: the model trained in Rust is not the JVM's file");
    }
    count
}

#[test]
fn every_neural_postag_golden_matches_the_jvm() {
    let mut totals = Totals::default();
    for t in each_run(&runs(), |run| {
        let mut totals = Totals::default();
        check_inference(run, &mut totals);
        totals
    }) {
        totals += t;
    }
    assert_eq!(totals.goldens, 16, "goldens");
    eprintln!(
        "{} runs, {} goldens, {} sentences: {} tags and {} feedforward context slots equal",
        totals.runs, totals.goldens, totals.sentences, totals.tags, totals.contexts
    );
    eprintln!("feedforward emissions: {}", totals.ff_emissions);
    eprintln!("BiLSTM emissions: {}", totals.bilstm_emissions);
    eprintln!("CRF marginals of the JVM's emissions: {}", totals.marginals_of_jvm_emissions);
    eprintln!("CRF marginals: {}", totals.marginals);
    eprintln!("tag probabilities: {}", totals.probabilities);
    eprintln!("sequence scores: {}", totals.sequence_scores);
}

#[test]
fn trainers_reproduce_the_jvm_models() {
    let runs = runs();
    let mut trained = Count::default();
    for c in each_run(&runs, check_training) {
        trained += c;
    }
    let feedforward = runs.iter().filter(|r| !r.is_bilstm()).count();
    eprintln!("{feedforward} feedforward models trained in Rust are the JVM's files byte for byte");
    eprintln!("emissions of the BiLSTM models trained in Rust: {trained}");
}
