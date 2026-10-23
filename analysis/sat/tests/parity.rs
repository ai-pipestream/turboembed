//! The SaT step against wtpsplit itself: the small random SaT model of
//! testdata/tiny-sat-bundle, and what wtpsplit 2.2.2 gave for it on the
//! texts of tests/data/texts.jsonl with three settings
//! (tests/data/t<threshold>-s<stride>-b<block>.jsonl, written by
//! analysis/sat/reference in its container; analysis/sat/README.md).
//!
//! For every text the tokens must be the same ids at the same offsets
//! (upstream's trimmed of whitespace, as the core's spans are), every
//! character's boundary probability within `PROB_TOLERANCE` of
//! wtpsplit's, the same characters over the threshold except those
//! whose probability in wtpsplit is within that tolerance of it, and,
//! where the same characters are over it, the pieces `split` returned.
//! Two known differences are allowed, and each fixture's count of them
//! and of the characters near the threshold is checked:
//!
//! - where upstream's offset ends inside a grapheme cluster that XLM-R's
//!   character map replaced whole, the core's span covers the cluster,
//!   and the score is compared where upstream reads it;
//! - a text with the pad token written in it is compared for its tokens
//!   only, as HF numbers positions around a pad token and the core does
//!   not (`compare`).

use std::path::{Path, PathBuf};

use serde_json::Value;
use turbo::bundle::Bundle;
use turbo_analysis::{Analyzer, Document, Span};
use turbo_analysis_embed::{Device, EncoderOptions, Precision};
use turbo_analysis_layers::Sentences;
use turbo_sat::{SatAnnotator, SatOptions, cuts};

/// wtpsplit sums the windows' scores in float16 and rounds the mean to
/// float16 (`extract`). Each of those roundings moves the mean score x
/// by at most about |x| / 2048; with at most five windows over a token
/// here (block 30, stride 7) there are six, and the logistic's slope
/// times |x| is at most 0.224, so a probability moves by less than
/// 0.224 * 6 / 2048, about 6.6e-4. The encoders' own float32
/// differences are far smaller.
const PROB_TOLERANCE: f64 = 1e-3;

/// The bundle's pad token, `<pad>`.
const PAD_ID: i64 = 1;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn bundle() -> PathBuf {
    root().join("testdata/tiny-sat-bundle")
}

struct Fixture {
    threshold: f64,
    stride: u32,
    block: u32,
    records: Vec<Value>,
}

/// The fixture `name` under tests/data, for the tiny bundle.
fn fixture(name: &str) -> Fixture {
    fixture_at(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(name), &bundle())
}

fn fixture_at(path: &Path, bundle: &Path) -> Fixture {
    let name = path.display();
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{name}: {e}"));
    let mut lines = text.lines().map(|l| serde_json::from_str::<Value>(l).unwrap());
    let header = lines.next().unwrap();
    // The fixtures were made from the weights the bundle carries.
    let b = Bundle::open(bundle).unwrap();
    let weights = b.manifest.files.iter().find(|f| f.path == "weights/model.safetensors").unwrap();
    assert_eq!(header["weights_sha256"], weights.sha256.as_str(), "{name}: made from other weights");
    Fixture {
        threshold: header["threshold"].as_f64().unwrap(),
        stride: header["stride"].as_u64().unwrap() as u32,
        block: header["block_size"].as_u64().unwrap() as u32,
        records: lines.collect(),
    }
}

/// The device TURBO_TEST_DEVICE names (`cpu`, `select`, a device index or
/// a backend name; the CPU when it is not set) at the precision TURBO_TEST_PRECISION
/// names (model by default): every fixture holds there as on the CPU.
fn encoder() -> EncoderOptions {
    let device = match std::env::var("TURBO_TEST_DEVICE") {
        Err(std::env::VarError::NotPresent) => Device::Cpu,
        Err(e) => panic!("TURBO_TEST_DEVICE: {e}"),
        Ok(d) if d == "cpu" => Device::Cpu,
        Ok(d) if d == "select" => Device::Select,
        Ok(d) => match d.parse() {
            Ok(i) => Device::Index(i),
            Err(_) => Device::Backend(d),
        },
    };
    let precision = match std::env::var("TURBO_TEST_PRECISION").as_deref() {
        Err(_) | Ok("model") => Precision::Model,
        Ok("fastest") => Precision::Fastest,
        Ok("exact") => Precision::Exact,
        Ok(p) => panic!("TURBO_TEST_PRECISION={p}: model, fastest or exact"),
    };
    EncoderOptions { device, precision, ..EncoderOptions::default() }
}

fn open(f: &Fixture) -> SatAnnotator {
    open_at(f, &bundle())
}

fn open_at(f: &Fixture, bundle: &Path) -> SatAnnotator {
    let o = SatOptions {
        encoder: encoder(),
        threshold: f.threshold,
        stride: f.stride,
        block_size: f.block,
        ..SatOptions::default()
    };
    SatAnnotator::open(bundle, &o).unwrap()
}

/// The code point offset of every byte offset that starts a character.
fn code_points(text: &str) -> Vec<u32> {
    let mut at = vec![0u32; text.len() + 1];
    let mut n = 0;
    for (i, c) in text.char_indices() {
        at[i] = n;
        n += 1;
        at[i + c.len_utf8()] = n;
    }
    at
}

/// Upstream's offset `[a, b)`, in code points, with whitespace at either
/// end left out unless that is all there is, as the core gives spans
/// (core/tests/tokenizer.rs compares them so).
fn trimmed(text: &str, [a, b]: [u32; 2]) -> [u32; 2] {
    let chars: Vec<char> = text.chars().skip(a as usize).take((b - a) as usize).collect();
    let lead = chars.iter().take_while(|c| c.is_whitespace()).count() as u32;
    if lead == b - a {
        return [a, b];
    }
    let trail = chars.iter().rev().take_while(|c| c.is_whitespace()).count() as u32;
    [a + lead, b - trail]
}

/// What the comparison found over one fixture.
#[derive(Default, Debug)]
struct Seen {
    texts: usize,
    worst_prob: f64,
    /// Characters over the threshold on one side only, each within the
    /// tolerance of it in wtpsplit.
    near_threshold: usize,
    widened: usize,
    pad_in_text: usize,
    /// Per text, whether a character is over the threshold on one side
    /// only, or the text has a pad token in it: the texts whose pieces
    /// are not compared.
    skipped: Vec<bool>,
}

fn compare(name: &str) -> Seen {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(name);
    compare_at(&path, &bundle())
}

/// Every text of the fixture at `path` through the step on `bundle`.
fn compare_at(path: &Path, bundle: &Path) -> Seen {
    let f = fixture_at(path, bundle);
    let sat = open_at(&f, bundle);
    let name = path.file_name().unwrap().to_string_lossy().into_owned();
    let mut seen = Seen::default();
    for r in &f.records {
        let text = r["text"].as_str().unwrap();
        seen.texts += 1;
        let (spans, scores) = sat.scores(&[text]).remove(0).unwrap();
        let blank = text.chars().all(cuts::is_python_space);
        // Where each token's score is read: its span, or upstream's end
        // where the core's span covers a whole cluster (below).
        let mut read = spans.clone();

        // The same tokens at the same offsets, counted in code points.
        if !blank {
            let ids: Vec<i64> = r["ids"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
            let cp = code_points(text);
            let offsets: Vec<[u32; 2]> = spans.iter().map(|&[a, b]| [cp[a as usize], cp[b as usize]]).collect();
            let theirs: Vec<[u32; 2]> = r["offsets"]
                .as_array()
                .unwrap()
                .iter()
                .map(|o| trimmed(text, [o[0].as_u64().unwrap() as u32, o[1].as_u64().unwrap() as u32]))
                .collect();
            assert_eq!(scores.len(), ids.len(), "{name}: {text:?}: token count");
            assert_eq!(spans.len(), theirs.len(), "{name}: {text:?}: token count");
            // Where upstream's offset ends inside a grapheme cluster that the
            // character map replaced whole, the core's span covers the
            // cluster (core/tests/tokenizer.rs, spans_near_upstream); the
            // score is then compared where upstream reads it.
            let g = cuts::grapheme_index(text);
            let byte: Vec<u32> = text.char_indices().map(|(i, _)| i as u32).chain([text.len() as u32]).collect();
            for (k, (o, t)) in offsets.iter().zip(&theirs).enumerate() {
                if o[1] != t[1] {
                    let (ours_end, their_end) = (byte[o[1] as usize], byte[t[1] as usize]);
                    assert!(
                        o[1] > t[1]
                            && o[0] <= t[0]
                            && g.binary_search(&ours_end).is_ok()
                            && g.binary_search(&their_end).is_err(),
                        "{name}: {text:?}: token {k} ends at {o:?} here, {t:?} upstream"
                    );
                    read[k][1] = their_end;
                    seen.widened += 1;
                }
                assert_eq!(o[0], t[0], "{name}: {text:?}: token {k} at {o:?} here, {t:?} upstream");
            }
        }

        // HF's XLM-R numbers the positions of the tokens that are not the
        // pad token, so a pad token written in the text gets the padding
        // position and every token after it the position before its own;
        // the core numbers every token. Such a text is compared only for
        // its tokens.
        if r["ids"].as_array().unwrap().iter().any(|v| v.as_i64() == Some(PAD_ID)) {
            seen.pad_in_text += 1;
            seen.skipped.push(true);
            continue;
        }

        // Every character's probability, as wtpsplit lays them out: a
        // token's on its last character, 0 where no token ends.
        let n = text.chars().count();
        let theirs: Vec<f64> = r["probs"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        // wtpsplit compares its float32 probabilities with the threshold
        // in float32 (NumPy 2 casts the Python float down).
        let over = |p: f64| p as f32 > f.threshold as f32;
        let mut differ = 0;
        if !blank {
            let cp = code_points(text);
            let mut ours = vec![0f64; n];
            for (&[a, b], &s) in read.iter().zip(&scores) {
                if b > a {
                    ours[cp[b as usize] as usize - 1] = f64::from(cuts::sigmoid(s));
                }
            }
            assert_eq!(theirs.len(), n, "{name}: {text:?}: a probability per character");
            for (i, (&o, &t)) in ours.iter().zip(&theirs).enumerate() {
                let d = (o - t).abs();
                seen.worst_prob = seen.worst_prob.max(d);
                assert!(d <= PROB_TOLERANCE, "{name}: {text:?}: character {i}: {o} here, {t} in wtpsplit");
                if over(o) != over(t) {
                    assert!(
                        (t - f.threshold).abs() <= PROB_TOLERANCE,
                        "{name}: {text:?}: character {i}: {o} here, {t} in wtpsplit, threshold {}",
                        f.threshold
                    );
                    differ += 1;
                }
            }
        }
        seen.near_threshold += differ;
        seen.skipped.push(differ > 0);

        // The same pieces, before snapping, which wtpsplit does not do.
        let mut b = Vec::new();
        cuts::boundaries(&read, &scores, f.threshold, &mut b);
        let mut c = Vec::new();
        cuts::cut_points(text, &b, &mut c);
        let mut p = Vec::new();
        cuts::pieces(text, &c, true, &mut p);
        let ours: Vec<&str> = p.iter().map(|s| &text[s.start as usize..s.end as usize]).collect();
        let theirs: Vec<&str> = r["split"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        if differ == 0 {
            assert_eq!(ours, theirs, "{name}: {text:?}: pieces");
        }
    }
    seen
}

#[test]
fn stride_16_block_62_matches_wtpsplit() {
    let s = compare("t0.5-s16-b62.jsonl");
    eprintln!("t0.5-s16-b62: {s:?}");
    assert_eq!((s.texts, s.near_threshold, s.widened, s.pad_in_text), (90, 0, 6, 1));
}

#[test]
fn windows_without_overlap_match_wtpsplit() {
    let s = compare("t0.25-s62-b62.jsonl");
    eprintln!("t0.25-s62-b62: {s:?}");
    assert_eq!((s.texts, s.near_threshold, s.widened, s.pad_in_text), (90, 1, 6, 1));
}

#[test]
fn many_overlapping_windows_match_wtpsplit() {
    let s = compare("t0.5-s7-b30.jsonl");
    eprintln!("t0.5-s7-b30: {s:?}");
    assert_eq!((s.texts, s.near_threshold, s.widened, s.pad_in_text), (90, 0, 6, 1));
}

/// A released checkpoint against wtpsplit: TURBO_TEST_SAT_BUNDLE names a
/// bundle sealed from it, and TURBO_TEST_SAT_FIXTURES the fixtures,
/// separated by commas, that the reference's `segment` made from the same
/// weights over tests/data/texts.jsonl (analysis/sat/README.md). Each is
/// held to the tiny model's checks, with no character near the threshold
/// and no more than the one text with a pad token in it left out. With
/// neither variable set it says it was skipped; one alone is an error.
#[test]
fn a_released_checkpoint_matches_wtpsplit() {
    use std::env::VarError::NotPresent;
    let (bundle, fixtures) = match (std::env::var("TURBO_TEST_SAT_BUNDLE"), std::env::var("TURBO_TEST_SAT_FIXTURES")) {
        (Ok(b), Ok(f)) => (b, f),
        (Err(NotPresent), Err(NotPresent)) => {
            println!("skipped: TURBO_TEST_SAT_BUNDLE and TURBO_TEST_SAT_FIXTURES are not set");
            return;
        }
        (b, f) => panic!("TURBO_TEST_SAT_BUNDLE ({b:?}) and TURBO_TEST_SAT_FIXTURES ({f:?}) go together"),
    };
    for path in fixtures.split(',') {
        let s = compare_at(Path::new(path), Path::new(&bundle));
        eprintln!("{path}: {s:?}");
        assert_eq!((s.texts, s.near_threshold, s.widened), (90, 0, 6), "{path}: texts, near the threshold, widened");
        assert!(s.pad_in_text <= 1, "{path}: {} texts compared for their tokens only", s.pad_in_text);
    }
}

/// The cuts between wtpsplit's pieces, as byte offsets: where each piece
/// ends, and past the line feed `split` dropped after it.
fn cuts_of(text: &str, pieces: &[&str]) -> Vec<u32> {
    let mut at = 0;
    let mut out = Vec::new();
    for (k, p) in pieces.iter().enumerate() {
        assert!(text[at..].starts_with(p), "{text:?}: piece {k} {p:?} is not at {at}");
        at += p.len();
        if k + 1 < pieces.len() {
            if text[at..].starts_with('\n') {
                at += 1;
            }
            out.push(at as u32);
        }
    }
    assert_eq!(at, text.len(), "{text:?}: the pieces cover the text");
    out.dedup();
    out
}

/// The layer the step writes holds wtpsplit's sentences with each cut
/// moved to the end of the grapheme cluster it falls in, trimmed of
/// whitespace, the empty ones left out: on every text but the one with a
/// pad token in it, whose scores differ, and those with a character on
/// the other side of the threshold (`compare` counts both). Returns the
/// texts compared and those with a cut that moved.
fn layer_matches(name: &str) -> (usize, usize) {
    let skipped = compare(name).skipped;
    let f = fixture(name);
    let analyzer = Analyzer::builder().add(open(&f)).build().unwrap();
    let mut docs: Vec<Document> =
        f.records.iter().map(|r| Document::new(r["text"].as_str().unwrap().to_owned())).collect();
    let results = analyzer.run(&mut docs);
    let (mut compared, mut snapped) = (0, 0);
    for (((r, d), res), &skip) in f.records.iter().zip(&docs).zip(results).zip(&skipped) {
        res.unwrap();
        let text = r["text"].as_str().unwrap();
        let s = d.get::<Sentences>().unwrap();
        let ours: Vec<&str> =
            s.spans.iter().map(|(_, sp): (_, Span)| &text[sp.start as usize..sp.end as usize]).collect();
        for w in &ours {
            assert!(!w.is_empty() && !w.starts_with(cuts::is_python_space) && !w.ends_with(cuts::is_python_space));
        }
        if skip {
            continue;
        }
        let pieces: Vec<&str> = r["split"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        let mut c = cuts_of(text, &pieces);
        let g = cuts::grapheme_index(text);
        let before = c.clone();
        cuts::snap(&g, &mut c);
        snapped += usize::from(before != c);
        let mut p = Vec::new();
        cuts::pieces(text, &c, true, &mut p);
        let theirs: Vec<&str> = p
            .into_iter()
            .filter_map(|sp| cuts::trim(text, &g, sp))
            .map(|sp| &text[sp.start as usize..sp.end as usize])
            .collect();
        assert_eq!(ours, theirs, "{name}: {text:?}");
        compared += 1;
    }
    (compared, snapped)
}

#[test]
fn the_layer_is_wtpsplit_s_pieces_snapped_and_trimmed() {
    for name in ["t0.5-s16-b62.jsonl", "t0.25-s62-b62.jsonl", "t0.5-s7-b30.jsonl"] {
        let (compared, snapped) = layer_matches(name);
        eprintln!("{name}: {compared} texts, {snapped} with a cut inside a cluster");
        let want = if name.starts_with("t0.25") { 88 } else { 89 };
        assert_eq!(compared, want, "{name}: texts compared");
        assert!(snapped > 5, "{name}: {snapped} texts have a cut inside a cluster");
    }
}
