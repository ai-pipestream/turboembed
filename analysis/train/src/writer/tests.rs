use super::*;
use crate::event::Event;
use crate::index::{DataIndex, IndexParams, IndexerKind};
use crate::monitor::TrainingConfiguration;
use crate::params::{Key, TrainingParameters};
use turbo_maxent::MaxentModel;

fn trained() -> GisModel {
    let events = [
        Event::from_strs("yes", &["a", "b"]),
        Event::from_strs("no", &["c", "b"]),
        Event::from_strs("maybe", &["d"]),
        Event::from_strs("yes", &["a", "d"]),
    ];
    let index = DataIndex::index(&events, &IndexParams::new(IndexerKind::TwoPass, 0, true)).unwrap();
    let mut params = TrainingParameters::new();
    params.put(Key::ITERATIONS, 10);
    crate::gis::train(&index, &params, &mut TrainingConfiguration::default()).unwrap()
}

#[test]
fn the_written_model_reads_back_with_the_same_parameters() {
    let model = trained();
    let bytes = write_gis(&model).unwrap();
    let read = MaxentModel::from_trained_bytes(&bytes).unwrap();
    assert_eq!(read.outcomes().cloned().collect::<Vec<_>>(), model.outcome_labels());
    assert_eq!(read.num_predicates(), model.num_predicates());
    for p in ["a", "b", "c", "d"] {
        assert!(read.has_predicate(p), "{p}");
    }
    // The written parameters evaluate as the trained ones.
    let mut out = vec![0.0; 3];
    read.eval(&["a", "b"], &mut out);
    let yes = model.outcome_labels().iter().position(|o| *o == "yes").unwrap();
    assert!(out[yes] > 0.5, "{out:?}");
}

#[test]
fn nul_and_supplementary_characters_are_transcoded_to_standard_utf8_on_the_wire() {
    use crate::params::Algorithm;
    let outcome = "yes\0😀";
    let events = [
        Event::from_strs(outcome, &["w0=😀\0", "t0=_"]),
        Event::from_strs("O", &["w0=x", "t0=_"]),
        Event::from_strs("O", &["w0=y", "t0=_"]),
    ];
    for (algorithm, kind_str) in [
        (Algorithm::Maxent, GIS_TYPE),
        (Algorithm::MaxentQn, QN_TYPE),
        (Algorithm::Perceptron, PERCEPTRON_TYPE),
        (Algorithm::NaiveBayes, NAIVE_BAYES_TYPE),
    ] {
        let mut params = TrainingParameters::new();
        params.put(Key::ALGORITHM, algorithm.name());
        params.put(Key::ITERATIONS, 5);
        params.put(Key::CUTOFF, 0);
        let (model, _) = crate::trainer::train_events(&events, &params, &mut TrainingConfiguration::default()).unwrap();
        let bytes = model.write().unwrap();
        let mut expected_prefix = Vec::from((kind_str.len() as u16).to_le_bytes());
        expected_prefix.extend_from_slice(kind_str.as_bytes());
        assert!(bytes.starts_with(&expected_prefix), "{algorithm:?}");
        let at = expected_prefix.len();
        assert_eq!(&bytes[at..at + 4], &2i32.to_le_bytes(), "{algorithm:?}");
        // "yes\0😀" is 3 + 1 + 4 = 8 bytes of standard UTF-8 (not 11 bytes of modified UTF-8).
        assert_eq!(&bytes[at + 4..at + 6], &(outcome.len() as u16).to_le_bytes(), "{algorithm:?}");
        assert_eq!(&bytes[at + 6..at + 6 + outcome.len()], outcome.as_bytes(), "{algorithm:?}");
        let read = MaxentModel::from_trained_bytes(&bytes).unwrap();
        assert_eq!(read.outcome_str(0), Some(outcome), "{algorithm:?}");
        assert_eq!(read.outcome_index(outcome), Some(0), "{algorithm:?}");
        assert!(read.has_predicate("t0=_"), "{algorithm:?}");
        if algorithm != Algorithm::Perceptron {
            assert!(read.has_predicate("w0=😀\0"), "{algorithm:?}");
        }
    }
}

#[test]
fn unpaired_surrogates_are_refused_at_the_wire_boundary() {
    use crate::params::Algorithm;
    let lone = JavaString::from_utf16(&[0x52, 0x31, 0xDE00, 0x5F, 0x44, 0x30, 0xD83D]);
    let events = [
        Event::new(lone, vec![JavaString::new("w0=😀"), JavaString::new("t0=_")]),
        Event::from_strs("O", &["w0=x", "t0=_"]),
        Event::from_strs("O", &["w0=y", "t0=_"]),
    ];
    for algorithm in [Algorithm::Maxent, Algorithm::MaxentQn, Algorithm::Perceptron, Algorithm::NaiveBayes] {
        let mut params = TrainingParameters::new();
        params.put(Key::ALGORITHM, algorithm.name());
        params.put(Key::ITERATIONS, 5);
        params.put(Key::CUTOFF, 0);
        let (model, _) = crate::trainer::train_events(&events, &params, &mut TrainingConfiguration::default()).unwrap();
        assert_eq!(model.write(), Err(Error::UnpairedSurrogate), "{algorithm:?}");
    }
}

#[test]
fn the_layout_is_persist_byte_for_byte() {
    let model = trained();
    let bytes = write_gis(&model).unwrap();
    let mut expect = Vec::new();
    expect.extend_from_slice(&(GIS_TYPE.len() as u16).to_le_bytes());
    expect.extend_from_slice(GIS_TYPE.as_bytes());
    expect.extend_from_slice(&3i32.to_le_bytes());
    for o in ["yes", "no", "maybe"] {
        expect.extend_from_slice(&(o.len() as u16).to_le_bytes());
        expect.extend_from_slice(o.as_bytes());
    }
    // Patterns in sorted order: a [0], b [0 1], d [0 2], c [1].
    expect.extend_from_slice(&4i32.to_le_bytes());
    for g in ["1 0", "1 0 1", "1 0 2", "1 1"] {
        expect.extend_from_slice(&(g.len() as u16).to_le_bytes());
        expect.extend_from_slice(g.as_bytes());
    }
    expect.extend_from_slice(&4i32.to_le_bytes());
    for p in ["a", "b", "d", "c"] {
        expect.extend_from_slice(&1u16.to_le_bytes());
        expect.push(p.as_bytes()[0]);
    }
    assert_eq!(&bytes[..expect.len()], &expect[..]);
    assert_eq!(bytes.len(), expect.len() + 6 * 8);
}

#[test]
fn strings_are_measured_in_standard_utf8_and_refused_above_max_utf_bytes() {
    // 16383 emoji are 98298 bytes in modified UTF-8, but 65532 bytes in standard UTF-8.
    let emojis = JavaString::new(&"\u{1F600}".repeat(16383));
    let mut out = ModelOutput { bytes: Vec::new() };
    out.write_utf(&emojis).unwrap();
    assert_eq!(out.bytes.len(), 2 + 16383 * 4);

    // 40000 NULs are 80000 bytes in modified UTF-8, but 40000 bytes in standard UTF-8.
    let nuls = JavaString::new(&"\0".repeat(40000));
    let mut out = ModelOutput { bytes: Vec::new() };
    out.write_utf(&nuls).unwrap();
    assert_eq!(out.bytes.len(), 2 + 40000);

    // Strings over 65535 UTF-8 bytes are refused without CHUNKED-MODEL-PARAMS wrapping.
    let too_long = JavaString::new(&"x".repeat(65536));
    let mut out = ModelOutput { bytes: Vec::new() };
    assert_eq!(out.write_utf(&too_long), Err(Error::StringTooLong { what: "model parameter", bytes: 65536 }));
}
