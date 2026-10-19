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
    let read = MaxentModel::read(&bytes).unwrap();
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
fn an_outcome_with_a_lone_surrogate_is_written_as_write_utf_writes_it() {
    // The class of "😀" with the lemma "_": R1<low>_D0<high>.
    let class = JavaString::from_utf16(&[0x52, 0x31, 0xDE00, 0x5F, 0x44, 0x30, 0xD83D]);
    let events = [
        Event::new(class.clone(), vec![JavaString::new("w0=😀"), JavaString::new("t0=_")]),
        Event::from_strs("O", &["w0=x", "t0=_"]),
        Event::from_strs("O", &["w0=y", "t0=_"]),
    ];
    let index = DataIndex::index(&events, &IndexParams::new(IndexerKind::TwoPass, 0, true)).unwrap();
    let mut params = TrainingParameters::new();
    params.put(Key::ITERATIONS, 5);
    let model = crate::gis::train(&index, &params, &mut TrainingConfiguration::default()).unwrap();
    let bytes = write_gis(&model).unwrap();
    // After "GIS", the int, the double and the outcome count: the class
    // as writeUTF writes it, each surrogate as three bytes.
    let at = 5 + 4 + 8 + 4;
    assert_eq!(&bytes[at..at + 2], &[0, 11]);
    assert_eq!(&bytes[at + 2..at + 13], b"R1\xED\xB8\x80_D0\xED\xA0\xBD");
    let read = MaxentModel::read(&bytes).unwrap();
    assert_eq!(read.outcome(0), &class);
    assert_eq!(read.outcome_index(&class), Some(0));
    assert_eq!(read.outcome_texts(), Err(&class));
}

#[test]
fn the_qn_and_perceptron_writers_write_a_lone_surrogate_outcome_as_write_utf_does() {
    use crate::params::Algorithm;
    let class = JavaString::from_utf16(&[0x52, 0x31, 0xDE00, 0x5F, 0x44, 0x30, 0xD83D]);
    let events = [
        Event::new(class.clone(), vec![JavaString::new("w0=😀"), JavaString::new("t0=_")]),
        Event::from_strs("O", &["w0=x", "t0=_"]),
        Event::from_strs("O", &["w0=y", "t0=_"]),
    ];
    for (algorithm, kind) in [(Algorithm::MaxentQn, &b"\0\x02QN"[..]), (Algorithm::Perceptron, b"\0\x0aPerceptron")] {
        let mut params = TrainingParameters::new();
        params.put(Key::ALGORITHM, algorithm.name());
        params.put(Key::ITERATIONS, 5);
        params.put(Key::CUTOFF, 0);
        let (model, _) = crate::trainer::train_events(&events, &params, &mut TrainingConfiguration::default()).unwrap();
        let bytes = model.write().unwrap();
        assert!(bytes.starts_with(kind), "{algorithm:?}");
        // The outcome count, then the class as writeUTF writes it, each
        // surrogate as three bytes.
        let at = kind.len();
        assert_eq!(&bytes[at..at + 4], &[0, 0, 0, 2], "{algorithm:?}");
        assert_eq!(&bytes[at + 4..at + 6], &[0, 11], "{algorithm:?}");
        assert_eq!(&bytes[at + 6..at + 17], b"R1\xED\xB8\x80_D0\xED\xA0\xBD", "{algorithm:?}");
        let read = MaxentModel::read(&bytes).unwrap();
        assert_eq!(read.outcome(0), &class, "{algorithm:?}");
        assert_eq!(read.outcome_index(&class), Some(0), "{algorithm:?}");
    }
}

#[test]
fn the_layout_is_persist_byte_for_byte() {
    let model = trained();
    let bytes = write_gis(&model).unwrap();
    let mut expect = Vec::new();
    expect.extend_from_slice(&[0, 3, b'G', b'I', b'S']);
    expect.extend_from_slice(&1i32.to_be_bytes());
    expect.extend_from_slice(&1.0f64.to_bits().to_be_bytes());
    expect.extend_from_slice(&3i32.to_be_bytes());
    for o in ["yes", "no", "maybe"] {
        expect.extend_from_slice(&(o.len() as u16).to_be_bytes());
        expect.extend_from_slice(o.as_bytes());
    }
    // Patterns in sorted order: a [0], b [0 1], d [0 2], c [1].
    expect.extend_from_slice(&4i32.to_be_bytes());
    for g in ["1 0", "1 0 1", "1 0 2", "1 1"] {
        expect.extend_from_slice(&(g.len() as u16).to_be_bytes());
        expect.extend_from_slice(g.as_bytes());
    }
    expect.extend_from_slice(&4i32.to_be_bytes());
    for p in ["a", "b", "d", "c"] {
        expect.extend_from_slice(&[0, 1, p.as_bytes()[0]]);
    }
    assert_eq!(&bytes[..expect.len()], &expect[..]);
    assert_eq!(bytes.len(), expect.len() + 6 * 8);
}

fn utf8_len(s: &JavaString) -> usize {
    s.to_str().unwrap().len()
}

#[test]
fn long_strings_are_chunked_by_utf8_bytes() {
    // 70000 ASCII bytes: two chunks of 65535 and 4465 bytes.
    let long = JavaString::new(&"x".repeat(70000));
    let chunks = split_by_utf8_length(&long);
    assert_eq!(chunks.iter().map(|c| c.len()).collect::<Vec<_>>(), [65535, 4465]);
    // Three byte characters do not straddle a chunk end.
    let long = JavaString::new(&"\u{20AC}".repeat(30000));
    let chunks = split_by_utf8_length(&long);
    assert_eq!(chunks.iter().map(|c| c.len()).collect::<Vec<_>>(), [65535, 24465]);
    assert_eq!(utf8_len(&long), 90000);
    let mut out = ModelOutput { bytes: Vec::new() };
    out.write_utf(&long).unwrap();
    let signature = b"CHUNKED-MODEL-PARAMS:2";
    assert_eq!(&out.bytes[..2 + signature.len()], [&[0, signature.len() as u8][..], signature].concat());
    assert_eq!(out.bytes.len(), 2 + signature.len() + 2 + 65535 + 2 + 24465);
}

#[test]
fn a_chunk_of_supplementary_characters_overflows_write_utf() {
    // 16383 emoji fit a 65532 byte UTF-8 chunk but are 98298 bytes of
    // modified UTF-8, which writeUTF refuses, as the JVM's does.
    let long = JavaString::new(&"\u{1F600}".repeat(17000));
    let chunks = split_by_utf8_length(&long);
    assert_eq!(chunks[0].len(), 16383 * 6);
    let mut out = ModelOutput { bytes: Vec::new() };
    assert_eq!(
        out.write_utf(&long),
        Err(Error::StringTooLong { what: "chunk of a model parameter", bytes: 16383 * 6 })
    );
}

#[test]
fn a_nul_is_one_utf8_byte_but_two_written() {
    assert_eq!(next_char(b"\xC0\x80a", 0), Some((2, 1)));
    assert_eq!(next_char(b"\xC3\xA9", 0), Some((2, 2)));
    assert_eq!(next_char(b"\xC0", 0), None);
    assert_eq!(next_char(b"\xE2\x82", 0), None);
    // 40000 NULs: 80000 bytes of modified UTF-8, so the string is
    // chunked, but 40000 bytes of UTF-8, so it is one chunk, whose
    // writeUTF throws in the JVM.
    let nuls = JavaString::new(&"\0".repeat(40000));
    assert_eq!(nuls.len(), 80000);
    let chunks = split_by_utf8_length(&nuls);
    assert_eq!(chunks.iter().map(|c| c.len()).collect::<Vec<_>>(), [80000]);
    let mut out = ModelOutput { bytes: Vec::new() };
    assert_eq!(out.write_utf(&nuls), Err(Error::StringTooLong { what: "chunk of a model parameter", bytes: 80000 }));
    // NULs after the first chunk make a second chunk of twice their count.
    let late = JavaString::new(&format!("{}{}", "x".repeat(65535), "\0".repeat(1000)));
    let chunks = split_by_utf8_length(&late);
    assert_eq!(chunks.iter().map(|c| c.len()).collect::<Vec<_>>(), [65535, 2000]);
    let mut out = ModelOutput { bytes: Vec::new() };
    out.write_utf(&late).unwrap();
    let signature = b"CHUNKED-MODEL-PARAMS:2";
    assert_eq!(out.bytes.len(), 2 + signature.len() + 2 + 65535 + 2 + 2000);
}

#[test]
fn a_lone_surrogate_cuts_a_chunked_string() {
    let mut units: Vec<u16> = vec![b'y' as u16; 70000];
    units[66000] = 0xD83D;
    let long = JavaString::from_utf16(&units);
    let chunks = split_by_utf8_length(&long);
    assert_eq!(chunks.iter().map(|c| c.len()).collect::<Vec<_>>(), [65535, 465]);

    units[0] = 0xD83D;
    let at_start = JavaString::from_utf16(&units);
    let chunks = split_by_utf8_length(&at_start);
    assert_eq!(chunks.iter().map(|c| c.len()).collect::<Vec<_>>(), [0]);
}
