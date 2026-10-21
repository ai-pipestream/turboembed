//! The reader on small checkpoints `bundle/reference/torch_fixture.py`
//! writes with torch itself, and on those checkpoints changed byte by
//! byte into what the reader refuses.

use super::*;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/torch");

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!("{DIR}/{name}")).unwrap()
}

fn error(bytes: &[u8]) -> String {
    match read(bytes) {
        Ok(_) => panic!("read accepted it"),
        Err(e) => e,
    }
}

#[test]
fn a_state_dict_reads_as_torch_load_reads_it() {
    let bytes = fixture("state.pt");
    let tensors = read(&bytes).unwrap();
    let want: serde_json::Value = serde_json::from_slice(&fixture("state.json")).unwrap();
    let want = want.as_object().unwrap();
    assert_eq!(tensors.keys().collect::<Vec<_>>(), {
        let mut k: Vec<_> = want.keys().collect();
        k.sort();
        k
    });
    for (name, t) in &tensors {
        let w = &want[name];
        let shape: Vec<usize> = w["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        assert_eq!(t.shape, shape, "{name}");
        let values: Vec<f32> = t.data.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        let expect: Vec<f32> = w["values"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
        assert_eq!(values, expect, "{name}");
    }
    // The same tensors as safetensors.
    let out = to_safetensors(&tensors, |k| Some(k.to_owned()));
    let file = turbo::safetensors::File::parse("converted", &out).unwrap();
    assert_eq!(file.get("shared.tail", turbo::safetensors::Dtype::F32, 1).unwrap().shape, [6]);
}

#[test]
fn what_a_state_dict_of_f32_tensors_does_not_hold_is_refused() {
    assert!(error(&fixture("noncontiguous.pt")).contains("not contiguous"));
    assert!(error(&fixture("float64.pt")).contains("global torch.DoubleStorage is not read"));
    assert!(error(&fixture("legacy.pt")).contains("legacy"));
}

/// The archive's entry `name`: where its bytes start and how long it is.
fn entry(bytes: &[u8], name: &str) -> (usize, usize) {
    let e = zip::entries(bytes).unwrap().into_iter().find(|e| e.name.ends_with(name)).unwrap();
    (e.data.as_ptr() as usize - bytes.as_ptr() as usize, e.data.len())
}

#[test]
fn a_changed_entry_fails_its_crc() {
    let mut bytes = fixture("state.pt");
    let (at, _) = entry(&bytes, "/data/0");
    bytes[at] ^= 1;
    assert!(error(&bytes).contains("CRC-32 mismatch"));
}

#[test]
fn a_compressed_entry_is_refused() {
    let mut bytes = fixture("state.pt");
    // The first central-directory entry's method field.
    let cd = bytes.windows(4).position(|w| w == [0x50, 0x4b, 0x01, 0x02]).unwrap();
    bytes[cd + 10] = 8;
    assert!(error(&bytes).contains("compressed"));
}

#[test]
fn a_truncated_archive_is_refused() {
    let bytes = fixture("state.pt");
    assert!(read(&bytes[..bytes.len() / 2]).is_err());
}

fn pickle_bytes() -> Vec<u8> {
    let bytes = fixture("state.pt");
    let (at, n) = entry(&bytes, "/data.pkl");
    bytes[at..at + n].to_vec()
}

fn replace(b: &[u8], from: &[u8], to: &[u8]) -> Vec<u8> {
    let at = b.windows(from.len()).position(|w| w == from).unwrap();
    [&b[..at], to, &b[at + from.len()..]].concat()
}

#[test]
fn the_pickle_runs_no_code() {
    let p = pickle_bytes();
    assert!(pickle::state_dict(&p).is_ok());
    // A global that would run code, in place of OrderedDict.
    let evil = replace(&p, b"collections\nOrderedDict\n", b"posix\nsystem\n");
    assert_eq!(pickle::state_dict(&evil).unwrap_err(), "pickle: global posix.system is not read");
    let evil = replace(&p, b"torch._utils\n_rebuild_tensor_v2\n", b"builtins\neval\n");
    assert_eq!(pickle::state_dict(&evil).unwrap_err(), "pickle: global builtins.eval is not read");
    // STACK_GLOBAL, which takes its names from the stack, is not an opcode
    // the reader knows.
    let mut stack_global = p.clone();
    stack_global.insert(2, 0x93);
    assert_eq!(pickle::state_dict(&stack_global).unwrap_err(), "pickle: opcode 0x93 is not read");
    // A truncated pickle never reaches STOP.
    assert!(pickle::state_dict(&p[..p.len() - 1]).is_err());
}
