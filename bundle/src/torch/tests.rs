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

/// A stored zip of `files`, as `torch.save` lays one out, without zip64.
fn archive(files: &[(&str, &[u8])]) -> Vec<u8> {
    let (mut out, mut central) = (Vec::new(), Vec::new());
    for (name, data) in files {
        let (at, crc, n) = (out.len() as u32, zip::crc32(data), data.len() as u32);
        let fixed = |sig: u32| -> Vec<u8> {
            let mut h = sig.to_le_bytes().to_vec();
            h.extend([20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            h.extend(crc.to_le_bytes());
            h.extend(n.to_le_bytes());
            h.extend(n.to_le_bytes());
            h.extend((name.len() as u16).to_le_bytes());
            h.extend([0, 0]);
            h
        };
        let mut local = fixed(0x0403_4b50);
        local.drain(4..6);
        out.extend(local);
        out.extend(name.as_bytes());
        out.extend(*data);
        let mut c = fixed(0x0201_4b50);
        c.extend([0; 10]);
        c.extend(at.to_le_bytes());
        c.extend(name.as_bytes());
        central.extend(c);
    }
    let (offset, size) = (out.len() as u32, central.len() as u32);
    out.extend(central);
    out.extend(0x0605_4b50u32.to_le_bytes());
    out.extend([0, 0, 0, 0]);
    out.extend((files.len() as u16).to_le_bytes());
    out.extend((files.len() as u16).to_le_bytes());
    out.extend(size.to_le_bytes());
    out.extend(offset.to_le_bytes());
    out.extend([0, 0]);
    out
}

/// A pickle that memoizes a tuple of the previous tuple twice, `depth`
/// times: copied out of the memo naively, it doubles at every level.
fn tuple_bomb(depth: u8) -> Vec<u8> {
    let mut p = vec![0x80, 2, b'N', b'q', 0];
    for i in 1..=depth {
        p.extend([b'h', i - 1, b'h', i - 1, 0x86, b'q', i]);
    }
    p.extend(*b"}.");
    p
}

#[test]
fn the_memo_cannot_grow_by_doubling() {
    // The archive itself is read; only its pickle is hostile.
    assert!(read(&archive(&[("archive/data.pkl", &[0x80, 2, b'}', b'.'])])).unwrap().is_empty());
    for depth in [14, 32] {
        let err = error(&archive(&[("archive/data.pkl", &tuple_bomb(depth))]));
        assert!(err.contains("the memo copies more than 65536 values"), "depth {depth}: {err}");
    }
}

#[test]
fn offsets_outside_the_archive_are_errors() {
    let bytes = fixture("state.pt");
    // The zip64 locator's offset of the zip64 end record.
    let locator = bytes.windows(4).rposition(|w| w == [0x50, 0x4b, 0x06, 0x07]).unwrap();
    let mut far = bytes.clone();
    far[locator + 8..locator + 16].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(read(&far).is_err());
    // A central-directory entry's local-header offset: full, so it would
    // need a zip64 field, and one short of full, past the archive.
    let cd = bytes.windows(4).position(|w| w == [0x50, 0x4b, 0x01, 0x02]).unwrap();
    for local in [u32::MAX, u32::MAX - 1] {
        let mut far = bytes.clone();
        far[cd + 42..cd + 46].copy_from_slice(&local.to_le_bytes());
        assert!(read(&far).is_err(), "{local}");
    }
}

/// `NONE, TUPLE2` repeated: each pair wraps the tuple before it, two bytes
/// a level, so 50,000 levels fit in 100 KB. The nesting limit refuses it
/// while it builds, before a deep value exists to drop.
#[test]
fn deep_nesting_is_refused_while_it_builds() {
    let mut p = vec![0x80, 2, b'N'];
    for _ in 0..50_000 {
        p.extend([b'N', 0x86]);
    }
    p.extend(*b"}.");
    let err = error(&archive(&[("archive/data.pkl", &p)]));
    assert!(err.contains("values nest deeper than 32"), "{err}");
}

/// A value memoized once and fetched `gets` times: each fetch copies all
/// of it.
fn fetched(value: &[u8], gets: usize) -> Vec<u8> {
    let mut p = Vec::from([0x80, 2]);
    p.extend(value);
    p.extend([b'q', 0]);
    p.extend([b'h', 0].repeat(gets));
    p.extend(*b"}.");
    p
}

fn string(len: u32) -> Vec<u8> {
    let mut v = vec![b'X'];
    v.extend(len.to_le_bytes());
    v.extend(std::iter::repeat_n(b'a', len as usize));
    v
}

#[test]
fn a_long_string_is_charged_by_its_bytes() {
    // 65,536 bytes fetched 60,000 times would copy 3.9 GB.
    let err = pickle::state_dict(&fetched(&string(1 << 16), 60_000)).unwrap_err();
    assert!(err.contains("the memo copies more than 65536 values"), "{err}");
    // Names as long as a state dict's are fetched as before.
    assert!(pickle::state_dict(&fetched(&string(200), 10_000)).is_ok());
}

#[test]
fn a_tensor_of_high_rank_is_charged_by_its_dimensions() {
    // _rebuild_tensor_v2(storage, 0, (1,) * 16000, (1,) * 16000, False, OrderedDict())
    let mut v = b"ctorch._utils\n_rebuild_tensor_v2\n(".to_vec();
    v.extend(b"(X\x07\x00\x00\x00storagectorch\nFloatStorage\nX\x01\x00\x00\x000X\x03\x00\x00\x00cpuK\x01tQ");
    v.push(b'K');
    v.push(0);
    for _ in 0..2 {
        v.push(b'(');
        v.extend([b'K', 1].repeat(16_000));
        v.push(b't');
    }
    v.extend(b"\x89ccollections\nOrderedDict\n)RtR");
    let err = pickle::state_dict(&fetched(&v, 60_000)).unwrap_err();
    assert!(err.contains("the memo copies more than 65536 values"), "{err}");
}
