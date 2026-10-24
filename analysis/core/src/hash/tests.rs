use std::hash::Hasher;

use super::*;
use crate::View;

#[test]
fn empty_and_known_vectors() {
    assert_eq!(fnv1a(b""), FNV1A_OFFSET_BASIS);
    assert_eq!(fnv1a(b"a"), 0xaf63_dc4c_8601_ec8c);
    assert_eq!(Fnv1a::new().bytes(b"foobar").finish(), fnv1a(b"foobar"));
    assert_eq!(Fnv1a::with_seed(fnv1a(b"foo")).bytes(b"bar").finish(), fnv1a(b"foobar"));
}

#[test]
fn string_framing_prevents_adjacent_collisions() {
    let ab_c = Fnv1a::new().str("ab").str("c").finish();
    let a_bc = Fnv1a::new().str("a").str("bc").finish();
    assert_ne!(ab_c, a_bc);
    assert_ne!(Fnv1a::new().str("").str("a").finish(), Fnv1a::new().str("a").finish());
}

#[test]
fn floats_are_canonicalized() {
    assert_eq!(Fnv1a::new().f64(-0.0).finish(), Fnv1a::new().f64(0.0).finish());
    assert_eq!(Fnv1a::new().f32(-0.0).finish(), Fnv1a::new().f32(0.0).finish());
    let nan_a = f64::from_bits(0x7ff8_0000_0000_0001);
    let nan_b = f64::from_bits(0xfff8_1234_5678_9abc);
    assert_eq!(Fnv1a::new().f64(nan_a).finish(), Fnv1a::new().f64(nan_b).finish());
    assert_ne!(Fnv1a::new().f64(f64::NAN).finish(), Fnv1a::new().f64(0.0).finish());
}

#[test]
fn keys_mix_layer_and_instance() {
    let default = Fnv1a::new().key(Key::<View>::DEFAULT).finish();
    let named = Fnv1a::new().key(Key::<View>::named("gold")).finish();
    assert_ne!(default, named);
    assert_eq!(default, Fnv1a::new().key_id(Key::<View>::DEFAULT.id()).finish());
}

#[test]
fn hasher_trait_agrees_with_bytes() {
    let mut h = Fnv1a::new();
    h.write(b"foo");
    h.write_u64(7);
    // Hasher::write_u64 default writes native-endian in std's
    // provided impl? It writes to_le via write; compare against
    // the same byte stream through bytes().
    let mut b = Fnv1a::new();
    b.bytes(b"foo").bytes(&7u64.to_le_bytes());
    assert_eq!(h.finish(), b.finish());
    assert_eq!(Fnv1a::new().usize(7).finish(), Fnv1a::new().u64(7).finish());
    assert_ne!(Fnv1a::new().bool(true).finish(), Fnv1a::new().bool(false).finish());
}
