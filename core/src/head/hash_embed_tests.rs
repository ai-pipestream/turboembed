//! Hashed word embeddings: the four features as the module comment
//! defines them, rows pinned so a change to the hashing shows, and a
//! tagger reading them checked against a plain reference.

use super::hash_embed::{FEATURES, SEEDS, Tables, features, rows, suffix};
use super::tagger::{TokenTagger, Work, tensors};
use crate::manifest::{
    Adapter, Bilstm, Decode, HashEmbed, HeadKind, HeadSpec, MAX_HASH_DIM, MAX_HASH_ROWS, Scheme, WordPooling,
};
use crate::status::{INVALID_ARGUMENT, INVALID_SHAPE};

const H: HashEmbed = HashEmbed { dim: 3, rows: [11, 5, 7, 6] };

#[test]
fn the_four_features_follow_the_module_comment() {
    assert_eq!(features("Dogs"), ["dogs", "D", "ogs", "Xxxx"].map(String::from));
    assert_eq!(features("1999"), ["1999", "1", "999", "dddd"].map(String::from));
    assert_eq!(features("U.S.A."), ["u.s.a.", "U", ".A.", "X.X.X."].map(String::from));
    assert_eq!(features("Zoë")[3], "Xxx");
    assert_eq!(features("aaaaaaa")[3], "xxxx", "a run is kept to four");
    assert_eq!(features("ab")[2], "ab", "a short word is its own suffix");
    assert_eq!(features(&"a".repeat(100))[3], "LONG");
    assert_eq!(features(""), ["", "", "", ""].map(String::from));
}

#[test]
fn rows_are_in_range_pinned_and_differ_by_seed() {
    for w in ["the", "The", "dog", "1999", "Zoë", ""] {
        for (k, seeds) in rows(&H, w).iter().enumerate() {
            assert!(seeds.iter().all(|&r| r < H.rows[k]), "{w} {k} {seeds:?}");
        }
    }
    let big = HashEmbed { dim: 1, rows: [1 << 20; 4] };
    let r = rows(&big, "dog");
    assert_eq!(r.len() * SEEDS, 16);
    assert!(r[0].iter().any(|&x| x != r[0][0]), "the seeds give different rows");
    // NORM ignores case; PREFIX and SHAPE do not.
    let (a, b) = (rows(&big, "Dog"), rows(&big, "dog"));
    assert_eq!(a[0], b[0]);
    assert_ne!(a[1], b[1]);
    assert_ne!(a[3], b[3]);
    assert_eq!(rows(&big, "dog"), PINNED, "the hashing changed: every trained hash table is void");
}

/// rows(&big, "dog") as first written; a change here invalidates trained heads.
const PINNED: [[u32; SEEDS]; 4] = [
    [128807, 217347, 436463, 869249],
    [76402, 614645, 372927, 839765],
    [128807, 217347, 436463, 869249],
    [741850, 14180, 399277, 178351],
];

fn spec(adapter: Option<(u32, u32)>) -> HeadSpec {
    HeadSpec {
        name: "t".into(),
        layer: "turboembed:pos".into(),
        kind: HeadKind::TokenTagger,
        labels: vec!["A".into(), "B".into()],
        scheme: Scheme::Tags,
        word_pooling: WordPooling::First,
        adapter: adapter.map(|(hidden, layers)| Adapter { bilstm: Bilstm { hidden, layers } }),
        decode: Decode::Argmax,
        biaffine: None,
        pos_input: None,
        hash_embed: Some(H),
        layer_mix: None,
        span: None,
        relation: None,
        coref: None,
        license_class: None,
        artifact: "weights-f32".into(),
        tensor_prefix: "heads.t".into(),
    }
}

fn weights(spec: &HeadSpec, hidden: usize) -> Vec<(String, Vec<f32>)> {
    let mut s = 7u32;
    tensors(spec, hidden)
        .into_iter()
        .map(|(name, shape)| {
            let n = shape.iter().product::<u64>() as usize;
            let v = (0..n)
                .map(|_| {
                    s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
                })
                .collect();
            (name, v)
        })
        .collect()
}

#[test]
fn a_tagger_joins_the_hashed_embedding_after_the_pooled_states() {
    let (hidden, sp) = (4, spec(None));
    let names: Vec<String> = tensors(&sp, hidden).into_iter().map(|(s, _)| s).collect();
    let tables: Vec<String> = (0..4).map(suffix).collect();
    assert_eq!(names[..4], tables[..], "the tables come first without an adapter");
    assert_eq!(tensors(&sp, hidden)[4].1, [2, hidden as u64 + 4 * 3]);
    let w = weights(&sp, hidden);
    let get = |s: &str| w.iter().find(|(k, _)| k == s).map(|(_, v)| v.clone());
    let t = TokenTagger::from_tensors(&sp, hidden, &get).unwrap();
    assert!(t.reads_forms());
    let states: Vec<f32> = (0..3 * hidden).map(|i| i as f32 * 0.1 - 0.4).collect();
    let words = [0, 1, 2];
    let forms = ["The", "dogs", "ran"];
    let (mut out, mut work) = (Vec::new(), Work::default());
    let e = t.scores(&states, &words, &mut out, &mut work).unwrap_err();
    assert_eq!(e.code, INVALID_ARGUMENT);
    let e = t.scores_forms(&states, &words, Some(&forms[..2]), &mut out, &mut work).unwrap_err();
    assert_eq!(e.code, INVALID_SHAPE);
    t.scores_forms(&states, &words, Some(&forms), &mut out, &mut work).unwrap();

    let table = |k: usize| get(&suffix(k)).unwrap();
    let (ow, ob) = (get("out.weight").unwrap(), get("out.bias").unwrap());
    let width = hidden + 12;
    for (i, form) in forms.iter().enumerate() {
        let mut x: Vec<f64> = states[i * hidden..][..hidden].iter().map(|&v| v as f64).collect();
        for (k, seeds) in rows(&H, form).iter().enumerate() {
            let tk = table(k);
            for d in 0..3 {
                x.push(seeds.iter().map(|&r| tk[r as usize * 3 + d] as f64).sum());
            }
        }
        for l in 0..2 {
            let want = ob[l] as f64 + (0..width).map(|j| ow[l * width + j] as f64 * x[j]).sum::<f64>();
            assert!((out[i * 2 + l] as f64 - want).abs() < 1e-5, "{form} {l}");
        }
    }
}

#[test]
fn with_an_adapter_the_tables_follow_it_and_feed_it() {
    let (hidden, sp) = (4, spec(Some((2, 1))));
    let shapes = tensors(&sp, hidden);
    assert_eq!(shapes[0], ("lstm.0.fwd.w".to_string(), vec![8, hidden as u64 + 12]));
    assert_eq!(shapes[6].0, suffix(0));
    let w = weights(&sp, hidden);
    let get = |s: &str| w.iter().find(|(k, _)| k == s).map(|(_, v)| v.clone());
    let t = TokenTagger::from_tensors(&sp, hidden, &get).unwrap();
    let (mut a, mut b, mut work) = (Vec::new(), Vec::new(), Work::default());
    let states = vec![0.25; 2 * hidden];
    t.scores_forms(&states, &[0, 1], Some(&["Ann", "ran"]), &mut a, &mut work).unwrap();
    t.scores_forms(&states, &[0, 1], Some(&["Bob", "ran"]), &mut b, &mut work).unwrap();
    assert_ne!(a, b, "the same states with other words score differently");
}

#[test]
fn bad_tables_and_specs_are_refused() {
    assert_eq!(
        Tables::new(H, [vec![0.0; 32], vec![0.0; 15], vec![0.0; 21], vec![0.0; 18]]).unwrap_err().code,
        INVALID_SHAPE
    );
    // A spec the manifest would refuse is refused here too, before a
    // lookup could take a row modulo zero.
    let empty = || std::array::from_fn(|_| Vec::new());
    for bad in [
        HashEmbed { dim: 1, rows: [0; 4] },
        HashEmbed { dim: 0, rows: [1; 4] },
        HashEmbed { dim: 1, rows: [1, 1, 0, 1] },
        HashEmbed { dim: MAX_HASH_DIM + 1, rows: [1; 4] },
        HashEmbed { dim: 1, rows: [MAX_HASH_ROWS + 1, 1, 1, 1] },
    ] {
        let e = Tables::new(bad, empty()).unwrap_err();
        assert_eq!(e.code, INVALID_ARGUMENT, "{bad:?}");
        assert!(e.message.contains("dim 1 to"), "{e:?}");
    }
    let mut sp = spec(None);
    sp.hash_embed = Some(HashEmbed { dim: 0, rows: [1; 4] });
    assert!(sp.validate("head").is_err());
    sp.hash_embed = Some(HashEmbed { dim: 2, rows: [1, 0, 1, 1] });
    assert!(sp.validate("head").is_err());
    sp.hash_embed = Some(H);
    sp.kind = HeadKind::BiaffineParser;
    sp.decode = Decode::Mst;
    sp.biaffine = Some(crate::manifest::Biaffine { arc: 2, rel: 2 });
    assert!(sp.validate("head").unwrap_err().message.contains("hash_embed"));
    assert_eq!(FEATURES.len(), 4);
}

#[test]
fn non_ascii_and_mixed_script_rows_are_pinned() {
    // Characters, not bytes: a suffix is the last three characters, and
    // an uncased letter (CJK) shapes as x, as spaCy's `is_upper` gives.
    assert_eq!(features("東京"), ["東京", "東", "東京", "xx"].map(String::from));
    assert_eq!(features("Tokyo東京2024"), ["tokyo東京2024", "T", "024", "Xxxxxdddd"].map(String::from));
    let big = HashEmbed { dim: 1, rows: [1 << 20; 4] };
    for (w, want) in PINNED_UNICODE {
        assert_eq!(rows(&big, w), want, "{w}: the hashing changed: every trained hash table is void");
    }
}

/// rows(&big, w) as first written for a Latin word with a diacritic, a
/// CJK word, and a word mixing scripts and digits.
const PINNED_UNICODE: [(&str, [[u32; SEEDS]; 4]); 3] = [
    (
        "Zoë",
        [
            [813574, 384992, 410414, 329289],
            [1023363, 199977, 1036419, 117924],
            [714545, 341061, 172524, 38245],
            [327476, 812623, 56975, 267625],
        ],
    ),
    (
        "東京",
        [
            [154146, 587550, 992401, 622288],
            [431298, 432282, 930494, 308778],
            [154146, 587550, 992401, 622288],
            [607117, 1003624, 645368, 518860],
        ],
    ),
    (
        "Tokyo東京2024",
        [
            [614402, 486703, 1048561, 935150],
            [712305, 405649, 931681, 741005],
            [785229, 672048, 714202, 857314],
            [297800, 96109, 15989, 988853],
        ],
    ),
];
