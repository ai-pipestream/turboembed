//! The treebank table: unique ids, each SPDX id the one its licence
//! names, and the class a head trained on each carries. Porttinari's
//! text rights are not settled, so its heads are local-only whatever
//! its CC BY 4.0 says (#1663).

use std::collections::BTreeSet;

use turbo::manifest::LicenseClass;
use turbo_trainer_recipes::ud::{TREEBANKS, Treebank, UdSplits};
use turbo_trainer_recipes::ud_heads::{UdHead, record};

/// The share-alike treebanks: EWT and the 23 languages whose treebanks
/// #1651 replaces with accept-first heads (Kazakh KTB is not here: 31
/// training sentences and no dev split).
const SHARE_ALIKE: [&str; 24] = [
    "ud-en-ewt",
    "ud-af-afribooms",
    "ud-hy-bsut",
    "ud-hr-set",
    "ud-da-ddt",
    "ud-nl-alpino",
    "ud-fr-gsd",
    "ud-ka-glc",
    "ud-de-gsd",
    "ud-is-icepahc",
    "ud-id-gsd",
    "ud-ga-idt",
    "ud-ko-kaist",
    "ud-lv-lvtb",
    "ud-no-bokmaal",
    "ud-fa-perdt",
    "ud-pt-gsd",
    "ud-ro-rrt",
    "ud-ru-gsd",
    "ud-sr-set",
    "ud-sk-snk",
    "ud-sl-ssj",
    "ud-sv-talbanken",
    "ud-tr-boun",
];

fn class(bank: &Treebank, head: UdHead) -> LicenseClass {
    record(bank, head, &UdSplits::default(), Vec::new(), true).class().unwrap()
}

#[test]
fn every_treebank_has_its_own_id_and_the_spdx_id_of_its_licence() {
    let ids: BTreeSet<&str> = TREEBANKS.iter().map(|b| b.id).collect();
    assert_eq!(ids.len(), TREEBANKS.len(), "a treebank id appears twice");
    for b in &TREEBANKS {
        let want = match b.licence {
            "CC BY 4.0" => "CC-BY-4.0",
            "CC BY-SA 4.0" => "CC-BY-SA-4.0",
            "CC BY-SA 3.0" => "CC-BY-SA-3.0",
            other => panic!("{}: licence {other} is not one the table expects", b.id),
        };
        assert_eq!(b.spdx, want, "{}", b.id);
    }
}

#[test]
fn each_treebank_s_heads_carry_its_class_and_porttinari_s_are_never_permissive() {
    let want = |id: &str| match id {
        "ud-es-ancora" | "ud-ca-ancora" | "ud-fi-ftb" | "ud-it-markit" => LicenseClass::Permissive,
        "ud-pt-porttinari" => LicenseClass::LocalOnly,
        other if SHARE_ALIKE.contains(&other) => LicenseClass::ShareAlike,
        other => panic!("{other}: no expected class"),
    };
    for b in &TREEBANKS {
        for head in [UdHead::Lemma, UdHead::Deps] {
            assert_eq!(class(b, head), want(b.id), "{} {head:?}", b.id);
        }
    }
    let p = TREEBANKS.iter().find(|b| b.id == "ud-pt-porttinari").unwrap();
    assert!(p.recipe_only && p.spdx == "CC-BY-4.0", "Porttinari is CC BY 4.0 and recipe-only");
    assert_ne!(class(p, UdHead::Lemma), LicenseClass::Permissive);
}

#[test]
fn every_share_alike_treebank_is_in_the_table_and_kazakh_is_not() {
    for id in SHARE_ALIKE {
        assert!(TREEBANKS.iter().any(|b| b.id == id), "{id} is not in the treebank table");
    }
    assert!(TREEBANKS.iter().all(|b| !b.id.starts_with("ud-kk-")), "KTB is too small to train on");
    let ga = TREEBANKS.iter().find(|b| b.id == "ud-ga-idt").unwrap();
    assert_eq!((ga.licence, ga.spdx), ("CC BY-SA 3.0", "CC-BY-SA-3.0"));
}

#[test]
fn every_treebank_pinned_in_a_banks_module_is_in_the_table() {
    for src in [
        include_str!("../src/ud_banks.rs"),
        include_str!("../src/ud_banks_sa.rs"),
        include_str!("../src/ud_banks_sa2.rs"),
    ] {
        let ids: Vec<&str> = src.lines().filter_map(|l| l.trim().strip_prefix("id: \"")?.strip_suffix("\",")).collect();
        assert_eq!(ids.len(), src.matches("pub const ").count(), "a const without an id line");
        for id in ids {
            assert!(TREEBANKS.iter().any(|b| b.id == id), "{id} is pinned but not in TREEBANKS");
        }
    }
}
