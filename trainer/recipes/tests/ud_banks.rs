//! The treebank table: unique ids, each SPDX id the one its licence
//! names, and the class a head trained on each carries. Porttinari's
//! text rights are not settled, so its heads are local-only whatever
//! its CC BY 4.0 says (#1663).

use std::collections::BTreeSet;

use turbo::manifest::LicenseClass;
use turbo_trainer_recipes::ud::{TREEBANKS, Treebank, UdSplits};
use turbo_trainer_recipes::ud_heads::{UdHead, record};

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
            other => panic!("{}: licence {other} is not one the table expects", b.id),
        };
        assert_eq!(b.spdx, want, "{}", b.id);
    }
}

#[test]
fn each_treebank_s_heads_carry_its_class_and_porttinari_s_are_never_permissive() {
    let want = |id: &str| match id {
        "ud-es-ancora" | "ud-ca-ancora" | "ud-fi-ftb" | "ud-it-markit" => LicenseClass::Permissive,
        "ud-en-ewt" | "ud-de-gsd" | "ud-fr-gsd" | "ud-nl-alpino" | "ud-ru-gsd" | "ud-pt-gsd" => {
            LicenseClass::ShareAlike
        }
        "ud-pt-porttinari" => LicenseClass::LocalOnly,
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
