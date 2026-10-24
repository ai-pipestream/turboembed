use super::LicenceClass::*;
use super::*;

#[test]
fn identifiers_are_unique_and_round_trip() {
    for (i, a) in Licence::ALL.iter().enumerate() {
        assert_eq!(a.spdx().parse::<Licence>(), Ok(*a));
        for b in &Licence::ALL[i + 1..] {
            assert_ne!(a.spdx(), b.spdx());
        }
        match a.legal_url() {
            Some(u) => assert!(u.starts_with("https://"), "{a}"),
            None => {
                assert!(a.spdx().starts_with("LicenseRef-") && (a.refusal().is_some() || a.class() == LocalOnly), "{a}")
            }
        }
    }
}

#[test]
fn classes_follow_the_policy() {
    assert_eq!(Licence::Apache2.class(), Permissive);
    assert_eq!(Licence::CcBy40.class(), Permissive);
    assert_eq!(Licence::OdcBy10.class(), Permissive);
    assert_eq!(Licence::Cc0.class(), Permissive);
    assert_eq!(Licence::CcBy20Fr.class(), Permissive);
    assert_eq!("LicenseRef-CC-BY-2.0-FR".parse::<Licence>(), Ok(Licence::CcBy20Fr));
    assert_eq!(Licence::CcBySa40.class(), ShareAlike);
    assert_eq!(Licence::CcBySa30.class(), ShareAlike);
    assert_eq!(Licence::Lgpl21Only.class(), Lgpl);
    assert_eq!(Licence::Gpl30Only.class(), Gpl);
    // Non-commercial data is local-only (Kristian, 2026-10-14), unless
    // it is also no-derivatives.
    assert_eq!(Licence::CcByNc40.class(), LocalOnly);
    assert_eq!(Licence::CcByNcSa30.class(), LocalOnly);
    assert_eq!(Licence::CcByNcSa40.class(), LocalOnly);
    assert_eq!(Licence::CcByNcNd40.class(), Refused);
    assert_eq!(Licence::CcByNd40.class(), Refused);
    assert_eq!(Licence::Unconfirmed.class(), Refused);
    assert_eq!(Licence::PlatformTerms.class(), Refused);
    assert_eq!(Licence::ResearchOnly.class(), Refused);
    assert_eq!(Licence::NoLicence.class(), Refused);
    assert_eq!(Licence::RestrictedPublic.class(), LocalOnly);
    for l in Licence::ALL {
        assert_eq!(l.class() == Refused, l.refusal().is_some(), "{l}");
        assert_eq!(l.class() == LocalOnly, l.local_only().is_some(), "{l}");
    }
}

#[test]
fn reasons_name_the_terms() {
    assert_eq!(Licence::CcByNcNd40.refusal(), Some(RefusalReason::NoDerivatives));
    assert_eq!(Licence::CcByNd40.refusal(), Some(RefusalReason::NoDerivatives));
    for l in [Licence::CcByNc40, Licence::CcByNcSa30, Licence::CcByNcSa40] {
        assert_eq!(l.local_only(), Some(LocalOnlyReason::NonCommercial), "{l}");
        assert_eq!(l.refusal(), None, "{l}");
    }
    assert_eq!(Licence::RestrictedPublic.local_only(), Some(LocalOnlyReason::HostTerms));
    assert_eq!(Licence::CcBySa40.local_only(), None);
    assert_eq!(Licence::CcByNcNd40.local_only(), None);
    // What the user reads before accepting keeps what the terms allow
    // apart from this project's policy, which the user agrees to.
    assert_eq!(
        LocalOnlyReason::NonCommercial.terms(),
        "The licence grants its rights for non-commercial purposes only."
    );
    assert_eq!(
        LocalOnlyReason::NonCommercial.policy(),
        "This project's policy: the zoo requests data under this licence only at addresses under its original \
             source, and the project never ships, bundles or redistributes the data or any model trained on it. By \
             accepting, you agree to treat any model you train on it as non-commercial and to keep that model on \
             your machine. Model weights under this licence are held to the same: the project never ships or \
             redistributes them or anything converted from them, and you agree to use them, and what you convert \
             from them, for non-commercial purposes only and on your machine."
    );
    assert_eq!(
        LocalOnlyReason::HostTerms.terms(),
        "These are the host's own terms, not an open licence; read them for what they allow."
    );
    assert_eq!(
        LocalOnlyReason::HostTerms.policy(),
        "This project's policy: the zoo requests data under them only at addresses on that host, for local \
             training and evaluation, and the project never ships, bundles or redistributes the data or anything \
             made from it. By accepting, you agree to keep the data and any model you train on it on your machine."
    );
    for r in [LocalOnlyReason::NonCommercial, LocalOnlyReason::HostTerms] {
        assert!(!r.terms().contains("policy") && !r.terms().contains("project"), "{r}");
    }
}

/// Non-commercial terms are accepted like CC BY-SA: one text, the
/// licence's legal code, rather than a host's terms page.
#[test]
fn non_commercial_licences_have_one_text() {
    for l in [Licence::CcByNc40, Licence::CcByNcSa30, Licence::CcByNcSa40] {
        let r = Requirement::of(l).unwrap();
        assert!(r.text.starts_with("https://creativecommons.org/licenses/by-nc"), "{r}");
        assert!(r.text.ends_with("/legalcode"), "{r}");
    }
    assert_eq!(Requirement::of(Licence::RestrictedPublic), None);
}

#[test]
fn non_commercial_data_needs_acceptance_and_never_leaves_the_machine() {
    let default = Policy::default();
    let lax = Policy { gpl: GplAcceptance::NotRequired };
    for l in [Licence::CcByNc40, Licence::CcByNcSa30, Licence::CcByNcSa40] {
        let c = l.class();
        assert!(c.may_fetch(), "{l}");
        assert!(c.needs_acceptance(default) && c.needs_acceptance(lax), "{l}");
        assert!(!c.may_bundle() && !c.may_redistribute(), "{l}");
    }
    for l in [Licence::CcByNcNd40, Licence::CcByNd40] {
        assert!(!l.class().may_fetch() && !l.class().may_bundle(), "{l}");
    }
}

#[test]
fn spdx_names_are_case_sensitive() {
    assert!("apache-2.0".parse::<Licence>().is_err());
    assert!("CC-BY-SA".parse::<Licence>().is_err());
}

#[test]
fn class_order_and_behaviour() {
    assert!(Permissive < Lgpl && Lgpl < ShareAlike && ShareAlike < Gpl && Gpl < LocalOnly && LocalOnly < Refused);
    let default = Policy::default();
    let lax = Policy { gpl: GplAcceptance::NotRequired };
    assert!(Permissive.may_bundle());
    for c in [Lgpl, ShareAlike, Gpl, LocalOnly, Refused] {
        assert!(!c.may_bundle(), "{c}");
    }
    for c in [Permissive, Lgpl, ShareAlike, Gpl] {
        assert!(c.may_redistribute(), "{c}");
    }
    assert!(!LocalOnly.may_redistribute() && !Refused.may_redistribute());
    assert!(LocalOnly.may_fetch());
    assert!(LocalOnly.needs_acceptance(default) && LocalOnly.needs_acceptance(lax));
    assert!(!Refused.may_fetch());
    assert!(!Permissive.needs_acceptance(default));
    assert!(Lgpl.needs_acceptance(default) && Lgpl.needs_acceptance(lax));
    assert!(ShareAlike.needs_acceptance(default) && ShareAlike.needs_acceptance(lax));
    assert!(Gpl.needs_acceptance(default));
    assert!(!Gpl.needs_acceptance(lax));
    for c in LicenceClass::ALL {
        assert_eq!(c.name().parse::<LicenceClass>(), Ok(c));
    }
}

#[test]
fn the_most_restrictive_licence_decides_the_class() {
    use Licence::*;
    assert_eq!(class_of([]), Permissive);
    assert_eq!(class_of([Apache2, CcBy40]), Permissive);
    assert_eq!(class_of([Apache2, CcBySa40]), ShareAlike);
    assert_eq!(class_of([Lgpl21Only, CcBySa30]), ShareAlike);
    assert_eq!(class_of([CcBySa40, Gpl30Only]), Gpl);
    // A model trained on restricted data is restricted, whatever its
    // own licence, and whatever else it was trained on short of refused
    // data.
    assert_eq!(class_of([Apache2, RestrictedPublic]), LocalOnly);
    assert_eq!(class_of([Apache2, CcBy40, RestrictedPublic]), LocalOnly);
    assert_eq!(class_of([Gpl30Only, RestrictedPublic, CcBySa40]), LocalOnly);
    assert_eq!(class_of([RestrictedPublic, ResearchOnly]), Refused);
    // A model trained on non-commercial data is local-only as well,
    // also beside share-alike or restricted data, and refused with
    // no-derivatives data.
    assert_eq!(class_of([Apache2, CcByNcSa40]), LocalOnly);
    assert_eq!(class_of([Apache2, CcByNc40]), LocalOnly);
    assert_eq!(class_of([Mit, CcBy40, CcByNcSa30]), LocalOnly);
    assert_eq!(class_of([Apache2, CcBySa40, CcByNcSa40]), LocalOnly);
    assert_eq!(class_of([RestrictedPublic, CcByNcSa40]), LocalOnly);
    assert_eq!(class_of([Apache2, CcByNcNd40]), Refused);
    assert_eq!(class_of([CcByNcSa40, CcByNd40]), Refused);
    assert_eq!(class_of([CcByNcSa40]), LocalOnly);
    // Order of the licences never matters.
    assert_eq!(class_of([RestrictedPublic, Apache2]), class_of([Apache2, RestrictedPublic]));
    assert_eq!(class_of([CcByNcSa40, Apache2]), class_of([Apache2, CcByNcSa40]));
}

#[test]
fn attribution_requirement_matches_licence_terms() {
    use Licence::*;
    for l in [
        Apache2,
        Mit,
        Bsd2,
        Bsd3,
        CcBy20Fr,
        CcBy25,
        CcBy30,
        CcBy30Us,
        CcBy40,
        OdcBy10,
        CcBySa30,
        CcBySa40,
        Odbl10,
        Lgpl21Only,
        Lgpl21OrLater,
        Lgpl30Only,
        Lgpl30OrLater,
        Gpl20Only,
        Gpl20OrLater,
        Gpl30Only,
        Gpl30OrLater,
        CcByNc40,
        CcByNcSa30,
        CcByNcSa40,
        CcByNd40,
        CcByNcNd40,
    ] {
        assert!(l.requires_attribution(), "{l}");
    }
    for l in
        [Cc0, Pddl10, PublicDomain, RestrictedPublic, ResearchOnly, PlatformTerms, NoLicence, Proprietary, Unconfirmed]
    {
        assert!(!l.requires_attribution(), "{l}");
    }
}
