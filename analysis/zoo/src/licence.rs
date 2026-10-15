//! Licences and the classes the zoo sorts them into.
//!
//! A [`Licence`] is one set of terms, named by its SPDX identifier, or by
//! a `LicenseRef-` identifier for terms SPDX has no name for (no licence
//! at all, research-only terms, a platform's terms of service). Each
//! licence belongs to one [`LicenceClass`], and the class decides what the
//! zoo does with an artifact: whether it may be bundled, whether it may
//! be fetched, and whether the user must accept the licence first.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// What the zoo does with an artifact, from least to most restrictive.
/// The order is the derived `Ord`: an artifact's class is the maximum of
/// its own licence's class and the classes of the data it was trained on.
///
/// | Class | Bundled and shipped | Fetched |
/// |---|---|---|
/// | `Permissive` | yes | yes |
/// | `Lgpl` | no | after the user accepts the licence |
/// | `ShareAlike` | no | after the user accepts the licence |
/// | `Gpl` | no | yes; by default only after the user accepts the licence ([`GplAcceptance`]) |
/// | `LocalOnly` | no, nor any model trained on it | only from addresses under its original source ([`crate::Artifact::origin`]), after the user accepts its terms (the host's own, or the non-commercial licence), under every policy; by this project's policy neither it nor any model trained on it is shipped, and such a model stays on the user's machine |
/// | `Refused` | no | never; the zoo does not list it either |
///
/// LGPL sits below CC BY-SA because its copyleft is weaker; GPL sits
/// above it because GPL is the strongest copyleft the zoo fetches at all.
/// Under the default policy those three classes behave the same, so the
/// order between them only decides which class an artifact reports.
/// `LocalOnly` sits above them: nothing in it, and nothing trained on it,
/// leaves the user's machine, which also covers what CC BY-NC-SA's
/// share-alike term asks. A refused licence anywhere still wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LicenceClass {
    /// MIT, BSD, Apache-2.0, CC0, CC BY, ODC-By, public domain.
    Permissive,
    /// LGPL: fetched after the user accepts it, never attributed to this
    /// project, never bundled.
    Lgpl,
    /// CC BY-SA and other share-alike terms (decided 2026-10-14): fetched
    /// after the user accepts them, never bundled. A model trained on
    /// share-alike data is in this class whatever its own licence says.
    ShareAlike,
    /// GPL: fetched on demand, never bundled or checked in.
    Gpl,
    /// Data the zoo may fetch but that this project never ships, bundles or
    /// redistributes, nor any model trained on it, which stays on the
    /// machine of the user who trained it. Two kinds of terms put data here
    /// ([`LocalOnlyReason`]):
    ///
    /// - restricted but public data (IMDB, Sentiment140, the SemEval
    ///   Twitter sets; Kristian, 2026-10-12, on #326), under its host's own
    ///   terms, `LicenseRef-RestrictedPublic`;
    /// - non-commercial data under CC BY-NC or CC BY-NC-SA, such as English
    ///   GUM in Universal Dependencies (Kristian, 2026-10-14). As OpenNLP
    ///   does with data it cannot distribute, code reads such data from
    ///   files the user supplies with nothing to accept; only a fetch
    ///   through the zoo needs the acceptance.
    ///
    /// Either way every address the index gives for the data lies under
    /// its original source ([`crate::Artifact::origin`]), and the zoo
    /// fetches it only after the user accepts its terms, under every
    /// policy. A model trained on it is in this class too, so it is never
    /// bundled, checked in, shipped or listed as a shipped model;
    /// evaluation numbers on it may be recorded, its files not. These
    /// rules are this project's policy, which the user agrees to by
    /// accepting ([`LocalOnlyReason::policy`]); for non-commercial data
    /// they go beyond the licences, which allow sharing the data and
    /// adapted material for non-commercial purposes.
    LocalOnly,
    /// No-derivatives terms (CC BY-ND 4.0, CC BY-NC-ND 4.0), by this
    /// project's default (2026-10-14) rather than by the licences
    /// ([`RefusalReason::NoDerivatives`]); research-only terms, platform
    /// terms, no licence, proprietary or unconfirmed terms, unless an entry
    /// places the data in `LocalOnly` by naming
    /// `LicenseRef-RestrictedPublic`.
    Refused,
}

impl LicenceClass {
    /// The constant the index writes for the class.
    pub const fn name(self) -> &'static str {
        match self {
            LicenceClass::Permissive => "CLASS_PERMISSIVE",
            LicenceClass::Lgpl => "CLASS_LGPL",
            LicenceClass::ShareAlike => "CLASS_SHARE_ALIKE",
            LicenceClass::Gpl => "CLASS_GPL",
            LicenceClass::LocalOnly => "CLASS_LOCAL_ONLY",
            LicenceClass::Refused => "CLASS_REFUSED",
        }
    }

    pub const ALL: [LicenceClass; 6] = [
        LicenceClass::Permissive,
        LicenceClass::Lgpl,
        LicenceClass::ShareAlike,
        LicenceClass::Gpl,
        LicenceClass::LocalOnly,
        LicenceClass::Refused,
    ];

    /// Whether an artifact of this class may be carried in a bundle,
    /// checked in or shipped. A model this project ships must be trained
    /// only on data of a class for which this holds.
    pub const fn may_bundle(self) -> bool {
        matches!(self, LicenceClass::Permissive)
    }

    /// Whether an artifact of this class may leave the machine of the user
    /// who fetched it or trained it: published, shared, or listed as a
    /// model another user can fetch from somewhere other than the original
    /// host. False for `LocalOnly` and `Refused`.
    pub const fn may_redistribute(self) -> bool {
        !matches!(self, LicenceClass::LocalOnly | LicenceClass::Refused)
    }

    /// Whether the zoo lists and fetches artifacts of this class at all.
    pub const fn may_fetch(self) -> bool {
        !matches!(self, LicenceClass::Refused)
    }

    /// Whether a licence of this class must be accepted by the user before
    /// an artifact under it is fetched.
    pub const fn needs_acceptance(self, policy: Policy) -> bool {
        match self {
            LicenceClass::Permissive | LicenceClass::Refused => false,
            LicenceClass::Lgpl | LicenceClass::ShareAlike | LicenceClass::LocalOnly => true,
            LicenceClass::Gpl => matches!(policy.gpl, GplAcceptance::Required),
        }
    }
}

impl fmt::Display for LicenceClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for LicenceClass {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        LicenceClass::ALL.into_iter().find(|c| c.name() == s).ok_or_else(|| format!("{s:?} is not a licence class"))
    }
}

/// Whether GPL artifacts need the user's acceptance before a fetch. The
/// policy on #326 says "download only" for GPL and leaves acceptance open;
/// the default asks for it, as for LGPL and CC BY-SA.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GplAcceptance {
    #[default]
    Required,
    NotRequired,
}

/// The choices the licence policy leaves to the deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Policy {
    pub gpl: GplAcceptance,
}

/// Why a licence is in [`LicenceClass::LocalOnly`]. Under this project's
/// policy neither kind of data, nor any model trained on it, is shipped or
/// redistributed, and such a model stays on the user's machine; the kinds
/// differ in what the user accepts and in what the terms themselves allow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalOnlyReason {
    /// `LicenseRef-RestrictedPublic`: public data under its host's own
    /// terms. The user accepts each host's terms page on its own.
    HostTerms,
    /// CC BY-NC and CC BY-NC-SA. The user accepts the licence's legal code,
    /// once per licence, as for CC BY-SA.
    NonCommercial,
}

impl LocalOnlyReason {
    /// What the terms themselves allow, as the user is told before
    /// accepting them, after the address of their text. It says nothing of
    /// this project's own rules, which are [`LocalOnlyReason::policy`].
    pub const fn terms(self) -> &'static str {
        match self {
            LocalOnlyReason::HostTerms => {
                "These are the host's own terms, not an open licence; read them for what they allow."
            }
            LocalOnlyReason::NonCommercial => "The licence grants its rights for non-commercial purposes only.",
        }
    }

    /// This project's policy for data under such terms, which the user
    /// agrees to by accepting them, as the user is told after
    /// [`LocalOnlyReason::terms`]. For non-commercial data it is stricter
    /// than the licence, which allows sharing the data and adapted material
    /// for non-commercial purposes.
    pub const fn policy(self) -> &'static str {
        match self {
            LocalOnlyReason::HostTerms => {
                "This project's policy: the zoo fetches data under them only from that host, for local training \
                 and evaluation, and the project never ships, bundles or redistributes the data or anything made \
                 from it. By accepting, you agree to keep the data and any model you train on it on your machine."
            }
            LocalOnlyReason::NonCommercial => {
                "This project's policy: the zoo fetches data under this licence only from its original source, \
                 and the project never ships, bundles or redistributes the data or any model trained on it. By \
                 accepting, you agree to treat any model you train on it as non-commercial and to keep that model \
                 on your machine."
            }
        }
    }
}

impl fmt::Display for LocalOnlyReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            LocalOnlyReason::HostTerms => "a host's own terms",
            LocalOnlyReason::NonCommercial => "non-commercial terms",
        })
    }
}

/// Why a licence is in [`LicenceClass::Refused`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefusalReason {
    /// CC BY-ND 4.0 and CC BY-NC-ND 4.0, refused by this project's default
    /// (2026-10-14), not by the licences: version 4.0 allows producing
    /// adapted material (CC BY-NC-ND 4.0 for non-commercial purposes) and
    /// forbids only sharing it, so it would allow a model trained on the
    /// data and kept on the user's machine. Earlier versions differ: CC
    /// BY-ND 3.0 grants no right to make adaptations (its section 3).
    NoDerivatives,
    ResearchOnly,
    PlatformTerms,
    NoLicence,
    Proprietary,
    /// The sources disagree about the terms (Leipzig corpora: one says
    /// CC BY, another CC BY-NC); refused until they are confirmed. Once
    /// confirmed as CC BY or CC BY-NC, such data would be permissive or
    /// local-only.
    Unconfirmed,
}

impl fmt::Display for RefusalReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            RefusalReason::NoDerivatives => "no-derivatives terms",
            RefusalReason::ResearchOnly => "research-only terms",
            RefusalReason::PlatformTerms => "a platform's terms of service",
            RefusalReason::NoLicence => "no licence",
            RefusalReason::Proprietary => "proprietary terms",
            RefusalReason::Unconfirmed => "unconfirmed terms",
        })
    }
}

macro_rules! licences {
    ($($variant:ident => $id:literal, $class:expr, $url:expr;)*) => {
        /// A licence, by its SPDX identifier (`LicenseRef-` for terms SPDX
        /// does not name).
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub enum Licence {
            $($variant,)*
        }

        impl Licence {
            pub const ALL: &'static [Licence] = &[$(Licence::$variant,)*];

            /// The identifier the index and the acceptance file write.
            pub const fn spdx(self) -> &'static str {
                match self {
                    $(Licence::$variant => $id,)*
                }
            }

            /// Where the licence's own text is published. An artifact's
            /// `terms` says where the artifact's publisher states that the
            /// licence applies; this is the licence itself. `None` for
            /// `LicenseRef-RestrictedPublic` and the `LicenseRef-`
            /// refusals, which name a kind of terms rather than one text:
            /// for those the artifact's own terms address is the text.
            pub const fn legal_url(self) -> Option<&'static str> {
                match self {
                    $(Licence::$variant => $url,)*
                }
            }

            /// The class the licence puts an artifact in.
            pub const fn class(self) -> LicenceClass {
                match self.refusal() {
                    Some(_) => LicenceClass::Refused,
                    None => match self {
                        $(Licence::$variant => $class,)*
                    },
                }
            }
        }
    };
}

use LicenceClass::{Gpl, Lgpl, LocalOnly, Permissive, Refused, ShareAlike};

licences! {
    Apache2 => "Apache-2.0", Permissive, Some("https://www.apache.org/licenses/LICENSE-2.0");
    Mit => "MIT", Permissive, Some("https://opensource.org/license/mit");
    Bsd2 => "BSD-2-Clause", Permissive, Some("https://opensource.org/license/bsd-2-clause");
    Bsd3 => "BSD-3-Clause", Permissive, Some("https://opensource.org/license/bsd-3-clause");
    Cc0 => "CC0-1.0", Permissive, Some("https://creativecommons.org/publicdomain/zero/1.0/legalcode");
    CcBy25 => "CC-BY-2.5", Permissive, Some("https://creativecommons.org/licenses/by/2.5/legalcode");
    CcBy30 => "CC-BY-3.0", Permissive, Some("https://creativecommons.org/licenses/by/3.0/legalcode");
    CcBy30Us => "CC-BY-3.0-US", Permissive, Some("https://creativecommons.org/licenses/by/3.0/us/legalcode");
    CcBy40 => "CC-BY-4.0", Permissive, Some("https://creativecommons.org/licenses/by/4.0/legalcode");
    OdcBy10 => "ODC-By-1.0", Permissive, Some("https://opendatacommons.org/licenses/by/1-0/");
    Pddl10 => "PDDL-1.0", Permissive, Some("https://opendatacommons.org/licenses/pddl/1-0/");
    PublicDomain => "LicenseRef-PublicDomain", Permissive, Some("https://creativecommons.org/publicdomain/mark/1.0/");
    CcBySa30 => "CC-BY-SA-3.0", ShareAlike, Some("https://creativecommons.org/licenses/by-sa/3.0/legalcode");
    CcBySa40 => "CC-BY-SA-4.0", ShareAlike, Some("https://creativecommons.org/licenses/by-sa/4.0/legalcode");
    Odbl10 => "ODbL-1.0", ShareAlike, Some("https://opendatacommons.org/licenses/odbl/1-0/");
    Lgpl21Only => "LGPL-2.1-only", Lgpl, Some("https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html");
    Lgpl21OrLater => "LGPL-2.1-or-later", Lgpl, Some("https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html");
    Lgpl30Only => "LGPL-3.0-only", Lgpl, Some("https://www.gnu.org/licenses/lgpl-3.0.html");
    Lgpl30OrLater => "LGPL-3.0-or-later", Lgpl, Some("https://www.gnu.org/licenses/lgpl-3.0.html");
    Gpl20Only => "GPL-2.0-only", Gpl, Some("https://www.gnu.org/licenses/old-licenses/gpl-2.0.html");
    Gpl20OrLater => "GPL-2.0-or-later", Gpl, Some("https://www.gnu.org/licenses/old-licenses/gpl-2.0.html");
    Gpl30Only => "GPL-3.0-only", Gpl, Some("https://www.gnu.org/licenses/gpl-3.0.html");
    Gpl30OrLater => "GPL-3.0-or-later", Gpl, Some("https://www.gnu.org/licenses/gpl-3.0.html");
    RestrictedPublic => "LicenseRef-RestrictedPublic", LocalOnly, None;
    CcByNc40 => "CC-BY-NC-4.0", LocalOnly, Some("https://creativecommons.org/licenses/by-nc/4.0/legalcode");
    CcByNcSa30 => "CC-BY-NC-SA-3.0", LocalOnly, Some("https://creativecommons.org/licenses/by-nc-sa/3.0/legalcode");
    CcByNcSa40 => "CC-BY-NC-SA-4.0", LocalOnly, Some("https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode");
    CcByNd40 => "CC-BY-ND-4.0", Refused, Some("https://creativecommons.org/licenses/by-nd/4.0/legalcode");
    CcByNcNd40 => "CC-BY-NC-ND-4.0", Refused, Some("https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode");
    ResearchOnly => "LicenseRef-ResearchOnly", Refused, None;
    PlatformTerms => "LicenseRef-PlatformTerms", Refused, None;
    NoLicence => "LicenseRef-NoLicence", Refused, None;
    Proprietary => "LicenseRef-Proprietary", Refused, None;
    Unconfirmed => "LicenseRef-Unconfirmed", Refused, None;
}

impl Licence {
    /// Why the licence is refused, or `None` when the zoo may fetch under
    /// it.
    pub const fn refusal(self) -> Option<RefusalReason> {
        match self {
            Licence::CcByNd40 | Licence::CcByNcNd40 => Some(RefusalReason::NoDerivatives),
            Licence::ResearchOnly => Some(RefusalReason::ResearchOnly),
            Licence::PlatformTerms => Some(RefusalReason::PlatformTerms),
            Licence::NoLicence => Some(RefusalReason::NoLicence),
            Licence::Proprietary => Some(RefusalReason::Proprietary),
            Licence::Unconfirmed => Some(RefusalReason::Unconfirmed),
            _ => None,
        }
    }

    /// Why the licence is local-only, or `None` when it is in another
    /// class.
    pub const fn local_only(self) -> Option<LocalOnlyReason> {
        match self {
            Licence::RestrictedPublic => Some(LocalOnlyReason::HostTerms),
            Licence::CcByNc40 | Licence::CcByNcSa30 | Licence::CcByNcSa40 => Some(LocalOnlyReason::NonCommercial),
            _ => None,
        }
    }
}

/// The class of an artifact under `licences`: its own licence and those of
/// everything it was trained on. The most restrictive wins; no licences at
/// all is `Permissive`. Use it for a model trained locally too: a model
/// trained on `LocalOnly` data, restricted or non-commercial, is
/// `LocalOnly` whatever its own licence.
pub fn class_of(licences: impl IntoIterator<Item = Licence>) -> LicenceClass {
    licences.into_iter().map(Licence::class).max().unwrap_or(LicenceClass::Permissive)
}

/// Terms a user must accept before a fetch: a licence and the address of
/// the text they are shown. For a licence with a `legal_url` that is the
/// address; for `LicenseRef-RestrictedPublic` it is the terms page of the
/// data's host, as the index records it, so accepting one host's terms
/// never covers another's.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Requirement {
    pub licence: Licence,
    pub text: String,
}

impl Requirement {
    /// The terms of a licence with a single text: the licence and its
    /// `legal_url`. `None` for a licence without one.
    pub fn of(licence: Licence) -> Option<Requirement> {
        licence.legal_url().map(|u| Requirement { licence, text: u.to_owned() })
    }
}

impl fmt::Display for Requirement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.licence, self.text)
    }
}

impl fmt::Display for Licence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.spdx())
    }
}

impl FromStr for Licence {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        Licence::ALL
            .iter()
            .copied()
            .find(|l| l.spdx() == s)
            .ok_or_else(|| format!("{s:?} is not a licence the zoo knows (docs/zoo.md lists them)"))
    }
}

impl Serialize for Licence {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.spdx())
    }
}

impl<'de> Deserialize<'de> for Licence {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

impl Serialize for LicenceClass {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.name())
    }
}

impl<'de> Deserialize<'de> for LicenceClass {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
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
                None => assert!(
                    a.spdx().starts_with("LicenseRef-") && (a.refusal().is_some() || a.class() == LocalOnly),
                    "{a}"
                ),
            }
        }
    }

    #[test]
    fn classes_follow_the_policy() {
        assert_eq!(Licence::Apache2.class(), Permissive);
        assert_eq!(Licence::CcBy40.class(), Permissive);
        assert_eq!(Licence::OdcBy10.class(), Permissive);
        assert_eq!(Licence::Cc0.class(), Permissive);
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
            "This project's policy: the zoo fetches data under this licence only from its original source, and the \
             project never ships, bundles or redistributes the data or any model trained on it. By accepting, you \
             agree to treat any model you train on it as non-commercial and to keep that model on your machine."
        );
        assert_eq!(
            LocalOnlyReason::HostTerms.terms(),
            "These are the host's own terms, not an open licence; read them for what they allow."
        );
        assert_eq!(
            LocalOnlyReason::HostTerms.policy(),
            "This project's policy: the zoo fetches data under them only from that host, for local training and \
             evaluation, and the project never ships, bundles or redistributes the data or anything made from it. \
             By accepting, you agree to keep the data and any model you train on it on your machine."
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
}
