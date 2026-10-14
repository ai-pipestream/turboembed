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
/// | `Refused` | no | never; the zoo does not list it either |
///
/// LGPL sits below CC BY-SA because its copyleft is weaker; GPL sits
/// above it because GPL is the strongest copyleft the zoo fetches at all.
/// Under the default policy the three middle classes behave the same, so
/// the order between them only decides which class an artifact reports.
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
    /// Non-commercial, no-derivatives, research-only, platform terms, no
    /// licence, proprietary, or unconfirmed terms.
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
            LicenceClass::Refused => "CLASS_REFUSED",
        }
    }

    pub const ALL: [LicenceClass; 5] = [
        LicenceClass::Permissive,
        LicenceClass::Lgpl,
        LicenceClass::ShareAlike,
        LicenceClass::Gpl,
        LicenceClass::Refused,
    ];

    /// Whether an artifact of this class may be carried in a bundle,
    /// checked in or shipped.
    pub const fn may_bundle(self) -> bool {
        matches!(self, LicenceClass::Permissive)
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
            LicenceClass::Lgpl | LicenceClass::ShareAlike => true,
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

/// Why a licence is in [`LicenceClass::Refused`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefusalReason {
    NonCommercial,
    NoDerivatives,
    ResearchOnly,
    PlatformTerms,
    NoLicence,
    Proprietary,
    /// The sources disagree about the terms (Leipzig corpora: one says
    /// CC BY, another CC BY-NC); refused until they are confirmed.
    Unconfirmed,
}

impl fmt::Display for RefusalReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            RefusalReason::NonCommercial => "non-commercial terms",
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
            /// licence applies; this is the licence itself. `None` for the
            /// `LicenseRef-` refusals, which name a kind of terms rather
            /// than one text.
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

use LicenceClass::{Gpl, Lgpl, Permissive, Refused, ShareAlike};

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
    CcByNc40 => "CC-BY-NC-4.0", Refused, Some("https://creativecommons.org/licenses/by-nc/4.0/legalcode");
    CcByNcSa30 => "CC-BY-NC-SA-3.0", Refused, Some("https://creativecommons.org/licenses/by-nc-sa/3.0/legalcode");
    CcByNcSa40 => "CC-BY-NC-SA-4.0", Refused, Some("https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode");
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
            Licence::CcByNc40 | Licence::CcByNcSa30 | Licence::CcByNcSa40 | Licence::CcByNcNd40 => {
                Some(RefusalReason::NonCommercial)
            }
            Licence::CcByNd40 => Some(RefusalReason::NoDerivatives),
            Licence::ResearchOnly => Some(RefusalReason::ResearchOnly),
            Licence::PlatformTerms => Some(RefusalReason::PlatformTerms),
            Licence::NoLicence => Some(RefusalReason::NoLicence),
            Licence::Proprietary => Some(RefusalReason::Proprietary),
            Licence::Unconfirmed => Some(RefusalReason::Unconfirmed),
            _ => None,
        }
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
                None => assert!(a.spdx().starts_with("LicenseRef-") && a.refusal().is_some(), "{a}"),
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
        assert_eq!(Licence::CcByNcSa40.class(), Refused);
        assert_eq!(Licence::CcByNd40.class(), Refused);
        assert_eq!(Licence::Unconfirmed.class(), Refused);
        assert_eq!(Licence::PlatformTerms.class(), Refused);
        for l in Licence::ALL {
            assert_eq!(l.class() == Refused, l.refusal().is_some(), "{l}");
        }
    }

    #[test]
    fn spdx_names_are_case_sensitive() {
        assert!("apache-2.0".parse::<Licence>().is_err());
        assert!("CC-BY-SA".parse::<Licence>().is_err());
    }

    #[test]
    fn class_order_and_behaviour() {
        assert!(Permissive < Lgpl && Lgpl < ShareAlike && ShareAlike < Gpl && Gpl < Refused);
        let default = Policy::default();
        let lax = Policy { gpl: GplAcceptance::NotRequired };
        assert!(Permissive.may_bundle());
        for c in [Lgpl, ShareAlike, Gpl, Refused] {
            assert!(!c.may_bundle(), "{c}");
        }
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
}
