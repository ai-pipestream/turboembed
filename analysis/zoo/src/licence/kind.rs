use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{LicenceClass, LocalOnlyReason, RefusalReason};

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
    // SPDX has no identifier for the French port of CC BY 2.0, the licence
    // of most Tatoeba sentences.
    CcBy20Fr => "LicenseRef-CC-BY-2.0-FR", Permissive, Some("https://creativecommons.org/licenses/by/2.0/fr/legalcode");
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
