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
/// | `LocalOnly` | no, nor any model trained on it | requested only at addresses under its original source ([`crate::Artifact::origin`]), after the user accepts its terms (the host's own, or the non-commercial licence), under every policy; by this project's policy neither it nor any model trained on it is shipped, and such a model stays on the user's machine |
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
                "This project's policy: the zoo requests data under them only at addresses on that host, for local \
                 training and evaluation, and the project never ships, bundles or redistributes the data or \
                 anything made from it. By accepting, you agree to keep the data and any model you train on it on \
                 your machine."
            }
            LocalOnlyReason::NonCommercial => {
                "This project's policy: the zoo requests data under this licence only at addresses under its \
                 original source, and the project never ships, bundles or redistributes the data or any model \
                 trained on it. By accepting, you agree to treat any model you train on it as non-commercial and \
                 to keep that model on your machine. Model weights under this licence are held to the same: the \
                 project never ships or redistributes them or anything converted from them, and you agree to use \
                 them, and what you convert from them, for non-commercial purposes only and on your machine."
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

mod kind;

pub use kind::{Licence, Requirement, class_of};

#[cfg(test)]
mod tests;
