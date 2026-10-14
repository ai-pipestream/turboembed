//! The model zoo: a licence-aware index of the models, classifier heads,
//! lexicons, label centroid sets and datasets the analysis add-ons use,
//! and a fetcher that enforces the licence policy recorded on #326.
//!
//! - [`Licence`] and [`LicenceClass`]: each artifact's licence, by SPDX
//!   identifier, and the class it falls in. An artifact's class is the
//!   most restrictive of its own licence and the licences of the data it
//!   was trained on; the index records both, and the class.
//! - [`Index`]: the artifacts, parsed strictly from the JSON format
//!   docs/zoo.md describes, which OpenNLP can read as well. Refused
//!   artifacts are never listed.
//! - [`Acceptances`]: which user accepted which licence, in a strict JSON
//!   file of its own.
//! - [`Zoo`]: hands out an artifact by id as verified local files, after
//!   checking the user accepted every licence its class needs. Every file
//!   is checked by size, SHA-256 and SHA-512, cached or fetched.
//!
//! docs/licences.md is the table of approved and refused datasets and
//! checkpoints the index is built from.

pub mod accept;
pub mod digest;
pub mod error;
pub mod fetch;
pub mod index;
pub mod licence;

pub use accept::{Acceptance, Acceptances, UserId};
pub use digest::{Sha256, Sha512};
pub use error::{Check, ZooError};
#[cfg(feature = "http")]
pub use fetch::Http;
pub use fetch::{Fetched, Offline, Transport, Zoo, verify};
pub use index::{Artifact, ArtifactId, Component, File, Format, Index, Kind, Language, TrainingData, Url};
pub use licence::{GplAcceptance, Licence, LicenceClass, Policy, RefusalReason};
