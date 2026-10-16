//! N-gram models, character n-gram models, n-gram probability utilities, and
//! Stupid Backoff language modeling (`opennlp.tools.ngram` and
//! `opennlp.tools.languagemodel`).
//!
//! - [`NGramModel`] counts token n-grams in insertion order, supports frequency
//!   cutoffs, converts to [`turbo_maxent::Dictionary`], and reads/writes
//!   OpenNLP's `<dictionary>` XML format with `count` attributes.
//! - [`NGramCharModel`] counts character n-grams lowercased with Java's
//!   `Character.toLowerCase(int)`.
//! - [`NGramGenerator`] and [`NGramUtils`] slice token and character sequences
//!   into n-grams and compute Maximum Likelihood, Laplace-smoothed,
//!   prior-smoothed, linearly interpolated, and missing-mass probabilities.
//! - [`LanguageModel`] and [`NGramLanguageModel`] score token sequences and
//!   predict next tokens using Stupid Backoff (`0.4` backoff weight) with
//!   `StrictMath.log` and `StrictMath.exp` from [`turbo_jvm::strict_math`].

pub mod char_model;
pub mod error;
pub mod language_model;
pub mod model;
pub mod utils;

pub use char_model::NGramCharModel;
pub use error::{Error, Result};
pub use language_model::{DEFAULT_N, LanguageModel, NGramLanguageModel};
pub use model::NGramModel;
pub use utils::{NGramGenerator, NGramUtils};
