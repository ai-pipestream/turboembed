//! Helpers that reproduce JVM behaviour bit for bit, for the Rust port of
//! Apache OpenNLP 3.x.
//!
//! OpenNLP runs on Java, and a few of its results depend on details of the
//! platform rather than of OpenNLP itself:
//!
//! - Context generators, `StringPattern` and character n-grams look at
//!   Java `char` values, which are UTF-16 code units, and
//!   `String.compareTo` orders strings by those units. [`chars`] gives the
//!   UTF-16 units of a `&str` with the UTF-8 byte offset each belongs to,
//!   and the JVM's string order.
//! - Maxent, perceptron and naive Bayes models compute probabilities with
//!   `StrictMath.exp` and `StrictMath.log`, which are fdlibm.
//!   [`strict_math`] ports fdlibm's `exp` and `log` bit for bit.
//! - Code outside the models calls `Math.log`, which HotSpot on x86-64
//!   computes with its own routine rather than fdlibm. [`math::log`] is a
//!   correctly rounded logarithm; its documentation says which JVM build
//!   it was compared with, on which arguments, and where they differ.
//! - Readers, cross-validation and trainers shuffle with a seeded
//!   `java.util.Random`. [`random::JavaRandom`] is that generator.
//! - Event files, model manifests and training reports hold numbers as
//!   `Double.toString` and `Float.toString` write them, and training
//!   parameters and real valued features are read with `parseDouble`,
//!   `parseFloat`, `parseInt` and `parseBoolean`. [`text`] is those
//!   conversions.
//!
//! Every function here is checked against vectors written by a real JVM
//! (`tests/vectors/`, made by `java/Vectors.java`), except [`math::log`],
//! whose JVM values are constants in its tests.

pub mod chars;
pub mod math;
pub mod random;
pub mod strict_math;
pub mod text;
