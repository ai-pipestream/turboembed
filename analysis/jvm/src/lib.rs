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
//!
//! Every function here is checked against vectors written by a real JVM
//! (`tests/vectors/`, made by `java/Vectors.java`).

pub mod chars;
