//! Unicode 18.0 character data and the operations the analysis crates
//! need on it: general category, the properties the normalizers test,
//! scripts, case mapping and folding, the normalization forms, confusable
//! prototypes and emoji sequences. The tables are generated from the
//! Unicode Character Database files in `ucd/` by `turbo-unicode-gen` and
//! committed; nothing is read at run time.

/// The Unicode version the tables were generated from.
pub const UNICODE_VERSION: (u8, u8, u8) = (18, 0, 0);
