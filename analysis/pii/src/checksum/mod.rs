//! The checksums and hashes the detectors validate with, written here so
//! the crate depends on no hashing library.

pub mod base58;
pub mod bech32;
pub mod bic;
pub mod es;
pub mod fr;
pub mod icao9303;
pub mod keccak;
pub mod luhn;
pub mod mod11;
pub mod mod97;
pub mod sha256;
