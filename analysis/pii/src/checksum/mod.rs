//! The checksums and hashes the detectors validate with, written here so
//! the crate depends on no hashing library (SHA-256 and its HMAC are
//! `turbo-sha256`'s).

pub mod base58;
pub mod bech32;
pub mod bic;
pub mod damm;
pub mod es;
pub mod fr;
pub mod icao9303;
pub mod keccak;
pub mod luhn;
pub mod mod11;
pub mod mod97;
pub mod parcel;
pub mod verhoeff;
pub mod vin;
