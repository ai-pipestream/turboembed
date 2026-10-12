//! URL and e-mail address removal: two normalizers that replace each URL
//! and each e-mail address with one space and report what they replaced
//! as placeholders.
//!
//! - [`Urls`] keeps the output of OpenNLP's `UrlCharSequenceNormalizer`,
//!   which the existing language detector models were trained with. It
//!   matches only lowercase `http://` and `https://` URLs and stops a URL
//!   at the first character outside a small ASCII set.
//! - [`BoundedUrls`] ports `BoundedUrlCharSequenceNormalizer`: a URL of
//!   any scheme is taken as a whole the way it is bounded in running text,
//!   and only `http` and `https` URLs are replaced.

mod bounded;
mod mail;
mod plain;

pub use bounded::BoundedUrls;
pub use plain::Urls;
