//! The AES-256-GCM and FF1 strategy arms (#1885): their messages to
//! `turbo-pii`'s encryptors, each under an encryption key registered with
//! `--pii-encryption-key` and named by reference, never carried.
//!
//! A server built without the `pii-reversible` feature knows neither arm:
//! the policy is refused as naming an unknown arm, or, with
//! `redact_and_flag_unknown`, the mentions are masked and flagged, as for
//! any arm this server does not know.

use turbo_pii::rewrite::RewriteStrategy;

use super::error::At;
use super::error::PolicyError;
use super::resolve::PolicyResolver;
use crate::proto::result::{PiiAesGcmStrategy, PiiFf1Strategy};

#[cfg(feature = "pii-reversible")]
mod enabled {
    use turbo_pii::PiiKind;
    use turbo_pii::rewrite::{AesGcmEncryptor, Ff1Alphabet, Ff1Encryptor, RewriteStrategy};

    use super::super::error::{At, PolicyError};
    use super::super::resolve::{PolicyResolver, check_key_id};
    use crate::proto::result::{PiiAesGcmStrategy, PiiFf1Alphabet, PiiFf1Strategy, PiiPolicyKeyRef};

    fn key<'a>(
        at: At,
        field: &'static str,
        key: Option<&PiiPolicyKeyRef>,
        resolver: &'a dyn PolicyResolver,
    ) -> Result<&'a [u8], PolicyError> {
        let key = key.ok_or(PolicyError::Missing { at, field })?;
        check_key_id(at, &key.id)?;
        resolver.encryption_key(&key.id, key.version).map_err(|e| PolicyError::Unresolved {
            at,
            field,
            reason: e.reason(),
        })
    }

    // CryptError names lengths only, never the key.
    fn invalid(at: At, e: impl std::fmt::Display) -> PolicyError {
        PolicyError::Invalid { at, reason: e.to_string() }
    }

    pub fn aes_gcm(
        at: At,
        s: &PiiAesGcmStrategy,
        resolver: &dyn PolicyResolver,
    ) -> Result<RewriteStrategy, PolicyError> {
        let bytes = key(at, "aes_256_gcm.key", s.key.as_ref(), resolver)?;
        let enc = AesGcmEncryptor::new(bytes).map_err(|e| invalid(at, e))?.with_context(s.context.as_bytes());
        Ok(enc.into())
    }

    pub fn ff1(at: At, s: &PiiFf1Strategy, resolver: &dyn PolicyResolver) -> Result<RewriteStrategy, PolicyError> {
        let alphabet = match PiiFf1Alphabet::try_from(s.alphabet) {
            Ok(PiiFf1Alphabet::Digits) => Ff1Alphabet::Digits,
            Ok(PiiFf1Alphabet::UpperAlphanumeric) => Ff1Alphabet::UpperAlphanumeric,
            Ok(PiiFf1Alphabet::Alphanumeric) => Ff1Alphabet::Alphanumeric,
            Ok(PiiFf1Alphabet::Unspecified) | Err(_) => {
                return Err(PolicyError::Unspecified { at, field: "ff1.alphabet", value: s.alphabet });
            }
        };
        let bytes = key(at, "ff1.key", s.key.as_ref(), resolver)?;
        let enc = Ff1Encryptor::new(bytes, alphabet).map_err(|e| invalid(at, e))?.with_tweak(s.tweak.as_bytes());
        Ok(enc.into())
    }

    /// The letters every value of a built-in kind holds, as its detector
    /// finds it. A kind not named here may be all digits, and FF1 checks
    /// each mention when it encrypts.
    enum Letters {
        /// At least one letter, of either case.
        Some,
        /// At least one lower case letter.
        Lower,
    }

    fn letters(kind: PiiKind) -> Option<Letters> {
        match kind {
            // `ghp_`, `gho_`, `ghu_`, `ghr_`, `ghs_` or `github_pat_`.
            PiiKind::GithubToken => Some(Letters::Lower),
            // `AKIA` or `ASIA`; a registered top level domain; a country
            // code; a bank code; a check letter; `X`, `Y` or `Z` and a
            // check letter; `0x` or `0X`; a scheme or `www.`.
            PiiKind::AwsAccessKey
            | PiiKind::Email
            | PiiKind::Iban
            | PiiKind::Bic
            | PiiKind::EsDni
            | PiiKind::EsNie
            | PiiKind::EthAddress
            | PiiKind::Url => Some(Letters::Some),
            _ => None,
        }
    }

    /// Refuses an FF1 arm whose alphabet cannot hold a character every
    /// value of `kind` has: the policy would load, then fail on each such
    /// mention.
    pub fn check_kind(at: At, s: &PiiFf1Strategy, kind: PiiKind) -> Result<(), PolicyError> {
        let short = match (PiiFf1Alphabet::try_from(s.alphabet), letters(kind)) {
            (Ok(PiiFf1Alphabet::Digits), Some(_)) => "a letter",
            (Ok(PiiFf1Alphabet::UpperAlphanumeric), Some(Letters::Lower)) => "a lower case letter",
            _ => return Ok(()),
        };
        let name = kind.builtin_name().unwrap_or("custom");
        Err(PolicyError::Invalid {
            at,
            reason: format!("ff1.alphabet cannot encrypt a {name} value: every one holds {short} outside it"),
        })
    }
}

/// The native strategy of an `aes_256_gcm` arm, or `None` on a server
/// built without reversible protection.
pub(super) fn aes_gcm(
    at: At,
    s: &PiiAesGcmStrategy,
    resolver: &dyn PolicyResolver,
) -> Result<Option<RewriteStrategy>, PolicyError> {
    #[cfg(feature = "pii-reversible")]
    {
        enabled::aes_gcm(at, s, resolver).map(Some)
    }
    #[cfg(not(feature = "pii-reversible"))]
    {
        let _ = (at, s, resolver);
        Ok(None)
    }
}

/// The native strategy of an `ff1` arm, or `None` on a server built
/// without reversible protection.
pub(super) fn ff1(
    at: At,
    s: &PiiFf1Strategy,
    resolver: &dyn PolicyResolver,
) -> Result<Option<RewriteStrategy>, PolicyError> {
    #[cfg(feature = "pii-reversible")]
    {
        enabled::ff1(at, s, resolver).map(Some)
    }
    #[cfg(not(feature = "pii-reversible"))]
    {
        let _ = (at, s, resolver);
        Ok(None)
    }
}

/// Refuses an `ff1` arm on a kind whose values its alphabet cannot hold. A
/// server without reversible protection does not know the arm, so there
/// is nothing to check.
pub(super) fn check_ff1_kind(at: At, s: &PiiFf1Strategy, kind: turbo_pii::PiiKind) -> Result<(), PolicyError> {
    #[cfg(feature = "pii-reversible")]
    {
        enabled::check_kind(at, s, kind)
    }
    #[cfg(not(feature = "pii-reversible"))]
    {
        let _ = (at, s, kind);
        Ok(())
    }
}
