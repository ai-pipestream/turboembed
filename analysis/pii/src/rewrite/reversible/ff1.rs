//! FF1 format-preserving encryption (NIST SP 800-38G) of each mention's
//! digits or letters ([`Ff1Encryptor`]). The characters of the alphabet
//! are the numerals, encrypted together under the caller's tweak; every
//! other character is a separator and keeps its place, so `123-45-6789`
//! encrypts to another `ddd-dd-dddd`. A letter or digit outside the
//! alphabet would stay in the clear, so the value is refused instead.
//!
//! One value under one key and tweak always gives the same ciphertext, so
//! ciphertexts are linkable, as [`crate::rewrite::HmacTokenizer`]'s labels
//! are.

use std::fmt;
use std::sync::Arc;

use aes_gcm::aes::Aes256;
use fpe::ff1::{FF1, FlexibleNumeralString};
use turbo_analysis::View;
use turbo_analysis_layers::{Pii, PiiStrategyKind};

use super::{CryptError, check_key, key_check};
use crate::rewrite::replace::EditOrigin;
use crate::rewrite::{PiiRewrite, RewriteError};

/// FF1 needs at least this many values in a domain.
const MIN_DOMAIN: u64 = 1_000_000;

/// The characters FF1 encrypts.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Ff1Alphabet {
    /// `0` to `9`, radix 10.
    Digits,
    /// `0` to `9` then `A` to `Z`, radix 36. A lower case letter is
    /// refused.
    UpperAlphanumeric,
    /// `0` to `9`, `A` to `Z` then `a` to `z`, radix 62.
    Alphanumeric,
}

impl Ff1Alphabet {
    /// The number of numerals.
    pub const fn radix(self) -> u32 {
        match self {
            Ff1Alphabet::Digits => 10,
            Ff1Alphabet::UpperAlphanumeric => 36,
            Ff1Alphabet::Alphanumeric => 62,
        }
    }

    /// The fewest numerals a value needs: a million values in its domain.
    pub fn min_numerals(self) -> usize {
        let radix = u64::from(self.radix());
        let (mut n, mut domain) = (0, 1u64);
        while domain < MIN_DOMAIN {
            domain *= radix;
            n += 1;
        }
        n
    }

    fn numeral(self, c: char) -> Option<u16> {
        let n = match c {
            '0'..='9' => c as u32 - '0' as u32,
            'A'..='Z' if self != Ff1Alphabet::Digits => c as u32 - 'A' as u32 + 10,
            'a'..='z' if self == Ff1Alphabet::Alphanumeric => c as u32 - 'a' as u32 + 36,
            _ => return None,
        };
        u16::try_from(n).ok()
    }

    fn char(self, n: u16) -> char {
        let n = u8::try_from(n).expect("a numeral below the radix");
        char::from(match n {
            0..=9 => b'0' + n,
            10..=35 => b'A' + n - 10,
            _ => b'a' + n - 36,
        })
    }
}

/// Encrypts the numerals of each mention with FF1 under a 32-byte AES-256
/// key and a caller tweak.
pub struct Ff1Encryptor {
    // Shared by clones: FF1 is not Clone, and the key schedule is
    // zeroised when the last clone drops.
    ff1: Arc<FF1<Aes256>>,
    key_check: [u8; 32],
    alphabet: Ff1Alphabet,
    tweak: Vec<u8>,
    refuse_overlaps: bool,
}

impl fmt::Debug for Ff1Encryptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The key stays out of logs.
        f.debug_struct("Ff1Encryptor")
            .field("alphabet", &self.alphabet)
            .field("tweak_bytes", &self.tweak.len())
            .field("refuse_overlaps", &self.refuse_overlaps)
            .finish_non_exhaustive()
    }
}

impl Clone for Ff1Encryptor {
    fn clone(&self) -> Ff1Encryptor {
        Ff1Encryptor {
            ff1: Arc::clone(&self.ff1),
            key_check: self.key_check,
            alphabet: self.alphabet,
            tweak: self.tweak.clone(),
            refuse_overlaps: self.refuse_overlaps,
        }
    }
}

/// Equal when the key, the alphabet, the tweak and the overlap rule are.
impl PartialEq for Ff1Encryptor {
    fn eq(&self, other: &Ff1Encryptor) -> bool {
        self.key_check == other.key_check
            && self.alphabet == other.alphabet
            && self.tweak == other.tweak
            && self.refuse_overlaps == other.refuse_overlaps
    }
}

impl Eq for Ff1Encryptor {}

impl Ff1Encryptor {
    /// Folds the key (as an HMAC under a fixed label, never itself), the
    /// alphabet, the tweak and the overlap rule into `h`.
    pub(crate) fn fold_config(&self, h: &mut turbo_analysis::Fnv1a) {
        let alphabet = u8::try_from(self.alphabet.radix()).expect("radix 10, 36 or 62");
        h.bytes(&self.key_check).u8(alphabet).usize(self.tweak.len()).bytes(&self.tweak);
        h.u8(u8::from(self.refuse_overlaps));
    }

    /// An encryptor of `alphabet` under the 32-byte AES-256 `key`, with an
    /// empty tweak.
    pub fn new(key: &[u8], alphabet: Ff1Alphabet) -> Result<Ff1Encryptor, CryptError> {
        check_key(key)?;
        let ff1 = Arc::new(FF1::<Aes256>::new(key, alphabet.radix()).expect("the radixes are 10, 36 and 62"));
        Ok(Ff1Encryptor { ff1, key_check: key_check(key), alphabet, tweak: Vec::new(), refuse_overlaps: false })
    }

    /// This encryptor under `tweak`, which decryption must then be given:
    /// a field name or a record id, say.
    pub fn with_tweak(mut self, tweak: &[u8]) -> Ff1Encryptor {
        self.tweak = tweak.to_vec();
        self
    }

    /// This encryptor, refusing overlapping mentions with
    /// [`RewriteError::Overlap`].
    pub fn refusing_overlaps(mut self) -> Ff1Encryptor {
        self.refuse_overlaps = true;
        self
    }

    /// Whether overlapping mentions are refused rather than coalesced.
    pub const fn refuses_overlaps(&self) -> bool {
        self.refuse_overlaps
    }

    /// The alphabet this encryptor encrypts.
    pub const fn alphabet(&self) -> Ff1Alphabet {
        self.alphabet
    }

    fn apply(&self, value: &str, decrypt: bool, out: &mut String) -> Result<(), CryptError> {
        if value.is_empty() {
            return Err(CryptError::EmptyValue);
        }
        let mut numerals = Vec::with_capacity(value.len());
        for (at, c) in value.char_indices() {
            match self.alphabet.numeral(c) {
                Some(n) => numerals.push(n),
                None if c.is_alphanumeric() => return Err(CryptError::OutsideAlphabet { at }),
                None => {}
            }
        }
        let need = self.alphabet.min_numerals();
        if numerals.len() < need {
            return Err(CryptError::TooFewNumerals { have: numerals.len(), need });
        }
        let ns = FlexibleNumeralString::from(numerals);
        let done = if decrypt { self.ff1.decrypt(&self.tweak, &ns) } else { self.ff1.encrypt(&self.tweak, &ns) };
        let done: Vec<u16> = done.map_err(|_| CryptError::TooLong)?.into();
        let mut next = done.into_iter();
        out.reserve(value.len());
        for c in value.chars() {
            if self.alphabet.numeral(c).is_some() {
                out.push(self.alphabet.char(next.next().expect("one numeral out for each in")));
            } else {
                out.push(c);
            }
        }
        Ok(())
    }

    /// Appends the encryption of `value` to `out`.
    pub fn write_encrypted(&self, value: &str, out: &mut String) -> Result<(), CryptError> {
        self.apply(value, false, out)
    }

    /// The encryption of `value`.
    pub fn encrypt(&self, value: &str) -> Result<String, CryptError> {
        let mut out = String::new();
        self.apply(value, false, &mut out)?;
        Ok(out)
    }

    /// The value that `encrypted`, made by an encryptor with this key,
    /// alphabet and tweak, stands for.
    pub fn decrypt(&self, encrypted: &str) -> Result<String, CryptError> {
        let mut out = String::new();
        self.apply(encrypted, true, &mut out)?;
        Ok(out)
    }

    /// Replaces every mention of `pii`, a layer over `src`, with the
    /// encryption of its surface, as a view derived from `src`.
    pub fn rewrite(&self, src: &View, pii: &Pii) -> Result<PiiRewrite, RewriteError> {
        let text = src.text();
        PiiRewrite::derive(
            src,
            pii,
            |_| self.refuse_overlaps,
            0,
            |c, out| {
                let span = c.merged.span;
                self.apply(&text[span.start as usize..span.end as usize], false, out)
                    .map_err(|e| RewriteError::Encrypt(span, e.to_string()))?;
                Ok(EditOrigin::only(PiiStrategyKind::Ff1))
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::Ff1Alphabet;

    #[test]
    fn the_alphabets_need_a_million_values() {
        assert_eq!(Ff1Alphabet::Digits.min_numerals(), 6);
        assert_eq!(Ff1Alphabet::UpperAlphanumeric.min_numerals(), 4);
        assert_eq!(Ff1Alphabet::Alphanumeric.min_numerals(), 4);
        for a in [Ff1Alphabet::Digits, Ff1Alphabet::UpperAlphanumeric, Ff1Alphabet::Alphanumeric] {
            for n in 0..u16::try_from(a.radix()).unwrap() {
                assert_eq!(a.numeral(a.char(n)), Some(n));
            }
        }
    }
}
