//! Morfologik base-form encoders and decoders (`NONE`, `SUFFIX`, `PREFIX`, `INFIX`).

use crate::error::MorfologikError;

/// Offset added to a control value before conversion to a byte (`'A'`).
pub const CONTROL_OFFSET: u8 = b'A';

/// Removal count for replacement of the complete surface form (`255`).
pub const REPLACE_FORM: usize = 255;

/// The base-form encoder declared by a dictionary's `fsa.dict.encoder`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BaseFormEncoding {
    /// The encoded bytes are the base form verbatim.
    None,
    /// The base form drops trailing bytes of the surface form, then appends the encoded rest.
    Suffix,
    /// As [`Self::Suffix`], and leading bytes of the surface form are dropped too.
    #[default]
    Prefix,
    /// As [`Self::Suffix`], and a run of bytes inside the surface form is dropped too.
    Infix,
}

impl BaseFormEncoding {
    /// Parses an encoder name case-insensitively (`NONE`, `SUFFIX`, `PREFIX`, `INFIX`).
    pub fn parse_name(name: &str) -> Result<Self, MorfologikError> {
        let trimmed = trim_java_string(name);
        if trimmed.eq_ignore_ascii_case("NONE") {
            Ok(Self::None)
        } else if trimmed.eq_ignore_ascii_case("SUFFIX") {
            Ok(Self::Suffix)
        } else if trimmed.eq_ignore_ascii_case("PREFIX") {
            Ok(Self::Prefix)
        } else if trimmed.eq_ignore_ascii_case("INFIX") {
            Ok(Self::Infix)
        } else {
            Err(MorfologikError::invalid_argument(format!("unknown fsa.dict.encoder: {name}")))
        }
    }

    /// Canonical uppercase name (`NONE`, `SUFFIX`, `PREFIX`, `INFIX`).
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "NONE",
            Self::Suffix => "SUFFIX",
            Self::Prefix => "PREFIX",
            Self::Infix => "INFIX",
        }
    }

    /// Number of control bytes preceding the appended literal bytes.
    pub const fn control_bytes(self) -> usize {
        match self {
            Self::None => 0,
            Self::Suffix => 1,
            Self::Prefix => 2,
            Self::Infix => 3,
        }
    }
}

/// Trims ASCII control characters and spaces (`c <= '\u{0020}'`), matching `java.lang.String.trim()`.
pub(crate) fn trim_java_string(s: &str) -> &str {
    s.trim_matches(|c: char| (c as u32) <= 0x20)
}

#[inline]
fn control(b: u8) -> usize {
    usize::from(b.wrapping_sub(CONTROL_OFFSET))
}

fn require_control_len(encoded: &[u8], prefix_bytes: usize, encoding: BaseFormEncoding) -> Result<(), MorfologikError> {
    if encoded.len() < prefix_bytes {
        return Err(MorfologikError::invalid_argument(format!(
            "{} encoded base needs at least {prefix_bytes} control byte(s)",
            encoding.as_str()
        )));
    }
    Ok(())
}

fn bounded(index: isize, form_len: usize) -> Result<usize, MorfologikError> {
    if index < 0 || (index as usize) > form_len {
        return Err(MorfologikError::invalid_argument(format!(
            "encoded base addresses byte {index} outside a form of length {form_len}"
        )));
    }
    Ok(index as usize)
}

fn require_ordered(start: usize, end: usize) -> Result<(), MorfologikError> {
    if start > end {
        return Err(MorfologikError::invalid_argument("encoded base removal ranges overlap"));
    }
    Ok(())
}

/// Recovers a base form from a surface form and its encoded representation, matching
/// `MorfologikDictionaryReader.decodeBaseForm`.
pub fn decode_base_form(form: &[u8], encoded: &[u8], encoding: BaseFormEncoding) -> Result<Vec<u8>, MorfologikError> {
    let prefix_bytes = encoding.control_bytes();
    require_control_len(encoded, prefix_bytes, encoding)?;
    match encoding {
        BaseFormEncoding::None => Ok(encoded.to_vec()),
        BaseFormEncoding::Suffix => {
            let suffix = control(encoded[0]);
            let keep = if suffix == REPLACE_FORM {
                0
            } else {
                bounded((form.len() as isize) - (suffix as isize), form.len())?
            };
            let mut out = Vec::with_capacity(keep + (encoded.len() - prefix_bytes));
            out.extend_from_slice(&form[..keep]);
            out.extend_from_slice(&encoded[prefix_bytes..]);
            Ok(out)
        }
        BaseFormEncoding::Prefix => {
            let prefix = control(encoded[0]);
            let suffix = control(encoded[1]);
            if prefix == REPLACE_FORM || suffix == REPLACE_FORM {
                return Ok(encoded[prefix_bytes..].to_vec());
            }
            let start = bounded(prefix as isize, form.len())?;
            let end = bounded((form.len() as isize) - (suffix as isize), form.len())?;
            require_ordered(start, end)?;
            let mut out = Vec::with_capacity((end - start) + (encoded.len() - prefix_bytes));
            out.extend_from_slice(&form[start..end]);
            out.extend_from_slice(&encoded[prefix_bytes..]);
            Ok(out)
        }
        BaseFormEncoding::Infix => {
            let position = control(encoded[0]);
            let length = control(encoded[1]);
            let suffix = control(encoded[2]);
            if length == REPLACE_FORM || suffix == REPLACE_FORM {
                return Ok(encoded[prefix_bytes..].to_vec());
            }
            let cut = bounded(position as isize, form.len())?;
            let resume = bounded((cut + length) as isize, form.len())?;
            let end = bounded((form.len() as isize) - (suffix as isize), form.len())?;
            require_ordered(resume, end)?;
            let mut out = Vec::with_capacity(cut + (end - resume) + (encoded.len() - prefix_bytes));
            out.extend_from_slice(&form[..cut]);
            out.extend_from_slice(&form[resume..end]);
            out.extend_from_slice(&encoded[prefix_bytes..]);
            Ok(out)
        }
    }
}

/// Encodes a `base` form relative to a surface `form` using `encoding`, matching
/// Morfologik's `NoEncoder`, `SuffixEncoder`, `PrefixEncoder`, and `InfixEncoder`.
pub fn encode_base_form(form: &[u8], base: &[u8], encoding: BaseFormEncoding) -> Vec<u8> {
    match encoding {
        BaseFormEncoding::None => base.to_vec(),
        BaseFormEncoding::Suffix => {
            let shared = common_prefix_len(form, base);
            let truncate = form.len() - shared;
            if truncate >= REPLACE_FORM {
                let mut out = Vec::with_capacity(1 + base.len());
                out.push(CONTROL_OFFSET.wrapping_add(REPLACE_FORM as u8));
                out.extend_from_slice(base);
                out
            } else {
                let mut out = Vec::with_capacity(1 + (base.len() - shared));
                out.push(CONTROL_OFFSET.wrapping_add(truncate as u8));
                out.extend_from_slice(&base[shared..]);
                out
            }
        }
        BaseFormEncoding::Prefix => {
            let mut best_prefix = 0usize;
            let mut best_shared = common_prefix_len(form, base);
            let max_prefix = form.len().min(REPLACE_FORM - 1);
            for p in 1..=max_prefix {
                let shared = common_prefix_len(&form[p..], base);
                if shared > best_shared {
                    best_shared = shared;
                    best_prefix = p;
                }
            }
            let truncate_suffix = form.len() - best_prefix - best_shared;
            if truncate_suffix >= REPLACE_FORM || best_shared == 0 {
                let suffix = if form.len() >= REPLACE_FORM { REPLACE_FORM } else { form.len() };
                let mut out = Vec::with_capacity(2 + base.len());
                out.push(CONTROL_OFFSET);
                out.push(CONTROL_OFFSET.wrapping_add(suffix as u8));
                out.extend_from_slice(base);
                out
            } else {
                let mut out = Vec::with_capacity(2 + (base.len() - best_shared));
                out.push(CONTROL_OFFSET.wrapping_add(best_prefix as u8));
                out.push(CONTROL_OFFSET.wrapping_add(truncate_suffix as u8));
                out.extend_from_slice(&base[best_shared..]);
                out
            }
        }
        BaseFormEncoding::Infix => {
            // Find the best (position, length) infix removal that maximizes shared prefix with `base`.
            let mut best_pos = 0usize;
            let mut best_len = 0usize;
            let mut best_shared = common_prefix_len(form, base);
            let max_pos = form.len().min(255);
            for pos in 0..max_pos {
                let head_match = common_prefix_len(&form[..pos], base);
                if head_match < pos {
                    break;
                }
                let max_del = (form.len() - pos).min(REPLACE_FORM - 1);
                for del in 1..=max_del {
                    let tail_match = common_prefix_len(&form[pos + del..], &base[pos..]);
                    let total_shared = pos + tail_match;
                    if total_shared > best_shared {
                        best_shared = total_shared;
                        best_pos = pos;
                        best_len = del;
                    }
                }
            }
            let truncate_suffix = form.len() - best_len - best_shared;
            if truncate_suffix >= REPLACE_FORM || best_shared == 0 {
                let suffix = if form.len() >= REPLACE_FORM { REPLACE_FORM } else { form.len() };
                let mut out = Vec::with_capacity(3 + base.len());
                out.push(CONTROL_OFFSET);
                out.push(CONTROL_OFFSET);
                out.push(CONTROL_OFFSET.wrapping_add(suffix as u8));
                out.extend_from_slice(base);
                out
            } else {
                let mut out = Vec::with_capacity(3 + (base.len() - best_shared));
                out.push(CONTROL_OFFSET.wrapping_add(best_pos as u8));
                out.push(CONTROL_OFFSET.wrapping_add(best_len as u8));
                out.push(CONTROL_OFFSET.wrapping_add(truncate_suffix as u8));
                out.extend_from_slice(&base[best_shared..]);
                out
            }
        }
    }
}

fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}
