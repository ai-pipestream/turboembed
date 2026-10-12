//! A reusable buffer of model features.
//!
//! A maxent model looks its predicates up by their Java modified UTF-8
//! bytes. The context generator writes every feature of one candidate
//! into a single byte buffer and records where each ends, so building a
//! context allocates nothing once the buffer has grown to fit.

/// The features of one context, as modified UTF-8, one after another.
#[derive(Clone, Debug, Default)]
pub struct Features {
    bytes: Vec<u8>,
    ends: Vec<u32>,
}

impl Features {
    pub fn new() -> Features {
        Features::default()
    }

    /// Removes every feature, keeping the memory.
    pub fn clear(&mut self) {
        self.bytes.clear();
        self.ends.clear();
    }

    /// The number of features.
    pub fn len(&self) -> usize {
        self.ends.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }

    /// Feature `i`'s bytes.
    ///
    /// # Panics
    ///
    /// When `i` is not below [`len`](Self::len).
    pub fn get(&self, i: usize) -> &[u8] {
        let start = if i == 0 { 0 } else { self.ends[i - 1] as usize };
        &self.bytes[start..self.ends[i] as usize]
    }

    /// Every feature in order, for `MaxentModel::eval_features`.
    pub fn iter(&self) -> impl Iterator<Item = &[u8]> + Clone + '_ {
        let mut start = 0;
        self.ends.iter().map(move |&end| {
            let feature = &self.bytes[start..end as usize];
            start = end as usize;
            feature
        })
    }

    /// The buffer the feature being built is appended to.
    #[inline]
    pub(crate) fn buf(&mut self) -> &mut Vec<u8> {
        &mut self.bytes
    }

    /// Ends the feature being built: everything appended since the
    /// previous feature ended.
    #[inline]
    pub(crate) fn finish(&mut self) {
        self.ends.push(self.bytes.len() as u32);
    }

    /// Adds a whole feature.
    #[inline]
    pub(crate) fn push(&mut self, feature: &[u8]) {
        self.bytes.extend_from_slice(feature);
        self.finish();
    }

    /// Adds a number as decimal digits, as `Integer.toString` writes a
    /// non-negative `int`.
    pub(crate) fn push_number(&mut self, mut n: u32) {
        let mut digits = [0u8; 10];
        let mut i = digits.len();
        loop {
            i -= 1;
            digits[i] = b'0' + (n % 10) as u8;
            n /= 10;
            if n == 0 {
                break;
            }
        }
        self.push(&digits[i..]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn features_in_order() {
        let mut f = Features::new();
        f.push(b"sn");
        f.buf().extend_from_slice(b"x=");
        f.buf().extend_from_slice(b"Mr");
        f.finish();
        f.push_number(0);
        f.push_number(42);
        f.push_number(u32::MAX);
        f.push(b"");
        assert_eq!(f.len(), 6);
        let all: Vec<&[u8]> = f.iter().collect();
        assert_eq!(all, [&b"sn"[..], b"x=Mr", b"0", b"42", b"4294967295", b""]);
        assert_eq!(f.get(1), b"x=Mr");
        assert_eq!(f.get(5), b"");
        f.clear();
        assert!(f.is_empty());
        assert_eq!(f.iter().count(), 0);
    }
}
