//! Every stemmer of the crate as one value, for the annotator and for
//! configuration.

use std::hash::Hasher as _;

use turbo_analysis::Fnv1a as Fnv;

use crate::light::LightStemmer;
use crate::porter::PorterStemmer;
use crate::snowball::SnowballStemmer;
use crate::stemmer::Stemmer;

/// A stemmer OpenNLP ships: its `PorterStemmer`, a `SnowballStemmer` or a
/// light or minimal stemmer.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Algorithm {
    Porter,
    Snowball(SnowballStemmer),
    Light(LightStemmer),
}

impl Algorithm {
    /// A hash of the choice that stays the same across runs and builds,
    /// for provenance.
    pub fn config_hash(&self) -> u64 {
        let mut h = Fnv::default();
        match *self {
            Algorithm::Porter => h.write(b"porter"),
            Algorithm::Snowball(s) => {
                h.write(b"snowball ");
                h.write(s.algorithm().java_name().as_bytes());
                h.write(&s.repeat().get().to_le_bytes());
            }
            Algorithm::Light(l) => {
                h.write(b"light ");
                h.write(l.java_name().as_bytes());
                if let LightStemmer::NorwegianLight(n) | LightStemmer::NorwegianMinimal(n) = l {
                    h.write(&[n as u8]);
                }
            }
        }
        h.finish()
    }
}

impl Stemmer for Algorithm {
    fn stem_utf16(&self, word: &[u16], out: &mut Vec<u16>) {
        match self {
            Algorithm::Porter => PorterStemmer.stem_utf16(word, out),
            Algorithm::Snowball(s) => s.stem_utf16(word, out),
            Algorithm::Light(l) => l.stem_utf16(word, out),
        }
    }

    fn config_hash(&self) -> u64 {
        Algorithm::config_hash(self)
    }
}
