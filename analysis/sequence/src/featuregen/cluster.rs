//! Word cluster resources (`BrownCluster`, `WordClusterDictionary`) and
//! the generators that read them.
//!
//! Both resources are text files stored in the model next to the
//! descriptor that names them. They are third-party data (cluster
//! lexicons computed from large corpora), so a model that carries one
//! carries its license too.

use std::collections::HashMap;
use std::sync::Arc;

use turbo_jvm::chars::utf16_len;

use super::FeatureGenerator;
use crate::error::{Error, Result};
use crate::features::Features;
use crate::java::{lines, lowercase_into, parse_int, split_trimmed};
use crate::pattern::{PatternMode, token_feature};

/// `BrownCluster`: each token's Brown cluster, a bit string naming its
/// place in the cluster hierarchy.
#[derive(Debug, Clone, Default)]
pub struct BrownCluster {
    clusters: HashMap<String, String>,
}

impl BrownCluster {
    /// Reads the tab-separated lexicon. A line of three fields is
    /// `cluster, token, frequency` and is kept when the frequency is above
    /// five; a line of two fields is `token, cluster`. Trailing empty
    /// fields do not count and other lines are skipped. A token listed
    /// twice keeps its last cluster. The bytes are UTF-8; a malformed
    /// sequence reads as U+FFFD, as Java's decoder replaces it.
    pub fn read(name: &str, bytes: &[u8]) -> Result<BrownCluster> {
        let text = String::from_utf8_lossy(bytes);
        let mut clusters = HashMap::new();
        for line in lines(&text) {
            let fields = split_trimmed(line, '\t');
            match fields[..] {
                [cluster, token, frequency] => {
                    let frequency = parse_int(frequency).ok_or_else(|| Error::Resource {
                        name: name.into(),
                        message: format!("the frequency {frequency:?} is not an integer"),
                    })?;
                    if frequency > 5 {
                        clusters.insert(token.to_owned(), cluster.to_owned());
                    }
                }
                [token, cluster] => {
                    clusters.insert(token.to_owned(), cluster.to_owned());
                }
                _ => {}
            }
        }
        Ok(BrownCluster { clusters })
    }

    /// The cluster of a token, if it has one.
    pub fn lookup(&self, token: &str) -> Option<&str> {
        self.clusters.get(token).map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.clusters.len()
    }

    pub fn is_empty(&self) -> bool {
        self.clusters.is_empty()
    }

    /// `BrownTokenClasses.getWordClasses`: the lengths, in UTF-16 code
    /// units, of the prefixes of the token's cluster that are its classes:
    /// up to 4, and then 6, 10 and 20 while the cluster is longer than the
    /// previous length. Empty when the token has no cluster.
    fn classes(&self, token: &str) -> Option<(&str, ClassLengths)> {
        let cluster = self.lookup(token)?;
        Some((cluster, ClassLengths::new(utf16_len(cluster))))
    }
}

/// The prefix lengths of one cluster (see [`BrownCluster::classes`]).
#[derive(Clone, Copy)]
struct ClassLengths {
    lengths: [usize; 4],
    count: usize,
}

impl ClassLengths {
    const PATH_LENGTHS: [usize; 4] = [4, 6, 10, 20];

    fn new(cluster_len: usize) -> ClassLengths {
        let mut lengths = [0; 4];
        lengths[0] = cluster_len.min(Self::PATH_LENGTHS[0]);
        let mut count = 1;
        for i in 1..Self::PATH_LENGTHS.len() {
            if Self::PATH_LENGTHS[i - 1] < cluster_len {
                lengths[count] = cluster_len.min(Self::PATH_LENGTHS[i]);
                count += 1;
            }
        }
        ClassLengths { lengths, count }
    }

    fn as_slice(&self) -> &[usize] {
        &self.lengths[..self.count]
    }
}

/// `BrownTokenFeatureGenerator`: `browncluster=` and each class of the
/// token.
pub struct BrownToken {
    cluster: Arc<BrownCluster>,
}

impl BrownToken {
    pub fn new(cluster: Arc<BrownCluster>) -> BrownToken {
        BrownToken { cluster }
    }
}

impl FeatureGenerator for BrownToken {
    fn create_features(&mut self, out: &mut Features, tokens: &[&str], index: usize, _previous: &[&str]) {
        if let Some((cluster, lengths)) = self.cluster.classes(tokens[index]) {
            for &len in lengths.as_slice() {
                out.push_str("browncluster=").push_units(cluster, 0, len).finish();
            }
        }
    }
}

/// `BrownTokenClassFeatureGenerator`: `c,browncluster=`, the token's class
/// ([`token_feature`]), a comma and each of its Brown classes.
pub struct BrownTokenClass {
    cluster: Arc<BrownCluster>,
    mode: PatternMode,
}

impl BrownTokenClass {
    pub fn new(cluster: Arc<BrownCluster>, mode: PatternMode) -> BrownTokenClass {
        BrownTokenClass { cluster, mode }
    }
}

impl FeatureGenerator for BrownTokenClass {
    fn create_features(&mut self, out: &mut Features, tokens: &[&str], index: usize, _previous: &[&str]) {
        let token = tokens[index];
        let shape = token_feature(token, self.mode);
        if let Some((cluster, lengths)) = self.cluster.classes(token) {
            for &len in lengths.as_slice() {
                out.push_str("c,browncluster=").push_str(shape).push_str(",").push_units(cluster, 0, len).finish();
            }
        }
    }
}

/// `BrownBigramFeatureGenerator`: the Brown classes of the token paired
/// with those of the token before it (`pbrowncluster,browncluster=`) and
/// after it (`browncluster,nbrowncluster=`), class by class, as many
/// pairs as the shorter list has.
pub struct BrownBigram {
    cluster: Arc<BrownCluster>,
}

impl BrownBigram {
    pub fn new(cluster: Arc<BrownCluster>) -> BrownBigram {
        BrownBigram { cluster }
    }
}

impl FeatureGenerator for BrownBigram {
    fn create_features(&mut self, out: &mut Features, tokens: &[&str], index: usize, _previous: &[&str]) {
        let cluster = &*self.cluster;
        let Some(current) = cluster.classes(tokens[index]) else {
            return;
        };
        let mut pairs = |prefix: &str, first: (&str, ClassLengths), second: (&str, ClassLengths)| {
            for (&a, &b) in first.1.as_slice().iter().zip(second.1.as_slice()) {
                out.push_str(prefix).push_units(first.0, 0, a).push_str(",").push_units(second.0, 0, b).finish();
            }
        };
        if index > 0
            && let Some(prev) = cluster.classes(tokens[index - 1])
        {
            pairs("pbrowncluster,browncluster=", prev, current);
        }
        if index + 1 < tokens.len()
            && let Some(next) = cluster.classes(tokens[index + 1])
        {
            pairs("browncluster,nbrowncluster=", current, next);
        }
    }
}

/// `WordClusterDictionary`: each token's cluster id.
#[derive(Debug, Clone, Default)]
pub struct WordClusterDictionary {
    clusters: HashMap<String, String>,
}

impl WordClusterDictionary {
    /// Reads the space-separated dictionary: the first two fields of a
    /// line of two or three fields are the token and its cluster id;
    /// trailing empty fields do not count and other lines are skipped. A
    /// token listed twice keeps its last id.
    pub fn read(bytes: &[u8]) -> WordClusterDictionary {
        let text = String::from_utf8_lossy(bytes);
        let mut clusters = HashMap::new();
        for line in lines(&text) {
            let fields = split_trimmed(line, ' ');
            if fields.len() == 2 || fields.len() == 3 {
                clusters.insert(fields[0].to_owned(), fields[1].to_owned());
            }
        }
        WordClusterDictionary { clusters }
    }

    /// The cluster id of a token, if it has one.
    pub fn lookup(&self, token: &str) -> Option<&str> {
        self.clusters.get(token).map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.clusters.len()
    }

    pub fn is_empty(&self) -> bool {
        self.clusters.is_empty()
    }
}

/// `WordClusterFeatureGenerator`: the resource's name followed by the
/// token's cluster id, when it has one. With `lowercase`, the token is
/// lowercased (`StringUtil.toLowerCase`) before the lookup.
pub struct WordCluster {
    dictionary: Arc<WordClusterDictionary>,
    name: String,
    lowercase: bool,
    scratch: String,
}

impl WordCluster {
    pub fn new(dictionary: Arc<WordClusterDictionary>, name: &str, lowercase: bool) -> WordCluster {
        WordCluster { dictionary, name: name.to_owned(), lowercase, scratch: String::new() }
    }
}

impl FeatureGenerator for WordCluster {
    fn create_features(&mut self, out: &mut Features, tokens: &[&str], index: usize, _previous: &[&str]) {
        let token = tokens[index];
        let key = if self.lowercase {
            self.scratch.clear();
            lowercase_into(&mut self.scratch, token);
            self.scratch.as_str()
        } else {
            token
        };
        if let Some(id) = self.dictionary.lookup(key) {
            out.push_str(&self.name).push_str(id).finish();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_lengths_follow_brown_token_classes() {
        let lengths = |n| ClassLengths::new(n).as_slice().to_vec();
        assert_eq!(lengths(0), [0]);
        assert_eq!(lengths(3), [3]);
        assert_eq!(lengths(4), [4]);
        assert_eq!(lengths(5), [4, 5]);
        assert_eq!(lengths(7), [4, 6, 7]);
        assert_eq!(lengths(12), [4, 6, 10, 12]);
        assert_eq!(lengths(25), [4, 6, 10, 20]);
    }

    #[test]
    fn brown_lines_of_three_fields_need_a_frequency_above_five() {
        let c = BrownCluster::read("b", b"0101\tthe\t100\r\n0110\trare\t5\nword\t111\n\nx\ty\tz\tw\n").unwrap();
        assert_eq!(c.lookup("the"), Some("0101"));
        assert_eq!(c.lookup("rare"), None);
        assert_eq!(c.lookup("word"), Some("111"));
        assert_eq!(c.len(), 2);
        assert!(BrownCluster::read("b", b"0\tw\tmany\n").is_err());
    }

    #[test]
    fn word_clusters_take_two_or_three_fields() {
        let d = WordClusterDictionary::read(b"the 12\nof 7 0.5\nbad\n a 3\n");
        assert_eq!(d.lookup("the"), Some("12"));
        assert_eq!(d.lookup("of"), Some("7"));
        assert_eq!(d.lookup(""), Some("a"));
        assert_eq!(d.len(), 3);
    }
}
