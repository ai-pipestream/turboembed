//! `MorfologikDictionary`, `MorfologikLemmatizer`, `MorfologikTagDictionary`, and
//! `MorfologikDictionaryBuilder`, ported from `opennlp-morfologik`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub use crate::lemmatizer::MorfologikLemmatizer;
use turbo_analysis::ModelRef;
use turbo_sequence::SequenceValidator;
use turbo_unicode::java::to_lowercase;

use crate::encoding::{decode_base_form, encode_base_form};
use crate::error::MorfologikError;
use crate::fsa::{FsaReader, MAGIC, VERSION_FSA5};
use crate::metadata::MorfologikMetadata;
use crate::reader::index_of;

/// One morphological entry returned by [`MorfologikDictionary::lookup`]
/// (`morfologik.stemming.WordData`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordData {
    /// The queried surface word.
    pub word: String,
    /// The decoded base form (stem / lemma).
    pub stem: String,
    /// The decoded POS tag, or `None` when the dictionary entry has no tag separator.
    pub tag: Option<String>,
}

/// A compiled Morfologik binary dictionary (`.dict` FSA5/CFSA2 automaton paired with
/// `.info` metadata), corresponding to `morfologik.stemming.Dictionary`.
#[derive(Debug, Clone)]
pub struct MorfologikDictionary {
    fsa: FsaReader,
    metadata: MorfologikMetadata,
    model_ref: ModelRef,
}

impl MorfologikDictionary {
    /// First matching lemma in automaton order, without collecting WordData
    /// records or copying the query and POS labels for every candidate.
    pub fn first_lemma(&self, raw_word: &str, postag: &str) -> Option<String> {
        use crate::charset::DictionaryCharset;
        use std::borrow::Cow;
        let converted = self.metadata.input_conversion.apply_cow(Cow::Borrowed(raw_word));
        let encode = |s: &str| self.metadata.charset.encode(s).ok();
        let word = if self.metadata.charset == DictionaryCharset::Utf8 {
            Cow::Borrowed(converted.as_bytes())
        } else {
            Cow::Owned(encode(&converted)?)
        };
        let tag = if self.metadata.charset == DictionaryCharset::Utf8 {
            Cow::Borrowed(postag.as_bytes())
        } else {
            Cow::Owned(encode(postag)?)
        };
        let separator = self.metadata.separator;
        if word.contains(&separator) || tag.is_empty() {
            return None;
        }
        let mut found = None;
        self.fsa.lookup_with_separator(&word, separator, |suffix| {
            if found.is_some() {
                return;
            }
            let Some(sep) = index_of(suffix, separator, self.metadata.encoder.control_bytes()) else {
                return;
            };
            if suffix[sep + 1..] != *tag {
                return;
            }
            if let Ok(base) = decode_base_form(&word, &suffix[..sep], self.metadata.encoder) {
                found = self.metadata.charset.decode(&base).ok();
            }
        });
        found
    }

    /// Reads a `.dict` file and its sibling `.info` metadata file from disk.
    pub fn read(dict_path: impl AsRef<Path>) -> Result<Self, MorfologikError> {
        let dict_path = dict_path.as_ref();
        let info_path = MorfologikMetadata::expected_metadata_path(dict_path);
        let dict_bytes = std::fs::read(dict_path)
            .map_err(|e| MorfologikError::format(format!("Could not read dictionary {}: {e}", dict_path.display())))?;
        let info_bytes = std::fs::read(&info_path).map_err(|e| {
            MorfologikError::format(format!("Could not read dictionary metadata {}: {e}", info_path.display()))
        })?;
        Self::from_bytes(&dict_bytes, &info_bytes)
    }

    /// Reads a Morfologik dictionary from `.dict` automaton bytes and `.info` metadata bytes.
    pub fn from_bytes(dict_bytes: &[u8], info_bytes: &[u8]) -> Result<Self, MorfologikError> {
        let metadata = MorfologikMetadata::parse(info_bytes)?;
        let fsa = FsaReader::from_bytes(dict_bytes)?;
        let model_ref = ModelRef {
            id: "opennlp:MorfologikDictionary".to_owned(),
            revision: String::new(),
            sha256: crate::identity::digest(&[dict_bytes, info_bytes]),
        };
        Ok(Self { fsa, metadata, model_ref })
    }

    /// The dictionary's `.info` metadata.
    pub fn metadata(&self) -> &MorfologikMetadata {
        &self.metadata
    }

    /// The underlying FSA automaton reader.
    pub fn fsa(&self) -> &FsaReader {
        &self.fsa
    }

    /// Provenance reference, hashing both the automaton and its metadata.
    pub fn model_ref(&self) -> &ModelRef {
        &self.model_ref
    }

    /// Looks up `raw_word` in the automaton, returning all matching `(stem, tag)` records
    /// (`morfologik.stemming.DictionaryLookup.lookup(CharSequence)`).
    pub fn lookup(&self, raw_word: &str) -> Vec<WordData> {
        let converted = if self.metadata.input_conversion.is_empty() {
            raw_word.to_owned()
        } else {
            self.metadata.input_conversion.apply(raw_word)
        };
        let Ok(word_bytes) = self.metadata.charset.encode(&converted) else {
            return Vec::new();
        };
        let separator = self.metadata.separator;
        // As in `DictionaryLookup`, a word holding the separator byte cannot be a surface
        // form; walking it would continue into the encoded lemma and tag of another entry.
        if word_bytes.contains(&separator) {
            return Vec::new();
        }
        let encoder = self.metadata.encoder;
        let charset = self.metadata.charset;
        let control_len = encoder.control_bytes();
        let mut results = Vec::new();

        self.fsa.lookup_with_separator(&word_bytes, separator, |suffix| {
            let second_sep = index_of(suffix, separator, control_len);
            let base_end = second_sep.unwrap_or(suffix.len());
            let encoded_base = &suffix[..base_end];
            let Ok(base_bytes) = decode_base_form(&word_bytes, encoded_base, encoder) else {
                return;
            };
            let Ok(stem) = charset.decode(&base_bytes) else {
                return;
            };
            let tag = match second_sep {
                None => None,
                Some(sec) => {
                    let tag_bytes = &suffix[sec + 1..];
                    if tag_bytes.is_empty() {
                        None
                    } else {
                        let Ok(tag) = charset.decode(tag_bytes) else {
                            return;
                        };
                        Some(tag)
                    }
                }
            };
            results.push(WordData { word: converted.clone(), stem, tag });
        });
        results
    }
}

/// POS tag dictionary backed by a [`MorfologikDictionary`], ported from
/// `opennlp.morfologik.tagdict.MorfologikTagDictionary`.
#[derive(Debug, Clone)]
pub struct MorfologikTagDictionary {
    dictionary: Arc<MorfologikDictionary>,
    case_sensitive: bool,
}

impl MorfologikTagDictionary {
    /// Initializes a case-sensitive `MorfologikTagDictionary`.
    pub fn new(dictionary: Arc<MorfologikDictionary>) -> Self {
        Self::with_case_sensitivity(dictionary, true)
    }

    /// Initializes a `MorfologikTagDictionary` with explicit case sensitivity.
    pub fn with_case_sensitivity(dictionary: Arc<MorfologikDictionary>, case_sensitive: bool) -> Self {
        Self { dictionary, case_sensitive }
    }

    /// Initializes a `MorfologikTagDictionary` from `.dict` and `.info` bytes
    /// (`MorfologikPOSTaggerFactory.createTagDictionary`).
    pub fn from_bytes(dict_bytes: &[u8], info_bytes: &[u8], case_sensitive: bool) -> Result<Self, MorfologikError> {
        let dict = MorfologikDictionary::from_bytes(dict_bytes, info_bytes)?;
        Ok(Self::with_case_sensitivity(Arc::new(dict), case_sensitive))
    }

    /// Whether lookups are case-sensitive (`TagDictionary.isCaseSensitive()`).
    pub fn is_case_sensitive(&self) -> bool {
        self.case_sensitive
    }

    /// Returns the list of POS tags for `word`, or `None` (`null` in Java) when `word` is
    /// not in the dictionary or has only tagless entries (`MorfologikTagDictionary.getTags`).
    pub fn tags(&self, word: &str) -> Option<Vec<String>> {
        let query: std::borrow::Cow<'_, str> = if self.case_sensitive {
            std::borrow::Cow::Borrowed(word)
        } else {
            let mut lower = String::with_capacity(word.len());
            to_lowercase(word, &mut lower);
            std::borrow::Cow::Owned(lower)
        };
        let data = self.dictionary.lookup(&query);
        if data.is_empty() {
            return None;
        }
        let mut tags = Vec::with_capacity(data.len());
        for entry in data {
            if let Some(tag) = entry.tag {
                tags.push(tag);
            }
        }
        if tags.is_empty() { None } else { Some(tags) }
    }
}

impl<'a> SequenceValidator<&'a str> for MorfologikTagDictionary {
    fn valid_sequence(&self, index: usize, sequence: &[&'a str], _prior: &[&str], outcome: &str) -> bool {
        match self.tags(sequence[index]) {
            None => true,
            Some(tags) => tags.iter().any(|t| t == outcome),
        }
    }
}

/// Compiles a separator-delimited morphological dictionary table (`base,inflected[,tag]`)
/// and its `.info` metadata into a binary FSA5 `.dict` file, ported from
/// `opennlp.morfologik.builder.MorfologikDictionaryBuilder`.
#[derive(Debug, Default, Clone, Copy)]
pub struct MorfologikDictionaryBuilder;

impl MorfologikDictionaryBuilder {
    /// Derives the compiled dictionary file name from a metadata file name by exchanging
    /// the trailing `.info` for `.dict`. A name without that trailing suffix is returned
    /// unchanged (`MorfologikDictionaryBuilder.toDictionaryFileName`).
    pub fn to_dictionary_file_name(metadata_file_name: &str) -> String {
        if let Some(stem) = metadata_file_name.strip_suffix(".info") {
            format!("{stem}.dict")
        } else {
            metadata_file_name.to_owned()
        }
    }

    /// Compiles a dictionary input file (`base,inflected[,tag]` rows) using its sibling
    /// `.info` metadata file and writes the resulting `.dict` file next to it.
    pub fn build(&self, input: impl AsRef<Path>) -> Result<PathBuf, MorfologikError> {
        let input = input.as_ref();
        let metadata_path = MorfologikMetadata::expected_metadata_path(input);
        let input_bytes = std::fs::read(input)
            .map_err(|e| MorfologikError::format(format!("failed to read {}: {e}", input.display())))?;
        let info_bytes = std::fs::read(&metadata_path)
            .map_err(|e| MorfologikError::format(format!("failed to read {}: {e}", metadata_path.display())))?;
        let dict_bytes = Self::compile_bytes(&input_bytes, &info_bytes)?;
        let dict_file_name = Self::to_dictionary_file_name(
            metadata_path.file_name().and_then(|n| n.to_str()).unwrap_or("dictionary.info"),
        );
        let output_path = metadata_path.with_file_name(dict_file_name);
        std::fs::write(&output_path, dict_bytes)
            .map_err(|e| MorfologikError::format(format!("failed to write {}: {e}", output_path.display())))?;
        Ok(output_path)
    }

    /// Compiles raw dictionary table bytes (`base SEP inflected [SEP tag]` per line, as
    /// consumed by Morfologik's `DictCompile`) and `.info` metadata bytes into an FSA5
    /// automaton byte vector.
    pub fn compile_bytes(input_bytes: &[u8], info_bytes: &[u8]) -> Result<Vec<u8>, MorfologikError> {
        let meta = MorfologikMetadata::parse(info_bytes)?;
        let separator = meta.separator;
        let encoder = meta.encoder;
        let mut sequences: Vec<Vec<u8>> = Vec::new();

        for mut line in input_bytes.split(|&b| b == b'\n') {
            if let Some(stripped) = line.strip_suffix(b"\r") {
                line = stripped;
            }
            if line.is_empty() {
                continue;
            }
            let Some(first_sep) = index_of(line, separator, 0) else {
                return Err(MorfologikError::format("dictionary builder row has no separator"));
            };
            let base = &line[..first_sep];
            let second_sep = index_of(line, separator, first_sep + 1);
            let inflected_end = second_sep.unwrap_or(line.len());
            let inflected = &line[first_sep + 1..inflected_end];
            let encoded_base = encode_base_form(inflected, base, encoder);

            let mut seq = Vec::with_capacity(line.len() + 4);
            seq.extend_from_slice(inflected);
            seq.push(separator);
            seq.extend_from_slice(&encoded_base);
            if let Some(sec) = second_sep {
                seq.push(separator);
                seq.extend_from_slice(&line[sec + 1..]);
            }
            sequences.push(seq);
        }

        sequences.sort();
        sequences.dedup();
        Ok(serialize_fsa5(&sequences, separator))
    }
}

/// Serializes sorted, deduplicated byte sequences into an FSA5 automaton (`nodeDataLength = 0`,
/// with the narrowest `gotoLength` that can address every arc).
pub fn serialize_fsa5(sequences: &[Vec<u8>], annotation_byte: u8) -> Vec<u8> {
    #[derive(Default)]
    struct TrieNode {
        children: BTreeMap<u8, (bool, usize)>,
    }

    let mut trie: Vec<TrieNode> = vec![TrieNode::default()];
    for seq in sequences {
        if seq.is_empty() {
            continue;
        }
        let mut cur = 0usize;
        for (i, &b) in seq.iter().enumerate() {
            let is_last_byte = i + 1 == seq.len();
            let next_len = trie.len();
            let (next_cur, need_push) = {
                let entry = trie[cur].children.entry(b).or_insert((false, 0));
                if is_last_byte {
                    entry.0 = true;
                    (entry.1, false)
                } else if entry.1 == 0 {
                    entry.1 = next_len;
                    (next_len, true)
                } else {
                    (entry.1, false)
                }
            };
            if need_push {
                trie.push(TrieNode::default());
            }
            cur = next_cur;
        }
    }

    // Every arc is a label byte followed by a little-endian address of `goto_length` bytes
    // whose low 3 bits hold the flags. Pick the narrowest width that can address every arc,
    // counting the dummy terminal node and the epsilon node ahead of the trie.
    let arc_count = 2 + trie.iter().map(|node| node.children.len()).sum::<usize>();
    let goto_length = (1..=8usize)
        .find(|&width| {
            let data_len = (arc_count as u128) * (1 + width as u128);
            ((data_len << 3) | 7) < (1u128 << (8 * width))
        })
        .expect("an 8-byte address covers any in-memory automaton");
    let arc_size = 1 + goto_length;

    // Header: `\fsa`, version 0x05, filler '_', annotation separator, goto length (node data length 0).
    let mut out = vec![MAGIC[0], MAGIC[1], MAGIC[2], MAGIC[3], VERSION_FSA5, b'_', annotation_byte, goto_length as u8];

    // The arc block starts with the dummy terminal node at offset 0 (label 0, BIT_LAST_ARC,
    // address 0) and the epsilon node at offset `arc_size`, whose single arc points to the
    // root node (or to 0 for an empty automaton).
    let mut arcs: Vec<u8> = Vec::with_capacity(arc_count * arc_size);
    push_fsa5_arc(&mut arcs, 0, 0x02, goto_length);
    push_fsa5_arc(&mut arcs, b'^', 0x02, goto_length);

    if trie[0].children.is_empty() {
        out.extend_from_slice(&arcs);
        return out;
    }

    // Emit nodes in post-order so every child's byte offset is known before its parent's arcs.
    let mut node_offsets = vec![0usize; trie.len()];
    let mut order = Vec::with_capacity(trie.len());
    fn collect_postorder(node: usize, trie: &[TrieNode], order: &mut Vec<usize>) {
        for &(_, target) in trie[node].children.values() {
            if target != 0 {
                collect_postorder(target, trie, order);
            }
        }
        order.push(node);
    }
    collect_postorder(0, &trie, &mut order);

    for node in order {
        let offset = arcs.len();
        node_offsets[node] = offset;
        let len = trie[node].children.len();
        for (idx, (&label, &(is_final, target))) in trie[node].children.iter().enumerate() {
            let is_last = idx + 1 == len;
            let target_offset = if target == 0 { 0 } else { node_offsets[target] };
            let mut flags = 0u64;
            if is_final {
                flags |= 0x01;
            }
            if is_last {
                flags |= 0x02;
            }
            push_fsa5_arc(&mut arcs, label, ((target_offset as u64) << 3) | flags, goto_length);
        }
    }

    let root_encoded = ((node_offsets[0] as u64) << 3) | 0x02;
    arcs[arc_size + 1..2 * arc_size].copy_from_slice(&root_encoded.to_le_bytes()[..goto_length]);

    out.extend_from_slice(&arcs);
    out
}

fn push_fsa5_arc(arcs: &mut Vec<u8>, label: u8, encoded_address: u64, goto_length: usize) {
    arcs.push(label);
    arcs.extend_from_slice(&encoded_address.to_le_bytes()[..goto_length]);
}
