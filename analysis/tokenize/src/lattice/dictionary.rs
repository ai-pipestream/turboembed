//! A MeCab dictionary directory, read as OpenNLP's `MecabDictionary.load`
//! reads it: `matrix.def` first (its dimensions bound every entry's
//! context ids), then the `*.csv` lexicon files in sorted name order, then
//! `char.def` and `unk.def`.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use turbo_analysis_layers::{Tag, TagSet};
use turbo_maxent::{Digest, sha256};
use turbo_unicode::WhitespaceMode;

use super::chardef::{AssignmentId, Category, CategoryId, CharDef, CharDefSource};
use super::entries::{Columns, Dimensions, Entry, FeatureStore, Record, read_entries};
use super::error::{DictionaryError, DictionaryFile};
use super::limits::DictionaryLimits;
use super::matrix::Matrix;
use super::prebuilt::Table;
use super::trie::SurfaceTrie;

/// The context id of the start and the end of a text.
pub(crate) const BOUNDARY: u32 = 0;

/// An entry of a dictionary: a lexicon entry or an unknown word template.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct EntryId(pub(crate) u32);

/// The files of a dictionary given as bytes, for a dictionary that is not
/// a directory. The lexicon files are read in the order given; each has
/// a name for the errors and the digest.
#[derive(Clone, Copy, Debug)]
pub struct DictionaryFiles<'a> {
    pub matrix: &'a [u8],
    pub lexicons: &'a [(&'a str, &'a [u8])],
    pub char_def: &'a [u8],
    pub unk_def: &'a [u8],
}

/// A loaded MeCab dictionary.
#[derive(Debug, Clone)]
pub struct MecabDictionary {
    pub(crate) matrix: Matrix,
    pub(crate) trie: SurfaceTrie,
    /// The entries of surface group `g` are `groups[g]..groups[g + 1]`.
    pub(super) groups: Table<u32>,
    /// The lexicon entries by surface group, each group in file order,
    /// then the unknown word templates by category, each in file order.
    pub(crate) entries: Table<Entry>,
    pub(super) lexicon_len: u32,
    /// The templates of each category, as a range of `entries`; empty when
    /// `unk.def` lists none.
    pub(super) unknown: Box<[(u32, u32)]>,
    pub(crate) chars: CharDef,
    pub(super) features: FeatureStore,
    pub(super) pos: Arc<TagSet>,
    pub(super) tags: Table<u16>,
    pub(super) whitespace: WhitespaceMode,
    pub(super) sha256: Digest,
}

/// Where the files come from: a directory, or bytes in memory.
trait Source {
    fn matrix(&mut self) -> Result<Cow<'_, [u8]>, DictionaryError>;
    fn lexicon_names(&mut self) -> Result<Vec<String>, DictionaryError>;
    fn lexicon(&mut self, i: usize) -> Result<Cow<'_, [u8]>, DictionaryError>;
    fn char_def(&mut self) -> Result<Cow<'_, [u8]>, DictionaryError>;
    fn unk_def(&mut self) -> Result<Cow<'_, [u8]>, DictionaryError>;
}

struct Directory {
    dir: PathBuf,
    lexicons: Vec<(String, PathBuf)>,
}

impl Directory {
    fn read(&self, path: &Path, file: DictionaryFile) -> Result<Cow<'static, [u8]>, DictionaryError> {
        // OpenNLP checks `Files.exists` and then reads; a path that exists
        // but cannot be read (a directory) is an I/O error.
        if !path.exists() {
            return Err(DictionaryError::Missing(file));
        }
        std::fs::read(path).map(Cow::Owned).map_err(|source| DictionaryError::Io { file, source })
    }
}

impl Source for Directory {
    fn matrix(&mut self) -> Result<Cow<'_, [u8]>, DictionaryError> {
        self.read(&self.dir.join("matrix.def"), DictionaryFile::Matrix)
    }

    /// Every entry whose name ends in `.csv`, as the glob `*.csv` of
    /// `Files.newDirectoryStream` matches, sorted by the bytes of the name
    /// as `Path.compareTo` sorts them. A name that is not UTF-8 is taken
    /// with its bad bytes replaced, for the errors and the digest.
    fn lexicon_names(&mut self) -> Result<Vec<String>, DictionaryError> {
        let io = |source| DictionaryError::Io { file: DictionaryFile::Lexicon("*.csv".into()), source };
        let mut found: Vec<(Vec<u8>, PathBuf)> = Vec::new();
        for e in std::fs::read_dir(&self.dir).map_err(io)? {
            let e = e.map_err(io)?;
            let name = e.file_name();
            let bytes = name.as_encoded_bytes();
            if bytes.ends_with(b".csv") {
                found.push((bytes.to_vec(), e.path()));
            }
        }
        found.sort();
        self.lexicons = found.into_iter().map(|(b, p)| (String::from_utf8_lossy(&b).into_owned(), p)).collect();
        Ok(self.lexicons.iter().map(|(n, _)| n.clone()).collect())
    }

    fn lexicon(&mut self, i: usize) -> Result<Cow<'_, [u8]>, DictionaryError> {
        let (name, path) = &self.lexicons[i];
        let file = DictionaryFile::Lexicon(name.clone());
        std::fs::read(path).map(Cow::Owned).map_err(|source| DictionaryError::Io { file, source })
    }

    fn char_def(&mut self) -> Result<Cow<'_, [u8]>, DictionaryError> {
        self.read(&self.dir.join("char.def"), DictionaryFile::CharDef)
    }

    fn unk_def(&mut self) -> Result<Cow<'_, [u8]>, DictionaryError> {
        self.read(&self.dir.join("unk.def"), DictionaryFile::UnkDef)
    }
}

impl Source for DictionaryFiles<'_> {
    fn matrix(&mut self) -> Result<Cow<'_, [u8]>, DictionaryError> {
        Ok(Cow::Borrowed(self.matrix))
    }
    fn lexicon_names(&mut self) -> Result<Vec<String>, DictionaryError> {
        Ok(self.lexicons.iter().map(|(n, _)| n.to_string()).collect())
    }
    fn lexicon(&mut self, i: usize) -> Result<Cow<'_, [u8]>, DictionaryError> {
        Ok(Cow::Borrowed(self.lexicons[i].1))
    }
    fn char_def(&mut self) -> Result<Cow<'_, [u8]>, DictionaryError> {
        Ok(Cow::Borrowed(self.char_def))
    }
    fn unk_def(&mut self) -> Result<Cow<'_, [u8]>, DictionaryError> {
        Ok(Cow::Borrowed(self.unk_def))
    }
}

fn utf8(bytes: &[u8], file: DictionaryFile) -> Result<&str, DictionaryError> {
    std::str::from_utf8(bytes).map_err(|e| DictionaryError::Utf8 { file, offset: e.valid_up_to() })
}

/// The lexicon entries as read, in file order, with their surfaces.
struct Lexicon {
    surfaces: String,
    /// Each entry's surface range in `surfaces`.
    spans: Vec<(u32, u32)>,
    entries: Vec<Entry>,
}

impl Lexicon {
    fn surface(&self, i: usize) -> &[u8] {
        let (a, b) = self.spans[i];
        &self.surfaces.as_bytes()[a as usize..b as usize]
    }
}

impl MecabDictionary {
    /// Loads a prebuilt image file or the dictionary directory `dir`, whose files are UTF-8, with
    /// lines split at whitespace in `whitespace` mode, within the default
    /// limits.
    pub fn load(dir: impl AsRef<Path>, whitespace: WhitespaceMode) -> Result<MecabDictionary, DictionaryError> {
        MecabDictionary::load_with_limits(dir, whitespace, DictionaryLimits::default())
    }

    /// [`load`](Self::load) within `limits`.
    pub fn load_with_limits(
        dir: impl AsRef<Path>,
        whitespace: WhitespaceMode,
        limits: DictionaryLimits,
    ) -> Result<MecabDictionary, DictionaryError> {
        if dir.as_ref().is_file() {
            let dict = Self::load_prebuilt(&dir, limits).map_err(DictionaryError::Prebuilt)?;
            if dict.whitespace != whitespace {
                return Err(DictionaryError::Prebuilt(super::PrebuiltError::Invalid("whitespace mode mismatch")));
            }
            return Ok(dict);
        }
        let mut source = Directory { dir: dir.as_ref().to_path_buf(), lexicons: Vec::new() };
        MecabDictionary::read(&mut source, whitespace, limits)
    }

    /// Reads a dictionary from the bytes of its files.
    pub fn from_files(
        files: DictionaryFiles<'_>,
        whitespace: WhitespaceMode,
        limits: DictionaryLimits,
    ) -> Result<MecabDictionary, DictionaryError> {
        let mut source = files;
        MecabDictionary::read(&mut source, whitespace, limits)
    }

    fn read(
        source: &mut dyn Source,
        whitespace: WhitespaceMode,
        limits: DictionaryLimits,
    ) -> Result<MecabDictionary, DictionaryError> {
        let mut manifest = String::new();
        let mut note = |name: &str, bytes: &[u8]| {
            manifest.push_str(name);
            manifest.push('\t');
            manifest.push_str(&sha256(bytes).to_hex());
            manifest.push('\n');
        };

        let bytes = source.matrix()?;
        note("matrix.def", &bytes);
        let matrix = Matrix::parse(utf8(&bytes, DictionaryFile::Matrix)?, whitespace, limits)?;
        drop(bytes);
        let dims = Dimensions { rows: matrix.rows(), cols: matrix.cols() };

        let mut features = FeatureStore::new();
        let mut lexicon = Lexicon { surfaces: String::new(), spans: Vec::new(), entries: Vec::new() };
        let mut count = 0;
        for (i, name) in source.lexicon_names()?.into_iter().enumerate() {
            let bytes = source.lexicon(i)?;
            note(&name, &bytes);
            let file = DictionaryFile::Lexicon(name);
            let text = utf8(&bytes, file.clone())?;
            let mut too_large = false;
            read_entries(text, &file, dims, limits.max_entries, &mut count, &mut features, |surface, entry| {
                let start = lexicon.surfaces.len();
                lexicon.surfaces.push_str(surface);
                match (u32::try_from(start), u32::try_from(lexicon.surfaces.len())) {
                    (Ok(a), Ok(b)) => lexicon.spans.push((a, b)),
                    _ => too_large = true,
                }
                lexicon.entries.push(entry);
            })?;
            if too_large {
                return Err(DictionaryError::TooLarge);
            }
        }
        if lexicon.entries.is_empty() {
            return Err(DictionaryError::NoLexiconEntries);
        }

        let bytes = source.char_def()?;
        note("char.def", &bytes);
        let chars = CharDefSource::parse(utf8(&bytes, DictionaryFile::CharDef)?, whitespace)?;
        drop(bytes);

        let bytes = source.unk_def()?;
        note("unk.def", &bytes);
        let mut unknown: Vec<(String, Entry)> = Vec::new();
        let mut unknown_count = 0;
        read_entries(
            utf8(&bytes, DictionaryFile::UnkDef)?,
            &DictionaryFile::UnkDef,
            dims,
            limits.max_entries,
            &mut unknown_count,
            &mut features,
            |surface, entry| unknown.push((surface.to_string(), entry)),
        )?;
        drop(bytes);
        let mut by_category: Vec<(CategoryId, Entry)> = Vec::with_capacity(unknown.len());
        for (name, entry) in unknown {
            let id = chars.category(&name).ok_or(DictionaryError::UnknownCategory(name))?;
            by_category.push((id, entry));
        }
        let categories = chars.len();

        // The lexicon by surface: a stable sort keeps each surface's
        // entries in file order, which decides ties.
        let mut order: Vec<u32> = (0..lexicon.entries.len() as u32).collect();
        order.sort_unstable_by(|&a, &b| lexicon.surface(a as usize).cmp(lexicon.surface(b as usize)).then(a.cmp(&b)));
        let mut entries: Vec<Entry> = Vec::with_capacity(lexicon.entries.len() + by_category.len());
        let mut groups: Vec<u32> = Vec::new();
        let mut keys: Vec<&[u8]> = Vec::new();
        for &i in &order {
            let surface = lexicon.surface(i as usize);
            if keys.last() != Some(&surface) {
                keys.push(surface);
                groups.push(entries.len() as u32);
            }
            entries.push(lexicon.entries[i as usize]);
        }
        groups.push(entries.len() as u32);
        let trie = SurfaceTrie::build(&keys).map_err(|_| DictionaryError::TooLarge)?;
        drop(keys);
        drop(lexicon);
        let lexicon_len = entries.len() as u32;

        // The templates by category, each category's in file order.
        by_category.sort_by_key(|&(id, _)| id.0);
        let mut unknown = vec![(0u32, 0u32); categories];
        for (id, entry) in by_category {
            let at = u32::try_from(entries.len()).map_err(|_| DictionaryError::TooLarge)?;
            let range = &mut unknown[id.0 as usize];
            if range.0 == range.1 {
                *range = (at, at);
            }
            range.1 = at + 1;
            entries.push(entry);
        }

        let chars = chars.resolve()?;
        let (pos, tags) = part_of_speech_tags(&entries, &features)?;
        features.shrink();
        Ok(MecabDictionary {
            matrix,
            trie,
            groups: groups.into(),
            entries: entries.into(),
            lexicon_len,
            unknown: unknown.into_boxed_slice(),
            chars,
            features,
            pos,
            tags: tags.into(),
            whitespace,
            sha256: sha256(manifest.as_bytes()),
        })
    }

    /// The feature columns of `entry`, in file order.
    pub fn features(&self, entry: EntryId) -> Columns<'_> {
        self.features.columns(self.entries[entry.0 as usize].features)
    }

    /// Feature column `i` of `entry` (counted from 0, the first after the
    /// cost), if it has one.
    #[inline]
    pub fn feature(&self, entry: EntryId, i: usize) -> Option<&str> {
        self.features.column(self.entries[entry.0 as usize].features, i)
    }

    /// The feature record of `entry`, for reading several of its columns
    /// with one lookup of the entry.
    #[inline]
    pub(crate) fn feature_record(&self, entry: EntryId) -> Record<'_> {
        self.features.record(self.entries[entry.0 as usize].features)
    }

    /// The number of feature columns of `entry`.
    pub fn feature_count(&self, entry: EntryId) -> usize {
        self.features.len(self.entries[entry.0 as usize].features)
    }

    /// The left and right context ids and the cost of `entry`.
    pub fn costs(&self, entry: EntryId) -> (u32, u32, i16) {
        let e = self.entries[entry.0 as usize];
        (e.left, e.right, e.cost)
    }

    /// Whether `entry` is an unknown word template of `unk.def`.
    pub fn is_unknown(&self, entry: EntryId) -> bool {
        entry.0 >= self.lexicon_len
    }

    /// The part of speech tags: every distinct first feature column, in the
    /// order the entries list them (an entry without features has the
    /// empty tag).
    pub fn pos_tags(&self) -> &Arc<TagSet> {
        &self.pos
    }

    /// The part of speech tag of `entry`, its first feature column.
    pub fn tag(&self, entry: EntryId) -> Tag {
        Tag(self.tags[entry.0 as usize])
    }

    /// The number of lexicon entries.
    pub fn lexicon_len(&self) -> usize {
        self.lexicon_len as usize
    }

    /// The number of distinct lexicon surfaces.
    pub fn surfaces(&self) -> usize {
        self.groups.len() - 1
    }

    /// The whitespace mode the files' lines were split in.
    pub fn whitespace(&self) -> WhitespaceMode {
        self.whitespace
    }

    /// The SHA-256 of the manifest of the files read: for each, in the
    /// order read, its name, a tab, its SHA-256 in lowercase hex and a line
    /// feed.
    pub fn sha256(&self) -> Digest {
        self.sha256
    }

    /// The entries of the surface group the trie gives as `value`.
    #[inline(always)]
    pub(crate) fn group(&self, value: u32) -> std::ops::Range<u32> {
        self.groups[value as usize]..self.groups[value as usize + 1]
    }

    /// The templates `unk.def` lists for a category, if any.
    #[inline(always)]
    pub(crate) fn templates(&self, category: CategoryId) -> Option<std::ops::Range<u32>> {
        let (a, b) = self.unknown[category.0 as usize];
        (a != b).then_some(a..b)
    }

    #[inline(always)]
    pub(crate) fn assignment_of(&self, cp: u32) -> AssignmentId {
        self.chars.assignment_of(cp)
    }

    #[inline(always)]
    pub(crate) fn category(&self, id: CategoryId) -> &Category {
        self.chars.category(id)
    }

    pub(crate) fn default_category(&self) -> CategoryId {
        self.chars.default
    }
}

/// The tag set of the entries' first feature columns, and each entry's
/// tag. More distinct tags than a [`Tag`] can number is refused.
fn part_of_speech_tags(
    entries: &[Entry],
    features: &FeatureStore,
) -> Result<(Arc<TagSet>, Box<[u16]>), DictionaryError> {
    let mut labels: Vec<&str> = Vec::new();
    let mut index: HashMap<&str, u16> = HashMap::new();
    let mut tags = Vec::with_capacity(entries.len());
    for e in entries {
        let first = features.columns(e.features).next().unwrap_or("");
        let tag = match index.get(first) {
            Some(&t) => t,
            None => {
                let t = u16::try_from(labels.len()).map_err(|_| DictionaryError::TooManyTags)?;
                labels.push(first);
                index.insert(first, t);
                t
            }
        };
        tags.push(tag);
    }
    Ok((TagSet::new("mecab", &labels), tags.into_boxed_slice()))
}

#[cfg(test)]
mod tests;
