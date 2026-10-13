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
use super::entries::{Columns, Dimensions, Entry, FeatureStore, read_entries};
use super::error::{DictionaryError, DictionaryFile};
use super::limits::DictionaryLimits;
use super::matrix::Matrix;
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
    groups: Box<[u32]>,
    /// The lexicon entries by surface group, each group in file order,
    /// then the unknown word templates by category, each in file order.
    pub(crate) entries: Box<[Entry]>,
    lexicon_len: u32,
    /// The templates of each category, as a range of `entries`; empty when
    /// `unk.def` lists none.
    unknown: Box<[(u32, u32)]>,
    pub(crate) chars: CharDef,
    features: FeatureStore,
    pos: Arc<TagSet>,
    tags: Box<[Tag]>,
    whitespace: WhitespaceMode,
    sha256: Digest,
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
    /// Loads the dictionary directory `dir`, whose files are UTF-8, with
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
            groups: groups.into_boxed_slice(),
            entries: entries.into_boxed_slice(),
            lexicon_len,
            unknown: unknown.into_boxed_slice(),
            chars,
            features,
            pos,
            tags,
            whitespace,
            sha256: sha256(manifest.as_bytes()),
        })
    }

    /// The feature columns of `entry`, in file order.
    pub fn features(&self, entry: EntryId) -> Columns<'_> {
        self.features.columns(self.entries[entry.0 as usize].features)
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
        self.tags[entry.0 as usize]
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
) -> Result<(Arc<TagSet>, Box<[Tag]>), DictionaryError> {
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
        tags.push(Tag(tag));
    }
    Ok((TagSet::new("mecab", &labels), tags.into_boxed_slice()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::LatticeTokenizer;

    const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../goldens/data/fixtures/mecab-fixture-ja");
    const MATRIX: &[u8] = include_bytes!("../../../goldens/data/fixtures/mecab-fixture-ja/matrix.def");
    const NOUNS: &[u8] = include_bytes!("../../../goldens/data/fixtures/mecab-fixture-ja/A-nouns.csv");
    const OTHERS: &[u8] = include_bytes!("../../../goldens/data/fixtures/mecab-fixture-ja/b-others.csv");
    const CHAR_DEF: &[u8] = include_bytes!("../../../goldens/data/fixtures/mecab-fixture-ja/char.def");
    const UNK_DEF: &[u8] = include_bytes!("../../../goldens/data/fixtures/mecab-fixture-ja/unk.def");

    fn from_files(
        matrix: &[u8],
        lexicons: &[(&str, &[u8])],
        char_def: &[u8],
        unk_def: &[u8],
    ) -> Result<MecabDictionary, DictionaryError> {
        let files = DictionaryFiles { matrix, lexicons, char_def, unk_def };
        MecabDictionary::from_files(files, WhitespaceMode::Unicode, DictionaryLimits::default())
    }

    fn fixture() -> MecabDictionary {
        from_files(MATRIX, &[("A-nouns.csv", NOUNS), ("b-others.csv", OTHERS)], CHAR_DEF, UNK_DEF).unwrap()
    }

    #[test]
    fn a_directory_and_its_bytes_load_the_same_dictionary() {
        let dir = MecabDictionary::load(FIXTURE, WhitespaceMode::Unicode).unwrap();
        let mem = fixture();
        assert_eq!(dir.sha256(), mem.sha256());
        assert_eq!(dir.lexicon_len(), mem.lexicon_len());
        assert_eq!(dir.surfaces(), mem.surfaces());
        assert_eq!(dir.pos_tags().len(), mem.pos_tags().len());
        assert_eq!(dir.tag(EntryId(0)), mem.tag(EntryId(0)));
    }

    #[test]
    fn missing_files_are_named_in_load_order() {
        let tmp = std::env::temp_dir().join(format!("turbo-lattice-missing-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let load = || MecabDictionary::load(&tmp, WhitespaceMode::Unicode).unwrap_err();
        assert!(matches!(load(), DictionaryError::Missing(DictionaryFile::Matrix)));
        std::fs::write(tmp.join("matrix.def"), MATRIX).unwrap();
        assert!(matches!(load(), DictionaryError::NoLexiconEntries));
        std::fs::write(tmp.join("A-nouns.csv"), NOUNS).unwrap();
        assert!(matches!(load(), DictionaryError::Missing(DictionaryFile::CharDef)));
        std::fs::write(tmp.join("char.def"), CHAR_DEF).unwrap();
        assert!(matches!(load(), DictionaryError::Missing(DictionaryFile::UnkDef)));
        // A path that exists but is not a file is an I/O error.
        std::fs::create_dir(tmp.join("unk.def")).unwrap();
        assert!(matches!(load(), DictionaryError::Io { file: DictionaryFile::UnkDef, .. }));
        std::fs::remove_dir(tmp.join("unk.def")).unwrap();
        std::fs::write(tmp.join("unk.def"), UNK_DEF).unwrap();
        // A file named like a lexicon but not ending in .csv is not read.
        std::fs::write(tmp.join("z.csv.bak"), b"not, a lexicon").unwrap();
        assert!(MecabDictionary::load(&tmp, WhitespaceMode::Unicode).is_ok());
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn bytes_that_are_not_utf8_are_refused_with_their_offset() {
        let err = from_files(b"1 1\n0 0 \xff\n", &[("a.csv", NOUNS)], CHAR_DEF, UNK_DEF).unwrap_err();
        assert!(matches!(err, DictionaryError::Utf8 { file: DictionaryFile::Matrix, offset: 8 }));
        let err = from_files(MATRIX, &[("a.csv", b"\xe6\x9d")], CHAR_DEF, UNK_DEF).unwrap_err();
        assert!(
            matches!(err, DictionaryError::Utf8 { file: DictionaryFile::Lexicon(ref n), offset: 0 } if n == "a.csv")
        );
        let err = from_files(MATRIX, &[("a.csv", NOUNS)], b"\xc0", UNK_DEF).unwrap_err();
        assert!(matches!(err, DictionaryError::Utf8 { file: DictionaryFile::CharDef, .. }));
        let err = from_files(MATRIX, &[("a.csv", NOUNS)], CHAR_DEF, b"\x80").unwrap_err();
        assert!(matches!(err, DictionaryError::Utf8 { file: DictionaryFile::UnkDef, .. }));
    }

    /// Deviation: where unk.def names several undefined categories, the
    /// Java reports the first in the iteration order of a `HashMap`, and
    /// the port the first in file order.
    #[test]
    fn unk_def_names_only_defined_categories() {
        let unk = b"DEFAULT,0,0,0,x\nNOSUCH,0,0,0,x\n";
        let err = from_files(MATRIX, &[("a.csv", NOUNS)], CHAR_DEF, unk).unwrap_err();
        assert!(matches!(err, DictionaryError::UnknownCategory(ref n) if n == "NOSUCH"));
        let unk = b"ZZZ,0,0,0,x\nDEFAULT,0,0,0,x\nAAA,0,0,0,x\n";
        let err = from_files(MATRIX, &[("a.csv", NOUNS)], CHAR_DEF, unk).unwrap_err();
        assert!(matches!(err, DictionaryError::UnknownCategory(ref n) if n == "ZZZ"));
    }

    /// Deviation: the port checks that a whole file is UTF-8 before it
    /// reads a line of it; the Java decodes as it reads, so a malformed
    /// line before the bad bytes is the Java's error and the bad bytes are
    /// the port's.
    #[test]
    fn a_file_is_checked_for_utf8_before_its_lines_are_read() {
        let err = from_files(b"1 1\nx\n\xff\n", &[("a.csv", NOUNS)], CHAR_DEF, UNK_DEF).unwrap_err();
        assert!(matches!(err, DictionaryError::Utf8 { file: DictionaryFile::Matrix, offset: 6 }));
        let err = from_files(b"1 1\nx\n", &[("a.csv", NOUNS)], CHAR_DEF, UNK_DEF).unwrap_err();
        assert!(matches!(err, DictionaryError::MatrixLine { line: 2 }));
    }

    /// Deviation: OpenNLP keeps any number of distinct part-of-speech
    /// columns; a [`Tag`] numbers at most 65536 of them, so more are
    /// refused rather than wrapped.
    #[test]
    fn more_tags_than_a_tag_can_number_are_refused() {
        let mut lexicon = String::new();
        for i in 0..=65536u32 {
            lexicon.push_str(&format!("w{i},0,0,0,t{i}\n"));
        }
        let char_def = b"DEFAULT 0 1 0\n";
        let unk = b"DEFAULT,0,0,0,t0\n";
        let err = from_files(b"1 1\n0 0 0\n", &[("a.csv", lexicon.as_bytes())], char_def, unk).unwrap_err();
        assert!(matches!(err, DictionaryError::TooManyTags));
        let fits: String = lexicon.lines().take(65535).map(|l| format!("{l}\n")).collect();
        let d = from_files(b"1 1\n0 0 0\n", &[("a.csv", fits.as_bytes())], char_def, unk).unwrap();
        assert_eq!(d.pos_tags().len(), 65535);
    }

    #[test]
    fn the_entry_limit_counts_the_lexicon_and_unk_def_apart() {
        // The fixture's A-nouns.csv has 20 entries and its unk.def 13.
        let files =
            DictionaryFiles { matrix: MATRIX, lexicons: &[("a.csv", NOUNS)], char_def: CHAR_DEF, unk_def: UNK_DEF };
        let limits = |max_entries| DictionaryLimits { max_entries, ..DictionaryLimits::default() };
        let err = MecabDictionary::from_files(files, WhitespaceMode::Unicode, limits(19)).unwrap_err();
        assert!(matches!(err, DictionaryError::TooManyEntries { limit: 19 }));
        assert!(MecabDictionary::from_files(files, WhitespaceMode::Unicode, limits(20)).is_ok());
    }

    /// Mutated fixture files load or fail with an error; whatever loads
    /// tokenizes text without a panic.
    #[test]
    fn mutated_files_never_panic() {
        let originals: [&[u8]; 5] = [MATRIX, NOUNS, OTHERS, CHAR_DEF, UNK_DEF];
        let texts = ["東京都に行く", "ｶﾀｶﾅ 123 abc", "𠀋😀αβγ", "  ", "一二三。", "\u{30FC}\u{30FC}x"];
        let limits = DictionaryLimits { max_entries: 10_000, max_matrix_cells: 10_000 };
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let (mut loaded, mut refused) = (0, 0);
        for _ in 0..3000 {
            let mut files: Vec<Vec<u8>> = originals.iter().map(|f| f.to_vec()).collect();
            for _ in 0..1 + next() % 4 {
                let f = &mut files[(next() % 5) as usize];
                let at = if f.is_empty() { 0 } else { (next() as usize) % f.len() };
                match next() % 6 {
                    0 if !f.is_empty() => f[at] = next() as u8,
                    1 => f.insert(at, b"0123456789-,. \n\t"[(next() % 16) as usize]),
                    2 if !f.is_empty() => {
                        f.remove(at);
                    }
                    3 => f.truncate(at),
                    4 => f.splice(at..at, *b"-2147483648").for_each(drop),
                    _ => f.splice(at..at, "\u{3000}東".bytes()).for_each(drop),
                }
            }
            let lexicons = [("A-nouns.csv", &files[1][..]), ("b-others.csv", &files[2][..])];
            let d = DictionaryFiles { matrix: &files[0], lexicons: &lexicons, char_def: &files[3], unk_def: &files[4] };
            match MecabDictionary::from_files(d, WhitespaceMode::Unicode, limits) {
                Ok(d) => {
                    loaded += 1;
                    let t = LatticeTokenizer::new(d, WhitespaceMode::Unicode);
                    let mut out = Vec::new();
                    for text in texts {
                        let _ = t.analyze(text, &mut out);
                    }
                }
                Err(e) => {
                    refused += 1;
                    let _ = e.to_string();
                }
            }
        }
        assert!(loaded > 0 && refused > 0, "loaded {loaded}, refused {refused}");
    }
}
