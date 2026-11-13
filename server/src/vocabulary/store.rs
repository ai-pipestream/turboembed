//! The vocabulary store: a directory holding each artifact's bytes and a
//! small descriptor file beside them, `dictionaries/<id>.data` and
//! `<id>.meta` (with `<id>.image` for a compiled glossary), and
//! `vocabularies/<id>.tsv` and `<id>.meta`.
//!
//! Every file is written under a temporary name, synced and renamed into
//! place, the descriptor last, so an artifact exists exactly when its
//! descriptor does: a write cut short leaves only temporary files, which
//! opening the store removes. Writers are serialized by the index's lock,
//! and only an id the index holds ever names a path.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::proto::vocabulary::{
    CorpusStatisticsArtifactDescriptor, DictionaryArtifactDescriptor, SuggestLexiconArtifactDescriptor,
    VocabularyArtifactDescriptor, VoronoiCentroidArtifactDescriptor,
};

use super::feedback::{corpus_stats_meta, read_corpus_stats, read_voronoi, voronoi_meta};

const DICTIONARIES: &str = "dictionaries";
const VOCABULARIES: &str = "vocabularies";
const SUGGEST: &str = "suggest";
const CORPUS_STATS: &str = "corpus_stats";
const VORONOI: &str = "centroids";
const TMP: &str = ".tmp-";

/// A stored dictionary: its descriptor and how its upload is read again.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Dictionary {
    pub(crate) descriptor: DictionaryArtifactDescriptor,
    pub(crate) has_header_row: bool,
}

#[derive(Default)]
struct Index {
    dictionaries: BTreeMap<String, Dictionary>,
    vocabularies: BTreeMap<String, VocabularyArtifactDescriptor>,
    suggest_lexicons: BTreeMap<String, SuggestLexiconArtifactDescriptor>,
    corpus_stats: BTreeMap<String, CorpusStatisticsArtifactDescriptor>,
    voronoi: BTreeMap<String, VoronoiCentroidArtifactDescriptor>,
}

pub(crate) struct Store {
    dir: PathBuf,
    index: Mutex<Index>,
    seq: AtomicU64,
}

/// One artifact's descriptor and bytes, as DownloadVocabulary sends them.
pub(crate) enum Stored {
    Dictionary(DictionaryArtifactDescriptor, Vec<u8>),
    Vocabulary(VocabularyArtifactDescriptor, Vec<u8>),
}

fn io(path: &Path, e: std::io::Error) -> String {
    format!("{}: {e}", path.display())
}

/// `key=value` lines. Values never hold a line break: names and
/// languages are refused with control characters before they get here.
pub(super) fn meta(fields: &[(&str, String)]) -> Vec<u8> {
    let mut out = String::new();
    for (k, v) in fields {
        out.push_str(k);
        out.push('=');
        out.push_str(v);
        out.push('\n');
    }
    out.into_bytes()
}

fn parse_meta(text: &str) -> BTreeMap<&str, &str> {
    text.lines().filter_map(|l| l.split_once('=')).collect()
}

pub(super) fn num(m: &BTreeMap<&str, &str>, k: &str) -> Result<u64, String> {
    m.get(k).and_then(|v| v.parse().ok()).ok_or_else(|| format!("no number {k}"))
}

pub(super) fn text(m: &BTreeMap<&str, &str>, k: &str) -> String {
    m.get(k).copied().unwrap_or_default().to_owned()
}

fn dictionary_meta(d: &Dictionary) -> Vec<u8> {
    let x = &d.descriptor;
    meta(&[
        ("artifact_id", x.artifact_id.clone()),
        ("format_id", x.format_id.clone()),
        ("language", x.language.clone()),
        ("name", x.name.clone()),
        ("entry_count", x.entry_count.to_string()),
        ("byte_size", x.byte_size.to_string()),
        ("sha256", x.sha256.clone()),
        ("has_header_row", u8::from(d.has_header_row).to_string()),
    ])
}

fn read_dictionary(text_: &str) -> Result<Dictionary, String> {
    let m = parse_meta(text_);
    let descriptor = DictionaryArtifactDescriptor {
        artifact_id: text(&m, "artifact_id"),
        format_id: text(&m, "format_id"),
        language: text(&m, "language"),
        name: text(&m, "name"),
        entry_count: num(&m, "entry_count")?,
        byte_size: num(&m, "byte_size")?,
        sha256: text(&m, "sha256"),
    };
    Ok(Dictionary { descriptor, has_header_row: num(&m, "has_header_row")? == 1 })
}

fn vocabulary_meta(v: &VocabularyArtifactDescriptor) -> Vec<u8> {
    meta(&[
        ("artifact_id", v.artifact_id.clone()),
        ("language", v.language.clone()),
        ("documents", v.documents.to_string()),
        ("tokens", v.tokens.to_string()),
        ("term_count", v.term_count.to_string()),
        ("ngram_count", v.ngram_count.to_string()),
        ("byte_size", v.byte_size.to_string()),
        ("sha256", v.sha256.clone()),
        ("seed_dictionary_artifact_id", v.seed_dictionary_artifact_id.clone()),
    ])
}

fn read_vocabulary(text_: &str) -> Result<VocabularyArtifactDescriptor, String> {
    let m = parse_meta(text_);
    Ok(VocabularyArtifactDescriptor {
        artifact_id: text(&m, "artifact_id"),
        language: text(&m, "language"),
        documents: num(&m, "documents")?,
        tokens: num(&m, "tokens")?,
        term_count: num(&m, "term_count")?,
        ngram_count: num(&m, "ngram_count")?,
        byte_size: num(&m, "byte_size")?,
        sha256: text(&m, "sha256"),
        seed_dictionary_artifact_id: text(&m, "seed_dictionary_artifact_id"),
    })
}

fn suggest_meta(s: &SuggestLexiconArtifactDescriptor) -> Vec<u8> {
    meta(&[
        ("artifact_id", s.artifact_id.clone()),
        ("language", s.language.clone()),
        ("entry_count", s.entry_count.to_string()),
        ("stem_count", s.stem_count.to_string()),
        ("byte_size", s.byte_size.to_string()),
        ("sha256", s.sha256.clone()),
    ])
}

fn read_suggest(text_: &str) -> Result<SuggestLexiconArtifactDescriptor, String> {
    let m = parse_meta(text_);
    Ok(SuggestLexiconArtifactDescriptor {
        artifact_id: text(&m, "artifact_id"),
        language: text(&m, "language"),
        entry_count: num(&m, "entry_count")?,
        stem_count: num(&m, "stem_count")?,
        byte_size: num(&m, "byte_size")?,
        sha256: text(&m, "sha256"),
    })
}

/// The descriptors in `dir`, temporary files removed. A descriptor that
/// does not parse, or whose id is not its file's name, is refused.
fn scan<T>(dir: &Path, parse: impl Fn(&str) -> Result<T, String>, id: impl Fn(&T) -> &str) -> Result<Vec<T>, String> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| io(dir, e))? {
        let path = entry.map_err(|e| io(dir, e))?.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        if name.starts_with(TMP) {
            fs::remove_file(&path).map_err(|e| io(&path, e))?;
            continue;
        }
        let Some(stem) = name.strip_suffix(".meta") else { continue };
        let body = fs::read_to_string(&path).map_err(|e| io(&path, e))?;
        let t = parse(&body).map_err(|e| format!("{}: {e}", path.display()))?;
        if id(&t) != stem {
            return Err(format!("{}: names artifact {}", path.display(), id(&t)));
        }
        out.push(t);
    }
    Ok(out)
}

impl Store {
    /// The store in `dir`, made when it does not exist.
    pub(crate) fn open(dir: &Path) -> Result<Store, String> {
        let mut index = Index::default();
        for sub in [DICTIONARIES, VOCABULARIES, SUGGEST, CORPUS_STATS, VORONOI] {
            let p = dir.join(sub);
            fs::create_dir_all(&p).map_err(|e| io(&p, e))?;
        }
        for d in scan(&dir.join(DICTIONARIES), read_dictionary, |d| &d.descriptor.artifact_id)? {
            index.dictionaries.insert(d.descriptor.artifact_id.clone(), d);
        }
        for v in scan(&dir.join(VOCABULARIES), read_vocabulary, |v| &v.artifact_id)? {
            index.vocabularies.insert(v.artifact_id.clone(), v);
        }
        for s in scan(&dir.join(SUGGEST), read_suggest, |s| &s.artifact_id)? {
            index.suggest_lexicons.insert(s.artifact_id.clone(), s);
        }
        for c in scan(&dir.join(CORPUS_STATS), read_corpus_stats, |c| &c.artifact_id)? {
            index.corpus_stats.insert(c.artifact_id.clone(), c);
        }
        for vc in scan(&dir.join(VORONOI), read_voronoi, |vc| &vc.artifact_id)? {
            index.voronoi.insert(vc.artifact_id.clone(), vc);
        }
        Ok(Store { dir: dir.to_owned(), index: Mutex::new(index), seq: AtomicU64::new(0) })
    }

    fn index(&self) -> std::sync::MutexGuard<'_, Index> {
        self.index.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub(crate) fn dictionaries(&self) -> Vec<DictionaryArtifactDescriptor> {
        self.index().dictionaries.values().map(|d| d.descriptor.clone()).collect()
    }

    pub(crate) fn vocabularies(&self) -> Vec<VocabularyArtifactDescriptor> {
        self.index().vocabularies.values().cloned().collect()
    }

    pub(crate) fn suggest_lexicons(&self) -> Vec<SuggestLexiconArtifactDescriptor> {
        self.index().suggest_lexicons.values().cloned().collect()
    }

    pub(crate) fn corpus_statistics(&self) -> Vec<CorpusStatisticsArtifactDescriptor> {
        self.index().corpus_stats.values().cloned().collect()
    }

    pub(crate) fn voronoi_centroids(&self) -> Vec<VoronoiCentroidArtifactDescriptor> {
        self.index().voronoi.values().cloned().collect()
    }

    pub(crate) fn dictionary(&self, id: &str) -> Option<Dictionary> {
        self.index().dictionaries.get(id).cloned()
    }

    pub(crate) fn suggest_file(&self, id: &str) -> Option<(SuggestLexiconArtifactDescriptor, PathBuf)> {
        let desc = self.index().suggest_lexicons.get(id)?.clone();
        Some((desc, self.dir.join(SUGGEST).join(format!("{id}.slex"))))
    }

    /// Checks whether `id` is in the store's index and whether its stored
    /// descriptor has `sha256`.
    pub(crate) fn check_digest(&self, id: &str, sha256: &str) -> Option<bool> {
        let index = self.index();
        if let Some(d) = index.dictionaries.get(id) {
            Some(d.descriptor.sha256 == sha256)
        } else if let Some(v) = index.vocabularies.get(id) {
            Some(v.sha256 == sha256)
        } else if let Some(c) = index.corpus_stats.get(id) {
            Some(c.sha256 == sha256)
        } else if let Some(vc) = index.voronoi.get(id) {
            Some(vc.sha256 == sha256)
        } else {
            index.suggest_lexicons.get(id).map(|s| s.sha256 == sha256)
        }
    }

    /// Writes `bytes` to `dir/name` through a synced temporary file.
    fn write(&self, dir: &str, name: &str, bytes: &[u8]) -> Result<(), String> {
        let n = self.seq.fetch_add(1, Ordering::Relaxed);
        let tmp = self.dir.join(dir).join(format!("{TMP}{}-{n}", std::process::id()));
        let dest = self.dir.join(dir).join(name);
        let mut f = fs::File::create(&tmp).map_err(|e| io(&tmp, e))?;
        f.write_all(bytes).and_then(|()| f.sync_all()).map_err(|e| io(&tmp, e))?;
        drop(f);
        fs::rename(&tmp, &dest).map_err(|e| {
            let _ = fs::remove_file(&tmp);
            io(&dest, e)
        })
    }

    /// Stores a dictionary, or answers with the one already stored under
    /// its id.
    pub(crate) fn put_dictionary(
        &self,
        d: Dictionary,
        bytes: &[u8],
        image: Option<&[u8]>,
    ) -> Result<DictionaryArtifactDescriptor, String> {
        let mut index = self.index();
        let id = d.descriptor.artifact_id.clone();
        if let Some(have) = index.dictionaries.get(&id) {
            return Ok(have.descriptor.clone());
        }
        self.write(DICTIONARIES, &format!("{id}.data"), bytes)?;
        if let Some(image) = image {
            self.write(DICTIONARIES, &format!("{id}.image"), image)?;
        }
        self.write(DICTIONARIES, &format!("{id}.meta"), &dictionary_meta(&d))?;
        let out = d.descriptor.clone();
        index.dictionaries.insert(id, d);
        Ok(out)
    }

    /// Stores a vocabulary, or answers with the one already stored under
    /// its id. The flag is whether this call stored it.
    pub(crate) fn put_vocabulary(
        &self,
        v: VocabularyArtifactDescriptor,
        bytes: &[u8],
    ) -> Result<(VocabularyArtifactDescriptor, bool), String> {
        let mut index = self.index();
        if let Some(have) = index.vocabularies.get(&v.artifact_id) {
            return Ok((have.clone(), false));
        }
        self.write(VOCABULARIES, &format!("{}.tsv", v.artifact_id), bytes)?;
        self.write(VOCABULARIES, &format!("{}.meta", v.artifact_id), &vocabulary_meta(&v))?;
        index.vocabularies.insert(v.artifact_id.clone(), v.clone());
        Ok((v, true))
    }

    pub(crate) fn put_suggest(
        &self,
        s: SuggestLexiconArtifactDescriptor,
        bytes: &[u8],
    ) -> Result<SuggestLexiconArtifactDescriptor, String> {
        let mut index = self.index();
        if let Some(have) = index.suggest_lexicons.get(&s.artifact_id) {
            return Ok(have.clone());
        }
        self.write(SUGGEST, &format!("{}.slex", s.artifact_id), bytes)?;
        self.write(SUGGEST, &format!("{}.meta", s.artifact_id), &suggest_meta(&s))?;
        index.suggest_lexicons.insert(s.artifact_id.clone(), s.clone());
        Ok(s)
    }

    pub(crate) fn put_corpus_stats(
        &self,
        c: CorpusStatisticsArtifactDescriptor,
        bytes: &[u8],
    ) -> Result<CorpusStatisticsArtifactDescriptor, String> {
        let mut index = self.index();
        if let Some(have) = index.corpus_stats.get(&c.artifact_id) {
            return Ok(have.clone());
        }
        self.write(CORPUS_STATS, &format!("{}.cidf", c.artifact_id), bytes)?;
        self.write(CORPUS_STATS, &format!("{}.meta", c.artifact_id), &corpus_stats_meta(&c))?;
        index.corpus_stats.insert(c.artifact_id.clone(), c.clone());
        Ok(c)
    }

    pub(crate) fn put_voronoi(
        &self,
        v: VoronoiCentroidArtifactDescriptor,
        bytes: &[u8],
    ) -> Result<VoronoiCentroidArtifactDescriptor, String> {
        let mut index = self.index();
        if let Some(have) = index.voronoi.get(&v.artifact_id) {
            return Ok(have.clone());
        }
        self.write(VORONOI, &format!("{}.tcent", v.artifact_id), bytes)?;
        self.write(VORONOI, &format!("{}.meta", v.artifact_id), &voronoi_meta(&v))?;
        index.voronoi.insert(v.artifact_id.clone(), v.clone());
        Ok(v)
    }

    /// Removes an artifact, its descriptor first: false when there is none.
    pub(crate) fn delete(&self, id: &str) -> Result<bool, String> {
        let mut index = self.index();
        let (dir, files): (&str, &[&str]) = if index.dictionaries.contains_key(id) {
            (DICTIONARIES, &["meta", "data", "image"])
        } else if index.vocabularies.contains_key(id) {
            (VOCABULARIES, &["meta", "tsv"])
        } else if index.suggest_lexicons.contains_key(id) {
            (SUGGEST, &["meta", "slex"])
        } else if index.corpus_stats.contains_key(id) {
            (CORPUS_STATS, &["meta", "cidf"])
        } else if index.voronoi.contains_key(id) {
            (VORONOI, &["meta", "tcent"])
        } else {
            return Ok(false);
        };
        for ext in files {
            let p = self.dir.join(dir).join(format!("{id}.{ext}"));
            match fs::remove_file(&p) {
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(io(&p, e)),
                _ => {}
            }
        }
        index.dictionaries.remove(id);
        index.vocabularies.remove(id);
        index.suggest_lexicons.remove(id);
        index.corpus_stats.remove(id);
        index.voronoi.remove(id);
        Ok(true)
    }

    pub(crate) fn read_feedback(
        &self,
        id: &str,
    ) -> Result<Option<(crate::proto::vocabulary::vocabulary_artifact_chunk::Frame, Vec<u8>)>, String> {
        use crate::proto::vocabulary::vocabulary_artifact_chunk::Frame;
        let (path, size, sha, frame) = {
            let index = self.index();
            if let Some(c) = index.corpus_stats.get(id) {
                (
                    self.dir.join(CORPUS_STATS).join(format!("{id}.cidf")),
                    c.byte_size,
                    c.sha256.clone(),
                    Frame::CorpusStatistics(c.clone()),
                )
            } else if let Some(v) = index.voronoi.get(id) {
                (
                    self.dir.join(VORONOI).join(format!("{id}.tcent")),
                    v.byte_size,
                    v.sha256.clone(),
                    Frame::VoronoiCentroids(v.clone()),
                )
            } else {
                return Ok(None);
            }
        };
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(io(&path, e)),
        };
        if bytes.len() as u64 != size || turbo_maxent::sha256(&bytes).to_hex() != sha {
            return Err(format!("{}: the stored bytes do not match their descriptor", path.display()));
        }
        Ok(Some((frame, bytes)))
    }

    pub(crate) fn read_suggest(&self, id: &str) -> Result<Option<(SuggestLexiconArtifactDescriptor, Vec<u8>)>, String> {
        let Some((desc, path)) = self.suggest_file(id) else {
            return Ok(None);
        };
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(io(&path, e)),
        };
        if bytes.len() as u64 != desc.byte_size || turbo_maxent::sha256(&bytes).to_hex() != desc.sha256 {
            return Err(format!("{}: the stored bytes do not match their descriptor", path.display()));
        }
        Ok(Some((desc, bytes)))
    }

    /// An artifact's descriptor and bytes, checked against the descriptor's
    /// size and digest. An artifact deleted while it is read is not found.
    pub(crate) fn read(&self, id: &str) -> Result<Option<Stored>, String> {
        let (path, empty) = {
            let index = self.index();
            if let Some(d) = index.dictionaries.get(id) {
                let path = self.dir.join(DICTIONARIES).join(format!("{id}.data"));
                (path, Stored::Dictionary(d.descriptor.clone(), Vec::new()))
            } else if let Some(v) = index.vocabularies.get(id) {
                (self.dir.join(VOCABULARIES).join(format!("{id}.tsv")), Stored::Vocabulary(v.clone(), Vec::new()))
            } else {
                return Ok(None);
            }
        };
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(io(&path, e)),
        };
        let (size, sha) = match &empty {
            Stored::Dictionary(d, _) => (d.byte_size, &d.sha256),
            Stored::Vocabulary(v, _) => (v.byte_size, &v.sha256),
        };
        if bytes.len() as u64 != size || turbo_maxent::sha256(&bytes).to_hex() != *sha {
            return Err(format!("{}: the stored bytes do not match their descriptor", path.display()));
        }
        Ok(Some(match empty {
            Stored::Dictionary(d, _) => Stored::Dictionary(d, bytes),
            Stored::Vocabulary(v, _) => Stored::Vocabulary(v, bytes),
        }))
    }
}
