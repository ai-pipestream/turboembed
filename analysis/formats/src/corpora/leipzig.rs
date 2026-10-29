//! The Leipzig Corpora Collection for language detection:
//! `LeipzigLanguageSampleStream` behind `SampleShuffleStream` and
//! `SampleSkipStream`, as `LeipzigLanguageSampleStreamFactory` stacks
//! them.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};

use turbo_analysis::Span;
use turbo_jvm::random::JavaRandom;

use super::DirectoryError;
use super::error::CorpusError;
use crate::error::FormatError;
use crate::sample::LanguageSample;
use crate::source::{Field, Lines};

/// The seed of both of the factory's `java.util.Random`s.
const SEED: i64 = 23;

/// The length of the language code a file name starts with.
const CODE_LENGTH: usize = 3;

/// The factory's parameters.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct LeipzigParams {
    /// `sentencesPerSample`.
    pub sentences_per_sample: u32,
    /// `samplesPerLanguage`.
    pub samples_per_language: u32,
    /// `samplesToSkip`, 0 by default.
    pub samples_to_skip: u32,
}

/// One language sample: the text of some sentences of one file, each
/// followed by a space.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeipzigSample {
    /// The language code, the first three characters of the file's name.
    pub language: String,
    /// The sentences, each followed by a space.
    pub context: String,
    /// The file the sentences are from, an index into
    /// [`LeipzigCorpus::files`].
    pub file: usize,
    /// The bytes of each sentence in its file: the line after its first
    /// tab.
    pub sentences: Vec<Span>,
}

impl LeipzigSample {
    /// The sample as a [`LanguageSample`]. The language is from the
    /// file's name, not its text, so its span is empty, at the start of
    /// the context's; the context's span and the source run from the first
    /// byte of the sentence that comes first in the file to the end of the
    /// one that comes last.
    pub fn language_sample(&self) -> LanguageSample<'_> {
        let start = self.sentences.iter().map(|s| s.start).min().unwrap_or(0);
        let end = self.sentences.iter().map(|s| s.end).max().unwrap_or(0);
        let hull = Span::new(start, end);
        LanguageSample {
            language: Field { text: Cow::Borrowed(&self.language), span: Span::new(start, start) },
            context: Field { text: Cow::Borrowed(&self.context), span: hull },
            source: hull,
        }
    }
}

/// `LanguageSample.toString`: the language, a tab and the context.
impl fmt::Display for LeipzigSample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}\t{}", self.language, self.context)
    }
}

/// The sentence files of a Leipzig directory, in the order the stream
/// reads them: the files (symbolic links followed) whose name does not
/// start with `.` and starts with three letters `a` to `z`, sorted by
/// path in Java's string order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeipzigCorpus {
    files: Vec<PathBuf>,
}

/// Whether `name` starts with three characters `a` to `z`.
fn has_language_code(name: &str) -> bool {
    name.len() >= CODE_LENGTH && name.as_bytes()[..CODE_LENGTH].iter().all(u8::is_ascii_lowercase)
}

impl LeipzigCorpus {
    /// The sentence files of the directory `dir`.
    ///
    /// # Errors
    ///
    /// [`DirectoryError::NotADirectory`] when `dir` is not a directory
    /// (OpenNLP ends the tool), and what listing it gives.
    pub fn open(dir: &Path) -> Result<LeipzigCorpus, DirectoryError> {
        if !dir.is_dir() {
            return Err(DirectoryError::NotADirectory { path: dir.to_owned() });
        }
        let io = |error| DirectoryError::Io { path: dir.to_owned(), error };
        let mut files = Vec::new();
        for entry in std::fs::read_dir(dir).map_err(io)? {
            let path = entry.map_err(io)?.path();
            let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            if !name.starts_with('.') && has_language_code(&name) && path.is_file() {
                files.push(path);
            }
        }
        Ok(LeipzigCorpus::from_files(files))
    }

    /// A corpus of the given sentence files, sorted as the stream sorts
    /// them. The files must have names [`LeipzigCorpus::open`] would
    /// keep.
    ///
    /// # Panics
    ///
    /// When a file's name does not start with a language code.
    pub fn from_files(mut files: Vec<PathBuf>) -> LeipzigCorpus {
        for f in &files {
            assert!(has_language_code(&name(f)), "{} has no language code", f.display());
        }
        files.sort_by(|a, b| turbo_jvm::chars::compare(&a.to_string_lossy(), &b.to_string_lossy()));
        LeipzigCorpus { files }
    }

    /// The files, in the order they are read.
    pub fn files(&self) -> &[PathBuf] {
        &self.files
    }

    /// The samples the factory's stream gives, reading each file's text
    /// with `read` (decoded as UTF-8).
    ///
    /// Each language gets `(samples_per_language + samples_to_skip) /
    /// files of the language` samples from each of its files. A file's
    /// stream shuffles the indexes of all its lines with the factory's
    /// `Random(23)`, which every file shares, keeps the lines among the
    /// first `sentences_per_sample` times that many indexes that have a
    /// tab (so a line without one makes a sample short), shuffles them
    /// again and cuts them into samples of `sentences_per_sample` lines
    /// in that order, a sentence being the line after its first tab.
    /// Then `SampleShuffleStream` shuffles all the samples with a new
    /// `Random(23)`, and `SampleSkipStream` drops the first
    /// `samples_to_skip`.
    ///
    /// # Errors
    ///
    /// [`CorpusError::LeipzigTooFewLines`] for a file with fewer lines
    /// than its samples need: OpenNLP throws it before any sample is read.
    /// [`CorpusError::LeipzigParameters`] when the counts overflow a Java
    /// `int`. What `read` gives.
    pub fn samples(
        &self,
        params: LeipzigParams,
        mut read: impl FnMut(&Path) -> std::io::Result<String>,
    ) -> Result<Vec<LeipzigSample>, DirectoryError> {
        let format = |e| DirectoryError::Format(FormatError::Corpus(e));
        let per_language = params
            .samples_per_language
            .checked_add(params.samples_to_skip)
            .filter(|&n| i32::try_from(n).is_ok())
            .ok_or(format(CorpusError::LeipzigParameters))?;
        let mut files_per_language: HashMap<String, u32> = HashMap::new();
        for f in &self.files {
            *files_per_language.entry(language(f)).or_default() += 1;
        }
        let mut random = JavaRandom::new(SEED);
        let mut samples = Vec::new();
        for (index, path) in self.files.iter().enumerate() {
            let lang = language(path);
            let quota = per_language / files_per_language[&lang];
            let required = u64::from(params.sentences_per_sample) * u64::from(quota);
            if i32::try_from(required).is_err() {
                return Err(format(CorpusError::LeipzigParameters));
            }
            let text = read(path).map_err(|error| DirectoryError::Io { path: path.clone(), error })?;
            let lines = Lines::new(&text).map_err(DirectoryError::Format)?;
            let total = lines.clone().count();
            if (total as u64) < required {
                let path = path.to_string_lossy().into_owned();
                return Err(format(CorpusError::LeipzigTooFewLines { path, lines: total, required }));
            }
            let mut indexes: Vec<u32> = (0..total as u32).collect();
            random.shuffle(&mut indexes);
            let selected: HashSet<u32> = indexes[..required as usize].iter().copied().collect();
            let mut sentences: Vec<Span> = Vec::with_capacity(required as usize);
            for (i, line) in lines.enumerate() {
                if let Some(tab) = line.text().find('\t')
                    && selected.contains(&(i as u32))
                {
                    let span = line.span();
                    sentences.push(Span::new(span.start + tab as u32 + 1, span.end));
                }
            }
            random.shuffle(&mut sentences);
            if params.sentences_per_sample == 0 {
                continue;
            }
            for chunk in sentences.chunks(params.sentences_per_sample as usize) {
                let mut context = String::new();
                for s in chunk {
                    context.push_str(&text[s.range()]);
                    context.push(' ');
                }
                samples.push(LeipzigSample { language: lang.clone(), context, file: index, sentences: chunk.to_vec() });
            }
        }
        JavaRandom::new(SEED).shuffle(&mut samples);
        samples.drain(..(params.samples_to_skip as usize).min(samples.len()));
        Ok(samples)
    }
}

fn name(path: &Path) -> String {
    path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// The language code of a sentence file.
fn language(path: &Path) -> String {
    name(path)[..CODE_LENGTH].to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(files: &[(&str, &str)], params: LeipzigParams) -> Result<Vec<String>, String> {
        let corpus = LeipzigCorpus::from_files(files.iter().map(|(n, _)| PathBuf::from(n)).collect());
        let texts: HashMap<PathBuf, String> = files.iter().map(|(n, t)| (PathBuf::from(n), t.to_string())).collect();
        corpus
            .samples(params, |p| Ok(texts[p].clone()))
            .map(|s| s.iter().map(ToString::to_string).collect())
            .map_err(|e| e.to_string())
    }

    #[test]
    fn samples_come_shuffled_and_skipped() {
        let deu = "1\teins\n2\tzwei\n3\tdrei\n4\tvier\n";
        let eng = "1\tone\n2\ttwo\nno tab\n4\tfour\n";
        let params = LeipzigParams { sentences_per_sample: 2, samples_per_language: 2, samples_to_skip: 0 };
        let all = read(&[("eng_news.txt", eng), ("deu_news.txt", deu)], params).unwrap();
        assert_eq!(all.len(), 4);
        assert!(all.iter().all(|s| s.starts_with("deu\t") || s.starts_with("eng\t")));
        let skipped = LeipzigParams { samples_per_language: 1, samples_to_skip: 1, ..params };
        assert_eq!(read(&[("eng_news.txt", eng), ("deu_news.txt", deu)], skipped).unwrap(), all[1..]);
    }

    #[test]
    fn too_few_lines() {
        let params = LeipzigParams { sentences_per_sample: 3, samples_per_language: 2, samples_to_skip: 0 };
        assert_eq!(
            read(&[("eng.txt", "1\ta\n")], params).unwrap_err(),
            "eng.txt does not contain enough lines (1 lines < 6 required lines)."
        );
    }

    #[test]
    fn language_codes() {
        assert!(has_language_code("eng_news"));
        assert!(!has_language_code("en"));
        assert!(!has_language_code("En_x"));
    }
}
