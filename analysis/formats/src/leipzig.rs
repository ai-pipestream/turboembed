//! Leipzig Corpora Collection language sample reader and stream adapters
//! (`opennlp.tools.formats.leipzig`).
//!
//! Port of `LeipzigLanguageSampleStream`, `LeipzigLanguageSampleStreamFactory`,
//! `SampleShuffleStream`, and `SampleSkipStream` from OpenNLP format utilities.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use turbo_analysis::Span;
use turbo_jvm::random::JavaRandom;

use crate::error::FormatError;
use crate::sample::LanguageSample;
use crate::source::{Field, Lines};

/// Length of the ISO 639-3 language prefix at the start of Leipzig file names.
pub const LANG_CODE_LENGTH: usize = 3;

/// Default seed used by OpenNLP's `LeipzigLanguageSampleStream` and `SampleShuffleStream`.
pub const LEIPZIG_RANDOM_SEED: i64 = 23;

/// Checks whether `file_name` starts with 3 ASCII lowercase letters (`'a'..='z'`).
///
/// Mirrors `LeipzigLanguageSampleStream.hasLanguageCodePrefix`.
#[inline]
pub fn has_language_code_prefix(file_name: &str) -> bool {
    file_name.as_bytes().get(..LANG_CODE_LENGTH).is_some_and(|prefix| prefix.iter().all(u8::is_ascii_lowercase))
}

/// Stream adapter that buffers all items from an inner iterator and shuffles them
/// with `new Random(23)` (`JavaRandom::new(23)`).
///
/// Mirrors `opennlp.tools.formats.leipzig.SampleShuffleStream`.
#[derive(Debug, Clone)]
pub struct SampleShuffleStream<T> {
    buffered_samples: Vec<T>,
    index: usize,
}

impl<T> SampleShuffleStream<T> {
    /// Collects all items from `samples` and shuffles them using `JavaRandom::new(23)`.
    pub fn new(samples: impl IntoIterator<Item = T>) -> Self {
        let mut buffered_samples: Vec<T> = samples.into_iter().collect();
        let mut rng = JavaRandom::new(LEIPZIG_RANDOM_SEED);
        rng.shuffle(&mut buffered_samples);
        Self { buffered_samples, index: 0 }
    }

    /// Collects fallible items from `samples` and shuffles them using `JavaRandom::new(23)`.
    pub fn try_new<E>(samples: impl IntoIterator<Item = Result<T, E>>) -> Result<Self, E> {
        let mut buffered_samples: Vec<T> = samples.into_iter().collect::<Result<Vec<_>, E>>()?;
        let mut rng = JavaRandom::new(LEIPZIG_RANDOM_SEED);
        rng.shuffle(&mut buffered_samples);
        Ok(Self { buffered_samples, index: 0 })
    }

    /// Resets the stream cursor to the beginning of the shuffled buffer.
    #[inline]
    pub fn reset(&mut self) {
        self.index = 0;
    }

    /// Returns the buffered shuffled slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.buffered_samples
    }
}

impl<T: Clone> Iterator for SampleShuffleStream<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let item = self.buffered_samples.get(self.index)?.clone();
        self.index += 1;
        Some(item)
    }
}

/// Stream adapter that skips the first `samples_to_skip` items of an iterator.
///
/// Mirrors `opennlp.tools.formats.leipzig.SampleSkipStream`.
#[derive(Debug, Clone)]
pub struct SampleSkipStream<I> {
    samples: I,
}

impl<I: Iterator> SampleSkipStream<I> {
    /// Creates a new stream after advancing `samples` by `samples_to_skip` items.
    pub fn new(mut samples: I, samples_to_skip: usize) -> Self {
        for _ in 0..samples_to_skip {
            if samples.next().is_none() {
                break;
            }
        }
        Self { samples }
    }
}

impl<I: Iterator> Iterator for SampleSkipStream<I> {
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.samples.next()
    }
}

/// Reader for Leipzig Corpora Collection sentence files in a directory.
///
/// Mirrors `opennlp.tools.formats.leipzig.LeipzigLanguageSampleStream`.
#[derive(Debug, Clone)]
pub struct LeipzigLanguageReader {
    sentences_per_sample: usize,
    lang_sample_counts: HashMap<String, usize>,
    sentences_files: Vec<PathBuf>,
    file_index: usize,
    current_lang: String,
    current_sentences: Vec<String>,
    sentence_index: usize,
    random: JavaRandom,
}

impl LeipzigLanguageReader {
    /// Opens a Leipzig corpus directory.
    ///
    /// Scans `leipzig_folder` for non-hidden regular files whose file name starts with
    /// 3 ASCII lowercase letters (`'a'..='z'`), sorts them by path, computes the per-file
    /// sample quota `samples_per_language / files_for_lang`, and seeds `JavaRandom::new(23)`.
    pub fn new(
        leipzig_folder: impl AsRef<Path>,
        sentences_per_sample: usize,
        samples_per_language: usize,
    ) -> Result<Self, FormatError> {
        let folder = leipzig_folder.as_ref();
        let read_dir = fs::read_dir(folder)
            .map_err(|err| FormatError::Io { path: folder.display().to_string(), message: err.to_string() })?;

        let mut sentences_files = Vec::new();
        for entry in read_dir {
            let entry = entry
                .map_err(|err| FormatError::Io { path: folder.display().to_string(), message: err.to_string() })?;
            let path = entry.path();
            let Some(file_name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if file_name.starts_with('.') {
                continue;
            }
            let Ok(meta) = entry.metadata() else {
                continue;
            };
            if meta.is_file() && has_language_code_prefix(file_name) {
                sentences_files.push(path);
            }
        }

        sentences_files.sort();

        let mut lang_counts: HashMap<String, usize> = HashMap::new();
        for file in &sentences_files {
            if let Some(name) = file.file_name().and_then(|n| n.to_str()) {
                let lang = name[..LANG_CODE_LENGTH].to_owned();
                *lang_counts.entry(lang).or_insert(0) += 1;
            }
        }

        let lang_sample_counts: HashMap<String, usize> =
            lang_counts.into_iter().map(|(lang, count)| (lang, samples_per_language / count)).collect();

        Ok(Self {
            sentences_per_sample,
            lang_sample_counts,
            sentences_files,
            file_index: 0,
            current_lang: String::new(),
            current_sentences: Vec::new(),
            sentence_index: 0,
            random: JavaRandom::new(LEIPZIG_RANDOM_SEED),
        })
    }

    /// Resets the file iterator to the start of the directory (keeping the RNG state
    /// just like `LeipzigLanguageSampleStream.reset()`).
    pub fn reset(&mut self) {
        self.file_index = 0;
        self.current_lang.clear();
        self.current_sentences.clear();
        self.sentence_index = 0;
    }

    fn load_file(&mut self, path: &Path) -> Result<(), FormatError> {
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        let lang = file_name[..LANG_CODE_LENGTH].to_owned();
        let number_of_samples = self.lang_sample_counts.get(&lang).copied().unwrap_or(0);
        let text = fs::read_to_string(path)
            .map_err(|err| FormatError::Io { path: path.display().to_string(), message: err.to_string() })?;

        let all_lines: Vec<&str> = Lines::new(&text)?.map(|l| l.text()).collect();
        let total_line_count = all_lines.len();
        let required_lines = self.sentences_per_sample * number_of_samples;

        if total_line_count < required_lines {
            return Err(FormatError::LeipzigNotEnoughLines {
                path: path.display().to_string(),
                total_lines: total_line_count,
                required_lines,
            });
        }

        let mut indexes: Vec<usize> = (0..total_line_count).collect();
        self.random.shuffle(&mut indexes);
        let selected_lines: HashSet<usize> = indexes[..required_lines].iter().copied().collect();

        let mut sentences = Vec::with_capacity(required_lines);
        for (line_index, line) in all_lines.into_iter().enumerate() {
            if line.contains('\t') && selected_lines.contains(&line_index) {
                sentences.push(line.to_owned());
            }
        }

        self.random.shuffle(&mut sentences);
        self.current_lang = lang;
        self.current_sentences = sentences;
        self.sentence_index = 0;
        Ok(())
    }

    fn read_from_current_file(&mut self) -> Option<LanguageSample<'static>> {
        if self.sentences_per_sample == 0 || self.sentence_index >= self.current_sentences.len() {
            return None;
        }
        let mut sample_string = String::new();
        let mut count = 0;
        while count < self.sentences_per_sample && self.sentence_index < self.current_sentences.len() {
            let line = &self.current_sentences[self.sentence_index];
            self.sentence_index += 1;
            if let Some(tab_idx) = line.find('\t') {
                sample_string.push_str(&line[tab_idx + 1..]);
                sample_string.push(' ');
            }
            count += 1;
        }
        if sample_string.is_empty() {
            None
        } else {
            let len = sample_string.len() as u32;
            Some(LanguageSample {
                language: Field::made(self.current_lang.clone(), Span::new(0, 0)),
                context: Field::made(sample_string, Span::new(0, len)),
                source: Span::new(0, len),
            })
        }
    }

    /// Reads the next [`LanguageSample`], or `None` when all files are exhausted.
    pub fn next_sample(&mut self) -> Option<Result<LanguageSample<'static>, FormatError>> {
        loop {
            if let Some(sample) = self.read_from_current_file() {
                return Some(Ok(sample));
            }
            if self.file_index >= self.sentences_files.len() {
                return None;
            }
            let path = self.sentences_files[self.file_index].clone();
            self.file_index += 1;
            if let Err(err) = self.load_file(&path) {
                return Some(Err(err));
            }
        }
    }
}

impl Iterator for LeipzigLanguageReader {
    type Item = Result<LanguageSample<'static>, FormatError>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.next_sample()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn make_temp_dir(label: &str) -> PathBuf {
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("turbo_formats_leipzig_{label}_{}_{id}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn populate_opennlp_sample_dir(dir: &Path) {
        fs::write(dir.join("123-skipped.txt"), "skip this file\n").unwrap();
        fs::write(
            dir.join("dan-sentences.txt"),
            "1\tDer var engang en mand.\n2\tDer boede i en spand.\n3\tSpanden var af ler.\n",
        )
        .unwrap();
        fs::create_dir_all(dir.join("dontread")).unwrap();
        fs::write(dir.join("eng-sentences.txt"), "1\tThis is a sentence.\n2\tThis is another sentences.\n").unwrap();
    }

    #[test]
    fn read_sentence_files_matches_opennlp() {
        let dir = make_temp_dir("basic");
        populate_opennlp_sample_dir(&dir);

        let reader = LeipzigLanguageReader::new(&dir, 1, 2).unwrap();
        let samples: Vec<LanguageSample<'static>> = reader.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(samples.len(), 4);
        assert_eq!(samples[0].language.as_str(), "dan");
        assert_eq!(samples[1].language.as_str(), "dan");
        assert_eq!(samples[2].language.as_str(), "eng");
        assert_eq!(samples[3].language.as_str(), "eng");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn not_enough_sentences_returns_error() {
        let dir = make_temp_dir("short");
        populate_opennlp_sample_dir(&dir);

        let reader = LeipzigLanguageReader::new(&dir, 2, 2).unwrap();
        let res: Result<Vec<_>, _> = reader.collect();
        assert!(matches!(res, Err(FormatError::LeipzigNotEnoughLines { .. })));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn only_lowercase_ascii_language_codes_are_read() {
        let dir = make_temp_dir("codes");
        let names = [
            "eng_news_2010_10K-sentences.txt",
            "deu_wikipedia_2016_10K-sentences.txt",
            "Eng_news_2010_10K-sentences.txt",
            "ENG_news_2010_10K-sentences.txt",
            "enG_news_2010_10K-sentences.txt",
            "en1_news_2010_10K-sentences.txt",
            "e-g_news_2010_10K-sentences.txt",
            "en_news_2010_10K-sentences.txt",
            "\u{00E9}ng_news_2010_10K-sentences.txt",
            "\u{0130}ng_news_2010_10K-sentences.txt",
            "\u{FF45}ng_news_2010_10K-sentences.txt",
            "\u{1D41A}bc_news_2010_10K-sentences.txt",
            "en",
        ];
        for name in names {
            fs::write(dir.join(name), "1\tThis is a sentence.\n2\tThis is another sentence.\n").unwrap();
        }
        let samples: Vec<LanguageSample<'static>> =
            LeipzigLanguageReader::new(&dir, 1, 2).unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        let langs: Vec<&str> = samples.iter().map(|s| s.language.as_str()).collect();
        assert_eq!(langs, vec!["deu", "deu", "eng", "eng"]);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn line_terminators_and_shuffle_skip_adapters() {
        for (idx, terminator) in ["\n", "\r\n", "\r"].iter().enumerate() {
            let dir = make_temp_dir(&format!("term_{idx}"));
            fs::write(
                dir.join("eng-sentences.txt"),
                format!("1\tFirst sentence.{terminator}2\tSecond sentence.{terminator}"),
            )
            .unwrap();
            let samples: Vec<LanguageSample<'static>> =
                LeipzigLanguageReader::new(&dir, 1, 2).unwrap().collect::<Result<Vec<_>, _>>().unwrap();
            let mut texts: Vec<&str> = samples.iter().map(|s| s.context.as_str()).collect();
            texts.sort_unstable();
            assert_eq!(texts, vec!["First sentence. ", "Second sentence. "]);
            let _ = fs::remove_dir_all(&dir);
        }

        let shuffled = SampleShuffleStream::new(vec![1, 2, 3, 4, 5]);
        let skipped: Vec<i32> = SampleSkipStream::new(shuffled, 2).collect();
        assert_eq!(skipped.len(), 3);
    }
}
