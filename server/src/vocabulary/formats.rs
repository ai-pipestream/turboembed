//! The dictionary formats the vocabulary store knows, and the reading of
//! an upload with the analysis crate that owns its format. A format is
//! accepted only when its whole file parses; what the reader found is the
//! artifact's entry count, and its headwords seed a learned vocabulary.

use turbo_glossary::{Case, DelimitedReader, Delimiter, FirstRecord, GlossaryFile, MappedGlossaryMatcher};
use turbo_glossary::{ImageBuf, MappedGlossaryWriter, TbxReader};

/// One known format: its id, what it is, whether one upload can carry it,
/// and whether reading it needs a language.
pub(crate) struct Format {
    pub(crate) id: &'static str,
    pub(crate) description: &'static str,
    pub(crate) importable: bool,
    pub(crate) needs_language: bool,
}

const fn format(id: &'static str, description: &'static str, importable: bool, needs_language: bool) -> Format {
    Format { id, description, importable, needs_language }
}

pub(crate) const FORMATS: [Format; 11] = [
    format("glossary-csv", "comma separated id and term records (turbo-glossary DelimitedReader)", true, false),
    format("glossary-tbx", "a TBX termbase, the header's language selected (turbo-glossary TbxReader)", true, true),
    format("glossary-mapped", "a compiled glossary automaton image (turbo-glossary MappedGlossaryWriter)", true, false),
    format("plain-word-list", "UTF-8 text, one word or phrase a line", true, false),
    format("headword-definition-tsv", "UTF-8 text, a headword, a tab and its definition a line", true, false),
    format("opennlp-dictionary-xml", "OpenNLP dictionary XML (turbo-maxent Dictionary)", true, false),
    format("pos-tag-dictionary-xml", "OpenNLP POS tag dictionary XML (turbo-sequence TagDictionary)", true, false),
    format("lemmatizer-dictionary", "word, tag and lemma tab separated lines (turbo-lemmatizer)", true, false),
    format("morfologik-fsa", "a Morfologik FSA5 or CFSA2 dictionary and its .info file", false, false),
    format("hunspell-aff-dic", "a Hunspell .aff and .dic pair", false, false),
    format("wordnet-lmf-xml", "a Global WordNet WN-LMF XML lexicon (turbo-wordnet WnLmfReader)", true, false),
];

pub(crate) fn find(id: &str) -> Option<&'static Format> {
    FORMATS.iter().find(|f| f.id == id)
}

/// What reading an upload found.
pub(crate) struct Read {
    pub(crate) entry_count: u64,
    /// The compiled automaton of a glossary read from text, stored beside
    /// the upload so a pipeline can map it without compiling.
    pub(crate) image: Option<Vec<u8>>,
}

fn utf8(bytes: &[u8]) -> Result<&str, String> {
    let bytes = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes);
    std::str::from_utf8(bytes).map_err(|e| format!("not UTF-8 at byte {}", e.valid_up_to()))
}

/// The lines of `text` without their line ends, blank ones skipped, each
/// with its 1-based line number.
fn lines(text: &str) -> impl Iterator<Item = (usize, &str)> {
    text.split('\n')
        .enumerate()
        .map(|(n, l)| (n + 1, l.strip_suffix('\r').unwrap_or(l)))
        .filter(|(_, l)| !l.trim().is_empty())
}

fn glossary(file: GlossaryFile) -> Result<Read, String> {
    let image = MappedGlossaryWriter::write(&file.glossary, Case::Sensitive).map_err(|e| e.to_string())?;
    Ok(Read { entry_count: file.glossary.len() as u64, image: Some(image) })
}

fn headword_tsv(text: &str) -> Result<u64, String> {
    let mut n = 0;
    for (line, l) in lines(text) {
        match l.split_once('\t') {
            Some((head, _)) if !head.trim().is_empty() => n += 1,
            Some(_) => return Err(format!("line {line}: a blank headword")),
            None => return Err(format!("line {line}: no tab between the headword and its definition")),
        }
    }
    Ok(n)
}

/// Reads `bytes` as `format`: the entries found, or why the file is
/// refused. An empty dictionary is refused.
pub(crate) fn read(format: &Format, bytes: &[u8], language: &str, has_header_row: bool) -> Result<Read, String> {
    let first = if has_header_row { FirstRecord::Header } else { FirstRecord::Data };
    let plain = |entry_count: usize| Read { entry_count: entry_count as u64, image: None };
    let read = match format.id {
        "glossary-csv" => {
            glossary(DelimitedReader::new(Delimiter::COMMA, first).read(bytes).map_err(|e| e.to_string())?)?
        }
        "glossary-tbx" => {
            let reader = TbxReader::new(language).ok_or("glossary-tbx needs a language")?;
            glossary(reader.read(bytes).map_err(|e| e.to_string())?)?
        }
        "glossary-mapped" => {
            let image = ImageBuf::copy_from(bytes);
            plain(MappedGlossaryMatcher::from_bytes(image.bytes()).map_err(|e| e.to_string())?.entries())
        }
        "plain-word-list" => plain(lines(utf8(bytes)?).count()),
        "headword-definition-tsv" => Read { entry_count: headword_tsv(utf8(bytes)?)?, image: None },
        "opennlp-dictionary-xml" => plain(turbo_maxent::Dictionary::read(bytes).map_err(|e| e.to_string())?.len()),
        "pos-tag-dictionary-xml" => {
            plain(turbo_sequence::postag::TagDictionary::read(bytes).map_err(|e| e.to_string())?.len())
        }
        "lemmatizer-dictionary" => {
            utf8(bytes)?;
            plain(turbo_lemmatizer::DictionaryLemmatizer::parse(bytes, "upload").map_err(|e| e.to_string())?.len())
        }
        "wordnet-lmf-xml" => plain(wordnet(bytes)?.sense_order().len()),
        id => return Err(format!("{id} is not importable as one upload")),
    };
    if read.entry_count == 0 {
        return Err("the dictionary has no entries".into());
    }
    Ok(read)
}

fn wordnet(bytes: &[u8]) -> Result<turbo_wordnet::InMemoryWordNetLexicon, String> {
    turbo_wordnet::WnLmfReader::read_named(bytes, "upload.xml").map_err(|e| e.to_string())
}

/// The headwords of a stored dictionary, in file order where the format
/// has one and byte order otherwise, for seeding a vocabulary. A phrase
/// keeps its words separated by single spaces.
pub(crate) fn headwords(
    format: &Format,
    bytes: &[u8],
    language: &str,
    has_header_row: bool,
) -> Result<Vec<String>, String> {
    let first = if has_header_row { FirstRecord::Header } else { FirstRecord::Data };
    let terms = |file: GlossaryFile| file.glossary.iter().map(|(_, e)| e.term.to_owned()).collect();
    let mut out: Vec<String> = match format.id {
        "glossary-csv" => terms(DelimitedReader::new(Delimiter::COMMA, first).read(bytes).map_err(|e| e.to_string())?),
        "glossary-tbx" => {
            let reader = TbxReader::new(language).ok_or("glossary-tbx needs a language")?;
            terms(reader.read(bytes).map_err(|e| e.to_string())?)
        }
        "glossary-mapped" => {
            let image = ImageBuf::copy_from(bytes);
            let m = MappedGlossaryMatcher::from_bytes(image.bytes()).map_err(|e| e.to_string())?;
            (0..m.entries()).filter_map(|i| m.term(turbo_glossary::EntryIx(i as u32)).map(str::to_owned)).collect()
        }
        "plain-word-list" => lines(utf8(bytes)?).map(|(_, l)| l.trim().to_owned()).collect(),
        "headword-definition-tsv" | "lemmatizer-dictionary" => {
            lines(utf8(bytes)?).filter_map(|(_, l)| l.split('\t').next()).map(|h| h.trim().to_owned()).collect()
        }
        "opennlp-dictionary-xml" => {
            let d = turbo_maxent::Dictionary::read(bytes).map_err(|e| e.to_string())?;
            d.entries().map(|tokens| tokens.collect::<Vec<_>>().join(" ")).collect()
        }
        "pos-tag-dictionary-xml" => {
            let d = turbo_sequence::postag::TagDictionary::read(bytes).map_err(|e| e.to_string())?;
            d.iter().map(|(w, _)| w.to_owned()).collect()
        }
        "wordnet-lmf-xml" => wordnet(bytes)?.sense_order().keys().map(|k| k.lemma().replace('_', " ")).collect(),
        id => return Err(format!("{id} has no headwords to seed with")),
    };
    for h in &mut out {
        *h = h.split_whitespace().collect::<Vec<_>>().join(" ");
    }
    out.retain(|h| !h.is_empty());
    Ok(out)
}
