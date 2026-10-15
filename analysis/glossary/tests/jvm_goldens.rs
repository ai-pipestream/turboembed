//! Compares the matchers and the readers with what a JVM running OpenNLP
//! returned: the goldens of `GlossaryGolden.java` in `analysis/goldens-tool`,
//! under `analysis/goldens/data/glossary`, described in its `README.md`.
//!
//! The matcher runs read the `text-` inputs with the fixture glossary
//! `glossary-terms.tsv`; the reader runs read the `csv-`, `tsv-` and `tbx-`
//! inputs.

use std::borrow::Cow;
use std::num::NonZeroU32;

use turbo_analysis::{Span, Spans, ViewId};
use turbo_analysis_layers::{Strs, Tokens};
use turbo_glossary::{
    Boundaries, Candidate, Case, DelimitedReader, EntryIx, FirstRecord, Glossary, GlossaryFile, GlossaryMatch,
    GlossaryMatcher, MatchInput, ReadError, TbxReader, TextMatcher, TokenMatcher, TokenValue, select,
};
use turbo_goldens::{ColumnSpec, ColumnType, Golden, GoldenAnchor, GoldenLayer, Shape};
use turbo_normalize::{CaseFold, Whitespace, apply, normalize};
use turbo_stem::{PorterStemmer, Stemmer};
use turbo_tokenize::{word_boundaries, word_tokens};

const SET: &str = "glossary";

const fn col(name: &'static str, ty: ColumnType) -> ColumnSpec {
    ColumnSpec::new(name, ty)
}

const STR: ColumnType = ColumnType::Str;
const U32: ColumnType = ColumnType::U32;

const TERMS: &[ColumnSpec] = &[col("start", U32), col("end", U32), col("entry", U32)];
const ANALYZED: &[ColumnSpec] = &[col("start", U32), col("end", U32), col("normalized", STR)];
const ENTRY_TOKENS: &[ColumnSpec] = &[col("entry", U32), col("token", STR)];
const ENTRIES: &[ColumnSpec] = &[col("id", STR), col("term", STR)];

/// The `text-` inputs.
const TEXTS: [&str; 5] = ["text-en", "text-multi", "text-space", "text-empty", "text-edge"];

fn golden(run: &str, input: &str) -> Golden {
    let shapes = [
        Shape { id: "turbo:glossary-terms".into(), anchor: GoldenAnchor::Text, columns: TERMS },
        Shape { id: "turbo:analyzed-terms".into(), anchor: GoldenAnchor::Text, columns: ANALYZED },
        Shape { id: "turbo:analyzed-entries".into(), anchor: GoldenAnchor::Document, columns: ENTRY_TOKENS },
        Shape { id: "turbo:glossary-entries".into(), anchor: GoldenAnchor::Document, columns: ENTRIES },
    ];
    let path = turbo_goldens::data_dir().join(SET).join(run).join(SET).join(format!("{input}.golden"));
    Golden::read_with(&path, &shapes).unwrap_or_else(|e| panic!("{e}"))
}

fn input(id: &str) -> Vec<u8> {
    let path = turbo_goldens::data_dir().join("inputs").join(SET).join(format!("{id}.txt"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn text(id: &str) -> String {
    String::from_utf8(input(id)).unwrap()
}

/// Every input of the set whose id starts with `prefix`.
fn inputs(prefix: &str) -> Vec<String> {
    let dir = turbo_goldens::data_dir().join("inputs").join(SET);
    let mut ids: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter_map(|name| name.strip_suffix(".txt").map(str::to_string))
        .filter(|id| id.starts_with(prefix))
        .collect();
    ids.sort();
    assert!(!ids.is_empty(), "no {prefix} inputs");
    ids
}

fn layer<'g>(g: &'g Golden, id: &str) -> &'g GoldenLayer {
    g.layer(id).unwrap_or_else(|| panic!("{}: no layer {id}", g.header.input.id))
}

fn u32s<'l>(l: &'l GoldenLayer, name: &str) -> &'l [u32] {
    l.u32s(name).unwrap_or_else(|| panic!("{}: no u32 column {name}", l.id))
}

fn strs(l: &GoldenLayer, name: &str) -> Vec<String> {
    l.strs(name).unwrap_or_else(|| panic!("{}: no str column {name}", l.id)).iter().map(str::to_string).collect()
}

/// The fixture glossary, read as the Java reads it.
fn fixture() -> Glossary {
    let path = turbo_goldens::data_dir().join("fixtures").join("glossary-terms.tsv");
    let bytes = std::fs::read(&path).unwrap();
    DelimitedReader::tsv(FirstRecord::Header).read(&bytes).unwrap().glossary
}

/// The golden's matches as (start, end, entry).
fn jvm_matches(g: &Golden) -> Vec<(u32, u32, u32)> {
    let l = layer(g, "turbo:glossary-terms");
    let (s, e, x) = (u32s(l, "start"), u32s(l, "end"), u32s(l, "entry"));
    (0..l.len()).map(|i| (s[i], e[i], x[i])).collect()
}

fn ours(found: &[GlossaryMatch]) -> Vec<(u32, u32, u32)> {
    found.iter().map(|m| (m.span.start, m.span.end, m.entry.0)).collect()
}

/// The UAX #29 word boundaries of `text`, where OpenNLP's character
/// matcher lets a match start and end.
fn uax29(text: &str) -> Boundaries {
    let mut b = Vec::new();
    word_boundaries(text, &mut b);
    Boundaries::from_offsets(text.len() as u32, b)
}

fn show(text: &str, found: &[(u32, u32, u32)]) -> Vec<String> {
    found.iter().map(|&(s, e, x)| format!("{s}..{e} {:?} #{x}", &text[s as usize..e as usize])).collect()
}

fn assert_matches(run: &str, id: &str, text: &str, got: &[GlossaryMatch]) {
    let want = jvm_matches(&golden(run, id));
    assert_eq!(show(text, &ours(got)), show(text, &want), "{run} on {id}");
}

#[test]
fn character_matcher_matches_as_the_jvm_does() {
    let g = fixture();
    for (run, case) in [("exact", Case::Sensitive), ("ignore-case", Case::Insensitive)] {
        let m = TextMatcher::new(&g, case).unwrap();
        for id in TEXTS {
            let text = text(id);
            let mut found = Vec::new();
            m.find(&text, &uax29(&text), &mut found);
            assert_matches(run, id, &text, &found);
        }
    }
}

/// OpenNLP folds the text inside the matcher and checks the boundaries
/// in the original; here the folded text is scanned, each hit mapped back
/// through the alignment, and the source boundaries checked.
#[test]
fn character_matcher_through_a_whitespace_normalizer_matches_as_the_jvm_does() {
    let g = fixture();
    let ws = Whitespace::collapse();
    let m = TextMatcher::with_normalizer(&g, Case::Insensitive, |t| apply(&ws, t)).unwrap();
    for id in TEXTS {
        let text = text(id);
        let (folded, alignment) = normalize(&ws, &text);
        let boundaries = uax29(&text);
        let mut candidates = Vec::new();
        m.scan(&folded, |span, entry| {
            let span = alignment.span_to_parent(span);
            if boundaries.around(span) {
                candidates.push(Candidate { span, tokens: 0, entry });
            }
        });
        let mut found = Vec::new();
        select(&mut candidates, &mut found);
        assert_matches("whitespace+ignore-case", id, &text, &found);
    }
}

/// OpenNLP's `TermAnalyzer` with case folding and the Porter stemmer: the
/// UAX #29 word tokens of `text`, each with its lowercased stem.
fn analyze(text: &str) -> Vec<(Span, String)> {
    let (mut spans, mut types) = (Vec::new(), Vec::new());
    word_tokens(text, NonZeroU32::new(255).unwrap(), &mut spans, &mut types);
    let fold = CaseFold::lowercase();
    spans
        .into_iter()
        .map(|s| {
            let lower: Cow<'_, str> = apply(&fold, &text[s.range()]);
            (s, PorterStemmer.stem(&lower))
        })
        .collect()
}

/// The entries the term analyzing matcher takes: those whose terms give
/// at least one token and no blank one.
fn analyzable(g: &Glossary) -> Vec<EntryIx> {
    g.iter()
        .filter(|(_, e)| {
            let terms = analyze(e.term);
            !terms.is_empty() && terms.iter().all(|(_, v)| !v.trim().is_empty())
        })
        .map(|(ix, _)| ix)
        .collect()
}

fn token_matcher(g: &Glossary) -> TokenMatcher {
    TokenMatcher::over(g, &analyzable(g), TokenValue::Values, |term, out| {
        out.extend(analyze(term).into_iter().map(|(_, v)| v))
    })
    .unwrap()
}

/// The text's tokens and their values, as the step would read them from
/// the token and stem layers.
fn tokens_and_values(text: &str) -> (Tokens, Strs) {
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut values = Strs::default();
    for (s, v) in analyze(text) {
        spans.push(s);
        values.push(&v);
    }
    (Tokens { spans, sentence: None }, values)
}

#[test]
fn analysis_is_the_jvms() {
    let g = fixture();
    for run in ["analyzed", "composite"] {
        let want = golden(run, "text-en");
        let l = layer(&want, "turbo:analyzed-entries");
        let jvm: Vec<(u32, String)> = u32s(l, "entry").iter().copied().zip(strs(l, "token")).collect();
        let mut got = Vec::new();
        for ix in analyzable(&g) {
            got.extend(analyze(g.term(ix)).into_iter().map(|(_, v)| (ix.0, v)));
        }
        assert_eq!(got, jvm, "{run}: the analyzed entries");
        for id in TEXTS {
            let text = text(id);
            let gold = golden(run, id);
            let l = layer(&gold, "turbo:analyzed-terms");
            let (s, e) = (u32s(l, "start"), u32s(l, "end"));
            let jvm: Vec<(u32, u32, String)> =
                strs(l, "normalized").into_iter().enumerate().map(|(i, v)| (s[i], e[i], v)).collect();
            let got: Vec<(u32, u32, String)> = analyze(&text).into_iter().map(|(s, v)| (s.start, s.end, v)).collect();
            assert_eq!(got, jvm, "{run} on {id}: the analyzed text");
        }
    }
}

#[test]
fn token_matcher_matches_as_the_jvm_does() {
    let g = fixture();
    let m = token_matcher(&g);
    for id in TEXTS {
        let text = text(id);
        let (tokens, values) = tokens_and_values(&text);
        let mut found = Vec::new();
        m.find(&text, &tokens, Some(&values), &mut found).unwrap();
        assert_matches("analyzed", id, &text, &found);
    }
}

#[test]
fn composite_matches_as_the_jvm_does() {
    let g = fixture();
    let m = GlossaryMatcher::composite(vec![
        GlossaryMatcher::Text(TextMatcher::new(&g, Case::Sensitive).unwrap()),
        GlossaryMatcher::Tokens(token_matcher(&g)),
    ])
    .unwrap();
    for id in TEXTS {
        let text = text(id);
        let (tokens, values) = tokens_and_values(&text);
        let boundaries = uax29(&text);
        let mut found = Vec::new();
        m.find(
            &MatchInput { text: &text, tokens: &tokens, boundaries: &boundaries, values: Some(&values) },
            &mut found,
        )
        .unwrap();
        assert_matches("composite", id, &text, &found);
    }
}

/// The line a JVM error message names ("... line 3 ...", or the XML
/// parser's "[row,col]:[3,14]"), if any.
fn jvm_line(message: &str) -> Option<u32> {
    let at = match message.find("[row,col]:[") {
        Some(at) => at + "[row,col]:[".len(),
        None => message.find("line ")? + "line ".len(),
    };
    let digits: String = message[at..].chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

fn assert_read(run: &str, id: &str, bytes: &[u8], got: Result<GlossaryFile, ReadError>) {
    let want = golden(run, id);
    match (&want.header.error, got) {
        (Some(e), Err(ours)) => {
            if let Some(line) = jvm_line(&e.message) {
                assert_eq!(ours.line(), Some(line), "{run} on {id}: {ours} against {}: {}", e.class, e.message);
            }
        }
        (Some(e), Ok(f)) => {
            panic!("{run} on {id}: the JVM refused it ({}: {}), read {:?}", e.class, e.message, f.glossary)
        }
        (None, Err(ours)) => panic!("{run} on {id}: the JVM read it, refused here: {ours}"),
        (None, Ok(f)) => {
            let l = layer(&want, "turbo:glossary-entries");
            let jvm: Vec<(String, String)> = strs(l, "id").into_iter().zip(strs(l, "term")).collect();
            let got: Vec<(String, String)> =
                f.glossary.iter().map(|(_, e)| (e.id.to_string(), e.term.to_string())).collect();
            assert_eq!(got, jvm, "{run} on {id}");
            assert_eq!(f.term_at.len(), f.glossary.len());
            for (span, (_, e)) in f.term_at.iter().zip(f.glossary.iter()) {
                let raw = std::str::from_utf8(&bytes[span.range()]).unwrap();
                // A term with no quote doubled, markup or reference in it
                // is its span's text, trimmed for TBX.
                if !raw.contains(['"', '<', '&']) {
                    let raw = if run.starts_with("tbx") { raw.trim() } else { raw };
                    assert_eq!(raw, e.term, "{run} on {id}: the span of {e:?}");
                }
            }
        }
    }
}

#[test]
fn delimited_reader_reads_as_the_jvm_does() {
    for (run, prefix, reader) in
        [("csv", "csv-", DelimitedReader::csv()), ("tsv-header", "tsv-", DelimitedReader::tsv(FirstRecord::Header))]
    {
        for id in inputs(prefix) {
            let bytes = input(&id);
            assert_read(run, &id, &bytes, reader.read(&bytes));
        }
    }
}

#[test]
fn tbx_reader_reads_as_the_jvm_does() {
    for (run, tag) in [("tbx-en", "en"), ("tbx-de", "de")] {
        let reader = TbxReader::new(tag).unwrap();
        for id in inputs("tbx-") {
            let bytes = input(&id);
            assert_read(run, &id, &bytes, reader.read(&bytes));
        }
    }
}
