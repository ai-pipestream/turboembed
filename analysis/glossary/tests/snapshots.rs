//! Reviewed glossary snapshots: matcher hits and reader entries/errors.
//!
//! Regenerate with `TURBO_BLESS=1 cargo test -p turbo-glossary --test snapshots`.
//! JVM goldens are not an oracle; snapshots record this crate's output.
//! Property checks in `properties.rs` and the hand cases below catch a
//! wrong current answer that a pure freeze would miss.

use std::borrow::Cow;
use std::fs;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use turbo_analysis::{Span, Spans, ViewId};
use turbo_analysis_layers::{Strs, Tokens};
use turbo_glossary::{
    Boundaries, Candidate, Case, DelimitedReader, EntryIx, FirstRecord, Glossary, GlossaryFile, GlossaryMatch,
    GlossaryMatcher, MatchInput, ReadError, TbxReader, TextMatcher, TokenMatcher, TokenValue, intersects, select,
};
use turbo_normalize::{CaseFold, Whitespace, apply, normalize};
use turbo_stem::{PorterStemmer, Stemmer};
use turbo_tokenize::{word_boundaries, word_tokens};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests")
}

fn inputs() -> PathBuf {
    root().join("inputs")
}

fn snapshots() -> PathBuf {
    root().join("snapshots")
}

fn fixture() -> Glossary {
    let path = root().join("fixtures/glossary-terms.tsv");
    let bytes = fs::read(&path).unwrap();
    DelimitedReader::tsv(FirstRecord::Header).read(&bytes).unwrap().glossary
}

fn text_of(id: &str) -> String {
    fs::read_to_string(inputs().join(format!("{id}.txt"))).unwrap()
}

fn bytes_of(id: &str) -> Vec<u8> {
    fs::read(inputs().join(format!("{id}.txt"))).unwrap()
}

fn uax29(text: &str) -> Boundaries {
    let mut b = Vec::new();
    word_boundaries(text, &mut b);
    Boundaries::from_offsets(text.len() as u32, b)
}

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

fn tokens_and_values(text: &str) -> (Tokens, Strs) {
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut values = Strs::default();
    for (s, v) in analyze(text) {
        spans.push(s);
        values.push(&v);
    }
    (Tokens { spans, sentence: None }, values)
}

fn match_lines(found: &[GlossaryMatch]) -> String {
    let mut body = String::new();
    for m in found {
        body.push_str(&format!("{}\t{}\t{}\n", m.span.start, m.span.end, m.entry.0));
    }
    body
}

fn entry_lines(file: &GlossaryFile) -> String {
    let mut body = String::new();
    for (_, e) in file.glossary.iter() {
        body.push_str(&format!("{}\t{}\n", escape(e.id), escape(e.term)));
    }
    body
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\t', "\\t").replace('\n', "\\n").replace('\r', "\\r")
}

fn write_snap(run: &str, id: &str, kind: &str, body: &str) {
    let path = snapshots().join(run).join(format!("{id}.{kind}"));
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, body).unwrap();
}

fn read_snap(run: &str, id: &str, kind: &str) -> String {
    fs::read_to_string(snapshots().join(run).join(format!("{id}.{kind}")))
        .unwrap_or_else(|e| panic!("missing snapshot {run}/{id}.{kind}: {e} (TURBO_BLESS=1 to write)"))
}

fn check_or_bless(bless: bool, run: &str, id: &str, kind: &str, got: &str) {
    if bless {
        write_snap(run, id, kind, got);
        return;
    }
    let want = read_snap(run, id, kind);
    assert_eq!(got, want, "{run}/{id}.{kind}");
}

fn reader_snap(bless: bool, run: &str, id: &str, got: Result<GlossaryFile, ReadError>) {
    match got {
        Ok(f) => {
            check_or_bless(bless, run, id, "entries.tsv", &entry_lines(&f));
            assert!(!Path::new(&snapshots().join(run).join(format!("{id}.error.txt"))).exists() || bless);
            if bless {
                let err_path = snapshots().join(run).join(format!("{id}.error.txt"));
                let _ = fs::remove_file(err_path);
            }
        }
        Err(e) => {
            let body = format!("{e:?}\n");
            check_or_bless(bless, run, id, "error.txt", &body);
            if bless {
                let _ = fs::remove_file(snapshots().join(run).join(format!("{id}.entries.tsv")));
            }
        }
    }
}

#[test]
fn matchers_match_the_snapshots() {
    let bless = std::env::var_os("TURBO_BLESS").is_some();
    let g = fixture();
    let texts = ["text-en", "text-multi", "text-space", "text-empty", "text-edge"];

    for (run, case) in [("exact", Case::Sensitive), ("ignore-case", Case::Insensitive)] {
        let m = TextMatcher::new(&g, case).unwrap();
        for id in texts {
            let text = text_of(id);
            let mut found = Vec::new();
            m.find(&text, &uax29(&text), &mut found);
            check_or_bless(bless, run, id, "matches.tsv", &match_lines(&found));
        }
    }

    let ws = Whitespace::collapse();
    let m = TextMatcher::with_normalizer(&g, Case::Insensitive, |t| apply(&ws, t)).unwrap();
    for id in texts {
        let text = text_of(id);
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
        check_or_bless(bless, "whitespace+ignore-case", id, "matches.tsv", &match_lines(&found));
    }

    let tm = token_matcher(&g);
    for id in texts {
        let text = text_of(id);
        let (tokens, values) = tokens_and_values(&text);
        let mut found = Vec::new();
        tm.find(&text, &tokens, Some(&values), &mut found).unwrap();
        check_or_bless(bless, "analyzed", id, "matches.tsv", &match_lines(&found));
        let terms: String =
            analyze(&text).into_iter().map(|(s, v)| format!("{}\t{}\t{}\n", s.start, s.end, escape(&v))).collect();
        check_or_bless(bless, "analyzed", id, "terms.tsv", &terms);
    }
    let mut entry_terms = String::new();
    for ix in analyzable(&g) {
        for (_, v) in analyze(g.term(ix)) {
            entry_terms.push_str(&format!("{}\t{}\n", ix.0, escape(&v)));
        }
    }
    check_or_bless(bless, "analyzed", "entries", "terms.tsv", &entry_terms);

    let m = GlossaryMatcher::composite(vec![
        GlossaryMatcher::Text(TextMatcher::new(&g, Case::Sensitive).unwrap()),
        GlossaryMatcher::Tokens(token_matcher(&g)),
    ])
    .unwrap();
    for id in texts {
        let text = text_of(id);
        let (tokens, values) = tokens_and_values(&text);
        let boundaries = uax29(&text);
        let mut found = Vec::new();
        m.find(
            &MatchInput { text: &text, tokens: &tokens, boundaries: &boundaries, values: Some(&values) },
            &mut found,
        )
        .unwrap();
        check_or_bless(bless, "composite", id, "matches.tsv", &match_lines(&found));
    }
}

#[test]
fn readers_match_the_snapshots() {
    let bless = std::env::var_os("TURBO_BLESS").is_some();
    for (run, prefix, reader) in
        [("csv", "csv-", DelimitedReader::csv()), ("tsv-header", "tsv-", DelimitedReader::tsv(FirstRecord::Header))]
    {
        for id in input_ids(prefix) {
            reader_snap(bless, run, &id, reader.read(&bytes_of(&id)));
        }
    }
    for (run, tag) in [("tbx-en", "en"), ("tbx-de", "de")] {
        let reader = TbxReader::new(tag).unwrap();
        for id in input_ids("tbx-") {
            reader_snap(bless, run, &id, reader.read(&bytes_of(&id)));
        }
    }
}

fn input_ids(prefix: &str) -> Vec<String> {
    let mut ids: Vec<String> = fs::read_dir(inputs())
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter_map(|n| n.strip_suffix(".txt").map(str::to_string))
        .filter(|id| id.starts_with(prefix))
        .collect();
    ids.sort();
    assert!(!ids.is_empty(), "no {prefix} inputs");
    ids
}

/// Hand-verified matches against the fixture glossary: independent of the
/// snapshot files, so a wrong bless would still fail here.
#[test]
fn hand_verified_exact_hits() {
    let g = fixture();
    let m = TextMatcher::new(&g, Case::Sensitive).unwrap();
    // Fixture terms: "hot dog" (c1), "New York City" (c3). Exact match must
    // hit those surfaces and the slice must equal the registered term.
    let text = "I ate a hot dog in New York City.";
    let mut found = Vec::new();
    m.find(text, &uax29(text), &mut found);
    let surfaces: Vec<&str> = found.iter().map(|h| &text[h.span.range()]).collect();
    assert!(surfaces.contains(&"hot dog"), "{surfaces:?}");
    assert!(surfaces.contains(&"New York City"), "{surfaces:?}");
    for hit in &found {
        assert!(hit.span.start < hit.span.end);
        assert!((hit.span.end as usize) <= text.len());
        assert!(g.try_get(hit.entry).is_some());
        assert_eq!(&text[hit.span.range()], g.term(hit.entry));
    }
}

#[test]
fn match_spans_are_boundaries_and_non_overlapping_after_select() {
    let g = fixture();
    let m = TextMatcher::new(&g, Case::Sensitive).unwrap();
    for id in ["text-en", "text-multi", "text-space", "text-edge"] {
        let text = text_of(id);
        let boundaries = uax29(&text);
        let mut found = Vec::new();
        m.find(&text, &boundaries, &mut found);
        for (i, a) in found.iter().enumerate() {
            assert!(boundaries.around(a.span), "{id} #{i}");
            assert!(text.is_char_boundary(a.span.start as usize));
            assert!(text.is_char_boundary(a.span.end as usize));
            for b in &found[i + 1..] {
                assert!(!intersects(a.span, b.span), "{id}: {:?} vs {:?}", a.span, b.span);
            }
        }
    }
}

#[test]
fn hand_verified_reader_basic_csv() {
    let bytes = bytes_of("csv-basic");
    let file = DelimitedReader::csv().read(&bytes).unwrap();
    assert!(!file.glossary.is_empty());
    for (span, (_, e)) in file.term_at.iter().zip(file.glossary.iter()) {
        let raw = std::str::from_utf8(&bytes[span.range()]).unwrap();
        if !raw.contains(['"', '<', '&']) {
            assert_eq!(raw, e.term);
        }
        assert!(!e.term.is_empty() || e.id.is_empty());
    }
}
