//! The OntoNotes readers against the JVM: every `ontonotes` golden of
//! `analysis/goldens` (written by `cargo run -p turbo-goldens-tool --
//! generate`, see analysis/goldens-tool/README.md and
//! `OntoNotesGolden.java`) is what one of OpenNLP's OntoNotes streams read
//! from a corpus directory bundled in one input of the `ontonotes` set.
//! Each test writes the directory out, reads it with the readers here and
//! rebuilds every layer: the samples as written, the errors and where they
//! came, the tokens, names and tags, the parses node by node, the lines,
//! and the files in the order they were read.
//!
//! The messages of the exceptions are compared, except those of the parse
//! errors, which are the JVM's own texts (as in turbo-parser's tests).
//! Beside the comparison, every token, tag and name is checked to point at
//! the bytes it was read from.

use std::path::Path;

use turbo_formats::ontonotes::{
    NameDocument, NameStream, OntoNotesError, OntoNotesFormat, ParseStream, PosStream, document_lines,
};
use turbo_goldens::Golden;
use turbo_parser::decode_token;
use turbo_unicode::WhitespaceMode;

mod bundle;
mod treebank;

use treebank::{Rebuilt, s};

fn error(out: &mut Rebuilt, after: usize, e: &OntoNotesError) {
    let message = (!matches!(e, OntoNotesError::Parse { .. })).then(|| e.to_string());
    out.error(after, e.java_class(), message);
}

fn rebuild(run: &str, root: &Path) -> Rebuilt {
    let mut out = Rebuilt::default();
    out.layer("turbo:read-errors");
    let format = OntoNotesFormat::new(root);
    let files = if run.starts_with("names") { format.name_files() } else { format.parse_files() }.unwrap();
    let texts: Vec<String> = files.iter().map(|f| std::fs::read_to_string(f).unwrap()).collect();
    match run {
        "names" | "names+legacy" => {
            let mode = if run == "names" { WhitespaceMode::Unicode } else { WhitespaceMode::Legacy };
            let docs: Vec<NameDocument<'_>> = texts.iter().map(|t| NameDocument::read(t, mode)).collect();
            for (doc, text) in docs.iter().zip(&texts) {
                for sample in doc.samples() {
                    for t in sample.tokens() {
                        if t.is_borrowed() {
                            assert_eq!(&text[t.span.range()], t.as_str());
                        } else {
                            assert!(text[t.span.range()].contains('<') || text[t.span.range()].starts_with('-'));
                        }
                    }
                    for n in sample.names() {
                        assert!(text[n.source.range()].starts_with("<ENAMEX"), "{}", &text[n.source.range()]);
                        assert!(text[n.source.range()].ends_with("</ENAMEX>"));
                    }
                }
            }
            let mut samples = Vec::new();
            for item in NameStream::new(&docs) {
                match item {
                    Ok(sample) => samples.push(sample),
                    Err(e) => error(&mut out, samples.len(), e),
                }
            }
            out.layer("turbo:samples");
            out.layer("turbo:sample-tokens");
            out.layer("turbo:names");
            for (i, sample) in samples.iter().enumerate() {
                out.row("turbo:samples", vec![s(sample)]);
                for t in sample.tokens() {
                    out.row("turbo:sample-tokens", vec![s(i), s(t)]);
                }
                for n in sample.names() {
                    out.row("turbo:names", vec![s(i), s(n.start), s(n.end), s(n.kind.unwrap_or(""))]);
                }
            }
        }
        "parse" => {
            let mut parses = Vec::new();
            for item in ParseStream::new(texts.iter().map(String::as_str)) {
                match item {
                    Ok(p) => {
                        let text = &texts[p.document()];
                        assert_eq!(p.tokens().len(), p.parse().token_nodes().len());
                        for t in p.tokens() {
                            assert_eq!(decode_token(&text[t.token.range()]), &p.parse().text()[t.text.range()]);
                            assert!(text[p.source().range()].contains(&text[t.label.range()]));
                        }
                        parses.push(p);
                    }
                    Err(e) => error(&mut out, parses.len(), &e),
                }
            }
            let trees: Vec<_> = parses.iter().map(|p| p.parse()).collect();
            out.parses(&trees);
        }
        "pos" => {
            let mut samples = Vec::new();
            for item in PosStream::new(texts.iter().map(String::as_str)) {
                match item {
                    Ok(sample) => samples.push(sample),
                    Err(e) => error(&mut out, samples.len(), &e),
                }
            }
            out.layer("turbo:samples");
            out.layer("turbo:sample-tokens");
            out.layer("turbo:sample-tags");
            for (i, sample) in samples.iter().enumerate() {
                out.row("turbo:samples", vec![s(sample)]);
                for (t, tag) in sample.tokens().iter().zip(sample.tags()) {
                    out.row("turbo:sample-tokens", vec![s(i), s(t)]);
                    out.row("turbo:sample-tags", vec![s(tag)]);
                }
            }
        }
        "lines" => {
            out.layer("turbo:samples");
            out.layer("turbo:lines");
            let mut count = 0;
            for text in &texts {
                match document_lines(text) {
                    Ok(lines) => {
                        for line in lines {
                            assert_eq!(&text[line.span.range()], line.text);
                            out.row("turbo:samples", vec![s(line.text)]);
                            out.row("turbo:lines", vec![s(line.text)]);
                            count += 1;
                        }
                    }
                    Err(e) => error(&mut out, count, &e),
                }
            }
        }
        _ => panic!("unknown ontonotes run {run}"),
    }
    out.listing(root, &files);
    out
}

#[test]
fn every_ontonotes_golden_matches() {
    let Some(all) = treebank::goldens("ontonotes") else { return };
    let data = turbo_goldens::data_dir();
    let shapes = treebank::shapes();
    assert!(all.len() >= 25, "only {} ontonotes goldens", all.len());
    for (run, path) in all {
        let golden = Golden::read_with(&path, &shapes).unwrap();
        let input = golden.input_text(&data).unwrap();
        let root = treebank::scratch("ontonotes-goldens", &path);
        bundle::write_bundle(&input, &root);
        treebank::compare(&path, &golden, rebuild(&run, &root));
    }
}

#[test]
fn an_invalid_directory_is_refused_as_the_factories_refuse_it() {
    let e = OntoNotesFormat::new("/nonexistent/ontonotes").name_files().unwrap_err();
    assert_eq!(e.java_class(), Some("opennlp.tools.cmdline.TerminateToolException"));
    assert_eq!(e.to_string(), "The specified OntoNotes directory is not valid!");
}
