//! Every glossary the readers take from the goldens' inputs, and the
//! fixture glossary, written as TBX in both dialects and read back, is the
//! same glossary.

use turbo_glossary::{DelimitedReader, FirstRecord, Glossary, TbxDialect, TbxReader, write_tbx};

fn inputs(prefix: &str) -> Vec<(String, Vec<u8>)> {
    let dir = turbo_goldens::data_dir().join("inputs").join("glossary");
    let mut out: Vec<(String, Vec<u8>)> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.file_name().unwrap().to_str().unwrap().starts_with(prefix))
        .map(|p| (p.display().to_string(), std::fs::read(&p).unwrap()))
        .collect();
    out.sort();
    assert!(!out.is_empty());
    out
}

fn round_trip(name: &str, g: &Glossary, language: &str) {
    for dialect in [TbxDialect::Martif, TbxDialect::Tbx] {
        let mut xml = String::new();
        write_tbx(g, language, dialect, &mut xml).unwrap_or_else(|e| panic!("{name}: {e}"));
        let back =
            TbxReader::new(language).unwrap().read(xml.as_bytes()).unwrap_or_else(|e| panic!("{name}: {e}\n{xml}"));
        assert_eq!(&back.glossary, g, "{name} {dialect:?}\n{xml}");
    }
}

#[test]
fn termbases_round_trip() {
    let mut read = 0;
    for (name, bytes) in inputs("tbx-") {
        for language in ["en", "de"] {
            if let Ok(f) = TbxReader::new(language).unwrap().read(&bytes) {
                round_trip(&name, &f.glossary, language);
                read += usize::from(!f.glossary.is_empty());
            }
        }
    }
    assert!(read >= 5, "{read}");
}

#[test]
fn delimited_lists_round_trip() {
    let fixture = turbo_goldens::data_dir().join("fixtures").join("glossary-terms.tsv");
    let g = DelimitedReader::tsv(FirstRecord::Header).read(&std::fs::read(&fixture).unwrap()).unwrap().glossary;
    round_trip("glossary-terms.tsv", &g, "en");
    for (name, bytes) in inputs("csv-") {
        if let Ok(f) = DelimitedReader::csv().read(&bytes) {
            // The reader keeps a field's spaces and TBX trims them.
            if f.glossary.iter().all(|(_, e)| e.term.trim() == e.term) {
                round_trip(&name, &f.glossary, "en");
            }
        }
    }
}
