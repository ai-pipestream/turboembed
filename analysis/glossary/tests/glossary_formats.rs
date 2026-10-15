//! Comprehensive CSV, TSV, and TBX reader, boundary, structure, security,
//! contract, and round-trip tests ported from OpenNLP's
//! `opennlp.tools.formats.glossary` test suite.

use std::io::Cursor;

use turbo_glossary::{
    AhoCorasickGlossaryMatcher, CsvGlossaryReader, GlossaryEntry, GlossaryMatcher, TbxGlossaryReader,
};

fn entry(id: &str, term: &str) -> GlossaryEntry {
    GlossaryEntry::new(id, term).unwrap()
}

// ============================================================================
// CsvGlossaryReaderTest & CsvGlossaryReaderBoundaryTest
// ============================================================================

#[test]
fn reads_csv_with_header_quotes_escaped_quotes_and_extra_columns() {
    let csv = concat!(
        "\u{FEFF}id,term,notes\r\n",
        "C1,machine learning,core concept\r\n",
        "C1,\"learning, machine\",\"inverted, \"\"quoted\"\" note\"\n",
        "\n",
        "C2,\"deep \"\"neural\"\" network\",extra\r",
        "C3,\"multi\r\nline\",ignored\n",
    );

    let reader = CsvGlossaryReader::csv(true);
    let entries = reader.read(Cursor::new(csv.as_bytes())).unwrap();

    assert_eq!(
        entries,
        vec![
            entry("C1", "machine learning"),
            entry("C1", "learning, machine"),
            entry("C2", "deep \"neural\" network"),
            entry("C3", "multi\nline"),
        ]
    );
}

#[test]
fn reads_tsv_without_header() {
    let tsv = "ID-1\tApache OpenNLP\nID-2\ttokenization\n";
    let reader = CsvGlossaryReader::tsv(false);
    let entries = reader.read(Cursor::new(tsv.as_bytes())).unwrap();

    assert_eq!(entries, vec![entry("ID-1", "Apache OpenNLP"), entry("ID-2", "tokenization"),]);
}

#[test]
fn csv_rejects_invalid_delimiters_and_malformed_records() {
    assert!(CsvGlossaryReader::new('"', false).is_err());
    assert!(CsvGlossaryReader::new('\n', false).is_err());
    assert!(CsvGlossaryReader::new('\r', false).is_err());

    let reader = CsvGlossaryReader::csv(false);

    // Fewer than 2 columns
    assert!(reader.read_str("only_one_column\n").is_err());

    // Blank id or term
    assert!(reader.read_str(" ,term\n").is_err());
    assert!(reader.read_str("id, \n").is_err());

    // Unterminated quote
    assert!(reader.read_str("id,\"unterminated\n").is_err());

    // Quote in unquoted field
    assert!(reader.read_str("id,bad\"quote\n").is_err());

    // Character after closing quote
    assert!(reader.read_str("id,\"quoted\"trailing\n").is_err());
}

#[test]
fn csv_boundary_cases_in_headers_extra_columns_and_line_numbers() {
    // Leading blank lines before header must not consume skip_header.
    let with_leading_blanks = "\n  \r\n\tid,term\nC1,alpha\n";
    let entries = CsvGlossaryReader::csv(true).read_str(with_leading_blanks).unwrap();
    assert_eq!(entries, vec![entry("C1", "alpha")]);

    // Malformed quoting in skipped header row must still fail.
    assert!(CsvGlossaryReader::csv(true).read_str("id,\"bad\"header\nC1,alpha\n").is_err());

    // Malformed quoting in ignored 3rd column must still fail.
    let reader = CsvGlossaryReader::csv(false);
    assert!(reader.read_str("C1,alpha,\"unterminated\n").is_err());
    assert!(reader.read_str("C1,alpha,bad\"quote\n").is_err());
    assert!(reader.read_str("C1,alpha,\"closed\"extra\n").is_err());

    // Physical line number after multi-line quoted record must be reported accurately.
    let multi_line_then_bad = "C1,\"line1\nline2\nline3\"\nC2,only_one_col_missing_delim\n";
    let err = reader.read_str("C1,\"line1\nline2\nline3\"\nonly_one_column\n").unwrap_err().to_string();
    assert!(err.contains("line 4"), "Expected error message to cite physical line 4, got: {err}");
    let _ = multi_line_then_bad;

    // Invalid UTF-8 bytes must fail.
    let bad_utf8: &[u8] = &[b'C', b'1', b',', 0xFF, 0xFE, b'\n'];
    assert!(reader.read(Cursor::new(bad_utf8)).is_err());
}

// ============================================================================
// TbxGlossaryReaderTest
// ============================================================================

#[test]
fn reads_tbx_v3_concept_entries_for_target_language() {
    let tbx = r#"<?xml version="1.0" encoding="UTF-8"?>
<tbx xmlns="urn:iso:std:iso:30042:ed-2" type="TBX-Core" style="dca" xml:lang="en">
  <tbxHeader>
    <fileDesc><titleStmt><title>Sample</title></titleStmt></fileDesc>
  </tbxHeader>
  <text>
    <body>
      <conceptEntry id="C-100">
        <langSec xml:lang="en-US">
          <termSec><term>natural language processing</term></termSec>
          <termSec><term>NLP</term></termSec>
        </langSec>
        <langSec xml:lang="de">
          <termSec><term>Sprachverarbeitung</term></termSec>
        </langSec>
      </conceptEntry>
      <conceptEntry id="C-200">
        <langSec xml:lang="EN">
          <termSec><term><hi>token</hi>ization</term></termSec>
        </langSec>
      </conceptEntry>
    </body>
  </text>
</tbx>"#;

    let en_reader = TbxGlossaryReader::new("en").unwrap();
    let en_entries = en_reader.read_str(tbx).unwrap();
    assert_eq!(
        en_entries,
        vec![entry("C-100", "natural language processing"), entry("C-100", "NLP"), entry("C-200", "tokenization"),]
    );

    let de_reader = TbxGlossaryReader::new("de").unwrap();
    let de_entries = de_reader.read_str(tbx).unwrap();
    assert_eq!(de_entries, vec![entry("C-100", "Sprachverarbeitung")]);
}

#[test]
fn reads_tbx_v2_martif_term_entries_with_tig_and_ntig() {
    let tbx_v2 = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE martif SYSTEM "http://example.invalid/TBXcoreStructV02.dtd">
<martif type="TBX" xml:lang="en">
  <text>
    <body>
      <termEntry id="TE-1">
        <langSet xml:lang="en">
          <tig><term>named entity</term></tig>
          <ntig><termGrp><term>NE</term></termGrp></ntig>
        </langSet>
      </termEntry>
    </body>
  </text>
</martif>"#;

    let reader = TbxGlossaryReader::new("en").unwrap();
    let entries = reader.read_str(tbx_v2).unwrap();
    assert_eq!(entries, vec![entry("TE-1", "named entity"), entry("TE-1", "NE")]);
}

#[test]
fn tbx_rejects_invalid_language_missing_id_blank_term_and_non_tbx_root() {
    assert!(TbxGlossaryReader::new(" ").is_err());

    let reader = TbxGlossaryReader::new("en").unwrap();

    // Non-TBX root element
    assert!(reader.read_str("<glossary/>").is_err());

    // Missing id on conceptEntry
    let missing_id = r#"<tbx><text><body><conceptEntry><langSec xml:lang="en"><termSec><term>x</term></termSec></langSec></conceptEntry></body></text></tbx>"#;
    assert!(reader.read_str(missing_id).is_err());

    // Blank <term>
    let blank_term = r#"<tbx><text><body><conceptEntry id="C1"><langSec xml:lang="en"><termSec><term>   </term></termSec></langSec></conceptEntry></body></text></tbx>"#;
    assert!(reader.read_str(blank_term).is_err());
}

// ============================================================================
// TbxGlossaryReaderStructureTest
// ============================================================================

#[test]
fn tbx_ignores_foreign_namespace_lookalike_elements_and_attributes() {
    let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<tbx xmlns="urn:iso:std:iso:30042:ed-2" xmlns:ext="urn:example:extension">
  <text>
    <body>
      <ext:conceptEntry id="FOREIGN-ENTRY">
        <ext:langSec xml:lang="en">
          <ext:termSec><ext:term>should be ignored</ext:term></ext:termSec>
        </ext:langSec>
      </ext:conceptEntry>
      <conceptEntry id="REAL-1" ext:id="WRONG-ID">
        <langSec xml:lang="en" ext:lang="de">
          <termSec>
            <ext:note><term>nested foreign term should be ignored</term></ext:note>
            <term>valid term</term>
          </termSec>
        </langSec>
      </conceptEntry>
    </body>
  </text>
</tbx>"#;

    let reader = TbxGlossaryReader::new("en").unwrap();
    let entries = reader.read_str(xml).unwrap();
    assert_eq!(entries, vec![entry("REAL-1", "valid term")]);
}

#[test]
fn tbx_rejects_entry_whose_only_id_attribute_is_namespace_qualified() {
    let xml = r#"<tbx xmlns="urn:iso:std:iso:30042:ed-2" xmlns:ext="urn:example:ext">
  <text><body>
    <conceptEntry ext:id="ONLY-FOREIGN">
      <langSec xml:lang="en"><termSec><term>x</term></termSec></langSec>
    </conceptEntry>
  </body></text>
</tbx>"#;

    let reader = TbxGlossaryReader::new("en").unwrap();
    assert!(reader.read_str(xml).is_err());
}

#[test]
fn tbx_ignores_elements_outside_body_and_mismatched_dialect_tags() {
    let xml = r#"<tbx xmlns="urn:iso:std:iso:30042:ed-2">
  <tbxHeader>
    <fileDesc><titleStmt><title>Header</title></titleStmt></fileDesc>
  </tbxHeader>
  <text>
    <body>
      <termEntry id="V2-IN-V3">
        <langSet xml:lang="en"><tig><term>wrong dialect</term></tig></langSet>
      </termEntry>
      <conceptEntry id="C1">
        <langSec xml:lang="en">
          <term>stray term outside termSec</term>
          <termSec>
            <note><term>term inside note</term></note>
            <term>accepted term</term>
          </termSec>
        </langSec>
      </conceptEntry>
    </body>
    <back>
      <conceptEntry id="IN-BACK">
        <langSec xml:lang="en"><termSec><term>back matter</term></termSec></langSec>
      </conceptEntry>
    </back>
  </text>
</tbx>"#;

    let reader = TbxGlossaryReader::new("en").unwrap();
    let entries = reader.read_str(xml).unwrap();
    assert_eq!(entries, vec![entry("C1", "accepted term")]);
}

#[test]
fn tbx_v2_ntig_requires_term_grp_wrapper_around_term() {
    let xml = r#"<martif type="TBX" xml:lang="en">
  <text><body>
    <termEntry id="TE-1">
      <langSet xml:lang="en">
        <ntig>
          <term>stray term directly under ntig</term>
          <termGrp><term>valid ntig term</term></termGrp>
        </ntig>
      </langSet>
    </termEntry>
  </body></text>
</martif>"#;

    let reader = TbxGlossaryReader::new("en").unwrap();
    let entries = reader.read_str(xml).unwrap();
    assert_eq!(entries, vec![entry("TE-1", "valid ntig term")]);
}

// ============================================================================
// TbxGlossaryReaderSecurityTest
// ============================================================================

#[test]
fn tbx_rejects_external_general_and_parameter_entities_even_when_unreferenced() {
    let reader = TbxGlossaryReader::new("en").unwrap();

    let xxe_referenced = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE tbx [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<tbx><text><body>
  <conceptEntry id="C1">
    <langSec xml:lang="en"><termSec><term>&xxe;</term></termSec></langSec>
  </conceptEntry>
</body></text></tbx>"#;
    assert!(reader.read_str(xxe_referenced).is_err());

    let xxe_unreferenced = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE tbx [
  <!ENTITY unused SYSTEM "http://127.0.0.1:1/never-connect">
]>
<tbx><text><body>
  <conceptEntry id="C1">
    <langSec xml:lang="en"><termSec><term>safe</term></termSec></langSec>
  </conceptEntry>
</body></text></tbx>"#;
    assert!(reader.read_str(xxe_unreferenced).is_err());

    let parameter_entity = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE tbx [
  <!ENTITY % pe SYSTEM "http://127.0.0.1:1/never-connect">
]>
<tbx><text><body>
  <conceptEntry id="C1">
    <langSec xml:lang="en"><termSec><term>safe</term></termSec></langSec>
  </conceptEntry>
</body></text></tbx>"#;
    assert!(reader.read_str(parameter_entity).is_err());
}

#[test]
fn tbx_expands_internal_general_entity_and_rejects_entity_expansion_bomb() {
    let reader = TbxGlossaryReader::new("en").unwrap();

    let internal_entity = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE tbx [
  <!ENTITY nlp "natural language processing">
]>
<tbx><text><body>
  <conceptEntry id="C1">
    <langSec xml:lang="en"><termSec><term>&nlp;</term></termSec></langSec>
  </conceptEntry>
</body></text></tbx>"#;
    let entries = reader.read_str(internal_entity).unwrap();
    assert_eq!(entries, vec![entry("C1", "natural language processing")]);

    let billion_laughs = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE tbx [
  <!ENTITY lol "lol">
  <!ENTITY lol1 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
  <!ENTITY lol2 "&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;">
  <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">
  <!ENTITY lol4 "&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;">
  <!ENTITY lol5 "&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;">
]>
<tbx><text><body>
  <conceptEntry id="C1">
    <langSec xml:lang="en"><termSec><term>&lol5;</term></termSec></langSec>
  </conceptEntry>
</body></text></tbx>"#;
    assert!(reader.read_str(billion_laughs).is_err());
}

// ============================================================================
// GlossaryReaderContractTest & Round-Trip Tests
// ============================================================================

#[test]
fn csv_tsv_and_tbx_readers_produce_identical_entries_and_round_trip() {
    let expected = vec![
        entry("C-001", "natural language processing"),
        entry("C-001", "NLP"),
        entry("C-002", "named-entity recognition"),
        entry("C-003", "caf\u{00E9} Latt\u{00E9}"),
        entry("C-004", "quote \"inside\" term, with comma"),
    ];

    // 1. CSV round-trip
    let csv_reader = CsvGlossaryReader::csv(true);
    let mut csv_buf = Vec::new();
    csv_reader.write(&expected, &mut csv_buf).unwrap();
    let from_csv = csv_reader.read(Cursor::new(&csv_buf)).unwrap();
    assert_eq!(from_csv, expected);

    // 2. TSV round-trip
    let tsv_reader = CsvGlossaryReader::tsv(true);
    let mut tsv_buf = Vec::new();
    tsv_reader.write(&expected, &mut tsv_buf).unwrap();
    let from_tsv = tsv_reader.read(Cursor::new(&tsv_buf)).unwrap();
    assert_eq!(from_tsv, expected);

    // 3. TBX v3 round-trip
    let tbx_reader = TbxGlossaryReader::new("en").unwrap();
    let mut tbx_v3_buf = Vec::new();
    tbx_reader.write_tbx_v3(&expected, &mut tbx_v3_buf).unwrap();
    let from_tbx_v3 = tbx_reader.read(Cursor::new(&tbx_v3_buf)).unwrap();
    assert_eq!(from_tbx_v3, expected);

    // 4. TBX v2 round-trip
    let mut tbx_v2_buf = Vec::new();
    tbx_reader.write_tbx_v2(&expected, &mut tbx_v2_buf).unwrap();
    let from_tbx_v2 = tbx_reader.read(Cursor::new(&tbx_v2_buf)).unwrap();
    assert_eq!(from_tbx_v2, expected);

    // 5. Match equivalence across all 4 formats
    let sample_text = "We use NLP and named-entity recognition while drinking caf\u{00E9} Latt\u{00E9}.";
    let m_csv = AhoCorasickGlossaryMatcher::new(&from_csv, false).unwrap().match_text(sample_text);
    let m_tsv = AhoCorasickGlossaryMatcher::new(&from_tsv, false).unwrap().match_text(sample_text);
    let m_tbx3 = AhoCorasickGlossaryMatcher::new(&from_tbx_v3, false).unwrap().match_text(sample_text);
    let m_tbx2 = AhoCorasickGlossaryMatcher::new(&from_tbx_v2, false).unwrap().match_text(sample_text);

    assert_eq!(m_csv, m_tsv);
    assert_eq!(m_csv, m_tbx3);
    assert_eq!(m_csv, m_tbx2);
    assert_eq!(m_csv.len(), 3);
}
