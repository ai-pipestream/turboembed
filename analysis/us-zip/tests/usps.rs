//! A caller's USPS locale list, in the layout of the `ZIP_Locale` sheet
//! saved as tab separated text. The rows are made up for the test.

use turbo_analysis::{Annotator, Document, Ix, View};
use turbo_pii::{Extractor, PiiDetector, Scanner};
use turbo_us_zip::{ATTRIBUTE_KEY_USPS_LOCALE, UspsLocales, ZipLocationAnnotator};

const HEADER: &str = "AREA NAME\tDISTRICT NAME\tDISTRICT NO\tDELIVERY ZIPCODE\tZIP CLASS CODE\tLOCALE NAME\tLOCALE KEY\t\
LOCALE TYPE\tPHYSICAL DELV ADDR\tPHYSICAL CITY\tPHYSICAL STATE\tPHYSICAL ZIP\tPHYSICAL ZIP 4";

fn sheet(rows: &[&str]) -> String {
    let mut s = format!("{HEADER}\n");
    for r in rows {
        s.push_str(r);
        s.push('\n');
    }
    s
}

const SPRINGFIELD: &str =
    "AREA\tDISTRICT  \t627\t62704\t\tSPRINGFIELD   \tK1\tP\t1 MAIN ST\tSPRINGFIELD\tIL\t62703\t9998";
const LELAND_GROVE: &str =
    "AREA\tDISTRICT\t627\t62704\tP\tLELAND GROVE\tK2\tS\t1 MAIN ST\tSPRINGFIELD\tIL\t62703\t9996";

#[test]
fn padded_fields_and_repeated_zip_codes_are_read() {
    let u = UspsLocales::parse(&sheet(&[SPRINGFIELD, LELAND_GROVE, SPRINGFIELD])).unwrap();
    assert_eq!(u.len(), 1);
    let got: Vec<(&str, &str)> = u.locales("62704-0001").collect();
    assert_eq!(got, [("SPRINGFIELD", "IL"), ("LELAND GROVE", "IL")]);
    assert_eq!(u.locales("62705").count(), 0);
}

#[test]
fn a_zip_code_stored_as_a_number_gets_its_zeros_back() {
    let row = "AREA\tPUERTO RICO\t006\t601\t\tADJUNTAS\tK3\tP\t1 CALLE\tADJUNTAS\tPR\t601\t9998";
    let u = UspsLocales::parse(&format!("\u{feff}{}", sheet(&[row]))).unwrap();
    assert_eq!(u.locales("00601").collect::<Vec<_>>(), [("ADJUNTAS", "PR")]);
}

#[test]
fn a_sheet_without_its_columns_or_with_bad_rows_is_refused() {
    let err = |text: &str| UspsLocales::parse(text).unwrap_err().to_string();
    assert!(err("ZIP\tCITY\n62704\tSPRINGFIELD\n").contains("no DELIVERY ZIPCODE column"));
    assert!(err(&sheet(&[&SPRINGFIELD.replace("62704", "62A04")])).contains("line 2: DELIVERY ZIPCODE"));
    assert!(err(&sheet(&[&SPRINGFIELD.replace("\tIL\t", "\tIll\t")])).contains("PHYSICAL STATE"));
    assert!(err(&sheet(&[&SPRINGFIELD.replace("SPRINGFIELD   ", "  ")])).contains("LOCALE NAME is blank"));
    assert!(err(&sheet(&["AREA\tDISTRICT\t627\t62704"])).contains("no LOCALE NAME field"));
}

#[test]
fn the_annotator_adds_locale_names_to_a_detected_zip_code() {
    let u = UspsLocales::parse(&sheet(&[SPRINGFIELD, LELAND_GROVE])).unwrap();
    let text = "Springfield, IL 62704";
    let doc = Document::new(text);
    let view = doc.get::<View>().unwrap().clone();
    let pii = PiiDetector::new(vec![Scanner::all(Extractor::UsIdentity)]).layer(text, view.id());
    let (locations,) = ZipLocationAnnotator::new().with_usps(u).annotate((&view, &pii)).unwrap();
    let (_, entry, _) = locations.resolution(Ix::new(0));
    assert_eq!(entry.alternate_names(), ["SPRINGFIELD", "LELAND GROVE"]);
    assert_eq!(entry.attribute(ATTRIBUTE_KEY_USPS_LOCALE).unwrap().value(), "SPRINGFIELD|LELAND GROVE");
    assert_eq!(entry.containment(), ["IL", "Sangamon County", "Springfield"]);
}

#[test]
fn lists_of_one_size_with_other_names_hash_apart() {
    let a = UspsLocales::parse(&sheet(&[SPRINGFIELD])).unwrap();
    let b = UspsLocales::parse(&sheet(&[LELAND_GROVE])).unwrap();
    assert_eq!(a.len(), b.len());
    assert_ne!(a.digest(), b.digest());
    let ha = ZipLocationAnnotator::new().with_usps(a).config_hash();
    let hb = ZipLocationAnnotator::new().with_usps(b).config_hash();
    assert_ne!(ha, hb);
    assert_ne!(ha, ZipLocationAnnotator::new().config_hash());
}

#[test]
fn a_locale_name_with_the_attribute_separator_is_refused() {
    let err = UspsLocales::parse(&sheet(&[&SPRINGFIELD.replace("SPRINGFIELD   ", "SPRING|FIELD")])).unwrap_err();
    assert!(err.to_string().contains("holds a `|`"), "{err}");
}
