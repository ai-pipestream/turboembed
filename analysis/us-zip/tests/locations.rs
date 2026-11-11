//! The ZIP Codes the PII detector finds, placed by the ZCTA tables.

use turbo_analysis::{Annotator, Document, View};
use turbo_pii::{Extractor, PiiDetector, Scanner};
use turbo_us_zip::{SOURCE_CENSUS_ZCTA, ZIP_LOCATIONS, ZipLocationAnnotator};

fn locate(text: &str) -> Vec<(String, String, Vec<String>)> {
    let doc = Document::new(text);
    let view = doc.get::<View>().unwrap().clone();
    let pii = PiiDetector::new(vec![Scanner::all(Extractor::UsIdentity)]).layer(text, view.id());
    let (locations,) = ZipLocationAnnotator::new().annotate((&view, &pii)).unwrap();
    (0..locations.spans.len() as u32)
        .map(|i| {
            let (span, entry, confidence) = locations.resolution(turbo_analysis::Ix::new(i));
            assert_eq!(confidence, 1.0);
            assert_eq!(entry.source(), SOURCE_CENSUS_ZCTA);
            (text[span.range()].to_string(), entry.record_id().to_string(), entry.containment().to_vec())
        })
        .collect()
}

#[test]
fn a_detected_zip_code_gets_its_state_county_and_place() {
    let got = locate("Mail it to 1 Main St, Springfield, IL 62704 by Friday.");
    assert_eq!(got.len(), 1);
    let (surface, zcta, containment) = &got[0];
    assert_eq!((surface.as_str(), zcta.as_str()), ("62704", "62704"));
    assert_eq!(containment, &["IL", "Sangamon County", "Springfield"]);
}

#[test]
fn zip_plus_four_and_several_codes_are_placed_in_text_order() {
    let got = locate("ZIP: 78701-4417 and later zip code 10001.");
    let zctas: Vec<&str> = got.iter().map(|g| g.1.as_str()).collect();
    assert_eq!(zctas, ["78701", "10001"]);
    assert_eq!(got[0].0, "78701-4417");
    assert_eq!(got[0].2[0], "TX");
    assert_eq!(got[1].2[0], "NY");
}

#[test]
fn a_zip_code_without_a_zcta_adds_nothing() {
    // A military ZIP Code: detected as PII, but it has no ZCTA.
    assert!(locate("APO AE 09012").is_empty());
    assert!(locate("no postal code here").is_empty());
}

#[test]
fn text_outside_ascii_around_the_zip_code_keeps_its_span() {
    let got = locate("Envíe a Cañon City, CO 81212, ¡gracias!");
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].0, "81212");
    assert_eq!(got[0].2[0], "CO");
}

#[test]
fn it_writes_the_zip_locations_key() {
    assert_eq!(ZipLocationAnnotator::new().produces(), &[ZIP_LOCATIONS.id()]);
}
