//! The Dutch time rules on texts of their own. The Recognizers-Text cases
//! are in `tests/spec/dutch.rs`.

use turbo_analysis_layers::DateTime;

use super::*;
use crate::engine::Parser;
use crate::time::resolve;

/// Each mention as `text|kind|timex|modifier`, with its preferred
/// reading, against the reference `at`.
fn found(text: &str, at: &str) -> Vec<String> {
    let now = DateTime::parse(at).unwrap();
    let mut p = Parser::new(&DUTCH_TIMES);
    p.parse(text);
    let mut ms = Vec::new();
    p.select(text, Output::Time, &mut ms);
    ms.iter()
        .map(|m| {
            let v = p.times().get(&m.value).unwrap();
            let r = resolve(&v, now).unwrap();
            let a = r.alternatives[r.preferred];
            format!("{}|{}|{}|{}", &text[m.start as usize..m.end as usize], r.kind.name(), a.timex, r.modifier)
        })
        .collect()
}

const AT: &str = "2021-06-15T10:00:00";

fn check(cases: &[(&str, &[&str])]) {
    for (text, want) in cases {
        assert_eq!(found(text, AT), *want, "{text}");
    }
}

#[test]
fn clock_times_count_from_the_half_and_quarter_hours() {
    check(&[
        ("Ik kom om kwart over drie", &["kwart over drie|time|T03:15|"]),
        ("We vertrekken om tien over half vijf", &["tien over half vijf|time|T04:40|"]),
        ("De trein vertrekt om 7.15 uur", &["om 7.15 uur|time|T07:15|"]),
        ("half acht", &["half acht|time|T07:30|"]),
        ("om half acht 's avonds", &["half acht 's avonds|time|T19:30|"]),
    ]);
}

#[test]
fn a_day_and_its_part_split_from_one_word() {
    check(&[
        ("Zaterdagochtend ga ik zwemmen", &["Zaterdagochtend|datetimerange|XXXX-WXX-6TMO|"]),
        ("Gisterochtend was het koud", &["Gisterochtend|datetimerange|2021-06-14TMO|"]),
        ("Vanmiddag regent het", &["Vanmiddag|datetimerange|2021-06-15TAF|"]),
        ("Morgenochtend om zeven uur", &["Morgenochtend om zeven uur|datetime|2021-06-16T07|"]),
        ("Kom je vrijdagavond om 8 uur?", &["vrijdagavond om 8 uur|datetime|XXXX-WXX-5T20|"]),
    ]);
}

#[test]
fn dates_read_day_first() {
    check(&[
        ("Ze is geboren op 12-03-1985", &["12-03-1985|date|1985-03-12|"]),
        ("dinsdag 4 mei", &["dinsdag 4 mei|date|XXXX-05-04|"]),
        ("op 4 mei om 14.00 uur", &["4 mei om 14.00 uur|datetime|XXXX-05-04T14:00|"]),
        ("Hij kwam eergisteren", &["eergisteren|date|2021-06-13|"]),
        ("Afgelopen zondag", &["Afgelopen zondag|date|2021-06-13|"]),
        ("vorige week dinsdag", &["vorige week dinsdag|date|2021-06-08|"]),
        ("de eerste maandag van september", &["de eerste maandag van september|date|XXXX-09-WXX-1-#1|"]),
    ]);
}

#[test]
fn ranges_and_portions() {
    check(&[
        (
            "Ik ben van 1 tot en met 5 juli op vakantie",
            &["van 1 tot en met 5 juli|daterange|(XXXX-07-01,XXXX-07-05,P4D)|"],
        ),
        ("tussen 3 en 5 uur", &["tussen 3 en 5 uur|timerange|(T03,T05,PT2H)|"]),
        ("maandag tot en met vrijdag", &["maandag tot en met vrijdag|daterange|(2021-06-14,2021-06-18,P4D)|"]),
        ("de komende drie dagen", &["komende drie dagen|daterange|(2021-06-16,2021-06-19,P3D)|"]),
        ("aan het einde van de week", &["aan het einde van de week|daterange|2021-W24|end"]),
        ("het eind van de maand", &["het eind van de maand|daterange|2021-06|end"]),
        ("in de zomer van 2019", &["zomer van 2019|daterange|2019-SU|"]),
    ]);
}

#[test]
fn durations_and_shifts() {
    check(&[
        ("Het duurt twee uur en een kwartier", &["twee uur en een kwartier|duration|PT2H15M|"]),
        ("Over een kwartier begint de film", &["Over een kwartier|datetime|2021-06-15T10:15:00|"]),
        ("een kwartier later", &["een kwartier later|datetime|2021-06-15T10:15:00|"]),
        ("Dat was drie jaar geleden", &["drie jaar geleden|date|2018-06-15|"]),
        ("twee weken na pasen", &["twee weken na pasen|date|2022-05-01|"]),
    ]);
}

#[test]
fn sets() {
    check(&[
        ("We spreken elke dinsdag om half tien af", &["elke dinsdag om half tien|set|XXXX-WXX-2T09:30|"]),
        ("Hij werkt dinsdagen en donderdagen", &["dinsdagen|set|XXXX-WXX-2|", "donderdagen|set|XXXX-WXX-4|"]),
        ("iedere ochtend", &["iedere ochtend|set|TMO|"]),
        ("om de twee weken", &["om de twee weken|set|P2W|"]),
    ]);
}

#[test]
fn words_with_other_meanings_are_no_times() {
    check(&[
        ("We zijn er zo", &[]),
        ("Jan komt morgen", &["morgen|date|2021-06-16|"]),
        ("Hij is vrij op maandag", &["maandag|date|XXXX-WXX-1|"]),
        ("Dit kost 15.00 euro", &[]),
        ("Het kost 3 euro per maand", &[]),
        ("Hij woont op nummer 12", &[]),
        ("Ze heeft 3 kinderen", &[]),
        ("tussen 3 en 5 kinderen", &[]),
        ("van 9 tot 11 leerlingen", &[]),
    ]);
}
