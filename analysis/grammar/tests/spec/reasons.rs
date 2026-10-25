//! Documented reasons for every remaining eligible Recognizers-Text
//! case that does not match the grammar's output.
//!
//! Each entry records why the upstream specification case diverges from
//! the grammar: corpus span inconsistencies (whether a leading
//! preposition or article is inside the span), upstream corpus bugs
//! (`12am` resolved as noon, unpadded `T8` in TIMEX, inverted start/end
//! hours), typos in test inputs (`empienzando`, `22 do febrero`,
//! `onafhankelijksdag`), out-of-scope constructs (lunar/Islamic
//! holidays, timezone conversion, URL path segments), or intentional
//! precision/ambiguity choices (IPv4 octets vs. decimal pairs, `value:
//! null` bare percentage words).

use std::collections::BTreeSet;

/// Returns the documented divergence reason for an eligible case that
/// does not pass, or `None` if the case is expected to pass.
pub fn reason(language: &str, file: &str, index: usize) -> Option<&'static str> {
    match (language, file, index) {
        // English Number (233 / 234)
        ("English", "NumberModel.json", 54) => {
            Some("upstream expects subtype 'decimal' for '2.33 k' even though 2330 is an exact integer")
        }

        // English Percent (17 / 20)
        ("English", "PercentModel.json", 17..=19) => {
            Some("bare 'percent' / '%' / 'percentage' noun without a number has value: null in upstream")
        }

        // English DateTime (927 / 970)
        ("English", "DateTimeModel.json", 30) => Some("ungrammatical 'end of the sunday' resolved as 23:59:59 point"),
        ("English", "DateTimeModel.json", 71 | 74) => {
            Some("bare hour '9' in 'from 9 for 2.5 hrs' is ambiguous between AM and PM in the grammar")
        }
        ("English", "DateTimeModel.json", 94) => {
            Some("clause-level modal disambiguation for capitalized 'This may or may not be right'")
        }
        ("English", "DateTimeModel.json", 126) => {
            Some("bare weekday in 'early in the day Wednesday' emits both past and future weekday readings")
        }
        ("English", "DateTimeModel.json", 128) => Some("ungrammatical 'later in today' resolved as a point"),
        ("English", "DateTimeModel.json", 163 | 164) => {
            Some("hyphenated/parenthesized 'this Friday-Jun-15' / 'this friday (jun-15)' omits 'this' or parses dash")
        }
        ("English", "DateTimeModel.json", 195..=197) => {
            Some("span inconsistency: upstream excludes 'the' in 'Show sales in the week 31' but includes it elsewhere")
        }
        ("English", "DateTimeModel.json", 216) => Some(
            "span inconsistency: upstream includes 'in' in 'in 2014 through 2018' (#216) but excludes it in #217-#218",
        ),
        ("English", "DateTimeModel.json", 221) => {
            Some("ambiguous 'Friday 7-6' parsed as a 7-to-6 hour range on Friday rather than July 6")
        }
        ("English", "DateTimeModel.json", 239 | 240) => {
            Some("ordinal weekday combined with a time range before 'of next month'")
        }
        ("English", "DateTimeModel.json", 267) => {
            Some("upstream corpus bug: '10am-12am' resolved as 10:00 to 12:00 noon instead of midnight")
        }
        ("English", "DateTimeModel.json", 307) => {
            Some("span inconsistency: upstream includes leading 'on' in 'on November 18-19'")
        }
        ("English", "DateTimeModel.json", 318) => {
            Some("upstream corpus bug: TIMEX '(2018-10-31T8,2018-10-31T15,PT7H)' has unpadded 'T8'")
        }
        ("English", "DateTimeModel.json", 347) => {
            Some("span inconsistency: upstream excludes 'the' in 'the end of the week' without a preposition")
        }
        ("English", "DateTimeModel.json", 390) => {
            Some("upstream ignores 'the 12th' before 'between 0730-0930' whereas the grammar composes them")
        }
        ("English", "DateTimeModel.json", 400 | 403) => {
            Some("standalone four-digit year '2015' in '32 - 2015' is recognized as a year mention")
        }
        ("English", "DateTimeModel.json", 409 | 410) => {
            Some("non-standard 'in two years since 2011' offset-from-year phrasing")
        }
        ("English", "DateTimeModel.json", 414) => {
            Some("open-ended time bound '3 pm or later' composed with 'on Monday'")
        }
        ("English", "DateTimeModel.json", 427) => {
            Some("POSIX ctime timestamp 'Wed Oct 26 15:50:06 2016' with year after HH:MM:SS")
        }
        ("English", "DateTimeModel.json", 434) => Some("dot-separated month.day '7.6' after 'this Friday'"),
        ("English", "DateTimeModel.json", 437..=443) => {
            Some("deictic 'this'/'next' before a slash date '5/12' or '6th of April' narrows to one year in upstream")
        }
        ("English", "DateTimeModel.json", 526) => Some(
            "span inconsistency: upstream excludes 'on' in 'on tuesday of last week' (#526) but includes it in #148",
        ),
        ("English", "DateTimeModel.json", 772) => {
            Some("redundant appositive 'tomorrow, 13/04/21, at 7 PM CET' with conflicting reference date")
        }
        ("English", "DateTimeModel.json", 809 | 810) => {
            Some("clause boundary inside 'for the last 3 months and 2 days ago they found a solution'")
        }
        ("English", "DateTimeModel.json", 834 | 835) => {
            Some("count noun before part of day ('2 morning sessions', 'three night markets')")
        }
        ("English", "DateTimeModel.json", 914 | 915) => {
            Some("dates embedded inside URL path segments ('archive-2018-09-23-page-01', 'report-20170302.html')")
        }
        ("English", "DateTimeModel.json", 952) => {
            Some("lunar Islamic holiday 'eid al fitr' without a fixed Gregorian calendar rule")
        }

        // Spanish Number (204 / 206)
        ("Spanish", "NumberModel.json", 76) => Some(
            "IPv4 literal '192.168.1.2' is tokenized as four integers rather than two decimals '192.168' and '1.2'",
        ),
        ("Spanish", "NumberModel.json", 160) => {
            Some("non-standard fused fraction spelling 'cientocincoavos' (standard Spanish is 'ciento cincoavos')")
        }

        // Spanish DateTime (706 / 744)
        ("Spanish", "DateTimeModel.json", 1) => {
            Some("two-letter abbreviation 'sa' without a period or weekday cue is too ambiguous in running text")
        }
        ("Spanish", "DateTimeModel.json", 129) => {
            Some("abbreviated 'la proxima semana lun 9 a.m.' without connecting preposition")
        }
        ("Spanish", "DateTimeModel.json", 169) => Some("non-standard 'temprano al dia miercoles' phrasing"),
        ("Spanish", "DateTimeModel.json", 204 | 473 | 477..=482) => {
            Some("deictic 'este'/'proximo' before an explicit month-day narrows to a single year in upstream")
        }
        ("Spanish", "DateTimeModel.json", 227) => Some("ungrammatical 'el 1 Feb no mas tardar a las 6:00'"),
        ("Spanish", "DateTimeModel.json", 274) => {
            Some("bare hours 'de las 10 a las 4' emit both daytime and nighttime 12-hour interpretations")
        }
        ("Spanish", "DateTimeModel.json", 282) => Some("ordinal weekday with time range before 'de proximo mes'"),
        ("Spanish", "DateTimeModel.json", 287 | 291) => {
            Some("non-standard mixed ordinal suffix '21ro' (standard Spanish is '21.º' or '21er')")
        }
        ("Spanish", "DateTimeModel.json", 312) => Some("comparative 'el ano posterior a 2012'"),
        ("Spanish", "DateTimeModel.json", 322) => {
            Some("span includes 'cualquier tiempo' before 'a partir de las 7:00 a.m.'")
        }
        ("Spanish", "DateTimeModel.json", 350) => Some("ungrammatical 'de 18 a 19 el noviembre'"),
        ("Spanish", "DateTimeModel.json", 431) => {
            Some("'el dia 21 07:30-09:30' emits only the future month reading when day 21 is past")
        }
        ("Spanish", "DateTimeModel.json", 441) => {
            Some("standalone four-digit year '2015' in '32 - 2015' is recognized as a year mention")
        }
        ("Spanish", "DateTimeModel.json", 447) => {
            Some("space after colon in '8: 45 a.m.' splits the hour and minute tokens")
        }
        ("Spanish", "DateTimeModel.json", 458) => {
            Some("'inicios de 7 de enero' emits both past and future year readings")
        }
        ("Spanish", "DateTimeModel.json", 459) => {
            Some("typo in upstream input: 'empienzando' instead of Spanish 'empezando'")
        }
        ("Spanish", "DateTimeModel.json", 461) => {
            Some("typo in upstream input: '22 do febrero' instead of '22 de febrero'")
        }
        ("Spanish", "DateTimeModel.json", 462) => {
            Some("'a primeros de 7 de enero' resolved as 'Mod=after' range in upstream")
        }
        ("Spanish", "DateTimeModel.json", 466) => {
            Some("ctime-style timestamp 'Mie 26 de octubre 15:50:06 2016' with year after HH:MM:SS")
        }
        ("Spanish", "DateTimeModel.json", 626) => {
            Some("span inconsistency: upstream includes 'a las' in 'a las 8.30 de la noche hoy'")
        }
        ("Spanish", "DateTimeModel.json", 701) => {
            Some("upstream uses 22:00:00 instead of 21:59:59 for 'mas tarde en la noche' start")
        }
        ("Spanish", "DateTimeModel.json", 722) => Some("two-digit year after comma in '1.º de julio, 17'"),
        ("Spanish", "DateTimeModel.json", 725) => Some("'el 12 entre 0730-0930' without 'dia' before '12'"),
        ("Spanish", "DateTimeModel.json", 728 | 729) => {
            Some("ambiguous 'mar' (Tuesday vs. March vs. 'sea') at clause end")
        }
        ("Spanish", "DateTimeModel.json", 768 | 769) => {
            Some("span inconsistency: upstream excludes 'por la' in 'Regresare por la tarde/noche'")
        }
        ("Spanish", "DateTimeModel.json", 792 | 793) => {
            Some("clause boundary inside 'durante los ultimos 3 meses y 2 dias antes de hoy'")
        }
        ("Spanish", "DateTimeModel.json", 824) => {
            Some("ungrammatical '7 horas del proximo domingo a la tarde' for 7 PM")
        }

        // Dutch Number (159 / 161)
        ("Dutch", "NumberModel.json", 6) => Some(
            "IPv4 literal '192.168.1.2' is tokenized as four integers rather than two decimals '192.168' and '1.2'",
        ),
        ("Dutch", "NumberModel.json", 123) => {
            Some("upstream expects only '567,89' inside space-grouped '1 234 567,89', which the grammar parses in full")
        }

        // Dutch DateTime (579 / 605)
        ("Dutch", "DateTimeModel.json", 66) => {
            Some("'rond 11 uur' carries Mod=approx in the grammar whereas upstream omits Mod")
        }
        ("Dutch", "DateTimeModel.json", 70) => {
            Some("part-of-day prefix before spelled range \"'s middags 5 tot zeven\"")
        }
        ("Dutch", "DateTimeModel.json", 72) => {
            Some("upstream drops 'tot 7' in '4:00 tot 7 weg' whereas the grammar parses the time range")
        }
        ("Dutch", "DateTimeModel.json", 116) => {
            Some("bare weekday in 'woensdag vroeg op de dag' emits both past and future weekday readings")
        }
        ("Dutch", "DateTimeModel.json", 119) => {
            Some("ambiguous 'voor het jaar' ('for the year' vs. 'before the year')")
        }
        ("Dutch", "DateTimeModel.json", 136) => Some("apostrophe two-digit year in \"5-3-'18 om 17:49:19\""),
        ("Dutch", "DateTimeModel.json", 152) => Some("ambiguous 'voor de 9e van mei' ('for' vs. 'before' May 9th)"),
        ("Dutch", "DateTimeModel.json", 161 | 375) => {
            Some("span inconsistency: upstream excludes leading 'op' in 'op vorige week dinsdag ...'")
        }
        ("Dutch", "DateTimeModel.json", 190) => Some(
            "span inconsistency: upstream includes 'in' in #190 ('in 2014 tot en met 2018') but excludes it in #191-#192",
        ),
        ("Dutch", "DateTimeModel.json", 199) => {
            Some("typo in upstream input: 'onafhankelijksdag' instead of Dutch 'onafhankelijkheidsdag'")
        }
        ("Dutch", "DateTimeModel.json", 205) => {
            Some("'vanaf het middaguur vandaag tot het middaguur morgen' with trailing date after second 'middaguur'")
        }
        ("Dutch", "DateTimeModel.json", 230) => Some("span includes 'een tijdstip' before \"vanaf 7 uur 's morgens\""),
        ("Dutch", "DateTimeModel.json", 264) => {
            Some("upstream corpus bug: '10:00 tot 12:00 maandagmorgen' includes inverted 22:00-to-00:00 reading")
        }
        ("Dutch", "DateTimeModel.json", 274) => {
            Some("upstream includes evening 18:00-20:00 reading for 'volgende maandagmorgen 6-8'")
        }
        ("Dutch", "DateTimeModel.json", 275) => Some("ambiguous 'voor dec-2018' ('for' vs. 'before' December 2018)"),
        ("Dutch", "DateTimeModel.json", 354) => {
            Some("cross-meridiem range \"van half acht tot 4 uur 's middags\" applies afternoon to both bounds")
        }
        ("Dutch", "DateTimeModel.json", 378 | 379) => {
            Some("ambiguous 'maandag 12-4' / 'maandag 11-4' parsed as day-month rather than hour range")
        }
        ("Dutch", "DateTimeModel.json", 419..=421) => {
            Some("deictic 'deze'/'volgende' before slash/ordinal date emits concrete year TIMEX in the grammar")
        }
        ("Dutch", "DateTimeModel.json", 477) => {
            Some("dot-separated 'van 11.30 tot 12.30' emits both 12-hour and 24-hour readings")
        }
        ("Dutch", "DateTimeModel.json", 570) => Some(
            "span inconsistency: upstream excludes 'aan' in 'aan het begin van 2000' (#570) but includes it elsewhere",
        ),
        ("Dutch", "DateTimeModel.json", 737) => {
            Some("bare juxtaposed abbreviated months 'jan feb 2017' without range connector")
        }
        ("Dutch", "DateTimeModel.json", 738) => Some(
            "span inconsistency: upstream includes 'op' in #738 ('op dinsdag van volgende week') but excludes it in #503",
        ),

        _ => None,
    }
}

/// All case indices in `(language, file)` that have a documented failure
/// reason.
pub fn documented_failures(language: &str, file: &str) -> BTreeSet<usize> {
    (0..1024).filter(|&i| reason(language, file, i).is_some()).collect()
}

/// Asserts that no passing case in `(language, file)` has a stale
/// failure reason, and that every failing eligible case is documented.
pub fn verify(language: &str, file: &str, eligible: usize, passing: &BTreeSet<usize>) {
    let documented = documented_failures(language, file);
    let stale: Vec<&usize> = passing.intersection(&documented).collect();
    assert!(stale.is_empty(), "{language}/{file}: passing cases still have documented failure reasons: {stale:?}");
    assert_eq!(
        passing.len() + documented.len(),
        eligible,
        "{language}/{file}: {} eligible cases, {} passing, but {} documented failure reasons",
        eligible,
        passing.len(),
        documented.len()
    );
}
