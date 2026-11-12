//! The padding, validation and contract cases of the ECB rate table, and
//! the edge cases of the port.

use super::*;

// EcbFxRatesPaddingTest

const HEADER: &str = "Date,USD,JPY\n";
const RECORD: &str = "2026-07-14,1.25,150\n";

#[test]
fn corrupted_field() {
    let cases = [
        format!("Date,\u{0}USD,JPY\n{RECORD}"),
        format!("Date,USD\u{0},JPY\n{RECORD}"),
        format!("Date,USD,JPY,\u{0}\n{RECORD}"),
        format!("{HEADER}\u{0}2026-07-14,1.25,150\n"),
        format!("{HEADER}2026-07-14\u{0},1.25,150\n"),
        format!("{HEADER}2026-07-14,\u{0}1.25,150\n"),
        format!("{HEADER}2026-07-14,1.25\u{0},150\n"),
        format!("{HEADER}2026-07-14,1.25,\u{0}N/A\n"),
        format!("{HEADER}2026-07-14,1.25,N/A\u{0}\n"),
        format!("{HEADER}2026-07-14,1.25,\u{0}\n"),
        format!("{HEADER}2026-07-14,1.25,150,\u{0}\n"),
    ];
    for csv in cases {
        assert!(EcbRates::parse(&csv).is_err(), "{csv:?}");
    }
}

#[test]
fn non_whitespace_padding() {
    for c in ['\u{0}', '\u{1}', '\u{7}', '\u{8}', '\u{E}', '\u{1B}', '\u{7F}', '\u{85}', '\u{200B}', '\u{FEFF}'] {
        assert!(EcbRates::parse(&format!("Date,{c}USD{c},JPY\n{RECORD}")).is_err(), "{c:?}");
    }
}

#[test]
fn whitespace_padding() {
    for c in ['\u{9}', '\u{B}', '\u{C}', '\u{1C}', '\u{1D}', '\u{1E}', '\u{1F}', ' '] {
        let p = c.to_string();
        let csv = format!("Date,{p}USD{p},{p}JPY{p},{p}\r\n{p}2026-07-14{p},{p}1.25{p},{p}N/A{p},{p}\r\n{p}\r\n");
        let r = load(&csv);
        assert_rate("1.25", rate(&r, "EUR", "USD", DATE));
        assert!(rate(&r, "EUR", "JPY", DATE).is_none());
    }
}

#[test]
fn unsupported_whitespace_padding() {
    for c in ['\u{A0}', '\u{2003}', '\u{202F}'] {
        assert!(EcbRates::parse(&format!("Date,{c}USD,JPY\n{RECORD}")).is_err(), "{c:?}");
    }
}

// EcbFxRatesValidationTest

#[test]
fn invalid_header() {
    for header in [
        "Date,",
        "Date,,",
        "Date,USD,USD",
        "Date,USD, USD ",
        "Date,USD,,JPY",
        "Date,usd",
        "Date,US",
        "Date,USDX",
        "Date,US1",
        "Date,US\u{301}",
        "Date,\u{20AC}UR",
    ] {
        let mut columns = header.matches(',').count();
        if header.ends_with(',') {
            columns -= 1;
        }
        let csv = format!("{header}\n2026-07-14{}\n", ",1.2".repeat(columns));
        assert!(EcbRates::parse(&csv).is_err(), "{header:?}");
    }
}

#[test]
fn invalid_column_count() {
    for row in
        ["2026-07-14", "2026-07-14,1.2", "2026-07-14,1.2,150,extra", "2026-07-14,1.2,150,,", "2026-07-14,1.2,150, , "]
    {
        let e = EcbRates::parse(&format!("Date,USD,JPY,\n{row}\n")).unwrap_err();
        assert!(matches!(e, EcbError::ColumnCount { columns: 3, .. }), "{row:?}: {e}");
    }
}

#[test]
fn duplicate_date() {
    for second in ["1.2", "1.3", "N/A"] {
        let e = EcbRates::parse(&format!("Date,USD\n2026-07-14,1.2\n2026-07-14,{second}\n")).unwrap_err();
        assert_eq!(e.to_string(), "duplicate reference history date: 2026-07-14");
    }
}

#[test]
fn invalid_rate() {
    for value in ["0", "0.0000", "-0", "-0.0000", "0E+20", "-1", "-0.001", "NaN", "Infinity", "1.2x"] {
        let e = EcbRates::parse(&format!("Date,USD\n2026-07-14,{value}\n")).unwrap_err();
        let message = e.to_string();
        assert!(message.contains("USD") && message.contains("2026-07-14"), "{value:?}: {message}");
    }
}

#[test]
fn invalid_base_rate() {
    for value in ["0.9", "1.1", "2"] {
        let e = EcbRates::parse(&format!("Date,EUR,USD\n2026-07-14,{value},1.2\n")).unwrap_err();
        assert!(matches!(e, EcbError::EuroNotOne { .. }), "{value:?}: {e}");
    }
}

#[test]
fn supported_layout() {
    for csv in [
        "Date,USD,JPY,\n2026-07-14,1.2,N/A,\n",
        "Date,USD,JPY\n2026-07-14,1.2,N/A\n",
        "Date,USD,JPY,\n2026-07-14,1.2,\n",
        "Date,USD,JPY\n2026-07-14,1.2,,\n",
        "Date, USD , JPY ,\r\n2026-07-14, 1.2 , ,\r\n",
        "Date,USD,EEK\n2026-07-14,1.2,N/A\n",
        "Date,EUR,USD\n2026-07-14,1.0000,1.2\n",
    ] {
        let r = load(csv);
        assert_rate("1.2", rate(&r, "EUR", "USD", DATE));
        assert!(rate(&r, "EUR", "JPY", DATE).is_none(), "{csv:?}");
    }
}

#[test]
fn positive_decimal_forms() {
    for value in ["1E-30", "1E+30", "+1.25", ".125"] {
        let r = load(&format!("Date,USD\n2026-07-14,{value}\n"));
        assert_rate(value, rate(&r, "EUR", "USD", DATE));
    }
}

#[test]
fn malformed_utf8() {
    for suffix in [false, true] {
        let mut bytes = if suffix { b"Date,USD\n2026-07-14,1.2\n".to_vec() } else { b"Date,US".to_vec() };
        let at = bytes.len();
        bytes.push(0xC3);
        assert!(matches!(EcbRates::from_bytes(&bytes), Err(EcbError::Utf8 { offset }) if offset == at));
    }
}

#[test]
fn path_loading() {
    let name = format!("turbo-numeric-ecb-{}", std::process::id());
    let dir = turbo_scratch::Scratch::at(std::env::temp_dir().join(name));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("rates.csv");
    std::fs::write(&path, "Date,USD\n2026-07-14,1.2\n").unwrap();
    assert_rate("1.2", EcbRates::load(&path).unwrap().rate(cur("EUR"), cur("USD"), day(DATE)).unwrap());
    assert!(matches!(EcbRates::load(dir.join("missing.csv")), Err(EcbError::Io(_))));
}

// FxRatesContractTest

/// A provider with a fixed rate that counts its calls.
struct Fixed {
    rate: Option<Decimal>,
    calls: Cell<u32>,
}

impl FxRates for Fixed {
    type Error = ScaleRangeError;

    fn rate(&self, from: Currency, to: Currency, as_of: Date) -> Result<Option<Decimal>, ScaleRangeError> {
        self.calls.set(self.calls.get() + 1);
        assert_eq!((from, to, as_of), (cur("EUR"), cur("USD"), day(DATE)));
        Ok(self.rate.clone())
    }
}

#[test]
fn valid_conversion_through_the_default_method() {
    let provider = Fixed { rate: Some(dec("1.25")), calls: Cell::new(0) };
    let converted = convert(&provider, &dec("-10.125"), "EUR", "USD", day(DATE)).unwrap();
    assert_eq!(provider.calls.get(), 1);
    // BigDecimal.equals: the value and the scale.
    assert_eq!(converted, dec("-12.65625"));
    let missing = Fixed { rate: None, calls: Cell::new(0) };
    assert!(convert(&missing, &dec("-10.125"), "EUR", "USD", day(DATE)).is_none());
}

// Edge cases of the port.

#[test]
fn rates_written_in_other_digits_and_signs() {
    let r = load("Date,USD,JPY\n2026-07-14,\u{0661}.\u{0662}\u{0665},+150\n");
    assert_rate("1.25", rate(&r, "EUR", "USD", DATE));
    assert_rate("150", rate(&r, "EUR", "JPY", DATE));
    // A digit outside the Basic Multilingual Plane is two Java chars.
    assert!(EcbRates::parse("Date,USD\n2026-07-14,\u{1D7D9}\n").is_err());
}

#[test]
fn an_explicit_euro_column_is_ignored_for_lookups() {
    let r = load("Date,EUR,USD\n2026-07-14,N/A,1.2\n");
    assert_rate("1.2", rate(&r, "EUR", "USD", DATE));
    assert_rate("1", rate(&r, "EUR", "EUR", DATE));
}

#[test]
fn blank_lines_and_terminators() {
    let r = load("Date,USD\r\r\n\u{2003}\n2026-07-14,1.2\r2026-07-07,1.1");
    assert_rate("1.2", rate(&r, "EUR", "USD", DATE));
    assert_rate("1.1", rate(&r, "EUR", "USD", "2026-07-13"));
    let dates: Vec<String> = r.dates().map(|d| d.to_string()).collect();
    assert_eq!(dates, ["2026-07-07", "2026-07-14"]);
    assert_eq!(r.currencies(), [cur("USD")]);
    // A no-break space is not blank: the line has one field.
    assert!(matches!(EcbRates::parse("Date,USD\n\u{A0}\n2026-07-14,1.2\n"), Err(EcbError::ColumnCount { .. })));
}

#[test]
fn errors_name_the_problem() {
    let message = |csv: &str| EcbRates::parse(csv).unwrap_err().to_string();
    assert_eq!(message("Date,USD,USD\n"), "duplicate reference history currency code: USD");
    assert_eq!(message("Date,US\n"), "invalid reference history currency code: US");
    assert_eq!(message("Date,\n"), "reference history header must name a currency");
    assert_eq!(message("\u{FEFF}Date,USD\n"), "not a reference history CSV: header is \u{FEFF}Date,USD");
    assert_eq!(message("Date,USD\n2026-07-14,\u{7}1\n"), "invalid control character at CSV offset 11");
    assert_eq!(
        message("Date,USD\n2026-07-14,-1\n"),
        "reference history rate must be positive for USD in row: 2026-07-14,-1"
    );
    assert_eq!(message("Date,EUR\n2026-07-14,2\n"), "reference history EUR rate must equal 1 in row: 2026-07-14,2");
    assert_eq!(message("Date,USD\n2026-07-14\n"), "expected 2 reference history columns in row: 2026-07-14");
}

#[test]
fn the_first_error_in_reading_order_wins() {
    // The bad rate comes before the repeated date is seen.
    let e = EcbRates::parse("Date,USD\n2026-07-14,1\n2026-07-14,x\n").unwrap_err();
    assert!(matches!(e, EcbError::InvalidRate { .. }), "{e}");
    let e = EcbRates::parse("Date,USD\n2026-07-14,1\n2026-07-14,1\n2026-07-15,x\n").unwrap_err();
    assert!(matches!(e, EcbError::DuplicateDate(_)), "{e}");
}
