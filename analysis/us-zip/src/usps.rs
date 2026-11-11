//! A caller's copy of the Postal Service's ZIP Code locale list, read as
//! tab separated text. The Postal Service's terms of use do not allow the
//! file to be redistributed, so it is never bundled: a caller downloads it
//! ("ZIP Codes by Area and District codes", `Zip_Locale_Detail.xlsx`, from
//! PostalPro), saves its `ZIP_Locale` sheet as tab separated text, and
//! reads that here.

use std::collections::HashMap;

use crate::TableError;
use crate::parse::digits;

/// A locale name and the USPS code of its office's state.
type Locale = (Box<str>, [u8; 2]);

const ZIP: &str = "DELIVERY ZIPCODE";
const LOCALE: &str = "LOCALE NAME";
const STATE: &str = "PHYSICAL STATE";

/// The locale names the Postal Service gives each delivery ZIP Code, the
/// name of the post office area its mail goes to (`SPRINGFIELD`), with
/// the state of that office.
#[derive(Debug, Default)]
pub struct UspsLocales {
    by_zip: HashMap<u32, Vec<Locale>>,
}

impl UspsLocales {
    /// Reads the `ZIP_Locale` sheet saved as tab separated text: a header
    /// row naming the `DELIVERY ZIPCODE`, `LOCALE NAME` and `PHYSICAL
    /// STATE` columns, then one row per locale. Fields are trimmed of the
    /// sheet's padding, and a ZIP Code a spreadsheet stored as a number
    /// (`601`) gets back its leading zeros. A ZIP Code listed under more
    /// than one locale keeps each distinct one, in file order.
    pub fn parse(text: &str) -> Result<UspsLocales, TableError> {
        let text = text.strip_prefix('\u{feff}').unwrap_or(text);
        let mut lines = text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty());
        let err = |line: usize, problem: String| TableError { table: "usps-locales", line, problem };
        let (_, header) = lines.next().ok_or_else(|| err(1, "no header row".into()))?;
        let col = |name: &str| {
            header.split('\t').position(|c| c.trim() == name).ok_or_else(|| err(1, format!("no {name} column")))
        };
        let (zc, lc, sc) = (col(ZIP)?, col(LOCALE)?, col(STATE)?);
        let mut out = UspsLocales::default();
        for (i, line) in lines {
            let n = i + 1;
            let f: Vec<&str> = line.split('\t').map(str::trim).collect();
            let get = |c: usize, what: &str| f.get(c).copied().ok_or_else(|| err(n, format!("no {what} field")));
            let zip = get(zc, ZIP)?;
            let code = (1..=5)
                .contains(&zip.len())
                .then(|| digits(zip, zip.len()))
                .flatten()
                .ok_or_else(|| err(n, format!("{ZIP} is not up to 5 ASCII digits: {zip:?}")))?;
            let locale = get(lc, LOCALE)?;
            if locale.is_empty() {
                return Err(err(n, format!("{LOCALE} is blank")));
            }
            let state = get(sc, STATE)?;
            let state: [u8; 2] = state
                .as_bytes()
                .try_into()
                .ok()
                .filter(|s: &[u8; 2]| s.iter().all(u8::is_ascii_uppercase))
                .ok_or_else(|| err(n, format!("{STATE} is not two capital letters: {state:?}")))?;
            let list = out.by_zip.entry(code).or_default();
            if !list.iter().any(|(l, s)| **l == *locale && *s == state) {
                list.push((locale.into(), state));
            }
        }
        Ok(out)
    }

    /// The number of ZIP Codes listed.
    pub fn len(&self) -> usize {
        self.by_zip.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_zip.is_empty()
    }

    /// The locales of a ZIP Code written as five ASCII digits or ZIP+4,
    /// as (locale name, USPS state code), in file order.
    pub fn locales(&self, zip: &str) -> impl Iterator<Item = (&str, &str)> {
        let list = crate::zip5(zip).and_then(|c| self.by_zip.get(&c));
        list.into_iter().flatten().map(|(l, s)| (&**l, std::str::from_utf8(s).unwrap_or("")))
    }
}
