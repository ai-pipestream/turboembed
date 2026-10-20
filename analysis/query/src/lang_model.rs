//! Compact, self-contained in-memory [`SpanDetector`] for short queries and
//! mixed-language clauses, covering Latin languages (`deu`, `eng`, `fra`,
//! `ita`, `nld`, `por`, `spa`) and script-distinct languages (`ara`, `ell`,
//! `heb`, `jpn`, `kor`, `rus`, `zho`).

use turbo_langdetect::{LanguageDetector, LanguageModel};
use turbo_langspan::decode::SwitchCost;
use turbo_langspan::{Settings, SpanDetector};
use turbo_maxent::{ModelFile, crc32};

/// Outcomes of the built-in query language model, in alphabetical order.
const OUTCOMES: &[&str] =
    &["ara", "deu", "ell", "eng", "fra", "heb", "ita", "jpn", "kor", "nld", "por", "rus", "spa", "zho"];

const IDX_DEU: u32 = 1;
const IDX_ENG: u32 = 3;
const IDX_FRA: u32 = 4;
const IDX_ITA: u32 = 6;
const IDX_NLD: u32 = 9;
const IDX_POR: u32 = 10;
const IDX_SPA: u32 = 12;

#[rustfmt::skip]
const PREDICATES: &[(&str, &[(u32, f64)])] = &[
    // English trigrams & short markers
    ("the", &[(IDX_ENG, 3.0)]), ("and", &[(IDX_ENG, 2.5)]), ("for", &[(IDX_ENG, 2.2)]),
    ("fli", &[(IDX_ENG, 3.0)]), ("lig", &[(IDX_ENG, 2.5)]), ("igh", &[(IDX_ENG, 3.2)]),
    ("ght", &[(IDX_ENG, 3.2)]), ("hts", &[(IDX_ENG, 2.8)]), ("che", &[(IDX_ENG, 2.4)]),
    ("hea", &[(IDX_ENG, 2.5)]), ("eap", &[(IDX_ENG, 3.0)]), ("tom", &[(IDX_ENG, 2.5)]),
    ("omo", &[(IDX_ENG, 2.5)]), ("orr", &[(IDX_ENG, 2.8)]), ("row", &[(IDX_ENG, 3.0)]),
    ("yes", &[(IDX_ENG, 2.8)]), ("day", &[(IDX_ENG, 2.8)]), ("wee", &[(IDX_ENG, 2.8)]),
    ("eek", &[(IDX_ENG, 2.8)]), ("und", &[(IDX_ENG, 2.0), (IDX_DEU, 2.5)]),
    ("der", &[(IDX_ENG, 1.8), (IDX_DEU, 2.4)]), ("wit", &[(IDX_ENG, 2.6)]),
    ("ith", &[(IDX_ENG, 2.8)]), ("fro", &[(IDX_ENG, 2.6)]), ("rom", &[(IDX_ENG, 2.4)]),
    ("nex", &[(IDX_ENG, 2.8)]), ("ext", &[(IDX_ENG, 2.6)]), (" to", &[(IDX_ENG, 2.2)]),
    ("to ", &[(IDX_ENG, 2.2)]), (" in", &[(IDX_ENG, 1.5)]),
    // Spanish trigrams & diacritic n-grams
    ("vue", &[(IDX_SPA, 3.5)]), ("uel", &[(IDX_SPA, 3.5)]), ("elo", &[(IDX_SPA, 3.2)]),
    ("los", &[(IDX_SPA, 3.2)]), ("bar", &[(IDX_SPA, 2.5)]), ("ara", &[(IDX_SPA, 2.8)]),
    ("rat", &[(IDX_SPA, 2.5)]), ("ato", &[(IDX_SPA, 3.0)]), ("tos", &[(IDX_SPA, 3.2)]),
    ("mañ", &[(IDX_SPA, 4.0)]), ("aña", &[(IDX_SPA, 4.0)]), ("ñan", &[(IDX_SPA, 4.0)]),
    ("ana", &[(IDX_SPA, 2.5)]), ("ñ", &[(IDX_SPA, 4.0)]), ("par", &[(IDX_SPA, 2.4)]),
    ("por", &[(IDX_SPA, 2.4), (IDX_POR, 2.4)]), ("con", &[(IDX_SPA, 2.2), (IDX_ITA, 2.0)]),
    ("las", &[(IDX_SPA, 3.0)]), ("del", &[(IDX_SPA, 2.8), (IDX_ITA, 2.2)]),
    ("una", &[(IDX_SPA, 2.8), (IDX_ITA, 2.2)]), ("hoy", &[(IDX_SPA, 3.5)]),
    ("aye", &[(IDX_SPA, 3.2)]), ("yer", &[(IDX_SPA, 3.0)]), ("sem", &[(IDX_SPA, 2.8)]),
    ("ema", &[(IDX_SPA, 2.6)]), ("tar", &[(IDX_SPA, 2.4)]), ("ard", &[(IDX_SPA, 2.4)]),
    ("rde", &[(IDX_SPA, 2.8)]),
    // French trigrams
    ("les", &[(IDX_FRA, 3.2)]), ("des", &[(IDX_FRA, 3.0)]), ("pou", &[(IDX_FRA, 3.0)]),
    ("our", &[(IDX_FRA, 2.4)]), ("dem", &[(IDX_FRA, 3.0)]), ("mai", &[(IDX_FRA, 2.8)]),
    ("ain", &[(IDX_FRA, 2.8)]), ("vol", &[(IDX_FRA, 2.8)]), ("ols", &[(IDX_FRA, 2.8)]),
    ("pas", &[(IDX_FRA, 2.5)]),
    // German trigrams
    ("flü", &[(IDX_DEU, 4.0)]), ("lüg", &[(IDX_DEU, 4.0)]), ("üge", &[(IDX_DEU, 4.0)]),
    ("nac", &[(IDX_DEU, 3.2)]), ("ach", &[(IDX_DEU, 3.2)]),
    ("mor", &[(IDX_DEU, 2.8), (IDX_NLD, 2.6)]), ("org", &[(IDX_DEU, 2.8), (IDX_NLD, 2.6)]),
    ("rge", &[(IDX_DEU, 2.8), (IDX_NLD, 2.6)]), ("gen", &[(IDX_DEU, 2.8), (IDX_NLD, 2.6)]),
    ("bil", &[(IDX_DEU, 2.6)]), ("lli", &[(IDX_DEU, 2.6)]), ("ige", &[(IDX_DEU, 2.6)]),
    // Dutch trigrams
    ("vlu", &[(IDX_NLD, 3.5)]), ("uch", &[(IDX_NLD, 3.2)]), ("cht", &[(IDX_NLD, 3.0)]),
    ("ten", &[(IDX_NLD, 2.6)]), ("naa", &[(IDX_NLD, 3.5)]), ("aar", &[(IDX_NLD, 3.2)]),
    ("van", &[(IDX_NLD, 3.0)]), ("het", &[(IDX_NLD, 3.2)]),
];

/// Builds a compact, self-contained [`SpanDetector`] tuned for short queries
/// and mixed-language clauses without requiring external model files.
pub fn default_span_detector() -> SpanDetector {
    let encoded: Vec<GisPredicate> =
        PREDICATES.iter().map(|(name, params)| (name.encode_utf16().collect(), params.to_vec())).collect();
    let classifier = build_gis_bytes(OUTCOMES, &encoded);
    let manifest = "Language=und\nfactory=opennlp.tools.langdetect.LanguageDetectorFactory\n\
Component-Name=LanguageDetectorME\nOpenNLP-Version=3.0.0-SNAPSHOT\nManifest-Version=1.0\n";
    let archive = build_zip(&[("manifest.properties", manifest.as_bytes()), ("langdetect.model", &classifier)]);
    let model = LanguageModel::open(ModelFile::open(archive).expect("valid in-memory GIS zip")).expect("valid model");
    let detector = LanguageDetector::new(model);
    let settings = Settings {
        switch: SwitchCost { space: 0.15, punctuation: 0.08, sentence: 0.04 },
        clause_letters: 4,
        rescore_margin: 0.0,
        shortlist: false,
    };
    SpanDetector::new(detector, settings)
}

type GisPredicate = (Vec<u16>, Vec<(u32, f64)>);

fn build_gis_bytes(outcomes: &[&str], predicates: &[GisPredicate]) -> Vec<u8> {
    let mut out = Vec::new();
    let write_utf = |buf: &mut Vec<u8>, units: &[u16]| {
        let mut mod_utf8 = Vec::with_capacity(units.len() * 2);
        for &u in units {
            turbo_unicode::modified_utf8::push_modified_utf8(&mut mod_utf8, u32::from(u));
        }
        buf.extend_from_slice(&(mod_utf8.len() as u16).to_be_bytes());
        buf.extend_from_slice(&mod_utf8);
    };
    let write_str = |buf: &mut Vec<u8>, s: &str| {
        let units: Vec<u16> = s.encode_utf16().collect();
        write_utf(buf, &units);
    };
    write_str(&mut out, "GIS");
    out.extend_from_slice(&1i32.to_be_bytes());
    out.extend_from_slice(&1.0f64.to_bits().to_be_bytes());
    out.extend_from_slice(&(outcomes.len() as i32).to_be_bytes());
    for o in outcomes {
        write_str(&mut out, o);
    }
    out.extend_from_slice(&(predicates.len() as i32).to_be_bytes());
    for (_, params) in predicates {
        let pat: Vec<String> =
            std::iter::once("1".to_owned()).chain(params.iter().map(|(o, _)| o.to_string())).collect();
        write_str(&mut out, &pat.join(" "));
    }
    out.extend_from_slice(&(predicates.len() as i32).to_be_bytes());
    for (name, _) in predicates {
        write_utf(&mut out, name);
    }
    for (_, params) in predicates {
        for &(_, w) in params {
            out.extend_from_slice(&w.to_bits().to_be_bytes());
        }
    }
    out
}

fn build_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut z = Vec::new();
    let mut central = Vec::new();
    for &(name, data) in entries {
        let offset = z.len() as u32;
        let head = |sig: u32, out: &mut Vec<u8>, is_central: bool| {
            out.extend_from_slice(&sig.to_le_bytes());
            if is_central {
                out.extend_from_slice(&20u16.to_le_bytes());
            }
            out.extend_from_slice(&[20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            out.extend_from_slice(&crc32(data).to_le_bytes());
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
        };
        head(0x0403_4b50, &mut z, false);
        z.extend_from_slice(name.as_bytes());
        z.extend_from_slice(data);
        head(0x0201_4b50, &mut central, true);
        central.extend_from_slice(&[0; 10]);
        central.extend_from_slice(&offset.to_le_bytes());
        central.extend_from_slice(name.as_bytes());
    }
    let dir = z.len() as u32;
    z.extend_from_slice(&central);
    z.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    let n = entries.len() as u16;
    z.extend_from_slice(&[0, 0, 0, 0]);
    z.extend_from_slice(&n.to_le_bytes());
    z.extend_from_slice(&n.to_le_bytes());
    z.extend_from_slice(&(central.len() as u32).to_le_bytes());
    z.extend_from_slice(&dir.to_le_bytes());
    z.extend_from_slice(&[0, 0]);
    z
}
