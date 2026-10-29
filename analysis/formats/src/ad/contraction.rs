//! Portuguese contractions: `PortugueseContractionUtility`.
//!
//! Floresta splits a contraction into its parts, the left one tagged
//! `<sam->` and the right one `<-sam>` (`de` and `o` for `do`). The name
//! stream joins them again, since names keep their contractions.

/// The contraction of `left` and `right`, from OpenNLP's table
/// (`CONTRACTIONS`). The keys keep the table's spelling: `com+nòs` and
/// `com+vòs` are written with a grave accent there.
pub(crate) fn contract(left: &str, right: &str) -> Option<&'static str> {
    Some(match (left, right) {
        ("a", "a") => "\u{e0}",
        ("a", "as") => "\u{e0}s",
        ("a", "aquele") => "\u{e0}quele",
        ("a", "aqueles") => "\u{e0}queles",
        ("a", "aquela") => "\u{e0}quela",
        ("a", "aquelas") => "\u{e0}quelas",
        ("a", "aquilo") => "\u{e0}quilo",
        ("a", "o") => "ao",
        ("a", "os") => "aos",
        ("com", "mim") => "comigo",
        ("com", "n\u{f2}s") => "conosco",
        ("com", "si") => "consigo",
        ("com", "ti") => "contigo",
        ("com", "v\u{f2}s") => "convosco",
        ("de", "a\u{ed}") => "da\u{ed}",
        ("de", "algu\u{e9}m") => "dalgu\u{e9}m",
        ("de", "algum") => "dalgum",
        ("de", "alguma") => "dalguma",
        ("de", "alguns") => "dalguns",
        ("de", "algumas") => "dalgumas",
        ("de", "ali") => "dali",
        ("de", "aqu\u{e9}m") => "daqu\u{e9}m",
        ("de", "aquele") => "daquele",
        ("de", "aquela") => "daquela",
        ("de", "aqueles") => "daqueles",
        ("de", "aquelas") => "daquelas",
        ("de", "aqui") => "daqui",
        ("de", "aquilo") => "daquilo",
        ("de", "ele") => "dele",
        ("de", "ela") => "dela",
        ("de", "eles") => "deles",
        ("de", "elas") => "delas",
        ("de", "entre") => "dentre",
        ("de", "esse") => "desse",
        ("de", "essa") => "dessa",
        ("de", "esses") => "desses",
        ("de", "essas") => "dessas",
        ("de", "este") => "deste",
        ("de", "esta") => "desta",
        ("de", "estes") => "destes",
        ("de", "estas") => "destas",
        ("de", "isso") => "disso",
        ("de", "isto") => "disto",
        ("de", "o") => "do",
        ("de", "a") => "da",
        ("de", "os") => "dos",
        ("de", "as") => "das",
        ("de", "outrem") => "doutrem",
        ("de", "outro") => "doutro",
        ("de", "outra") => "doutra",
        ("de", "outros") => "doutros",
        ("de", "outras") => "doutras",
        ("de", "um") => "dum",
        ("de", "uma") => "duma",
        ("de", "uns") => "duns",
        ("de", "umas") => "dumas",
        ("esse", "outro") => "essoutro",
        ("essa", "outra") => "essoutra",
        ("este", "outro") => "estoutro",
        ("este", "outra") => "estoutra",
        ("ele", "o") => "lho",
        ("ele", "a") => "lha",
        ("ele", "os") => "lhos",
        ("ele", "as") => "lhas",
        ("em", "algum") => "nalgum",
        ("em", "alguma") => "nalguma",
        ("em", "alguns") => "nalguns",
        ("em", "algumas") => "nalgumas",
        ("em", "aquele") => "naquele",
        ("em", "aquela") => "naquela",
        ("em", "aqueles") => "naqueles",
        ("em", "aquelas") => "naquelas",
        ("em", "aquilo") => "naquilo",
        ("em", "ele") => "nele",
        ("em", "ela") => "nela",
        ("em", "eles") => "neles",
        ("em", "elas") => "nelas",
        ("em", "esse") => "nesse",
        ("em", "essa") => "nessa",
        ("em", "esses") => "nesses",
        ("em", "essas") => "nessas",
        ("em", "este") => "neste",
        ("em", "esta") => "nesta",
        ("em", "estes") => "nestes",
        ("em", "estas") => "nestas",
        ("em", "isso") => "nisso",
        ("em", "isto") => "nisto",
        ("em", "o") => "no",
        ("em", "a") => "na",
        ("em", "os") => "nos",
        ("em", "as") => "nas",
        ("em", "outro") => "noutro",
        ("em", "outra") => "noutra",
        ("em", "outros") => "noutros",
        ("em", "outras") => "noutras",
        ("em", "um") => "num",
        ("em", "uma") => "numa",
        ("em", "uns") => "nuns",
        ("em", "umas") => "numas",
        ("por", "o") => "pelo",
        ("por", "a") => "pela",
        ("por", "os") => "pelos",
        ("por", "as") => "pelas",
        ("para", "a") => "pra",
        ("para", "o") => "pro",
        ("para", "as") => "pras",
        ("para", "os") => "pros",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_has_its_spelling() {
        assert_eq!(contract("de", "o"), Some("do"));
        assert_eq!(contract("com", "n\u{f2}s"), Some("conosco"));
        assert_eq!(contract("com", "n\u{f3}s"), None);
        assert_eq!(contract("De", "o"), None);
    }
}
