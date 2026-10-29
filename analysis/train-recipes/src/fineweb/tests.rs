use std::collections::HashSet;

use super::*;
use crate::langdetect::LANGUAGES;

#[test]
fn the_web_languages_are_the_recipe_s_in_its_order() {
    let codes: Vec<&str> = WEB_LANGUAGES.iter().map(|l| l.tatoeba).collect();
    let recipe: Vec<&str> = LANGUAGES.iter().map(|p| p.tatoeba).collect();
    assert_eq!(codes, recipe);
    let mut configs = HashSet::new();
    for l in &WEB_LANGUAGES {
        assert!(!l.sources.is_empty(), "{}", l.tatoeba);
        for s in l.sources {
            assert!(configs.insert(format!("{s:?}")), "{} repeats {s:?}", l.tatoeba);
            if let Source::FineWeb2(c) = s {
                let (code, script) = c.split_once('_').unwrap();
                assert!(code.len() == 3 && code.bytes().all(|b| b.is_ascii_lowercase()), "{c}");
                assert!(script.len() == 4 && script.starts_with(|c: char| c.is_ascii_uppercase()), "{c}");
                assert!(!script.ends_with("_removed"), "{c}");
            }
        }
    }
    assert_eq!(WEB_LANGUAGES.iter().filter(|l| l.sources == [Source::FineWeb]).count(), 1);
}

#[test]
fn documents_are_shared_among_sources() {
    let wants = WEB.wants();
    assert_eq!(wants.len(), 2 * 102);
    let srp: Vec<usize> =
        wants.iter().filter(|w| WEB_LANGUAGES[w.language].tatoeba == "srp").map(|w| w.documents).collect();
    assert_eq!(srp, [500, 500, 250, 250]);
    let odd = WebRecipe { split: WebSplit { training: 7, held_out: 3, ..WEB_SPLIT }, ..WEB };
    let uzb: Vec<(Side, usize)> = odd
        .wants()
        .iter()
        .filter(|w| WEB_LANGUAGES[w.language].tatoeba == "uzb")
        .map(|w| (w.side, w.documents))
        .collect();
    assert_eq!(uzb, [(Side::Training, 4), (Side::Training, 3), (Side::HeldOut, 2), (Side::HeldOut, 1)]);
    // Every training want comes before every held-out one.
    let first_held_out = wants.iter().position(|w| w.side == Side::HeldOut).unwrap();
    assert!(wants[first_held_out..].iter().all(|w| w.side == Side::HeldOut));
    assert_eq!(WEB_SPLIT.host_limit(1_000), 50);
    assert_eq!(WEB_SPLIT.host_limit(250), 12);
    assert_eq!(WEB_SPLIT.host_limit(3), 1);
    let zero_share = WebSplit { host_share: 0, ..WEB_SPLIT };
    assert_eq!(zero_share.host_limit(25), 25);
    assert_eq!(zero_share.host_limit(0), 1);
}

#[test]
fn hosts_and_windows() {
    assert_eq!(host("https://www.Example.org/a/b?c#d"), "example.org");
    assert_eq!(host("http://user:pw@de.wikipedia.org:8080/wiki"), "de.wikipedia.org");
    assert_eq!(host("http://[2001:db8::1]:80/x"), "[2001:db8::1]");
    assert_eq!(host("http://[2001:db8::1]/x"), "[2001:db8::1]");
    assert_eq!(host("example.com/path"), "example.com");
    assert_eq!(host("tatoeba.org/redirect?to=https://example.org/x"), "tatoeba.org");
    assert!(excluded_host(&host("tatoeba.org/redirect?to=https://example.org/x")));
    assert_eq!(host("//Example.com./x"), "example.com");
    assert_eq!(host("https://Example.com.../x"), "example.com");
    assert_eq!(host("https://ÜBER.example/"), "über.example");
    assert_eq!(host(""), "");
    assert!(excluded_host("tatoeba.org") && excluded_host("en.tatoeba.org"));
    assert!(!excluded_host("nottatoeba.org") && !excluded_host("tatoeba.org.example"));
    assert_eq!(window("héllo wörld", 5), "héllo");
    assert_eq!(window("short", 300), "short");
    assert_eq!(window("日本語のテキスト", 3), "日本語");
}

#[test]
fn span_and_overlap_boundary_cases() {
    let inverted = Span { start: 20, end: 10 };
    assert_eq!(inverted.len(), 0);
    assert!(inverted.is_empty());

    let o = Overlap::new(
        2,
        16,
        [
            (0, "Ein Satz über München."),
            (0, "zu kurz"),
            (1, "another language's sentence"),
            (99, "out of bounds language index sentence"),
        ],
    );
    assert!(o.found(0, "Er sagte: Ein Satz über München. Und mehr."));
    assert!(o.found(0, "Ein Satz über München."));
    assert!(!o.found(0, "Ein Satz über München"));
    assert!(!o.found(0, "ein satz über münchen."));
    // Too short to be looked for.
    assert!(!o.found(0, "zu kurz, und noch etwas mehr"));
    assert!(!o.found(0, "another language's sentence"));
    assert!(o.found(1, "and another language's sentence"));
    assert!(!o.found(99, "out of bounds language index sentence"));
    assert!(!Overlap::none(2).found(0, "Ein Satz über München."));
}

#[test]
fn lists_are_read_strictly() {
    let sha = "a".repeat(64);
    let line =
        format!("{}\t{}\tdata/afr_Latn/train/000_00000.parquet\t100\t{sha}\n", FINEWEB_2.repo, FINEWEB_2.revision);
    let files = parse_files("f", &format!("# header\n\n{line}")).unwrap();
    assert_eq!(files[0].size, 100);
    assert_eq!(parse_files("f", &write_files(&files)).unwrap(), files);
    let e = parse_files("f", &format!("{line}{line}")).unwrap_err();
    assert_eq!(e.to_string(), format!("f:2: {} data/afr_Latn/train/000_00000.parquet is listed twice", FINEWEB_2.repo));
    let e = parse_files("f", &line.replace(FINEWEB_2.revision, "main")).unwrap_err();
    assert!(e.to_string().starts_with("f:1: HuggingFaceFW/fineweb-2 at main, not the recipe's revision"), "{e}");
    let e = parse_files("f", &line.replace("\t100\t", "\t0100\t")).unwrap_err();
    assert_eq!(e.to_string(), "f:1: the size \"0100\" is not a number");
    let e = parse_files("f", &line.replace(&sha, &"A".repeat(64))).unwrap_err();
    assert!(e.to_string().contains("not a SHA-256 in lowercase hex"), "{e}");
    let e = parse_files("f", "a\tb\n").unwrap_err();
    assert_eq!(e.to_string(), "f:1: 2 fields, not 5");

    let r = format!("{}\tdata/x.parquet\t10\t20\t{sha}\n", FINEWEB.repo);
    let ranges = parse_ranges("r", &r).unwrap();
    assert_eq!(ranges[0].span, Span { start: 10, end: 20 });
    assert_eq!(parse_ranges("r", &write_ranges(&ranges)).unwrap(), ranges);
    let e = parse_ranges("r", &r.replace("\t20\t", "\t10\t")).unwrap_err();
    assert_eq!(e.to_string(), "r:1: the range 10-10 is empty");
    let e = parse_ranges("r", &format!("{r}{r}")).unwrap_err();
    assert!(e.to_string().ends_with("is listed twice"), "{e}");
}

#[test]
fn listed_ranges_must_be_the_ones_made() {
    let r = |start, end| WebRange {
        repo: FINEWEB.repo.to_owned(),
        path: "p".to_owned(),
        span: Span { start, end },
        sha256: "0".repeat(64),
    };
    assert_eq!(check_ranges(&[r(0, 4), r(4, 8)], &[r(0, 4), r(4, 8)]), Ok(()));
    let e = check_ranges(&[r(0, 4)], &[r(0, 4), r(4, 8)]).unwrap_err();
    assert_eq!(e.to_string(), "fineweb-ranges.txt: 1 ranges are listed, and the files give 2");
    let zeros = "0".repeat(64);
    let e = check_ranges(&[r(0, 5)], &[r(0, 4)]).unwrap_err();
    let want =
        format!("range 1 is HuggingFaceFW/fineweb p 0-5 {zeros}, the files give HuggingFaceFW/fineweb p 0-4 {zeros}");
    assert!(e.to_string().contains(&want), "{e}");
    // A range that differs only in its digest names both.
    let other = WebRange { sha256: "1".repeat(64), ..r(0, 4) };
    let e = check_ranges(&[other], &[r(0, 4)]).unwrap_err();
    assert!(e.to_string().contains(&format!("p 0-4 {}, the files give", "1".repeat(64))), "{e}");
}

#[test]
fn the_committed_lists_are_read() {
    let files = parse_files(FILES_NAME, FILES).unwrap();
    let ranges = parse_ranges(RANGES_NAME, RANGES).unwrap();
    // The files are exactly the ones the recipe reads, and each has
    // its three ranges.
    assert_eq!((files.len(), ranges.len()), (204, 612));
    files_of(&WEB.wants(), &files).unwrap();
}
