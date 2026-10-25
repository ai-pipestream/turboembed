use super::*;

fn tagdict_case(name: &str, g: &Golden) -> PosDictionary {
    match name {
        "fixture" => tag_dictionary(g).unwrap(),
        "case-sensitive" => {
            PosDictionary::new(true, tagged(&[("Run", &["NN", "VB"]), ("run", &["VBP"]), ("RUN", &["NNP"])]))
        }
        "collapsed" => PosDictionary::new(
            false,
            tagged(&[("Run", &["NN", "VB"]), ("run", &["VBP"]), ("RUN", &["NNP"]), ("Paris", &["NNP"])]),
        ),
        "special-characters" => PosDictionary::new(
            true,
            tagged(&[
                ("a&b<c>d\"e'f", &["X"]),
                ("tab\there", &["Y"]),
                ("cr\rhere", &["Z"]),
                ("\u{1F600}.", &["SYM"]),
                ("q", &["A\"B", "C<D&E"]),
                ("none", &[]),
            ]),
        ),
        "empty" => PosDictionary::new(false, Vec::new()),
        "forty" => PosDictionary::new(false, (1..=40).map(|i| (format!("w{i}"), vec!["T".to_string()]))),
        "unicode-case" => PosDictionary::new(
            false,
            tagged(&[
                ("\u{130}stanbul", &["NNP"]),
                ("\u{3A3}\u{39F}\u{3A6}\u{39F}\u{3A3}", &["NNP"]),
                ("Stra\u{DF}e", &["NN"]),
                ("\u{1C5}emal", &["NNP"]),
            ]),
        ),
        other => panic!("{other}"),
    }
}

pub(super) fn check_tagdict_cases(path: &Path, g: &Golden) -> usize {
    let Some(l) = common::layer_opt(g, "turbo:tagdict-cases") else {
        return 0;
    };
    let (names, xmls) = (l.strs("name").unwrap(), l.strs("xml").unwrap());
    for i in 0..l.len() {
        let name = names.get(i);
        let expected = common::lossless_text(xmls.get(i));
        let actual = String::from_utf8(tagdict_case(name, g).serialize().unwrap()).unwrap();
        assert_eq!(expected, actual, "{}: tag dictionary case {name}", path.display());
    }
    l.len()
}
