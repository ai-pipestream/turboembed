use turbo_zoo::LicenceClass;

use super::*;

const UD_STATEMENT: &str = "The treebank is licensed under the Creative Commons License Attribution 4.0 \
                            International.\n\nThe complete license text is available at:\n\
                            https://creativecommons.org/licenses/by/4.0/legalcode\n";
const README: &str = "# Summary\n\n<pre>\n=== Machine-readable metadata ===\nLicense: CC BY 4.0\n</pre>\n";

#[test]
fn ud_s_cc_by_statement_passes() {
    assert_eq!(check_licence_text(Grant::CcBy40, UD_STATEMENT, README), Ok(()));
    assert_eq!(check_licence_text(Grant::CcBy40, &UD_STATEMENT.replace("https", "http"), README), Ok(()));
}

#[test]
fn other_creative_commons_licences_are_refused() {
    let refused = [
        "This work is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License. \
         https://creativecommons.org/licenses/by-sa/4.0/",
        "The treebank is released under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 licence \
         (CC BY-NC-SA 3.0). https://creativecommons.org/licenses/by-nc-sa/3.0/",
        "Creative Commons Attribution-NoDerivatives 4.0 International, https://creativecommons.org/licenses/by-nd/4.0/",
        "Creative Commons Attribution 3.0 Unported, https://creativecommons.org/licenses/by/3.0/",
        "GNU General Public License v3",
        "",
    ];
    for text in refused {
        assert!(check_licence_text(Grant::CcBy40, text, README).is_err(), "{text}");
    }
    // CC BY 4.0 named beside a ShareAlike element is refused too.
    let both = format!("{UD_STATEMENT}\nSome files: CC BY-SA 4.0.");
    assert!(check_licence_text(Grant::CcBy40, &both, README).is_err());
}

#[test]
fn a_gnu_licence_passes_only_as_an_option_beside_cc_by() {
    let dual = "You may use it under the GNU Lesser General Public License, version 3 or later.\n\n\
                Alternatively, at your option, it is available under the Creative Commons Attribution 4.0 \
                International (CC BY 4.0) license. See <http://creativecommons.org/licenses/by/4.0/>.\n\n\
                GNU GENERAL PUBLIC LICENSE ... only occasionally and noncommercially ...";
    let option = Grant::CcBy40OrOther { other: Licence::Lgpl30OrLater };
    assert_eq!(check_licence_text(option, dual, README), Ok(()));
    assert!(check_licence_text(Grant::CcBy40, dual, README).is_err());
    let both = dual.replace("Alternatively, at your option, it", "It");
    assert!(check_licence_text(option, &both, README).is_err());
    // A dual-licence grant without a GNU licence in LICENSE.txt or with another licence variant is refused.
    let without_gnu = "Alternatively, at your option, Creative Commons Attribution 4.0 International \
                       (https://creativecommons.org/licenses/by/4.0/).";
    assert!(check_licence_text(option, without_gnu, README).is_err());
    assert!(check_licence_text(Grant::CcBy40OrOther { other: Licence::Apache2 }, dual, README).is_err());
}

#[test]
fn the_readme_must_say_cc_by_4_0() {
    for readme in [
        "License: CC BY-SA 4.0\n",
        "# Summary\n",
        "Licence: CC BY 4.0\n",
        "License: CC BY 4.0\nLicense: CC BY-NC-SA 4.0\n",
        "License: CC BY 4.0\nLicence: GPL-3.0\n",
    ] {
        assert!(check_licence_text(Grant::CcBy40, UD_STATEMENT, readme).is_err(), "{readme}");
    }
}

#[test]
fn the_treebanks_are_recorded_as_permissive_datasets_at_the_release() {
    for t in TREEBANKS {
        let a = t.dataset().unwrap();
        assert_eq!(a.class, LicenceClass::Permissive);
        assert_eq!(a.computed_class(), LicenceClass::Permissive);
        assert_eq!(a.license, Licence::CcBy40);
        assert_eq!(a.files.len(), 5);
        for (f, recorded) in a.files.iter().zip(t.files()) {
            assert_eq!(
                f.url.as_str(),
                format!(
                    "https://raw.githubusercontent.com/UniversalDependencies/{}/r2.16/{}",
                    t.repository, recorded.name
                )
            );
        }
        assert_eq!(t.train.name, format!("{}_{}-ud-train.conllu", t.language, t.name));
        assert_eq!(t.dev.name, format!("{}_{}-ud-dev.conllu", t.language, t.name));
        assert_eq!(t.test.name, format!("{}_{}-ud-test.conllu", t.language, t.name));
        assert_eq!(t.training_data().license, Licence::CcBy40);
        assert!(t.attribution().contains(t.repository) && t.attribution().contains("by/4.0"));
        assert!(!t.citation.is_empty() && !t.contributors.is_empty());
        // Only Porttinari's text is another's with no licence shown.
        let settled = t.text_rights == TextRights::Granted;
        assert_eq!(settled, t.repository != "UD_Portuguese-Porttinari", "{}", t.repository);
        assert_eq!(Index::builtin().get(&t.dataset_id()).ok(), settled.then_some(&a), "{}", t.repository);
        // Only FTB's LICENSE.txt gives a copyright notice, which the
        // attribution keeps.
        assert_eq!(t.copyright.is_some(), t.repository == "UD_Finnish-FTB", "{}", t.repository);
    }
    assert_eq!(SPANISH_ANCORA.dataset_id(), "ud-spanish-ancora-r2.16");
    assert!(FINNISH_FTB.attribution().contains(
        "(https://creativecommons.org/licenses/by/4.0/). FTB1U is Copyright 2014 FIN-CLARIN. The model is derived"
    ));
    assert!(SPANISH_ANCORA.attribution().contains("(https://creativecommons.org/licenses/by/4.0/). The model"));
    assert_eq!(FINNISH_FTB.grant, Grant::CcBy40OrOther { other: Licence::Lgpl30OrLater });
    assert!(matches!(PORTUGUESE_PORTTINARI.text_rights, TextRights::NotSettled { source } if source.contains("Folha")));
}

/// UD's CC BY 4.0 statement as Porttinari's and MarkIT's
/// `LICENSE.txt` have it at r2.16, with an `http` address.
const UD_STATEMENT_HTTP: &str = "The treebank is licensed under the Creative Commons License Attribution 4.0 \
                                 International.\n\nThe complete license text is available at:\n\
                                 http://creativecommons.org/licenses/by/4.0/legalcode\n";

/// FTB's `LICENSE.txt` at r2.16, up to the GPL 3's text: its own
/// statement. The texts of the GPL 3 and the LGPL 3 follow it in the
/// file (44,134 bytes in all), and are not quoted.
const FTB_STATEMENT: &str = "FTB1U is derived from FinnTreeBank 1 2014 by automated mapping of
labels and some restructuring in an attempt to conform to the new
Universal Dependency Finnish model.

FTB1U is Copyright 2014 FIN-CLARIN

FTB1U is Free Software. You may use, study, modify, and distribute it
under the conditions of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version (LGPLv3+). You should
have received a copy of the license with FTB1U. See
<http://www.gnu.org/licenses/>.

Alternatively, at your option, FTB1U is a Free Cultural Work which you
are free to share and adapt for any purpose, even commercially, under
the Creative Commons Attribution 4.0 International (CC BY 4.0) license
as long as you give proper attribution and apply no additional legal
restrictions. See the human-readable deed and actual license at
<http://creativecommons.org/licenses/by/4.0/>.

Unannotated linguistic material in FinnTreeBank 1 2014 was adapted
from the examples in Ison suomen kieliopin verkkoversio (The Web
Version of the Large Grammar of Finnish, VISK), available on-line as
<http://scripta.kotus.fi/visk>, and annotations originally produced
and further revised in FIN-CLARIN projects in the Department of Modern
Languages, University of Helsinki.
";

/// The licence lines of both AnCora READMEs at r2.16: the paragraph
/// left from before release 2.9, the changelog's entry for it, and the
/// metadata line.
const ANCORA_README: &str = "The GNU license is inherited from the original dataset, downloaded from
the AnCora website. Any license-related questions have to be directed to
the original data providers at the University of Barcelona (that is, not
to the UD contact address listed at the end of this README file).

* 2021-11-15 v2.9
  * The license changed to CC BY 4.0 (https://doi.org/10.5281/zenodo.4762030).

License: CC BY 4.0
";

/// Each treebank's `LICENSE.txt` and README licence lines at r2.16,
/// quoted as they are, pass the licence check with the recorded grant.
/// The four whole `LICENSE.txt` files quoted are the recorded ones, by
/// size and SHA-256. The AnCora READMEs' GNU paragraph does not fail
/// the check: the check reads licence names in `LICENSE.txt` and only
/// the metadata line in the README.
#[test]
fn the_treebanks_real_licence_texts_pass_with_their_recorded_grants() {
    let metadata = "License: CC BY 4.0\n";
    let texts = [
        (SPANISH_ANCORA, UD_STATEMENT, ANCORA_README),
        (CATALAN_ANCORA, UD_STATEMENT, ANCORA_README),
        (PORTUGUESE_PORTTINARI, UD_STATEMENT_HTTP, metadata),
        (FINNISH_FTB, FTB_STATEMENT, metadata),
        (ITALIAN_MARKIT, UD_STATEMENT_HTTP, metadata),
    ];
    assert_eq!(texts.map(|(t, ..)| t), TREEBANKS);
    for (t, licence, readme) in texts {
        assert_eq!(check_licence_text(t.grant, licence, readme), Ok(()), "{}", t.repository);
        if t.grant == Grant::CcBy40 {
            assert_eq!(licence.len() as u64, t.licence_file.size, "{}", t.repository);
            assert_eq!(Sha256::of(licence.as_bytes()).to_string(), t.licence_file.sha256, "{}", t.repository);
        }
    }
    // FTB's statement names the GNU licence, so it passes only as the
    // option it is.
    assert!(check_licence_text(Grant::CcBy40, FTB_STATEMENT, metadata).is_err());
    assert!(FTB_STATEMENT.len() < FINNISH_FTB.licence_file.size as usize);
}

#[test]
fn a_file_that_is_not_the_recorded_one_is_refused() {
    let dir = std::env::temp_dir().join(format!("turbo-train-recipes-treebank-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("LICENSE.txt"), UD_STATEMENT).unwrap();
    std::fs::write(dir.join("README.md"), README).unwrap();
    // The texts pass the licence check but are not the recorded files.
    assert!(matches!(
        ITALIAN_MARKIT.read(&dir),
        Err(Error::TreebankFile { file: "LICENSE.txt", treebank: "UD_Italian-MarkIT", .. })
    ));
    std::fs::write(dir.join("LICENSE.txt"), "Creative Commons Attribution-ShareAlike 4.0").unwrap();
    assert!(matches!(ITALIAN_MARKIT.read(&dir), Err(Error::Licence { .. })));
    std::fs::remove_dir_all(&dir).unwrap();
}
