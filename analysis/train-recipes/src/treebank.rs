//! The Universal Dependencies treebanks licensed CC BY 4.0 that the UD
//! recipes train on (issue #415), at release [`RELEASE`]: each
//! repository's `LICENSE.txt`, `README.md` and UD's own `train`, `dev`
//! and `test` files, with their sizes and digests, and the licence check
//! that reads `LICENSE.txt` and the README's metadata and refuses anything
//! but CC BY 4.0.
//!
//! The files are fetched through the zoo by digest from
//! `raw.githubusercontent.com` at the release tag, as a permissive dataset
//! (no acceptance is asked for), and are never checked in.
//!
//! Each licence was read in the repository's own `LICENSE.txt` and README
//! at the release, on 2026-10-14:
//!
//! - Spanish AnCora, Catalan AnCora, Portuguese Porttinari and Italian
//!   MarkIT: UD's CC BY 4.0 statement, and `License: CC BY 4.0` in the
//!   README's metadata. Both AnCora READMEs still have a paragraph that
//!   says a GNU licence is inherited from AnCora; their changelogs say the
//!   licence became CC BY 4.0 in release 2.9, and `LICENSE.txt` and the
//!   metadata say CC BY 4.0.
//! - Finnish FTB: `LICENSE.txt` offers LGPLv3 or later or, "at your
//!   option", CC BY 4.0, followed by the GPL's text, and the README's
//!   metadata says CC BY 4.0 ([`Grant::CcBy40OrOther`]). The recipes take
//!   CC BY 4.0.
//!
//! Portuguese DANTEStocks is CC BY 4.0 too, but its text is tweets
//! (`README.md`: "a collection of Brazilian Portuguese tweets"), which
//! the data policy keeps out with other data under a platform's terms, so
//! it has no recipe.

use std::path::{Path, PathBuf};

use turbo_zoo::{
    Acceptances, Artifact, File, Http, Index, Kind, Licence, Policy, Sha256, Sha512, TrainingData, UserId, Zoo,
    class_of, index::Format,
};

use crate::error::{Error, Result, io};

/// The Universal Dependencies release the recipes train on: the tag of
/// each repository.
pub const RELEASE: &str = "r2.16";
/// The treebanks' licence.
pub const LICENCE: Licence = Licence::CcBy40;
/// The README metadata line that states the licence.
pub const README_LICENCE_LINE: &str = "License: CC BY 4.0";

/// One file of a treebank repository at the release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TreebankFile {
    /// The file's name in the repository.
    pub name: &'static str,
    pub size: u64,
    pub sha256: &'static str,
    pub sha512: &'static str,
}

impl TreebankFile {
    /// Checks `bytes` against the recorded size, SHA-256 and SHA-512.
    fn check(&self, treebank: &Treebank, bytes: &[u8]) -> Result<()> {
        let sha256 = Sha256::of(bytes).to_string();
        let sha512 = Sha512::of(bytes).to_string();
        if bytes.len() as u64 == self.size && sha256 == self.sha256 && sha512 == self.sha512 {
            Ok(())
        } else {
            Err(Error::TreebankFile {
                treebank: treebank.repository,
                file: self.name,
                expected: format!("{} bytes, SHA-256 {}", self.size, self.sha256),
                found: format!("{} bytes, SHA-256 {sha256}", bytes.len()),
            })
        }
    }
}

/// How a treebank's `LICENSE.txt` grants CC BY 4.0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grant {
    /// CC BY 4.0 alone, in UD's statement: "The treebank is licensed under
    /// the Creative Commons License Attribution 4.0 International." The
    /// text names no other licence.
    CcBy40,
    /// Another licence or, at the user's option, CC BY 4.0; the recipes
    /// take CC BY 4.0. `other` names the other licence.
    CcBy40OrOther { other: &'static str },
}

/// One CC BY 4.0 treebank at the release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Treebank {
    /// The repository under `github.com/UniversalDependencies`.
    pub repository: &'static str,
    /// The language code the models are trained for.
    pub language: &'static str,
    /// The treebank's name in model ids (`ancora`).
    pub name: &'static str,
    pub grant: Grant,
    pub licence_file: TreebankFile,
    pub readme: TreebankFile,
    pub train: TreebankFile,
    pub dev: TreebankFile,
    pub test: TreebankFile,
    /// The contributors the README's metadata names.
    pub contributors: &'static str,
    /// The work the README asks to be cited, or, where it names none, the
    /// treebank as the README and `LICENSE.txt` describe it.
    pub citation: &'static str,
}

/// The text of a treebank's `train` and `test` files, read and checked.
/// The `dev` file is checked but not kept: no recipe reads it.
#[derive(Debug, Clone)]
pub struct TreebankText {
    pub train: String,
    pub test: String,
}

/// `s` lowercased with each run of characters other than ASCII letters
/// and digits made one space, with a space at both ends, so a phrase is
/// found as whole words.
fn words(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push(' ');
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with(' ') {
            out.push(' ');
        }
    }
    if !out.ends_with(' ') {
        out.push(' ');
    }
    out
}

/// Phrases of Creative Commons licence elements other than Attribution,
/// as [`words`] writes them: in a licence's name or address.
const RESTRICTED: [&str; 10] = [
    " noncommercial ",
    " non commercial ",
    " sharealike ",
    " share alike ",
    " noderivatives ",
    " noderivs ",
    " no derivatives ",
    " by nc ",
    " by sa ",
    " by nd ",
];

/// Checks the text of a treebank's `LICENSE.txt` and README against
/// `grant`: the licence text must name the Creative Commons Attribution
/// 4.0 International licence and its address, must name no NonCommercial,
/// ShareAlike or NoDerivatives element, and must name a GNU licence only
/// where `grant` offers another licence beside CC BY 4.0, and then "at
/// your option"; the README must have the metadata line
/// [`README_LICENCE_LINE`]. Returns why it refuses.
pub fn check_licence_text(grant: Grant, licence: &str, readme: &str) -> std::result::Result<(), &'static str> {
    let text = words(licence);
    if !(text.contains(" attribution 4 0 international ") && text.contains(" creativecommons org licenses by 4 0 ")) {
        return Err("LICENSE.txt does not grant the Creative Commons Attribution 4.0 International licence");
    }
    if RESTRICTED.iter().any(|p| text.contains(p)) {
        return Err("LICENSE.txt names a NonCommercial, ShareAlike or NoDerivatives licence element");
    }
    let gnu = text.contains(" gnu ") || text.contains(" general public license ");
    match grant {
        Grant::CcBy40 if gnu => return Err("LICENSE.txt names a GNU licence beside CC BY 4.0"),
        Grant::CcBy40OrOther { .. } if !text.contains(" alternatively at your option ") => {
            return Err("LICENSE.txt does not offer CC BY 4.0 at the user's option");
        }
        _ => {}
    }
    if !readme.lines().any(|l| l.trim() == README_LICENCE_LINE) {
        return Err("the README's metadata does not say License: CC BY 4.0");
    }
    Ok(())
}

impl Treebank {
    /// The five files, in the order the dataset lists them.
    pub fn files(&self) -> [TreebankFile; 5] {
        [self.licence_file, self.readme, self.train, self.dev, self.test]
    }

    /// A file's address at the release.
    pub fn url(&self, file: &TreebankFile) -> String {
        format!("https://raw.githubusercontent.com/UniversalDependencies/{}/{RELEASE}/{}", self.repository, file.name)
    }

    /// Where the licence is stated: `LICENSE.txt` at the release.
    pub fn licence_terms(&self) -> String {
        format!(
            "https://github.com/UniversalDependencies/{}/blob/{RELEASE}/{}",
            self.repository, self.licence_file.name
        )
    }

    /// The treebank and its release, as a model's training data names it.
    pub fn release_name(&self) -> String {
        format!("{} {RELEASE}", self.repository)
    }

    /// The zoo id of the dataset: the repository name lowercased with
    /// `-` for `_`, and the release (`ud-spanish-ancora-r2.16`).
    pub fn dataset_id(&self) -> String {
        format!("{}-{RELEASE}", self.repository.to_ascii_lowercase().replace('_', "-"))
    }

    /// The treebank as the zoo's permissive training data entry of a model.
    pub fn training_data(&self) -> TrainingData {
        TrainingData {
            name: self.release_name(),
            license: LICENCE,
            license_terms: self.licence_terms().parse().expect("a valid address"),
        }
    }

    /// What a model trained on the treebank credits, as CC BY asks: the
    /// treebank and its release, its contributors, where it is, its
    /// licence, and that the model is made from it.
    pub fn attribution(&self) -> String {
        format!(
            "Trained on {} of Universal Dependencies (https://github.com/UniversalDependencies/{}/tree/{RELEASE}), \
             by {}, under the Creative Commons Attribution 4.0 International licence \
             (https://creativecommons.org/licenses/by/4.0/). The model is derived from the treebank; the treebank \
             itself is not included.",
            self.release_name(),
            self.repository,
            self.contributors
        )
    }

    /// The release's files as a zoo dataset artifact: `LICENSE.txt`, the
    /// README and the three CoNLL-U files.
    ///
    /// # Errors
    ///
    /// A recorded field the zoo index refuses.
    pub fn dataset(&self) -> Result<Artifact> {
        let bad =
            |field: &'static str, value: String| move |message: String| Error::InvalidRelease { field, value, message };
        let id = self.dataset_id();
        let terms = self.licence_terms();
        let mut files = Vec::with_capacity(5);
        for f in self.files() {
            let url = self.url(&f);
            files.push(File {
                path: f.name.parse().map_err(bad("file", f.name.to_owned()))?,
                url: url.parse().map_err(bad("url", url.clone()))?,
                size: f.size,
                sha256: f.sha256.parse().map_err(bad("sha256", f.sha256.to_owned()))?,
                sha512: f.sha512.parse().map_err(bad("sha512", f.sha512.to_owned()))?,
            });
        }
        Ok(Artifact {
            id: id.parse().map_err(bad("id", id.clone()))?,
            kind: Kind::Dataset,
            component: None,
            language: Some(self.language.parse().map_err(bad("language", self.language.to_owned()))?),
            format: Format::Conllu,
            version: RELEASE.to_owned(),
            license: LICENCE,
            license_terms: terms.parse().map_err(bad("terms", terms.clone()))?,
            trained_on: Vec::new(),
            class: class_of([LICENCE]),
            origin: None,
            files,
        })
    }

    /// The licence check: [`check_licence_text`] on the bytes of
    /// `LICENSE.txt` and the README.
    ///
    /// # Errors
    ///
    /// [`Error::Licence`] with the reason.
    pub fn check_licence(&self, licence: &[u8], readme: &[u8]) -> Result<()> {
        let refuse = |reason: &'static str| Error::Licence { treebank: self.repository, reason };
        let licence = std::str::from_utf8(licence).map_err(|_| refuse("LICENSE.txt is not UTF-8"))?;
        let readme = std::str::from_utf8(readme).map_err(|_| refuse("the README is not UTF-8"))?;
        check_licence_text(self.grant, licence, readme).map_err(refuse)
    }

    /// Reads the treebank's files from `dir`: the licence check first,
    /// then every file against its recorded size and digests.
    ///
    /// # Errors
    ///
    /// A file that cannot be read, [`Error::Licence`], or
    /// [`Error::TreebankFile`] for a file that is not the recorded one.
    pub fn read(&self, dir: &Path) -> Result<TreebankText> {
        let read = |f: &TreebankFile| {
            let path = dir.join(f.name);
            std::fs::read(&path).map_err(io(&path))
        };
        let (licence, readme) = (read(&self.licence_file)?, read(&self.readme)?);
        self.check_licence(&licence, &readme)?;
        self.licence_file.check(self, &licence)?;
        self.readme.check(self, &readme)?;
        let text = |f: &TreebankFile| -> Result<String> {
            let bytes = read(f)?;
            f.check(self, &bytes)?;
            String::from_utf8(bytes).map_err(|e| Error::Io { path: dir.join(f.name), message: e.to_string() })
        };
        let train = text(&self.train)?;
        text(&self.dev)?;
        let test = text(&self.test)?;
        Ok(TreebankText { train, test })
    }

    /// Fetches the release's files through the zoo into
    /// `home/artifacts/<dataset id>/`, by digest, and returns that
    /// directory. CC BY 4.0 is permissive, so no acceptance is asked for.
    ///
    /// # Errors
    ///
    /// What the zoo fails on.
    pub fn fetch(&self, home: &Path, user: &UserId) -> Result<PathBuf> {
        let artifact = self.dataset()?;
        let id = artifact.id.to_string();
        let index = Index::parse(Index::to_canonical_json(std::slice::from_ref(&artifact)).as_bytes())?;
        let zoo = Zoo::new(index, home.join("artifacts"), Policy::default(), &Http);
        Ok(zoo.fetch_with(&id, user, || Ok(Acceptances::default()))?.dir)
    }
}

pub const SPANISH_ANCORA: Treebank = Treebank {
    repository: "UD_Spanish-AnCora",
    language: "es",
    name: "ancora",
    grant: Grant::CcBy40,
    licence_file: TreebankFile {
        name: "LICENSE.txt",
        size: 189,
        sha256: "653702c7fdee69f7d7a439c9ba4c178c26e745479f6e6d2943bed1a9b9c7bf67",
        sha512: "22412f70abfa24f53c5eaccf97e2660bc1091bb62a983294e533a967347118309a8952a6deafe8b79c6930b2b297e88a1cae269bf6d45d8b8f307f6adbef596d",
    },
    readme: TreebankFile {
        name: "README.md",
        size: 5623,
        sha256: "94041e7d42b86a189fc7866799e1119eea1d32ce02444c26b251d518548a7c1c",
        sha512: "a09395a7e62b9a5561d950b99e4c7d93ffa47fa7d843af20e0aab3418c740df97cd62b1bf5c9fb3dc11d8cd2c6c4c31be2780b186f055b91628c79b041326dd8",
    },
    train: TreebankFile {
        name: "es_ancora-ud-train.conllu",
        size: 42958918,
        sha256: "79e7463700e75bdf921c85b6e04495af118462a2b032bbff590fb82aa6ad8d5d",
        sha512: "9014bfc581e2f4140d92b6cbfa20c68939c53b830cecbb965d5b62ef45e1ed056c2babade29c63af0f8733d32d4be05a43bdc182d6e9adcdb298fe9da3541df2",
    },
    dev: TreebankFile {
        name: "es_ancora-ud-dev.conllu",
        size: 5104844,
        sha256: "e257745f4c339be53d5158867a28a2bfbcb98d890db86faa324e5f45f0284afc",
        sha512: "03c6e5e5511ecba1025f07bd3e55103dadb2e0dcd66de989913eb825c927808e5efd232c4000448beee7d0aefbf72b13d232e854e3781abadf025e033d055676",
    },
    test: TreebankFile {
        name: "es_ancora-ud-test.conllu",
        size: 5106509,
        sha256: "2a24316503a56e171f07b38cf2ec13c22e3b62a337c6dbf2d9d1a54ada2b012b",
        sha512: "9ff7e8cd44f95bc7821260bfda6e45444528547a0368cae484ae25e22b704d16d173c0b3af1174fd7d5b53104fd02d05d9763699f5672eda93660cb5743f81c1",
    },
    contributors: "Héctor Martínez Alonso and Daniel Zeman",
    citation: "Taulé, M., M.A. Martí, M. Recasens (2008) 'Ancora: Multilevel Annotated Corpora for Catalan and \
               Spanish', Proceedings of 6th International Conference on Language Resources and Evaluation. \
               Marrakesh (Morocco).",
};

pub const CATALAN_ANCORA: Treebank = Treebank {
    repository: "UD_Catalan-AnCora",
    language: "ca",
    name: "ancora",
    grant: Grant::CcBy40,
    licence_file: TreebankFile {
        name: "LICENSE.txt",
        size: 189,
        sha256: "653702c7fdee69f7d7a439c9ba4c178c26e745479f6e6d2943bed1a9b9c7bf67",
        sha512: "22412f70abfa24f53c5eaccf97e2660bc1091bb62a983294e533a967347118309a8952a6deafe8b79c6930b2b297e88a1cae269bf6d45d8b8f307f6adbef596d",
    },
    readme: TreebankFile {
        name: "README.md",
        size: 4726,
        sha256: "655c2547a513087644b94315fca88a38a105ae11a1673f7f701d1659194ac1e3",
        sha512: "c107336a2b8ecc33b8f29c6e8b689245531036f35e76670bd14e06fa52ed87e512a89ce6c6893ad50ab256e2b76ae8f93e72d06ced429c217ab5d11470219572",
    },
    train: TreebankFile {
        name: "ca_ancora-ud-train.conllu",
        size: 39714229,
        sha256: "bb5d06fb14336bae2c3aa64aa7e65afe183295a4669ce4f45bf0ed57c18ccccd",
        sha512: "ceffa3d603470e86640b6f2cbdbcedea0ac4582136eb01ea63169f6f30b985e889adc392677a6677d703caec74b0199dbf0a5f7de35f1d86bf25ebc1c6e8ee00",
    },
    dev: TreebankFile {
        name: "ca_ancora-ud-dev.conllu",
        size: 5421739,
        sha256: "bfcb95de30b6ea0b250619386a281add3e2bbd153804a89da4dbe2938ad28199",
        sha512: "be53e9a60a36f81b8a2eea14523c4d2cd2ab27c054be4a8bfde24a4f94d48df18420b1948ffdea0c994a8bdc97af85631b03a3a0f11880616c5972959d682481",
    },
    test: TreebankFile {
        name: "ca_ancora-ud-test.conllu",
        size: 5515247,
        sha256: "f68ff102b9bf8bc197be3262a1caf25682683c4db4c648663c7a791bfbfa115f",
        sha512: "f32af8db3d37698e1af3ff8d4a4940fafe3a28a7ae8d2ad46f9935f5224a01a852b04fb6cec63446f30af5ceaa7ca5e765cd8dd7c36e595a2328b6308d97c7a8",
    },
    contributors: "Héctor Martínez Alonso, Elena Pascual and Daniel Zeman",
    citation: "Taulé, M., M.A. Martí, M. Recasens (2008) 'Ancora: Multilevel Annotated Corpora for Catalan and \
               Spanish', Proceedings of 6th International Conference on Language Resources and Evaluation. \
               Marrakesh (Morocco).",
};

pub const PORTUGUESE_PORTTINARI: Treebank = Treebank {
    repository: "UD_Portuguese-Porttinari",
    language: "pt",
    name: "porttinari",
    grant: Grant::CcBy40,
    licence_file: TreebankFile {
        name: "LICENSE.txt",
        size: 188,
        sha256: "5cbfe2d4d12938f9961bf5a0028390f4b6b60749b075e228a5740848afcb4a9b",
        sha512: "10ef47a6ae80a0011f995aaf0ab6a40eded49199ed1bfe798d6a290fa3ad9f0ca24d30743775c916b12a73e0440538a5e962b9f1f50f0e7fbf09e52d12200475",
    },
    readme: TreebankFile {
        name: "README.md",
        size: 4619,
        sha256: "15b9172c825fefe9157d0942a793d9615368ae714e8c9e3b7bc9d8a1005bfed3",
        sha512: "14b47ed8d1e6adb78f674b63fbe16c832080d0dee06df23c443224f27d7146a071a94ce221fdb50695c1db4f866272aecd4ebb45945091532fe0d26f8aa4c40b",
    },
    train: TreebankFile {
        name: "pt_porttinari-ud-train.conllu",
        size: 8194835,
        sha256: "d9d3d76ac9ac8cb569c42a698cc60721e088db7c7e90401582aab68aa9ca85d3",
        sha512: "ff3711ad7dda691b4d50416bb14869ec5c3c3450d18433b8c7fe563e66ebbea7da59cea43ca057d4e3ed5099c388e889584ba47b21c0c3953a2fda681f894139",
    },
    dev: TreebankFile {
        name: "pt_porttinari-ud-dev.conllu",
        size: 1151582,
        sha256: "c5efbb16458c94d8803ebf8bb9f53f9a692c0a06c77d76326642ad4139a6cf09",
        sha512: "401b6740591dd06f7c7406cced3b8f64fde0d5ec58c746247aa95229d8178354d3a353fd1caf2be8226a59a4cc119b9db88d4dd9dc3d4258a39c0e9dac8190ff",
    },
    test: TreebankFile {
        name: "pt_porttinari-ud-test.conllu",
        size: 2333603,
        sha256: "08e51c3687f2800ea45445cb48c7030832984c316b1c39757606499b8e7577b9",
        sha512: "39d705b329663236e0d0ee529ce60d217956b3d667e9ed70e79a23063374c743da1204a0da06246f20f5fa638ae269745d991d8987c442793199b63b1fc67219",
    },
    contributors: "Magali Sanches Duran, Lucelene Lopes, Maria das Graças Volpe Nunes and Thiago Alexandre \
                   Salgueiro Pardo",
    citation: "Duran, M.S.; Lopes, L.; Nunes, M.G.V.; Pardo, T.A.S. (2023). The Dawn of the Porttinari Multigenre \
               Treebank: Introducing its Journalistic Portion. In the Proceedings of the 14th Symposium in \
               Information and Human Language Technology (STIL), pp. 115-124.",
};

pub const FINNISH_FTB: Treebank = Treebank {
    repository: "UD_Finnish-FTB",
    language: "fi",
    name: "ftb",
    grant: Grant::CcBy40OrOther { other: "LGPL-3.0-or-later" },
    licence_file: TreebankFile {
        name: "LICENSE.txt",
        size: 44134,
        sha256: "492fc0da8e9e5beecf59f5740e90482a256c8b59da6a46766aa778475969c507",
        sha512: "2cde368f977c49d493ed1f2696d53aede83ad1100cab5b6f9341d25de8f7ffdedb1d2bb95ad60827bb990887512b147b8fae65aab16b94fd710fbc431656a202",
    },
    readme: TreebankFile {
        name: "README.md",
        size: 5531,
        sha256: "9bb99c1912f08475aa6344a8399d2b86db99f9c6f36fc4548ce08c33ee7b59ce",
        sha512: "2c6b4504ef19320fb6dd5d43d3cf461e6374754d5ac72ca93f6870cfba123e6b07a5b790bebdea062e7660f2e6157e6b3eab848b6ee5739d78fb27baed066184",
    },
    train: TreebankFile {
        name: "fi_ftb-ud-train.conllu",
        size: 10505459,
        sha256: "0cc04932f18613c4491ace06f597b7b8caed36e557168b9b6ef154da7ae2d71b",
        sha512: "3d15a8473440c42960254438f761608afdbb6072918aba145d83095c876611c2e3fdab0bab8ff920309ddee09ece3746873433b3ca7a6185db267aebbeee6d5d",
    },
    dev: TreebankFile {
        name: "fi_ftb-ud-dev.conllu",
        size: 1296909,
        sha256: "1e46f7f88b6f1fa4f39b010b4ac8b8664b1be2c899f531f21b7bdc3fcf2e9142",
        sha512: "ca0469394a38185b00a5acd20cd6feda94820849edfcdd4317c79728b4cf212ebd365529e1cc8d903711e9ff4d7ddb44cb4a0ddbcdab7885b74f38eca294c590",
    },
    test: TreebankFile {
        name: "fi_ftb-ud-test.conllu",
        size: 1342602,
        sha256: "3aca7fbd727846c0955e9db581571a0fca89d46f825ee7f2a32fab1ea0bc31ba",
        sha512: "cfa0708711a3d8deea4720ea36e82a6387f8946a465269a53dc63bfba49698d2108ae9bd55bd9b72ed1b1670cb0e2041d646aba08552b37330b596e0b4c7ba65",
    },
    contributors: "Jussi Piitulainen, Hanna Nurmi and Jack Rueter",
    citation: "FTB1U, the UD version of FinnTreeBank 1 2014, Copyright 2014 FIN-CLARIN: grammatical examples \
               adapted from Iso suomen kielioppi, verkkoversio (VISK, http://scripta.kotus.fi/visk), annotated in \
               FIN-CLARIN projects at the Department of Modern Languages, University of Helsinki. The README names \
               no paper to cite.",
};

pub const ITALIAN_MARKIT: Treebank = Treebank {
    repository: "UD_Italian-MarkIT",
    language: "it",
    name: "markit",
    grant: Grant::CcBy40,
    licence_file: TreebankFile {
        name: "LICENSE.txt",
        size: 188,
        sha256: "5cbfe2d4d12938f9961bf5a0028390f4b6b60749b075e228a5740848afcb4a9b",
        sha512: "10ef47a6ae80a0011f995aaf0ab6a40eded49199ed1bfe798d6a290fa3ad9f0ca24d30743775c916b12a73e0440538a5e962b9f1f50f0e7fbf09e52d12200475",
    },
    readme: TreebankFile {
        name: "README.md",
        size: 1955,
        sha256: "3efbd63f62359b2be61af8822c6dfd27af67f93c184fb1a9018208267e726732",
        sha512: "0f26b24181605ddd54644385445fde8397d6b89cecadfd815932ee01f68c17a9da452a5090390a146e1d035aff639ef67325da06306d1ab58f5227104c86d179",
    },
    train: TreebankFile {
        name: "it_markit-ud-train.conllu",
        size: 1266080,
        sha256: "4f6fc3423acb00d49e1e167bd21b11ebbc23e22275fd6f5eeb4d75cfbbccc5c6",
        sha512: "7c6243a80f7a147360cf21fd03a8bc01f90b47d4230b3c22a203e4b3db6d96acf52ba99be3e27205e0ae73c94d12719f42f386af533243f8b1e79c320df9fedf",
    },
    dev: TreebankFile {
        name: "it_markit-ud-dev.conllu",
        size: 665430,
        sha256: "633b6186f47ecfebfcff4ff9c7e19a0cbffd1f655f14fc1e9265b12bda597f97",
        sha512: "85976b63e421e917d8c5ddfe99f1417b1dc7383b36c5fdbd4e42339b6b9301c708888d165767d601de58e45f7b5386944f6d561b85ea4ec699048c3fec9c79dd",
    },
    test: TreebankFile {
        name: "it_markit-ud-test.conllu",
        size: 658412,
        sha256: "9048ed0064e3989e2c37377755a28c25c7e462b1ad916e5808dc0a7ce95fc3fc",
        sha512: "ce160a7f7077c00d1a5111167bf5d1e8d5767208dc0f174f82160bf7c6e7b19666340f95a97d2df4746de2ce8f37d892f022656344f8d1f2d31ac3915aeba732",
    },
    contributors: "Teresa Paccosi, Alessio Palmero Aprosio and Sara Tonelli",
    citation: "It is MarkIT That is New: An Italian Treebank of Marked Constructions. Teresa Paccosi, Alessio \
               Palmero Aprosio and Sara Tonelli, Proceedings of the Eighth Italian Conference on Computational \
               Linguistics 2022 (CLIC-it 2021).",
};

/// The treebanks with recipes.
pub const TREEBANKS: [Treebank; 5] =
    [SPANISH_ANCORA, CATALAN_ANCORA, PORTUGUESE_PORTTINARI, FINNISH_FTB, ITALIAN_MARKIT];

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_zoo::LicenceClass;

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
        let option = Grant::CcBy40OrOther { other: "LGPL-3.0-or-later" };
        assert_eq!(check_licence_text(option, dual, README), Ok(()));
        assert!(check_licence_text(Grant::CcBy40, dual, README).is_err());
        let both = dual.replace("Alternatively, at your option, it", "It");
        assert!(check_licence_text(option, &both, README).is_err());
    }

    #[test]
    fn the_readme_must_say_cc_by_4_0() {
        for readme in ["License: CC BY-SA 4.0\n", "# Summary\n", "Licence: CC BY 4.0\n"] {
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
        }
        assert_eq!(SPANISH_ANCORA.dataset_id(), "ud-spanish-ancora-r2.16");
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
}
