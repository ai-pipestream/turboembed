//! The feature columns of the published MeCab dictionaries, by format, and
//! the reading, pronunciation and lemma each one has.
//!
//! A MeCab dictionary entry is `surface,left id,right id,cost` followed by
//! feature columns whose meaning the dictionary decides. OpenNLP keeps the
//! features as one string. Here each format has an enum of its columns in
//! file order, so the column a reading comes from is a name, checked by the
//! compiler, and [`DictionaryFormat::map`] says which of them hold the
//! reading, the pronunciation and the lemma. The format is given, never
//! guessed; [`DictionaryFormat::check`] refuses a dictionary whose lexicon
//! entries do not have the format's number of columns.

use std::fmt;

use super::dictionary::{EntryId, MecabDictionary};

/// The feature columns of IPADIC (mecab-ipadic 2.7.0), in file order.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum IpadicColumn {
    /// 品詞: the part of speech.
    PartOfSpeech,
    /// 品詞細分類1 to 3: its subcategories.
    Subcategory1,
    Subcategory2,
    Subcategory3,
    /// 活用型: the conjugation type.
    ConjugationType,
    /// 活用形: the conjugated form.
    ConjugationForm,
    /// 原形: the base form.
    BaseForm,
    /// 読み: the reading, in katakana.
    Reading,
    /// 発音: the pronunciation, in katakana.
    Pronunciation,
}

impl IpadicColumn {
    /// The number of columns of a lexicon entry.
    pub const COUNT: usize = 9;
}

/// The feature columns of mecab-ko-dic (2.1.1), in file order.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum KoDicColumn {
    /// 품사 태그: the part of speech tag (Sejong), `+` joined for an
    /// inflected entry.
    Tag,
    /// 의미 부류: the semantic class.
    SemanticClass,
    /// 종성 유무: `T` or `F`, whether the last syllable has a final
    /// consonant.
    FinalConsonant,
    /// 읽기: the reading, in Hangul.
    Reading,
    /// 타입: `Inflect`, `Compound`, `Preanalysis` or `*`.
    Type,
    /// 첫번째 품사 and 마지막 품사: the first and last tag of a
    /// decomposed entry.
    FirstTag,
    LastTag,
    /// 표현: the decomposition, `morpheme/tag/semantic class` joined by `+`.
    Expression,
}

impl KoDicColumn {
    /// The number of columns of a lexicon entry.
    pub const COUNT: usize = 8;
}

/// The feature columns every UniDic layout starts with, in file order. The
/// names are UniDic's.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum UnidicColumn {
    /// `pos1` to `pos4`: the part of speech and its subcategories.
    Pos1,
    Pos2,
    Pos3,
    Pos4,
    /// `cType`: the conjugation type.
    ConjugationType,
    /// `cForm`: the conjugated form.
    ConjugationForm,
    /// `lForm`: the reading of the lexeme, in katakana.
    LexemeReading,
    /// `lemma`: the lexeme, in its standard spelling, with an English
    /// gloss after `-` for some loanwords (`テスト-test`).
    Lexeme,
    /// `orth`: the surface as written.
    Orthography,
    /// `pron`: the pronunciation of the surface, in katakana.
    Pronunciation,
    /// `orthBase`: the base form as written.
    OrthographyBase,
    /// `pronBase`: the pronunciation of the base form.
    PronunciationBase,
    /// `goshu`: the word's origin (和, 漢, 外, ...).
    WordOrigin,
    /// `iType`, `iForm`, `fType`, `fForm`: the initial and final sound
    /// changes.
    InitialChangeType,
    InitialChangeForm,
    FinalChangeType,
    FinalChangeForm,
}

impl UnidicColumn {
    /// The number of columns every layout has.
    pub const COUNT: usize = 17;
}

/// The column layouts UniDic for MeCab is published in. They share the
/// first 17 columns ([`UnidicColumn`]) and differ after them.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum UnidicLayout {
    /// The 17 columns of the 2.1.2 source distribution
    /// (`unidic-mecab-2.1.2_src`), which has no kana column.
    Source17,
    /// The 26 columns of the 2.1.2 binary distribution (and unidic-lite):
    /// after the 17, `kana`, `kanaBase`, `form`, `formBase`, `iConType`,
    /// `fConType`, `aType`, `aConType`, `aModType`.
    Full26,
    /// The 29 columns of 2.2.0 and later: after the 17, `iConType`,
    /// `fConType`, `type`, `kana`, `kanaBase`, `form`, `formBase`,
    /// `aType`, `aConType`, `aModType`, `lid`, `lemma_id`.
    Full29,
}

impl UnidicLayout {
    /// The number of columns of a lexicon entry.
    pub const fn count(self) -> usize {
        match self {
            UnidicLayout::Source17 => 17,
            UnidicLayout::Full26 => 26,
            UnidicLayout::Full29 => 29,
        }
    }

    /// The `kana` column: the surface as read, in katakana.
    pub const fn kana(self) -> Option<usize> {
        match self {
            UnidicLayout::Source17 => None,
            UnidicLayout::Full26 => Some(17),
            UnidicLayout::Full29 => Some(20),
        }
    }
}

/// The format of a MeCab dictionary's feature columns.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DictionaryFormat {
    Ipadic,
    MecabKoDic,
    Unidic(UnidicLayout),
}

/// Which feature columns of a format hold the reading, the pronunciation
/// and the lemma, and how many columns a lexicon entry has.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FeatureMap {
    pub columns: usize,
    pub reading: Option<usize>,
    pub pronunciation: Option<usize>,
    pub lemma: Option<usize>,
}

impl DictionaryFormat {
    /// The columns of the format.
    ///
    /// - IPADIC: 読み, 発音 and 原形.
    /// - mecab-ko-dic: 읽기 only; it has no lemma or pronunciation column.
    /// - UniDic: `kana` (none in the 17-column layout), `pron` and
    ///   `orthBase`. The lemma is `orthBase`, the base form as written, as
    ///   IPADIC's 原形 is; UniDic's `lemma` column is the lexeme in a
    ///   standard spelling (為る for する) and is left in the features.
    pub const fn map(self) -> FeatureMap {
        match self {
            DictionaryFormat::Ipadic => FeatureMap {
                columns: IpadicColumn::COUNT,
                reading: Some(IpadicColumn::Reading as usize),
                pronunciation: Some(IpadicColumn::Pronunciation as usize),
                lemma: Some(IpadicColumn::BaseForm as usize),
            },
            DictionaryFormat::MecabKoDic => FeatureMap {
                columns: KoDicColumn::COUNT,
                reading: Some(KoDicColumn::Reading as usize),
                pronunciation: None,
                lemma: None,
            },
            DictionaryFormat::Unidic(layout) => FeatureMap {
                columns: layout.count(),
                reading: layout.kana(),
                pronunciation: Some(UnidicColumn::Pronunciation as usize),
                lemma: Some(UnidicColumn::OrthographyBase as usize),
            },
        }
    }

    /// The format's name, for messages.
    pub const fn name(self) -> &'static str {
        match self {
            DictionaryFormat::Ipadic => "IPADIC",
            DictionaryFormat::MecabKoDic => "mecab-ko-dic",
            DictionaryFormat::Unidic(UnidicLayout::Source17) => "UniDic (17 columns)",
            DictionaryFormat::Unidic(UnidicLayout::Full26) => "UniDic (26 columns)",
            DictionaryFormat::Unidic(UnidicLayout::Full29) => "UniDic (29 columns)",
        }
    }

    /// Checks that every lexicon entry of `dictionary` has the format's
    /// number of feature columns. The `unk.def` templates are not checked:
    /// they usually have fewer, and the columns they lack are absent.
    pub fn check(self, dictionary: &MecabDictionary) -> Result<(), FormatError> {
        let want = self.map().columns;
        for i in 0..dictionary.lexicon_len() {
            let entry = EntryId(i as u32);
            let found = dictionary.feature_count(entry);
            if found != want {
                return Err(FormatError { format: self, entry, found });
            }
        }
        Ok(())
    }
}

/// A lexicon entry whose number of feature columns is not its format's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FormatError {
    pub format: DictionaryFormat,
    pub entry: EntryId,
    pub found: usize,
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "a lexicon entry has {} feature columns where {} has {}",
            self.found,
            self.format.name(),
            self.format.map().columns
        )
    }
}

impl std::error::Error for FormatError {}

/// The value of column `column` of `entry`: `None` when the format has no
/// such column, the entry has fewer columns (an `unk.def` template), or
/// the value is empty or `*`, which the published dictionaries write for
/// none.
#[inline]
pub(crate) fn value(dictionary: &MecabDictionary, entry: EntryId, column: Option<usize>) -> Option<&str> {
    dictionary.feature(entry, column?).filter(|v| !v.is_empty() && *v != "*")
}

impl FeatureMap {
    /// The reading of `entry`, if the format and the entry have one.
    pub fn reading<'d>(&self, dictionary: &'d MecabDictionary, entry: EntryId) -> Option<&'d str> {
        value(dictionary, entry, self.reading)
    }

    /// The pronunciation of `entry`, if the format and the entry have one.
    pub fn pronunciation<'d>(&self, dictionary: &'d MecabDictionary, entry: EntryId) -> Option<&'d str> {
        value(dictionary, entry, self.pronunciation)
    }

    /// The lemma of `entry`, if the format and the entry have one.
    pub fn lemma<'d>(&self, dictionary: &'d MecabDictionary, entry: EntryId) -> Option<&'d str> {
        value(dictionary, entry, self.lemma)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::{DictionaryFiles, DictionaryLimits, LatticeTokenizer};
    use turbo_unicode::WhitespaceMode;

    /// One unknown character a morpheme.
    const CHAR_DEF: &str = "DEFAULT 0 0 1\n";

    fn dictionary(lexicon: &str, unk: &str) -> MecabDictionary {
        let lexicons = [("lex.csv", lexicon.as_bytes())];
        let files = DictionaryFiles {
            matrix: b"1 1\n0 0 0\n",
            lexicons: &lexicons,
            char_def: CHAR_DEF.as_bytes(),
            unk_def: unk.as_bytes(),
        };
        MecabDictionary::from_files(files, WhitespaceMode::Unicode, DictionaryLimits::default()).unwrap()
    }

    /// Each morpheme of `text` as its surface, reading, pronunciation and
    /// lemma.
    type Seen<'a> = (&'a str, Option<&'a str>, Option<&'a str>, Option<&'a str>);

    fn analyze<'a>(t: &'a LatticeTokenizer, format: DictionaryFormat, text: &'a str) -> Vec<Seen<'a>> {
        format.check(t.dictionary()).unwrap();
        let map = format.map();
        let d = t.dictionary();
        let mut out = Vec::new();
        t.analyze(text, &mut out).unwrap();
        out.iter()
            .map(|m| {
                (&text[m.span.range()], map.reading(d, m.entry), map.pronunciation(d, m.entry), map.lemma(d, m.entry))
            })
            .collect()
    }

    #[test]
    fn column_enums_are_in_file_order() {
        assert_eq!(IpadicColumn::Pronunciation as usize + 1, IpadicColumn::COUNT);
        assert_eq!(KoDicColumn::Expression as usize + 1, KoDicColumn::COUNT);
        assert_eq!(UnidicColumn::FinalChangeForm as usize + 1, UnidicColumn::COUNT);
        assert_eq!(UnidicColumn::LexemeReading as usize, 6);
        assert_eq!(UnidicColumn::Pronunciation as usize, 9);
        assert_eq!(UnidicColumn::OrthographyBase as usize, 10);
        for layout in [UnidicLayout::Source17, UnidicLayout::Full26, UnidicLayout::Full29] {
            assert!(layout.count() >= UnidicColumn::COUNT);
            assert!(layout.kana().is_none_or(|k| (UnidicColumn::COUNT..layout.count()).contains(&k)));
        }
    }

    /// IPADIC's layout: 読み, 発音 and 原形, and nothing from an unknown
    /// word, whose template has seven columns.
    #[test]
    fn ipadic_columns() {
        let lexicon = "東京,0,0,100,名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トーキョー\n\
            行っ,0,0,100,動詞,自立,*,*,五段・カ行促音便,連用タ接続,行く,イッ,イッ\n\
            た,0,0,100,助動詞,*,*,*,特殊・タ,基本形,た,タ,タ\n";
        let t = LatticeTokenizer::new(
            dictionary(lexicon, "DEFAULT,0,0,500,記号,一般,*,*,*,*,*\n"),
            WhitespaceMode::Unicode,
        );
        let got = analyze(&t, DictionaryFormat::Ipadic, "東京へ行った");
        assert_eq!(
            got,
            [
                ("東京", Some("トウキョウ"), Some("トーキョー"), Some("東京")),
                ("へ", None, None, None),
                ("行っ", Some("イッ"), Some("イッ"), Some("行く")),
                ("た", Some("タ"), Some("タ"), Some("た")),
            ]
        );
    }

    /// mecab-ko-dic's layout: 읽기, which differs from the surface for a
    /// Hanja entry, and `*` for none; no lemma or pronunciation column.
    #[test]
    fn ko_dic_columns() {
        let lexicon = "學校,0,0,100,NNG,*,F,학교,*,*,*,*\n\
            에,0,0,100,JKB,*,F,에,*,*,*,*\n\
            !,0,0,100,SF,*,*,*,*,*,*,*\n";
        let unk = "DEFAULT,0,0,500,SY,*,*,*,*,*,*,*\n";
        let t = LatticeTokenizer::new(dictionary(lexicon, unk), WhitespaceMode::Unicode);
        let got = analyze(&t, DictionaryFormat::MecabKoDic, "學校에!");
        assert_eq!(got, [("學校", Some("학교"), None, None), ("에", Some("에"), None, None), ("!", None, None, None)]);
    }

    fn unidic_entry(surface: &str, lexeme: &str, pron: &str, base: &str, kana: &str, layout: UnidicLayout) -> String {
        // pos1-4, cType, cForm, lForm, lemma, orth, pron, orthBase,
        // pronBase, goshu, iType, iForm, fType, fForm.
        let mut cols: Vec<String> = [
            "動詞",
            "一般",
            "*",
            "*",
            "五段-カ行",
            "連用形-促音便",
            "イク",
            lexeme,
            surface,
            pron,
            base,
            "イク",
            "和",
            "*",
            "*",
            "*",
            "*",
        ]
        .map(str::to_string)
        .to_vec();
        cols.resize(layout.count(), "*".to_string());
        if let Some(k) = layout.kana() {
            cols[k] = kana.to_string();
        }
        format!("{surface},0,0,100,{}\n", cols.join(","))
    }

    /// UniDic's layouts: the reading from `kana` where the layout has one,
    /// the pronunciation from `pron`, the lemma from `orthBase` and not
    /// from the lexeme column.
    #[test]
    fn unidic_columns() {
        for layout in [UnidicLayout::Source17, UnidicLayout::Full26, UnidicLayout::Full29] {
            let lexicon = unidic_entry("行っ", "行く", "イッ", "行く", "イッ", layout)
                + &unidic_entry("為", "為る", "シ", "する", "シ", layout);
            let t = LatticeTokenizer::new(
                dictionary(&lexicon, "DEFAULT,0,0,500,補助記号,一般,*,*,*,*\n"),
                WhitespaceMode::Unicode,
            );
            let format = DictionaryFormat::Unidic(layout);
            let got = analyze(&t, format, "行っ為");
            let kana = |k| layout.kana().map(|_| k);
            assert_eq!(
                got,
                [("行っ", kana("イッ"), Some("イッ"), Some("行く")), ("為", kana("シ"), Some("シ"), Some("する"))],
                "{layout:?}"
            );
            // The other layouts are refused.
            for other in [UnidicLayout::Source17, UnidicLayout::Full26, UnidicLayout::Full29] {
                assert_eq!(DictionaryFormat::Unidic(other).check(t.dictionary()).is_ok(), other == layout);
            }
        }
    }

    #[test]
    fn a_dictionary_of_another_format_is_refused() {
        let ko = dictionary("에,0,0,100,JKB,*,F,에,*,*,*,*\n", "DEFAULT,0,0,500,SY\n");
        let err = DictionaryFormat::Ipadic.check(&ko).unwrap_err();
        assert_eq!((err.found, err.entry), (8, EntryId(0)));
        assert_eq!(err.to_string(), "a lexicon entry has 8 feature columns where IPADIC has 9");
        assert!(DictionaryFormat::MecabKoDic.check(&ko).is_ok());
        // One entry with a column too many is enough.
        let mixed = dictionary("a,0,0,1,NNG,*,F,a,*,*,*,*\nb,0,0,1,NNG,*,F,b,*,*,*,*,extra\n", "DEFAULT,0,0,5,SY\n");
        let err = DictionaryFormat::MecabKoDic.check(&mixed).unwrap_err();
        assert_eq!((err.found, err.entry), (9, EntryId(1)));
    }

    #[test]
    fn empty_and_star_values_are_absent() {
        let d = dictionary("a,0,0,1,名詞,*,*,*,*,*,,*,ア\n", "DEFAULT,0,0,5,記号\n");
        let map = DictionaryFormat::Ipadic.map();
        assert_eq!(map.lemma(&d, EntryId(0)), None);
        assert_eq!(map.reading(&d, EntryId(0)), None);
        assert_eq!(map.pronunciation(&d, EntryId(0)), Some("ア"));
        // The unk.def template has one column.
        let unknown = EntryId(d.lexicon_len() as u32);
        assert!(d.is_unknown(unknown));
        assert_eq!(map.lemma(&d, unknown), None);
        assert_eq!(d.feature(unknown, 0), Some("記号"));
    }
}
