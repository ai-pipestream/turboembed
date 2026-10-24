use super::Divergence;

pub(super) const SECTION_QUOTES: &str = "docs/normalizers.md, \"Quotes\"";
pub(super) const SECTION_BULLETS: &str = "docs/normalizers.md, \"Bullets\"";
pub(super) const SECTION_ELLIPSIS: &str = "docs/normalizers.md, \"Ellipsis\"";
pub(super) const SECTION_INVISIBLE: &str = "docs/normalizers.md, \"Invisible characters\"";
pub(super) const SECTION_DEHYPHENATION: &str = "docs/normalizers.md, \"Dehyphenation\"";
pub(super) const SECTION_URLS: &str = "docs/normalizers.md, \"URLs and e-mail addresses (`url/`)\"";
pub(super) const SECTION_CONTRACTIONS: &str = "docs/normalizers.md, \"English contractions (`contractions/`)\"";
pub(super) const SECTION_SYMBOLS: &str = "docs/normalizers.md, \"Symbols (`symbols/`)\"";
pub(super) const SECTION_CONFUSABLES: &str = "docs/normalizers.md, \"Confusable skeleton (`confusables.rs`)\"";
pub(super) const SECTION_CASEFOLD: &str = "docs/normalizers.md, \"Case folding (`case_fold.rs`)\"";
pub(super) const SECTION_CHAIN_QUOTES: &str = SECTION_QUOTES;
pub(super) const SECTION_CHAIN: &str = SECTION_INVISIBLE;

pub(super) const QUOTE: &[Divergence] = &[Divergence {
    input: "quotes-reversed-9",
    section: SECTION_QUOTES,
    rust: "Das Wort \"Wort\" und \"Wort\" stehen nebeneinander.\n",
}];

pub(super) const BULLET: &[Divergence] = &[Divergence {
    input: "bullets-extra",
    section: SECTION_BULLETS,
    rust: "  black left\n  black right\n  inverse bullet\n  circled white\n  circled bullet\n  the plain bullet\n",
}];

pub(super) const ELLIPSIS: &[Divergence] = &[Divergence {
    input: "ellipsis-extra",
    section: SECTION_ELLIPSIS,
    rust: "One dot leader a.b, vertical ellipsis ... and vertical two dot .. and the usual ... and .. here.\n",
}];

pub(super) const INVISIBLE: &[Divergence] = &[Divergence {
    input: "invisible",
    section: SECTION_INVISIBLE,
    rust: "zerowidthspace, joiner\u{200d}and non\u{200c}joiner, bominside, wordjoiner here.\nMarks  left-to-right and  right-to-left and  embedding  and isolate  .\nControls \u{1} \u{2} \u{7f} \u{80} \u{9f} and softhyphen, variation\u{fe0f} selectors, \u{180e} Mongolian,  .\n",
}];

pub(super) const DEHYPHENATION: &[Divergence] = &[Divergence {
    input: "dehyphenation-marks",
    section: SECTION_DEHYPHENATION,
    rust: "The word cafe\u{301}teria is in decomposed form, and the Hindi किताबीका word ends in a vowel sign.\nPlain caféteria joins as before.\n",
}];

pub(super) const EMOJI: &[Divergence] = &[];

pub(super) const BOUNDED_URL: &[Divergence] = &[Divergence {
    input: "emails-international",
    section: SECTION_URLS,
    rust: "Mail   or   or   today.\nPlain   is the same in both.\n",
}];

pub(super) const ENGLISH_CONTRACTION: &[Divergence] = &[
    Divergence {
        input: "contractions",
        section: SECTION_CONTRACTIONS,
        rust: "I do not know. We can not go and they will not come. I am sure you are right; he's here, she's there.\nThey have left, we have won, I will try, you will see, he'd go, I'd say. Shall not, ain't, you all, o'clock, ma'am.\nDO NOT SHOUT, Is not it? Would not, could not, should not, must not, need not, has not, have not, had not, are not, was not, were not, does not, did not.\n",
    },
    Divergence {
        input: "contractions-more",
        section: SECTION_CONTRACTIONS,
        rust: "Here are the keys, these are ours, those are yours. May have been, there have been, what have we done, where have you gone.\nHere will do, this will work, these will fit, those will go, which will one? YOU ALL come, you all come, You all.\n",
    },
];

pub(super) const SYMBOL_JOINER: &[Divergence] = &[
    Divergence {
        input: "emoji-text-default",
        section: SECTION_SYMBOLS,
        rust: "Text default: ☺ ☺\u{fe0f} ❤ ❤\u{fe0f} ✔ ✔\u{fe0f} copyright ©\u{fe0f} trademark ™\u{fe0f} ☀ ☀\u{fe0f} ☂ ☂\u{fe0f}.\nWith text selector: ❤\u{fe0e} ☺\u{fe0e} ®\u{fe0e} and digits 5 5\u{fe0f} 5\u{fe0f}\u{20e3}.\nEmoji default: ✅ ❌ ⭐ ⌚ 😀 💩 🚀.\n",
    },
    Divergence {
        input: "symbols-lone",
        section: SECTION_SYMBOLS,
        rust: "AT&T and R&D but Q and A and a lone and here. 50% and 50 percent and a lone percent there.\ncopyright 2024 Example™ and Brand®, a lone copyright or trademark or ®. Temperature 20°C, section section 3, paragraph 4.\nx plus y = z, a < b > c, 5 * 6, 7 / 8, a ~ b, c ^ d, e | f, g # h, i at j, k $ l.\n",
    },
];

pub(super) const CONFUSABLE_SKELETON: &[Divergence] = &[
    Divergence {
        input: "bullets",
        section: SECTION_CONFUSABLES,
        rust: "· first itern\n° second itern\n∎ third itern\n‣ fourth itern\n- fifth itern\n· sixth itern\n· seventh\n● eighth\n- dash itern\n* star itern\n",
    },
    Divergence {
        input: "casefold-special",
        section: SECTION_CONFUSABLES,
        rust: "STRASSE Straße ß l\u{307} i ffi 'n j\u{306} i\u{308}\u{301} u\u{308}\u{301} h\u{331} u\u{313} a\u{328}\u{303} p\u{313}.\nMixed Case: Hello World, MiXeD cAsE, E\u{301}COLE, N\u{303}ANDU, ABΓ, Ab\u{304}B, D ᴅ.\nTitlecase Dz\u{306} Lj Nj Dz and ligature st մն եi.\n",
    },
    Divergence {
        input: "cjk-mixed",
        section: SECTION_CONFUSABLES,
        rust: "ᄉーᄀ天天气很好˳ 我们去公园玩! ''你好''˳\n日本語のテキス卜 と 空白˳ABC l23˳\n하ᄂ구ᄀoᅥ 터丨ᄀᄉーᄐー、rnixed with English text... and ー dashes.\n",
    },
    Divergence {
        input: "confusables-misc",
        section: SECTION_CONFUSABLES,
        rust: "llllll lc, l o ᴀ i j, coʍ, paypal.\nMath: ABC abc and AB. Hebrew l vs ASCll l, l vs l.\nArrnenian o u and Cherokee A R S look like Latin letters.\n",
    },
    Divergence {
        input: "dehyphenation-marks",
        section: SECTION_CONFUSABLES,
        rust: "The word cafe\u{301}-\nteria is in decornposed forrn, and the Hindi कিत𑧑बी-\nक𑧑 word ends in a vowel sign.\nPlain cafe\u{301}-\nteria joins as before.\n",
    },
    Divergence {
        input: "emails-international",
        section: SECTION_CONFUSABLES,
        rust: "Mail Jose\u{301}@exarnple.corn or 用戶@例子.广告 or e\u{301}a@b.co today.\nPlain ascii@exarnple.org is the sarne in both.\n",
    },
    Divergence {
        input: "fullwidth-kana",
        section: SECTION_CONFUSABLES,
        rust: "Fullwidth ABC abc Ol2 !? and ideographic cornrna、full stop˳\nHalfwidth katakana 力\u{ff9e}キ\u{ff9e} 八\u{30a}匕\u{30a} ア亻ウ and voiced rnark sequences 力\u{ff9e} 八\u{30a}.\nSquare ㌀ ㌃ ㍿ and parenthesized (株), ㊤.\n",
    },
    Divergence {
        input: "hangul",
        section: SECTION_CONFUSABLES,
        rust: "하ᄂᄀーᄅ cornposed, and decornposed 하ᄂᄀーᄅ jarno.\nMixed: 가 가 가ᄀ and 다丨하ᄂᄆ丨ᄂ구ᄀ 다丨하ᄂᄆ丨ᄂ구ᄀ.\nCornpatibility jarno 가 and halfwidth 가 stay different under NFC.\n",
    },
    Divergence {
        input: "kana-small",
        section: SECTION_CONFUSABLES,
        rust: "ア亻ウ ー あいう ウ\u{ff9e} ウ\u{ff9e} か\u{ff9e}き❬\u{ff9e} か\u{ff9e}き\u{ff9e}❬\u{ff9e}\u{ff9e} y\u{308} Y\u{308}.\nCornbining dakuten after kana and ASCll voiced: 力\u{ff9e}キ\u{ff9e}ク\u{ff9e} and 八\u{30a}匕\u{30a}.\n",
    },
    Divergence {
        input: "ligatures-nfkc",
        section: SECTION_CONFUSABLES,
        rust: "The first flight offered an efficient office; st and ft are ligatures too.\nDutch ij: ij, Serbian dz\u{306} lj, units kg MHz ㎡, TM ™, SM ℠, No No, ➀➁.\nSuperscript xᒾ and E=rncᒾ, l/5 fifth, l/4 quarter, ... ellipsis,   nbsp.\n",
    },
    Divergence {
        input: "mixed-article",
        section: SECTION_CONFUSABLES,
        rust: "''The re-\nsults,'' wrote Dr. Mu\u{308}llerーa well-known expertー''were surprising...''\nSee https://exarnple.corn/paper.pdf. Cost: £l9.99 (≈ Ꞓ23). Contact: rn.rnueller@exarnple.org\n· itern one with ideographic space\n· itern two   with  extra   spaces\r\nlt's lOO°/₀ done :) 😀 #win\n",
    },
    Divergence {
        input: "numbers-basic",
        section: SECTION_CONFUSABLES,
        rust: "ln 2O24 there were l,234,567 visitors, 3.l4l59 ratio, l2:3O tirne, l/2 cup, -5 degrees, +7 points.\nPhone 555-l234 or +l (8OO) 555-Ol99. Price $l9.99, 2O°/₀ off, 5elO and OxlF, l,5 in Gerrnan style.\nVersion 2.O.l, lOth, 2lst, 3rd place, H2O, Bl2, 4K, 24/7, lOOrn.\n",
    },
    Divergence {
        input: "numbers-unicode",
        section: SECTION_CONFUSABLES,
        rust: "Arabic-lndic .l٢٣٤, extended o٦V, Devanagari ٩२3, Thai ๑๒๓.\nFullwidth l23 and 456, superscripts ᒾ³ᣳ, fractions l/2 l/3, Rornan lll and circled ➀.\nMixed l2٣ and 45 in one token, Bengali o১২ and Tarnil oக.\n",
    },
    Divergence { input: "symbol-percent", section: SECTION_CONFUSABLES, rust: "°/₀" },
    Divergence {
        input: "symbols-lone",
        section: SECTION_CONFUSABLES,
        rust: "AT&T and R&D but Q & A and a lone & here. 5O°/₀ and 5O °/₀ and a lone °/₀ there.\n© 2O24 Exarnple™ and Brand®, a lone © or ™ or ®. Ternperature 2O°C, section § 3, ¶ 4.\nx + y = z, a < b > c, 5 * 6, 7 / 8, a ~ b, c ^ d, e l f, g # h, i @ j, k $ l.\n",
    },
    Divergence {
        input: "whitespace-unicode",
        section: SECTION_CONFUSABLES,
        rust: "no break space, ideographic space, ern space, thin space, narrow no-break, hair space.\nFigure space and en space and four-per-ern six-per-ern.\n  leading and trailing   \n",
    },
];

pub(super) const CASE_FOLD_TR: &[Divergence] = &[
    Divergence {
        input: "casefold-special",
        section: SECTION_CASEFOLD,
        rust: "strasse straße ß i ı ﬃ ŉ ǰ ΐ ΰ ẖ ὐ ᾷ ῤ.\nmixed case: hello world, mixed case, école, ñandu, αβγ, абв, Ꭰ Ꭰ.\ntitlecase ǆ ǉ ǌ ǳ and ligature ﬆ ﬓ և.\n",
    },
    Divergence {
        input: "confusables-misc",
        section: SECTION_CASEFOLD,
        rust: "ⅰⅱⅲ ⅼⅽ, ℓ ℴ ᴀ ı ȷ, ᴄᴏᴍ, ｐａｙｐａｌ.\nmath: 𝐀𝐁𝐂 𝐚𝐛𝐜 and 𝔸𝔹. hebrew ו vs ascıı |, ן vs l.\narmenian օ ս and cherokee Ꭺ Ꭱ Ꮪ look like latin letters.\n",
    },
    Divergence {
        input: "greek-sigma",
        section: SECTION_CASEFOLD,
        rust: "οδυσσευσ and σασ and σπιτι, ανθρωποσ.\nfinal sigma cases: αβσ ασ σ σ. σσ (ασ) ασα and αΐσ.\ntonos: άέήίόύώ, άέήίόύώ, ΐ ΰ, ἀ ἀ ᾳ ᾳ.\n",
    },
    Divergence {
        input: "ligatures-nfkc",
        section: SECTION_CASEFOLD,
        rust: "the ﬁrst ﬂight oﬀered an eﬃcient oﬃce; ﬆ and ﬆ are ligatures too.\ndutch ij: ĳ, serbian ǆ ǉ, units ㎏ ㎒ ㎡, tm ™, sm ℠, no №, ①②.\nsuperscript x² and e=mc², ⅕ fifth, ¼ quarter, … ellipsis, \u{a0} nbsp.\n",
    },
];

pub(super) const DEFAULT_CHAIN: &[Divergence] = &[
    Divergence {
        input: "invisible",
        section: SECTION_CHAIN,
        rust: "zerowidthspace, joiner\u{200d}and non\u{200c}joiner, bominside, wordjoiner here. marks left-to-right and right-to-left and embedding and isolate . controls \u{1} \u{2} \u{7f} \u{80} \u{9f} and softhyphen, variation selectors, \u{180e} mongolian, .",
    },
    Divergence {
        input: "quotes-reversed-9",
        section: SECTION_CHAIN_QUOTES,
        rust: "das wort \"wort\" und \"wort\" stehen nebeneinander.",
    },
];
