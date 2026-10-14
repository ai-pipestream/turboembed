//! The fragments the generator builds texts from, and the builder itself.
//!
//! A text is a few lines; a line is usually one or more sentences built
//! from a small grammar (a subject, a verb, an object and trimmings such
//! as an abbreviation, a number, a sum of money, a quantity, a URL, a quote
//! or a date), so that the sentence and token models see contexts like the
//! ones they were trained on, with the hard cases placed inside them. The
//! other kinds of line are the ones that break segmenters on their own: an
//! empty line, a whitespace only line, a very long token, a run of CJK or
//! Thai without spaces, right to left text with bidi controls, a heading, a
//! list item, a table row. The whole text is then mutated: zero width and
//! format characters inside words, combining marks on letters, Unicode
//! spaces for ASCII ones, CR LF or lone CR line ends, a byte order mark, no
//! final line end.
//!
//! Every character used is in the Unicode 15.0 repertoire, which is what
//! JDK 21 implements, so a difference between the two sides is never one of
//! Unicode version. The texts are valid UTF-8 by construction, so no lone
//! surrogate can be in an input: the Java side reads inputs as UTF-8 and
//! refuses a malformed file.

use super::rng::Rng;

const EN_WORDS: &[&str] = &[
    "the", "a", "an", "this", "that", "our", "their", "some", "every", "no", "quick", "brown", "lazy", "old", "new",
    "little", "large", "red", "early", "late", "cold", "fox", "dog", "report", "meeting", "price", "city", "river",
    "house", "car", "team", "year", "day", "night", "water", "paper", "data", "model", "test", "server", "network",
    "machine", "answer", "question", "letter", "line", "word", "sentence", "token", "offset", "boundary",
];
const EN_VERBS: &[&str] = &[
    "is",
    "was",
    "jumps over",
    "runs to",
    "sees",
    "saw",
    "found",
    "finds",
    "wrote",
    "writes",
    "reads",
    "read",
    "costs",
    "cost",
    "weighs",
    "measures",
    "takes",
    "took",
    "sends",
    "sent",
    "paid",
    "pays",
    "said",
    "says",
    "arrived at",
    "left",
    "mentions",
    "shows",
    "returns",
    "splits",
    "joins",
    "counts",
    "has",
    "had",
    "needs",
];
const EN_NAMES: &[&str] = &[
    "Smith",
    "Jones",
    "Müller",
    "García",
    "O'Brien",
    "Zhang",
    "Kim",
    "Ivanov",
    "Dubois",
    "Rossi",
    "Nakamura",
    "Al-Sayed",
    "Cohen",
    "Patel",
    "Nguyen",
    "van der Berg",
    "McDonald",
    "D'Angelo",
    "Søren",
    "Łukasz",
];
const TITLES: &[&str] = &["Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Sen.", "Gen.", "St.", "Capt.", "Rev."];
const ABBREVIATIONS: &[&str] = &[
    "U.S.", "U.K.", "e.g.", "i.e.", "etc.", "Inc.", "Ltd.", "Co.", "Corp.", "Jan.", "Feb.", "Dec.", "No.", "vs.",
    "approx.", "Fig.", "p.m.", "a.m.", "Ph.D.", "cf.", "Mt.", "Ave.", "Blvd.", "Jr.", "Sr.", "et al.", "ca.", "z.B.",
    "usw.", "bzw.", "Nr.", "Str.", "S.A.", "Sra.", "Sr.", "p.ej.", "M.", "Mme", "Mlle", "cf.",
];
const DE_WORDS: &[&str] = &[
    "der",
    "die",
    "das",
    "Straße",
    "Mädchen",
    "über",
    "Übung",
    "Größe",
    "schön",
    "Köln",
    "Zürich",
    "Fußball",
    "Weiß",
    "fünf",
    "Bücher",
    "Äpfel",
    "Österreich",
    "Jürgen",
    "Grüße",
    "naß",
];
const FR_WORDS: &[&str] = &[
    "le",
    "la",
    "les",
    "été",
    "école",
    "à",
    "où",
    "français",
    "garçon",
    "château",
    "naïve",
    "cœur",
    "œuvre",
    "Noël",
    "très",
    "déjà",
    "l'homme",
    "d'accord",
    "aujourd'hui",
    "qu'il",
];
const ES_PT_WORDS: &[&str] = &[
    "niño",
    "mañana",
    "señor",
    "año",
    "corazón",
    "música",
    "São",
    "Paulo",
    "ação",
    "coração",
    "não",
    "pão",
    "está",
    "también",
    "¿Qué?",
    "¡Hola!",
    "jalapeño",
    "piñata",
    "avô",
    "avó",
];
const GREEK_WORDS: &[&str] =
    &["Αθήνα", "ΟΔΥΣΣΕΥΣ", "λόγος", "σοφός", "ΣΟΦΟΣ", "θάλασσα", "άνθρωπος", "Ελλάδα", "σας", "ΑΣ", "ΐ", "ΰ"];
const CYRILLIC_WORDS: &[&str] =
    &["Москва", "привет", "мир", "Россия", "ёлка", "Київ", "Україна", "българия", "Ђорђе", "Љубљана", "ЁЖ"];
const ARABIC_WORDS: &[&str] =
    &["السلام", "عليكم", "مرحبا", "القاهرة", "بغداد", "كتاب", "مدرسة", "٢٠٢٤", "١٢٣", "العربية", "شكرًا", "اللّغة"];
const HEBREW_WORDS: &[&str] = &["שלום", "עולם", "ירושלים", "ספר", "בית", "תּוֹרָה", "ישראל", "ברוך", "הבא"];
const DEVANAGARI_WORDS: &[&str] =
    &["नमस्ते", "भारत", "किताब", "हिन्दी", "दिल्ली", "क्षत्रिय", "संस्कृत", "९२३", "ज्ञान", "श्री", "मुम्बई"];
const THAI_WORDS: &[&str] = &["สวัสดี", "ประเทศไทย", "ภาษาไทย", "กรุงเทพ", "ขอบคุณ", "น้ำ", "เด็ก", "๑๒๓"];
const CJK_WORDS: &[&str] = &[
    "北京",
    "上海",
    "我们",
    "研究生",
    "生命",
    "人民大会",
    "中文",
    "你好",
    "世界",
    "東京",
    "日本語",
    "の",
    "を",
    "です",
    "ます",
    "カタカナ",
    "ひらがな",
    "漢字",
    "서울",
    "한국어",
    "안녕하세요",
    "감사합니다",
    "大韩民国",
    "𠀋",
    "𪚲",
    "한",
    "글",
];
const CJK_PUNCT: &[&str] =
    &["。", "、", "，", "！", "？", "「", "」", "『", "』", "（", "）", "：", "；", "…", "〜", "・"];
const UNITS: &[&str] = &[
    "kg", "g", "mg", "km", "m", "cm", "mm", "mi", "ft", "in", "l", "ml", "h", "min", "s", "ms", "°C", "°F", "%", "kWh",
    "MB", "GB", "km/h", "mph", "lb", "oz", "ha", "m²", "m³", "V", "W", "Hz",
];
const CURRENCY_SYMBOLS: &[&str] = &["$", "€", "£", "¥", "₹", "₽", "₩", "CHF ", "USD ", "EUR ", "US$", "A$", "C$", "R$"];
const CURRENCY_CODES: &[&str] = &["USD", "EUR", "GBP", "JPY", "CHF", "CNY", "INR", "RUB", "BRL", "CAD", "AUD"];
const SCALE_WORDS: &[&str] =
    &["k", "K", "m", "M", "bn", "b", "B", "thousand", "million", "billion", "trillion", "Mio.", "Mrd."];
const DOMAINS: &[&str] = &[
    "example.com",
    "example.org",
    "test.co.uk",
    "sub.domain.example",
    "localhost",
    "münchen.de",
    "例子.测试",
    "xn--mnchen-3ya.de",
    "a.b.c.d.e.f",
    "192.168.0.1",
    "[::1]",
];
const PATHS: &[&str] = &[
    "",
    "/",
    "/path",
    "/path/to/file.html",
    "/a?b=1&c=2",
    "/x#frag",
    "/p?q=a%20b",
    "/~user",
    "/路径",
    "/(paren)",
    "/path.",
    "/path,",
    "/path)",
    ":8080/x",
    "/very/long/path/with/many/segments/and/a/file.tar.gz",
];
const EMAIL_LOCAL: &[&str] = &["john.doe", "jane", "a", "first+tag", "name_1", "José", "用戶", "o'brien", "x.y.z"];
const EMOJI: &[&str] = &[
    "😀",
    "😂",
    "🙂",
    "😀😀😀",
    "👍",
    "👍🏽",
    "👩‍💻",
    "👨‍👩‍👧‍👦",
    "🏳️‍🌈",
    "🇺🇸",
    "🇯🇵",
    "🇩🇪🇫🇷",
    "1️⃣",
    "#️⃣",
    "❤️",
    "❤",
    "☺︎",
    "☺️",
    "✈️",
    "⭐",
    "🚀",
    "💩",
    "🤷🏻‍♀️",
    "🧑🏿‍🚀",
    "©️",
    "™️",
    "😀.",
    "'😀",
    ",😀",
    "&😀",
    "(😀)",
    "😀!",
    "x😀y",
];
const ZERO_WIDTH: &[&str] = &[
    "\u{200B}", "\u{200C}", "\u{200D}", "\u{2060}", "\u{FEFF}", "\u{00AD}", "\u{034F}", "\u{180E}", "\u{FE0F}",
    "\u{FE0E}", "\u{061C}",
];
const BIDI: &[&str] = &[
    "\u{200E}", "\u{200F}", "\u{202A}", "\u{202B}", "\u{202C}", "\u{202D}", "\u{202E}", "\u{2066}", "\u{2067}",
    "\u{2068}", "\u{2069}",
];
const COMBINING: &[&str] = &[
    "\u{0301}",
    "\u{0300}",
    "\u{0308}",
    "\u{0327}",
    "\u{0303}",
    "\u{0306}",
    "\u{030C}",
    "\u{0323}",
    "\u{0331}",
    "\u{0338}",
    "\u{20E3}",
    "\u{20DD}",
    "\u{0940}",
    "\u{094D}",
    "\u{0E31}",
    "\u{0E48}",
    "\u{05B8}",
    "\u{064E}",
    "\u{3099}",
    "\u{309A}",
    "\u{0301}\u{0323}",
    "\u{0323}\u{0301}",
    "\u{0308}\u{0301}\u{0323}",
];
const SPACES: &[&str] = &[
    "\u{00A0}", "\u{2002}", "\u{2003}", "\u{2009}", "\u{200A}", "\u{202F}", "\u{205F}", "\u{3000}", "\u{1680}",
    "\u{2000}", "\u{2007}", "\u{2008}", "\t", "\u{000B}", "\u{000C}", "\u{0085}", "\u{2028}", "\u{2029}", "\u{001C}",
    "\u{001D}", "\u{001E}", "\u{001F}", "  ", "   ",
];
const CONTROLS: &[&str] = &["\u{0001}", "\u{0002}", "\u{001B}", "\u{007F}", "\u{0080}", "\u{009F}", "\u{0000}"];
const OPEN_QUOTES: &[&str] = &["\"", "“", "‘", "'", "«", "„", "‹", "『", "「", "`", "``"];
const CLOSE_QUOTES: &[&str] = &["\"", "”", "’", "'", "»", "“", "›", "』", "」", "'", "''"];
const DASHES: &[&str] = &["-", "–", "—", "―", "‐", "‑", "−", "--", "---", " - ", " – ", " — "];
const TERMINALS: &[&str] = &[
    ".", ".", ".", ".", "!", "?", "...", "…", "?!", "!!!", "..", ".)", ").", ".\"", ".”", "\".", "?\"", "!)", ".’",
    ":", ";", ",", "", "", ". . .", "。", "！", "？",
];
const SEPARATORS: &[&str] = &[" ", " ", " ", " ", "  ", "\n", "\u{00A0}", "\t", "", " \n", "\u{3000}", "\u{2028}"];
const CONTRACTIONS: &[&str] = &[
    "don't",
    "can't",
    "won't",
    "I'm",
    "we'll",
    "they've",
    "she'd",
    "o'clock",
    "ma'am",
    "y'all",
    "ain't",
    "it's",
    "don’t",
    "can’t",
    "I’m",
    "rock'n'roll",
    "'tis",
    "'90s",
    "Hallowe'en",
    "l'école",
    "d'Artagnan",
];
const SOCIAL: &[&str] = &[
    "#hashtag",
    "#Ünïcödé",
    "#日本",
    "@user",
    "@user_name",
    "@a.b",
    ":)",
    ":-(",
    ";-)",
    ":D",
    "<3",
    "xD",
    ":P",
    "RT",
    "#1",
    "#",
    "@",
];
const SYMBOLS: &[&str] = &[
    "&", "%", "©", "®", "™", "°", "§", "¶", "+", "=", "<", ">", "*", "/", "~", "^", "|", "#", "@", "$", "€", "¢", "½",
    "¼", "²", "³", "Ⅳ", "ⅷ", "№", "℃", "㎏", "㍿", "㌀", "ﬁ", "ﬂ", "ﬃ", "ﬆ", "Ǆ", "ǅ", "ǳ", "ẞ", "ß", "İ", "ı", "ſ",
    "ǰ", "ΐ", "ﬀ", "Ａ", "ｂ", "１", "２", "！", "？", "ｱ", "ｲ", "ｳ", "ｶﾞ", "ﾊﾟ", "pаypal", "соm", "ΑΒΓ",
];
const UNICODE_DIGITS: &[&str] = &["٠١٢٣", "٤٥٦", "۷۸۹", "९२३", "๑๒๓", "１２３", "४५६", "০১২", "௧௨௩", "𝟏𝟐𝟑"];
const MATH_LETTERS: &[&str] = &["𝐀𝐁𝐂", "𝑎𝑏𝑐", "𝔸𝔹", "𝒳", "𝟙", "𐐷", "𑀓", "𐌰𐌱", "𓀀𓀁", "𝄞", "𝕳𝖊𝖑𝖑𝖔"];

/// A fragment builder over one random source. Every method appends to the
/// text; the document method is the entry point.
pub struct Builder<'r> {
    rng: &'r mut Rng,
    out: String,
}

impl<'r> Builder<'r> {
    pub fn new(rng: &'r mut Rng) -> Builder<'r> {
        Builder { rng, out: String::new() }
    }

    /// A whole text, mutated.
    pub fn document(mut self) -> String {
        let lines = match self.rng.below(100) {
            0..=30 => 1,
            31..=60 => 2,
            61..=80 => 3,
            81..=92 => self.rng.range(4, 6),
            _ => self.rng.range(7, 14),
        };
        for i in 0..lines {
            self.line();
            if i + 1 < lines || self.rng.chance(70) {
                self.out.push('\n');
            }
        }
        let text = std::mem::take(&mut self.out);
        self.mutate(text)
    }

    fn line(&mut self) {
        match self.rng.below(100) {
            0..=44 => {
                let n = self.rng.range(1, 4);
                for i in 0..n {
                    self.sentence();
                    if i + 1 < n {
                        let sep = *self.rng.pick(SEPARATORS);
                        self.out.push_str(sep);
                    }
                }
            }
            45..=52 => self.fragment(),
            53..=57 => {}
            58..=62 => {
                let n = self.rng.range(1, 5);
                for _ in 0..n {
                    let s = *self.rng.pick(SPACES);
                    self.out.push_str(if s.len() > 3 { " " } else { s });
                }
            }
            63..=67 => self.long_token(),
            68..=72 => self.cjk_run(),
            73..=76 => self.rtl_line(),
            77..=79 => self.heading(),
            80..=84 => self.list_item(),
            85..=87 => self.table_row(),
            88..=90 => self.code_line(),
            91..=93 => self.hyphenated_lines(),
            94..=96 => self.mixed_script_line(),
            _ => self.dense_punctuation(),
        }
        if self.rng.chance(10) {
            let s = *self.rng.pick(SPACES);
            self.out.push_str(s);
        }
    }

    /// Subject, verb, object and trimmings, then a terminal.
    fn sentence(&mut self) {
        if self.rng.chance(8) {
            let q = *self.rng.pick(OPEN_QUOTES);
            self.out.push_str(q);
        }
        self.subject();
        self.out.push(' ');
        let verb = *self.rng.pick(EN_VERBS);
        self.out.push_str(verb);
        self.out.push(' ');
        self.object();
        let trimmings = self.rng.range(0, 3);
        for _ in 0..trimmings {
            if self.rng.chance(30) {
                self.out.push(',');
            }
            self.out.push(' ');
            self.trimming();
        }
        if self.rng.chance(10) {
            self.out.push(' ');
            let q = *self.rng.pick(CLOSE_QUOTES);
            self.out.push_str(q);
        }
        let terminal = *self.rng.pick(TERMINALS);
        self.out.push_str(terminal);
    }

    fn subject(&mut self) {
        match self.rng.below(100) {
            0..=34 => {
                if self.rng.chance(60) {
                    let t = *self.rng.pick(TITLES);
                    self.out.push_str(t);
                    self.out.push(' ');
                }
                let n = *self.rng.pick(EN_NAMES);
                self.out.push_str(n);
            }
            35..=69 => {
                let w = *self.rng.pick(EN_WORDS);
                self.push_capitalized(w);
                self.out.push(' ');
                let w = *self.rng.pick(EN_WORDS);
                self.out.push_str(w);
            }
            70..=79 => self.foreign_word(),
            80..=89 => self.number(),
            90..=94 => {
                let a = *self.rng.pick(ABBREVIATIONS);
                self.out.push_str(a);
            }
            _ => {
                let e = *self.rng.pick(EMOJI);
                self.out.push_str(e);
            }
        }
    }

    fn object(&mut self) {
        match self.rng.below(100) {
            0..=39 => {
                let w = *self.rng.pick(EN_WORDS);
                self.out.push_str(w);
                if self.rng.chance(50) {
                    self.out.push(' ');
                    let w = *self.rng.pick(EN_WORDS);
                    self.out.push_str(w);
                }
            }
            40..=54 => self.money(),
            55..=69 => self.quantity(),
            70..=79 => self.number(),
            80..=89 => self.url_or_email(),
            _ => self.foreign_word(),
        }
    }

    /// One more thing a sentence can carry.
    fn trimming(&mut self) {
        match self.rng.below(100) {
            0..=9 => {
                let a = *self.rng.pick(ABBREVIATIONS);
                self.out.push_str(a);
                self.out.push(' ');
                let w = *self.rng.pick(EN_WORDS);
                self.out.push_str(w);
            }
            10..=19 => self.number(),
            20..=27 => self.money(),
            28..=35 => self.quantity(),
            36..=43 => self.url_or_email(),
            44..=51 => self.date_or_time(),
            52..=59 => {
                let q = *self.rng.pick(OPEN_QUOTES);
                self.out.push_str(q);
                let w = *self.rng.pick(EN_WORDS);
                self.out.push_str(w);
                if self.rng.chance(40) {
                    let t = *self.rng.pick(TERMINALS);
                    self.out.push_str(t);
                }
                let q = *self.rng.pick(CLOSE_QUOTES);
                self.out.push_str(q);
            }
            60..=66 => {
                self.out.push('(');
                let w = *self.rng.pick(EN_WORDS);
                self.out.push_str(w);
                if self.rng.chance(40) {
                    self.out.push(' ');
                    self.number();
                }
                self.out.push(')');
            }
            67..=72 => {
                let c = *self.rng.pick(CONTRACTIONS);
                self.out.push_str(c);
            }
            73..=78 => {
                let s = *self.rng.pick(SOCIAL);
                self.out.push_str(s);
            }
            79..=84 => {
                let e = *self.rng.pick(EMOJI);
                self.out.push_str(e);
            }
            85..=89 => {
                let d = *self.rng.pick(DASHES);
                self.out.push_str(d);
                let w = *self.rng.pick(EN_WORDS);
                self.out.push_str(w);
            }
            90..=94 => {
                let s = *self.rng.pick(SYMBOLS);
                self.out.push_str(s);
            }
            _ => self.foreign_word(),
        }
    }

    fn fragment(&mut self) {
        let n = self.rng.range(1, 5);
        for i in 0..n {
            if i > 0 {
                self.out.push(' ');
            }
            match self.rng.below(5) {
                0 => {
                    let w = *self.rng.pick(EN_WORDS);
                    self.out.push_str(w);
                }
                1 => self.number(),
                2 => self.foreign_word(),
                3 => {
                    let a = *self.rng.pick(ABBREVIATIONS);
                    self.out.push_str(a);
                }
                _ => {
                    let s = *self.rng.pick(SYMBOLS);
                    self.out.push_str(s);
                }
            }
        }
    }

    fn foreign_word(&mut self) {
        let list: &[&str] = match self.rng.below(10) {
            0 => DE_WORDS,
            1 => FR_WORDS,
            2 => ES_PT_WORDS,
            3 => GREEK_WORDS,
            4 => CYRILLIC_WORDS,
            5 => ARABIC_WORDS,
            6 => HEBREW_WORDS,
            7 => DEVANAGARI_WORDS,
            8 => THAI_WORDS,
            _ => CJK_WORDS,
        };
        let w = *self.rng.pick(list);
        self.out.push_str(w);
    }

    fn digits(&mut self, n: usize) {
        for _ in 0..n {
            let d = b'0' + self.rng.below(10) as u8;
            self.out.push(d as char);
        }
    }

    /// Integers, grouped either way, decimals, signs, scale words, Unicode digits.
    fn number(&mut self) {
        match self.rng.below(100) {
            0..=24 => {
                let n = self.rng.range(1, 7);
                self.digits(n);
            }
            25..=39 => {
                // Grouped: US or EU, grouping and decimal marks.
                let eu = self.rng.chance(50);
                let groups = self.rng.range(2, 4);
                {
                    let n = self.rng.range(1, 3);
                    self.digits(n);
                }
                for _ in 1..groups {
                    self.out.push(if eu { '.' } else { ',' });
                    self.digits(3);
                }
                if self.rng.chance(50) {
                    self.out.push(if eu { ',' } else { '.' });
                    {
                        let n = self.rng.range(1, 3);
                        self.digits(n);
                    }
                }
            }
            40..=54 => {
                if self.rng.chance(40) {
                    let s = *self.rng.pick(&["-", "+", "−", "–"]);
                    self.out.push_str(s);
                }
                {
                    let n = self.rng.range(1, 4);
                    self.digits(n);
                }
                let mark = *self.rng.pick(&[".", ",", "·", "٫"]);
                self.out.push_str(mark);
                {
                    let n = self.rng.range(1, 6);
                    self.digits(n);
                }
            }
            55..=64 => {
                {
                    let n = self.rng.range(1, 3);
                    self.digits(n);
                }
                if self.rng.chance(50) {
                    self.out.push(' ');
                }
                let s = *self.rng.pick(SCALE_WORDS);
                self.out.push_str(s);
            }
            65..=74 => {
                let d = *self.rng.pick(UNICODE_DIGITS);
                self.out.push_str(d);
            }
            75..=84 => {
                {
                    let n = self.rng.range(1, 2);
                    self.digits(n);
                }
                let s = *self.rng.pick(&["st", "nd", "rd", "th", ".", "º", "ª", "e", "er"]);
                self.out.push_str(s);
            }
            85..=89 => {
                self.digits(1);
                let s = *self.rng.pick(&["/", "⁄", " / "]);
                self.out.push_str(s);
                self.digits(1);
            }
            90..=94 => {
                let s = *self.rng.pick(&["0x1F", "1e10", "2E-3", "0.", ".5", "1..2", "1.2.3", "3,", ",5", "--5", "5-"]);
                self.out.push_str(s);
            }
            _ => {
                // A long run of digits.
                let n = self.rng.range(20, 60);
                self.digits(n);
            }
        }
    }

    fn money(&mut self) {
        if self.rng.chance(20) {
            self.out.push('-');
        }
        if self.rng.chance(65) {
            let s = *self.rng.pick(CURRENCY_SYMBOLS);
            self.out.push_str(s);
            if self.rng.chance(15) {
                self.out.push(' ');
            }
            self.number();
        } else {
            self.number();
            let s = *self.rng.pick(&[" ", "", "\u{00A0}"]);
            self.out.push_str(s);
            if self.rng.chance(50) {
                let c = *self.rng.pick(CURRENCY_CODES);
                self.out.push_str(c);
            } else {
                let c = *self.rng.pick(CURRENCY_SYMBOLS);
                self.out.push_str(c.trim());
            }
        }
    }

    fn quantity(&mut self) {
        if self.rng.chance(15) {
            self.out.push('-');
        }
        self.number();
        let s = *self.rng.pick(&[" ", " ", "", "\u{00A0}", "\u{202F}"]);
        self.out.push_str(s);
        let u = *self.rng.pick(UNITS);
        self.out.push_str(u);
    }

    fn url_or_email(&mut self) {
        if self.rng.chance(35) {
            let l = *self.rng.pick(EMAIL_LOCAL);
            self.out.push_str(l);
            self.out.push('@');
            let d = *self.rng.pick(DOMAINS);
            self.out.push_str(d);
            return;
        }
        let scheme = *self.rng.pick(&["https://", "http://", "www.", "", "ftp://", "HTTPS://", "mailto:"]);
        self.out.push_str(scheme);
        let d = *self.rng.pick(DOMAINS);
        self.out.push_str(d);
        let p = *self.rng.pick(PATHS);
        self.out.push_str(p);
    }

    fn date_or_time(&mut self) {
        match self.rng.below(6) {
            0 => {
                self.out.push_str("20");
                self.digits(2);
                self.out.push('-');
                self.out.push('0');
                self.digits(1);
                self.out.push('-');
                self.out.push('1');
                self.digits(1);
            }
            1 => {
                self.digits(1);
                self.out.push('/');
                self.digits(2);
                self.out.push('/');
                self.digits(4);
            }
            2 => {
                self.digits(2);
                self.out.push('.');
                self.digits(2);
                self.out.push('.');
                self.digits(4);
            }
            3 => {
                self.digits(2);
                self.out.push(':');
                self.digits(2);
                if self.rng.chance(50) {
                    let s = *self.rng.pick(&[" p.m.", " a.m.", "pm", " PM", ":00"]);
                    self.out.push_str(s);
                }
            }
            4 => {
                let m = *self.rng.pick(&["Jan.", "Feb.", "March", "Dec.", "Sept.", "Oct"]);
                self.out.push_str(m);
                self.out.push(' ');
                {
                    let n = self.rng.range(1, 2);
                    self.digits(n);
                }
                if self.rng.chance(50) {
                    self.out.push_str(", 20");
                    self.digits(2);
                }
            }
            _ => {
                let s = *self.rng.pick(&["+1 (800) 555-0199", "555-1234", "+44 20 7946 0958", "1-800-FLOWERS"]);
                self.out.push_str(s);
            }
        }
    }

    /// A token far longer than any model saw: 300 to 4000 bytes of one
    /// kind of character, or a long URL, or one emoji many times.
    fn long_token(&mut self) {
        let kind = self.rng.below(7);
        let n = self.rng.range(300, 1500);
        match kind {
            0 => {
                for i in 0..n {
                    self.out.push((b'a' + (i % 26) as u8) as char);
                }
            }
            1 => {
                for i in 0..n {
                    self.out.push(if i % 7 == 6 { '-' } else { 'x' });
                }
            }
            2 => {
                self.out.push_str("https://example.com/");
                for i in 0..n {
                    self.out.push(if i % 11 == 10 { '/' } else { (b'a' + (i % 26) as u8) as char });
                }
            }
            3 => {
                let e = *self.rng.pick(EMOJI);
                for _ in 0..n / 8 {
                    self.out.push_str(e);
                }
            }
            4 => {
                for i in 0..n {
                    let w = CJK_WORDS[i % CJK_WORDS.len()];
                    self.out.push_str(w);
                }
            }
            5 => {
                self.out.push('a');
                let m = *self.rng.pick(COMBINING);
                for _ in 0..n / 4 {
                    self.out.push_str(m);
                }
            }
            _ => {
                // 254 letters and then a character outside the BMP, so a
                // cut after 255 UTF-16 units falls inside a pair.
                let k = *self.rng.pick(&[254usize, 255, 253, 509, 510]);
                for _ in 0..k {
                    self.out.push('b');
                }
                self.out.push('😀');
                self.out.push('c');
            }
        }
    }

    /// Chinese, Japanese or Korean without spaces, with its own punctuation.
    fn cjk_run(&mut self) {
        let n = self.rng.range(3, 30);
        for _ in 0..n {
            if self.rng.chance(15) {
                let p = *self.rng.pick(CJK_PUNCT);
                self.out.push_str(p);
            }
            let w = *self.rng.pick(CJK_WORDS);
            self.out.push_str(w);
            if self.rng.chance(10) {
                let w = *self.rng.pick(EN_WORDS);
                self.out.push_str(w);
            }
            if self.rng.chance(10) {
                self.number();
            }
        }
        if self.rng.chance(60) {
            let p = *self.rng.pick(&["。", "！", "？", "…", "．", "。」"]);
            self.out.push_str(p);
        }
    }

    /// Arabic or Hebrew words with digits and Latin inside, with bidi
    /// controls around parts of it.
    fn rtl_line(&mut self) {
        let n = self.rng.range(2, 10);
        let list = if self.rng.chance(50) { ARABIC_WORDS } else { HEBREW_WORDS };
        for i in 0..n {
            if i > 0 {
                self.out.push(' ');
            }
            if self.rng.chance(25) {
                let b = *self.rng.pick(BIDI);
                self.out.push_str(b);
            }
            match self.rng.below(10) {
                0 => self.number(),
                1 => {
                    let w = *self.rng.pick(EN_WORDS);
                    self.out.push_str(w);
                }
                _ => {
                    let w = *self.rng.pick(list);
                    self.out.push_str(w);
                }
            }
            if self.rng.chance(15) {
                let b = *self.rng.pick(BIDI);
                self.out.push_str(b);
            }
        }
        let t = *self.rng.pick(&[".", "؟", "،", "!", "", "۔"]);
        self.out.push_str(t);
    }

    fn heading(&mut self) {
        let n = self.rng.range(1, 5);
        for i in 0..n {
            if i > 0 {
                self.out.push(' ');
            }
            let w = *self.rng.pick(EN_WORDS);
            self.out.push_str(&w.to_uppercase());
        }
        if self.rng.chance(30) {
            self.out.push(':');
        }
    }

    fn list_item(&mut self) {
        let m = *self.rng.pick(&["- ", "* ", "• ", "· ", "1. ", "2) ", "a. ", "IV. ", "– ", "—", "\u{2023} ", "①"]);
        self.out.push_str(m);
        self.sentence();
    }

    fn table_row(&mut self) {
        let n = self.rng.range(2, 5);
        self.out.push('|');
        for _ in 0..n {
            self.out.push(' ');
            match self.rng.below(3) {
                0 => {
                    let w = *self.rng.pick(EN_WORDS);
                    self.out.push_str(w);
                }
                1 => self.number(),
                _ => self.money(),
            }
            self.out.push_str(" |");
        }
    }

    fn code_line(&mut self) {
        let s = *self.rng.pick(&[
            "x = 5 * 6;",
            "if (a < b && c > d) { return a.b(c); }",
            "SELECT * FROM t WHERE x = 'y';",
            "let s = \"Hello, world!\";",
            "a[0] = b[1] + c[2];",
            "printf(\"%d\\n\", n);",
            "C:\\Users\\name\\file.txt",
            "/usr/local/bin/turbo-goldens",
            "{\"key\": [1, 2.5, -3e4, null]}",
            "<a href=\"https://example.com\">link</a>",
            "f(x) = x² + 2x − 1",
            "∀x∈ℝ: x² ≥ 0",
        ]);
        self.out.push_str(s);
    }

    /// A word broken at a line end with a hyphen or a soft hyphen.
    fn hyphenated_lines(&mut self) {
        let w = *self.rng.pick(EN_WORDS);
        self.out.push_str(w);
        self.out.push(' ');
        let (a, b) = self.rng.pick(&[("exam", "ple"), ("sen", "tence"), ("cafe\u{0301}", "teria"), ("Fuß", "ball")]);
        self.out.push_str(a);
        let h = *self.rng.pick(&["-", "\u{00AD}", "‐", "–"]);
        self.out.push_str(h);
        if self.rng.chance(20) {
            self.out.push(' ');
        }
        let le = *self.rng.pick(&["\n", "\r\n", "\r", "\n\n", "\u{2028}"]);
        self.out.push_str(le);
        self.out.push_str(b);
        self.out.push(' ');
        let w = *self.rng.pick(EN_WORDS);
        self.out.push_str(w);
        self.out.push('.');
    }

    /// Words of several scripts and symbol runs side by side.
    fn mixed_script_line(&mut self) {
        let n = self.rng.range(3, 12);
        for i in 0..n {
            if i > 0 && self.rng.chance(85) {
                self.out.push(' ');
            }
            match self.rng.below(6) {
                0 | 1 => self.foreign_word(),
                2 => {
                    let w = *self.rng.pick(EN_WORDS);
                    self.out.push_str(w);
                }
                3 => {
                    let s = *self.rng.pick(SYMBOLS);
                    self.out.push_str(s);
                }
                4 => {
                    let m = *self.rng.pick(MATH_LETTERS);
                    self.out.push_str(m);
                }
                _ => {
                    let e = *self.rng.pick(EMOJI);
                    self.out.push_str(e);
                }
            }
        }
        let t = *self.rng.pick(TERMINALS);
        self.out.push_str(t);
    }

    /// Punctuation on its own and piled up.
    fn dense_punctuation(&mut self) {
        let n = self.rng.range(2, 12);
        for i in 0..n {
            if i > 0 && self.rng.chance(50) {
                self.out.push(' ');
            }
            match self.rng.below(5) {
                0 => {
                    let t = *self.rng.pick(TERMINALS);
                    self.out.push_str(t);
                }
                1 => {
                    let d = *self.rng.pick(DASHES);
                    self.out.push_str(d);
                }
                2 => {
                    let q = *self.rng.pick(OPEN_QUOTES);
                    self.out.push_str(q);
                    let q = *self.rng.pick(CLOSE_QUOTES);
                    self.out.push_str(q);
                }
                3 => {
                    let s = *self.rng.pick(SYMBOLS);
                    self.out.push_str(s);
                }
                _ => {
                    let w = *self.rng.pick(EN_WORDS);
                    self.out.push_str(w);
                }
            }
        }
    }

    fn push_capitalized(&mut self, w: &str) {
        let mut chars = w.chars();
        if let Some(c) = chars.next() {
            self.out.extend(c.to_uppercase());
            self.out.push_str(chars.as_str());
        }
    }

    /// The whole text changed in ways that reach into tokens: zero width
    /// and format characters inside words, combining marks after letters,
    /// Unicode spaces for ASCII spaces, control characters, other line ends,
    /// a byte order mark, repeated characters.
    fn mutate(&mut self, text: String) -> String {
        let mut text = text;
        if self.rng.chance(20) {
            text = self.insert_between_letters(&text, ZERO_WIDTH, 6);
        }
        if self.rng.chance(12) {
            text = self.insert_between_letters(&text, BIDI, 4);
        }
        if self.rng.chance(22) {
            text = self.insert_between_letters(&text, COMBINING, 8);
        }
        if self.rng.chance(18) {
            let space = *self.rng.pick(SPACES);
            let every = self.rng.range(1, 4);
            let mut n = 0;
            text = text
                .chars()
                .flat_map(|c| {
                    if c == ' ' {
                        n += 1;
                        if n % every == 0 {
                            return space.chars().collect::<Vec<_>>();
                        }
                    }
                    vec![c]
                })
                .collect();
        }
        if self.rng.chance(5) {
            text = self.insert_between_letters(&text, CONTROLS, 2);
        }
        match self.rng.below(100) {
            0..=19 => text = text.replace('\n', "\r\n"),
            20..=25 => text = text.replace('\n', "\r"),
            26..=29 => text = text.replace('\n', "\n\n"),
            30..=32 => text = text.replace('\n', " \n"),
            _ => {}
        }
        if self.rng.chance(5) {
            text.insert(0, '\u{FEFF}');
        }
        if self.rng.chance(8) {
            let lead = *self.rng.pick(SPACES);
            text.insert_str(0, lead);
        }
        text
    }

    /// `pieces` dropped in after letters, about one in `one_in` of them,
    /// so that the insertions land inside words.
    fn insert_between_letters(&mut self, text: &str, pieces: &[&str], per_hundred: usize) -> String {
        let mut out = String::with_capacity(text.len() + 16);
        for c in text.chars() {
            out.push(c);
            if c.is_alphabetic() && self.rng.chance(per_hundred) {
                let p = *self.rng.pick(pieces);
                out.push_str(p);
            }
        }
        out
    }
}
