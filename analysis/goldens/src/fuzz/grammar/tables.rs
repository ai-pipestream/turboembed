//! Word, symbol, and character tables for the fuzz grammar generator.

pub(super) const EN_WORDS: &[&str] = &[
    "the", "a", "an", "this", "that", "our", "their", "some", "every", "no", "quick", "brown", "lazy", "old", "new",
    "little", "large", "red", "early", "late", "cold", "fox", "dog", "report", "meeting", "price", "city", "river",
    "house", "car", "team", "year", "day", "night", "water", "paper", "data", "model", "test", "server", "network",
    "machine", "answer", "question", "letter", "line", "word", "sentence", "token", "offset", "boundary",
];
pub(super) const EN_VERBS: &[&str] = &[
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
pub(super) const EN_NAMES: &[&str] = &[
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
pub(super) const TITLES: &[&str] = &["Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Sen.", "Gen.", "St.", "Capt.", "Rev."];
pub(super) const ABBREVIATIONS: &[&str] = &[
    "U.S.", "U.K.", "e.g.", "i.e.", "etc.", "Inc.", "Ltd.", "Co.", "Corp.", "Jan.", "Feb.", "Dec.", "No.", "vs.",
    "approx.", "Fig.", "p.m.", "a.m.", "Ph.D.", "cf.", "Mt.", "Ave.", "Blvd.", "Jr.", "Sr.", "et al.", "ca.", "z.B.",
    "usw.", "bzw.", "Nr.", "Str.", "S.A.", "Sra.", "Sr.", "p.ej.", "M.", "Mme", "Mlle", "cf.",
];
pub(super) const DE_WORDS: &[&str] = &[
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
pub(super) const FR_WORDS: &[&str] = &[
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
pub(super) const ES_PT_WORDS: &[&str] = &[
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
pub(super) const GREEK_WORDS: &[&str] =
    &["Αθήνα", "ΟΔΥΣΣΕΥΣ", "λόγος", "σοφός", "ΣΟΦΟΣ", "θάλασσα", "άνθρωπος", "Ελλάδα", "σας", "ΑΣ", "ΐ", "ΰ"];
pub(super) const CYRILLIC_WORDS: &[&str] =
    &["Москва", "привет", "мир", "Россия", "ёлка", "Київ", "Україна", "българия", "Ђорђе", "Љубљана", "ЁЖ"];
pub(super) const ARABIC_WORDS: &[&str] =
    &["السلام", "عليكم", "مرحبا", "القاهرة", "بغداد", "كتاب", "مدرسة", "٢٠٢٤", "١٢٣", "العربية", "شكرًا", "اللّغة"];
pub(super) const HEBREW_WORDS: &[&str] = &["שלום", "עולם", "ירושלים", "ספר", "בית", "תּוֹרָה", "ישראל", "ברוך", "הבא"];
pub(super) const DEVANAGARI_WORDS: &[&str] =
    &["नमस्ते", "भारत", "किताब", "हिन्दी", "दिल्ली", "क्षत्रिय", "संस्कृत", "९२३", "ज्ञान", "श्री", "मुम्बई"];
pub(super) const THAI_WORDS: &[&str] = &["สวัสดี", "ประเทศไทย", "ภาษาไทย", "กรุงเทพ", "ขอบคุณ", "น้ำ", "เด็ก", "๑๒๓"];
pub(super) const CJK_WORDS: &[&str] = &[
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
pub(super) const CJK_PUNCT: &[&str] =
    &["。", "、", "，", "！", "？", "「", "」", "『", "』", "（", "）", "：", "；", "…", "〜", "・"];
pub(super) const UNITS: &[&str] = &[
    "kg", "g", "mg", "km", "m", "cm", "mm", "mi", "ft", "in", "l", "ml", "h", "min", "s", "ms", "°C", "°F", "%", "kWh",
    "MB", "GB", "km/h", "mph", "lb", "oz", "ha", "m²", "m³", "V", "W", "Hz",
];
pub(super) const CURRENCY_SYMBOLS: &[&str] =
    &["$", "€", "£", "¥", "₹", "₽", "₩", "CHF ", "USD ", "EUR ", "US$", "A$", "C$", "R$"];
pub(super) const CURRENCY_CODES: &[&str] =
    &["USD", "EUR", "GBP", "JPY", "CHF", "CNY", "INR", "RUB", "BRL", "CAD", "AUD"];
pub(super) const SCALE_WORDS: &[&str] =
    &["k", "K", "m", "M", "bn", "b", "B", "thousand", "million", "billion", "trillion", "Mio.", "Mrd."];
pub(super) const DOMAINS: &[&str] = &[
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
pub(super) const PATHS: &[&str] = &[
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
pub(super) const EMAIL_LOCAL: &[&str] =
    &["john.doe", "jane", "a", "first+tag", "name_1", "José", "用戶", "o'brien", "x.y.z"];
pub(super) const EMOJI: &[&str] = &[
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
pub(super) const ZERO_WIDTH: &[&str] = &[
    "\u{200B}", "\u{200C}", "\u{200D}", "\u{2060}", "\u{FEFF}", "\u{00AD}", "\u{034F}", "\u{180E}", "\u{FE0F}",
    "\u{FE0E}", "\u{061C}",
];
pub(super) const BIDI: &[&str] = &[
    "\u{200E}", "\u{200F}", "\u{202A}", "\u{202B}", "\u{202C}", "\u{202D}", "\u{202E}", "\u{2066}", "\u{2067}",
    "\u{2068}", "\u{2069}",
];
pub(super) const COMBINING: &[&str] = &[
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
pub(super) const SPACES: &[&str] = &[
    "\u{00A0}", "\u{2002}", "\u{2003}", "\u{2009}", "\u{200A}", "\u{202F}", "\u{205F}", "\u{3000}", "\u{1680}",
    "\u{2000}", "\u{2007}", "\u{2008}", "\t", "\u{000B}", "\u{000C}", "\u{0085}", "\u{2028}", "\u{2029}", "\u{001C}",
    "\u{001D}", "\u{001E}", "\u{001F}", "  ", "   ",
];
pub(super) const CONTROLS: &[&str] =
    &["\u{0001}", "\u{0002}", "\u{001B}", "\u{007F}", "\u{0080}", "\u{009F}", "\u{0000}"];
pub(super) const OPEN_QUOTES: &[&str] = &["\"", "“", "‘", "'", "«", "„", "‹", "『", "「", "`", "``"];
pub(super) const CLOSE_QUOTES: &[&str] = &["\"", "”", "’", "'", "»", "“", "›", "』", "」", "'", "''"];
pub(super) const DASHES: &[&str] = &["-", "–", "—", "―", "‐", "‑", "−", "--", "---", " - ", " – ", " — "];
pub(super) const TERMINALS: &[&str] = &[
    ".", ".", ".", ".", "!", "?", "...", "…", "?!", "!!!", "..", ".)", ").", ".\"", ".”", "\".", "?\"", "!)", ".’",
    ":", ";", ",", "", "", ". . .", "。", "！", "？",
];
pub(super) const SEPARATORS: &[&str] =
    &[" ", " ", " ", " ", "  ", "\n", "\u{00A0}", "\t", "", " \n", "\u{3000}", "\u{2028}"];
pub(super) const CONTRACTIONS: &[&str] = &[
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
pub(super) const SOCIAL: &[&str] = &[
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
pub(super) const SYMBOLS: &[&str] = &[
    "&", "%", "©", "®", "™", "°", "§", "¶", "+", "=", "<", ">", "*", "/", "~", "^", "|", "#", "@", "$", "€", "¢", "½",
    "¼", "²", "³", "Ⅳ", "ⅷ", "№", "℃", "㎏", "㍿", "㌀", "ﬁ", "ﬂ", "ﬃ", "ﬆ", "Ǆ", "ǅ", "ǳ", "ẞ", "ß", "İ", "ı", "ſ",
    "ǰ", "ΐ", "ﬀ", "Ａ", "ｂ", "１", "２", "！", "？", "ｱ", "ｲ", "ｳ", "ｶﾞ", "ﾊﾟ", "pаypal", "соm", "ΑΒΓ",
];
pub(super) const UNICODE_DIGITS: &[&str] = &["٠١٢٣", "٤٥٦", "۷۸۹", "९२३", "๑๒๓", "１２３", "४५६", "০১২", "௧௨௩", "𝟏𝟐𝟑"];
pub(super) const MATH_LETTERS: &[&str] = &["𝐀𝐁𝐂", "𝑎𝑏𝑐", "𝔸𝔹", "𝒳", "𝟙", "𐐷", "𑀓", "𐌰𐌱", "𓀀𓀁", "𝄞", "𝕳𝖊𝖑𝖑𝖔"];
