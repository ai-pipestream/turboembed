//! Project-authored boundary targets, not output captured from a detector.

pub struct Target {
    pub name: &'static str,
    pub language: &'static str,
    pub text: &'static str,
    pub sentences: &'static [&'static str],
}

pub const TARGETS: &[Target] = &[
    Target {
        name: "titles_and_decimal",
        language: "en",
        text: "Dr. Reed paid 3.50 euros. Next came lunch!",
        sentences: &["Dr. Reed paid 3.50 euros.", "Next came lunch!"],
    },
    Target {
        name: "repeated_content",
        language: "en",
        text: "Go now. Go now. Go now.",
        sentences: &["Go now.", "Go now.", "Go now."],
    },
    Target {
        name: "astral_and_combining",
        language: "en",
        text: "😀 Cafe\u{301} opens. 𐐀 Waits outside!",
        sentences: &["😀 Cafe\u{301} opens.", "𐐀 Waits outside!"],
    },
    Target {
        name: "quoted_terminator",
        language: "en",
        text: "She said \"Wait!\" Then left.",
        sentences: &["She said \"Wait!\"", "Then left."],
    },
    Target {
        name: "soft_line",
        language: "en",
        text: "The lantern\r\nstays lit. Next room.",
        sentences: &["The lantern\r\nstays lit.", "Next room."],
    },
    Target {
        name: "paragraph_without_punctuation",
        language: "en",
        text: "First room\n\nSecond room",
        sentences: &["First room", "Second room"],
    },
    Target {
        name: "unicode_paragraph",
        language: "en",
        text: "First room\u{2029}Second room",
        sentences: &["First room", "Second room"],
    },
    Target {
        name: "french",
        language: "fr",
        text: "La porte est ouverte. Entrez!",
        sentences: &["La porte est ouverte.", "Entrez!"],
    },
    Target {
        name: "german",
        language: "de",
        text: "Die Tür ist offen. Komm herein!",
        sentences: &["Die Tür ist offen.", "Komm herein!"],
    },
    Target {
        name: "spanish",
        language: "es",
        text: "La puerta está abierta. Entra!",
        sentences: &["La puerta está abierta.", "Entra!"],
    },
    Target {
        name: "japanese", language: "ja", text: "猫が来た。犬も来た！", sentences: &["猫が来た。", "犬も来た！"]
    },
    Target {
        name: "chinese", language: "zh", text: "门开了。请进！", sentences: &["门开了。", "请进！"]
    },
    Target { name: "empty", language: "en", text: "", sentences: &[] },
    Target { name: "only_whitespace", language: "en", text: " \t\r\n\u{a0}\u{3000}", sentences: &[] },
];
