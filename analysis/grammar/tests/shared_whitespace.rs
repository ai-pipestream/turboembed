use turbo_grammar::engine::{Kind, Lexed, NumeralStyle};

#[test]
fn lexer_skips_unicode_spaces_but_keeps_information_separator_punctuation() {
    let text = "Dr. Smith\u{b}paid\u{85}10\u{a0}euros\u{3000}yesterday.\u{1c}Fine!";
    let style =
        NumeralStyle { mixed_separators: false, space_groups: false, indian_groups: false, decimal_comma: false };
    let mut lexed = Lexed::default();
    lexed.lex(text, &style);
    let actual: Vec<_> = lexed.lexemes.iter().map(|l| (&text[l.start as usize..l.end as usize], l.kind)).collect();
    assert_eq!(
        actual,
        [
            ("Dr", Kind::Word),
            (".", Kind::Punct),
            ("Smith", Kind::Word),
            ("paid", Kind::Word),
            ("10", Kind::Numeral),
            ("euros", Kind::Word),
            ("yesterday", Kind::Word),
            (".", Kind::Punct),
            ("\u{1c}", Kind::Punct),
            ("Fine", Kind::Word),
            ("!", Kind::Punct),
        ]
    );
}
