use super::*;

fn piece(s: &str) -> Piece<'_> {
    Piece { text: s, span: Span::new(0, s.len() as u32) }
}

fn leaf(s: &str) -> (u32, Leaf<'_>) {
    match parse_line(piece(s)) {
        Some((level, Element::Leaf(l))) => (level, l),
        other => panic!("{s}: {other:?}"),
    }
}

fn text(p: Option<Piece<'_>>) -> Option<&str> {
    p.map(|p| p.text)
}

#[test]
fn nodes_take_the_tag_after_the_level() {
    let Some((level, Element::Node { tag, .. })) = parse_line(piece("==P<:np")) else { panic!() };
    assert_eq!((level, tag.unwrap().text), (3, "P<:np"));
    let Some((level, Element::Node { tag, .. })) = parse_line(piece("STA:fcl(x) (<y>)")) else { panic!() };
    assert_eq!((level, tag.unwrap().text), (1, "STA:fcl"));
    assert!(parse_line(piece("STA:fcl (x)")).is_some_and(|(_, e)| matches!(e, Element::Leaf(_))));
    // A hyphen at the end of the prefix moves into the tag when the line
    // does not read otherwise.
    let Some((level, Element::Node { tag, .. })) = parse_line(piece("=-:np")) else { panic!() };
    assert_eq!((level, tag.unwrap().text), (2, "-:np"));
}

#[test]
fn leaves_split_their_parts() {
    let (level, l) = leaf("===H:n(\"dia\" <np-def> <temp> M S)\tdia");
    assert_eq!(level, 4);
    assert_eq!(text(l.syntactic), Some("H"));
    assert_eq!(text(l.functional), Some("n"));
    assert_eq!(text(l.lemma), Some("dia"));
    assert_eq!(text(l.secondary), Some("<np-def> <temp>"));
    assert_eq!(text(l.morphological), Some("M S"));
    assert_eq!(l.lexeme.text, "dia");
    assert_eq!(l.lexeme.span, Span::new(34, 37));
    let (_, l) = leaf("=H:prp('de')  de");
    assert_eq!((text(l.secondary), text(l.morphological)), (Some(""), None));
}

#[test]
fn punctuation_and_odd_lines() {
    let (level, l) = leaf("==,");
    assert_eq!((level, l.lexeme.text, l.functional), (3, ",", None));
    let (level, l) = leaf("==");
    assert_eq!((level, l.lexeme.text), (2, "="));
    assert!(parse_line(piece("_")).is_none());
    assert!(parse_line(piece("=a.b")).is_none());
    let (level, l) = leaf("=é=xyz");
    assert_eq!((level, l.lexeme.text, text(l.functional)), (4, "xyz", Some("")));
    let (level, l) = leaf("loose words");
    assert_eq!((level, l.lexeme.text), (1, "loose words"));
    let (level, l) = leaf("=H==CJT:num(\"818\" M) 818");
    assert_eq!(
        (level, text(l.syntactic), text(l.functional), text(l.lemma)),
        (2, Some("H==CJT"), Some("num"), Some("818"))
    );
}
