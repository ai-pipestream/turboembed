use super::*;
struct Sink;
impl Handler for Sink {
    fn start_element(&mut self, _: &str, _: &[Attribute<'_>], _: usize) -> Result<()> {
        Ok(())
    }
    fn end_element(&mut self, _: &str, _: usize) -> Result<()> {
        Ok(())
    }
    fn characters(&mut self, _: &str) {}
}

#[test]
fn malformed_declarations_are_refused() {
    for text in [
        "<?xml ?><a/>",
        "<?xml version='1.0?><a/>",
        "<?xml version='1.0' junk='x'?><a/>",
        "<?xml encoding='UTF-8' version='1.0'?><a/>",
    ] {
        assert!(parse(text, &mut Sink).is_err(), "accepted {text}");
    }
}

#[test]
fn references_and_unclosed_markup_are_refused() {
    for text in [
        "<a>&#xD800;</a>",
        "<a>&#1114112;</a>",
        "<a>&#0;</a>",
        "<a>&#xFFFFFFFFFFFFFFFF;</a>",
        "<a x='abc>",
        "<a>",
        "<a/>junk",
    ] {
        assert!(parse(text, &mut Sink).is_err(), "accepted {text}");
    }
    assert!(parse("<?xml version='1.0' encoding='UTF-8' standalone='yes'?><a>&#x10FFFF;</a>", &mut Sink).is_ok());
    let text = format!("{}{}", "<a>".repeat(10000), "</a>".repeat(10000));
    assert!(parse(&text, &mut Sink).is_ok());
}
