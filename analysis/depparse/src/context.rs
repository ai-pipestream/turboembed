use std::borrow::Cow;

use crate::{ArcStandardState, Position, Result, error};
use turbo_analysis_layers::WordOrTokenAnchor;

pub struct ParserInput;
impl ParserInput {
    pub fn check(tokens: &[&str], tags: &[&str]) -> Result<()> {
        if tokens.is_empty() || tokens.len() != tags.len() || tokens.len() > u32::MAX as usize {
            return Err(error("nonempty, aligned tokens and tags required"));
        }
        Ok(())
    }
}
pub struct DependencyContextGenerator;
impl DependencyContextGenerator {
    pub fn context<A: WordOrTokenAnchor>(
        state: &ArcStandardState<A>,
        tokens: &[&str],
        tags: &[&str],
    ) -> Result<Vec<String>> {
        ParserInput::check(tokens, tags)?;
        if tokens.len() != state.token_count() {
            return Err(error("input does not match state"));
        }
        let (s0, s1, s2) = (state.stack(0), state.stack(1), state.stack(2));
        let (b0, b1, b2) = (state.buffer(0), state.buffer(1), state.buffer(2));
        let s0w = value(tokens, s0);
        let s0t = value(tags, s0);
        let s1w = value(tokens, s1);
        let s1t = value(tags, s1);
        let s2t = value(tags, s2);
        let b0w = value(tokens, b0);
        let b0t = value(tags, b0);
        let b1w = value(tokens, b1);
        let b1t = value(tags, b1);
        let b2t = value(tags, b2);
        let child = |p, left| match p {
            Position::Node(i) => {
                if left {
                    state.left[i.get()]
                } else {
                    state.right[i.get()]
                }
            }
            _ => None,
        };
        let child_tag = |p, left| value(tags, child(p, left).map_or(Position::Missing, Position::Node));
        let child_rel = |p, left| child(p, left).and_then(|i| state.relations[i.get()].as_deref()).unwrap_or("*NULL*");
        let s0lct = child_tag(s0, true);
        let s0rct = child_tag(s0, false);
        let s1lct = child_tag(s1, true);
        let s1rct = child_tag(s1, false);
        let s0lcl = child_rel(s0, true);
        let s0rcl = child_rel(s0, false);
        let s1rcl = child_rel(s1, false);
        let count = |p| match p {
            Position::Node(i) => state.counts[i.get()].min(3).to_string(),
            _ => "*NULL*".into(),
        };
        let s0deps = count(s0);
        let s1deps = count(s1);
        let dist = match (s0, b0) {
            (Position::Node(s), Position::Node(b)) => {
                let d = b.get() as i64 - s.get() as i64;
                if d >= 4 {
                    "4+".into()
                } else if d <= -4 {
                    "-4+".into()
                } else {
                    d.to_string()
                }
            }
            _ => "*NULL*".into(),
        };
        // The values the combined features join, with the separators
        // in them escaped so different configurations cannot combine
        // into the same feature. The single-value features below keep
        // the values as they are.
        let es0w = escape(s0w);
        let es0t = escape(s0t);
        let es1w = escape(s1w);
        let es1t = escape(s1t);
        let es2t = escape(s2t);
        let eb0w = escape(b0w);
        let eb0t = escape(b0t);
        let eb1t = escape(b1t);
        let es0lct = escape(s0lct);
        let es1rct = escape(s1rct);
        Ok(vec![
            format!("s0w={s0w}"),
            format!("s0t={s0t}"),
            format!("s1w={s1w}"),
            format!("s1t={s1t}"),
            format!("s2t={s2t}"),
            format!("b0w={b0w}"),
            format!("b0t={b0t}"),
            format!("b1w={b1w}"),
            format!("b1t={b1t}"),
            format!("b2t={b2t}"),
            format!("s0wt={es0w}/{es0t}"),
            format!("s1wt={es1w}/{es1t}"),
            format!("b0wt={eb0w}/{eb0t}"),
            format!("s0w,b0w={es0w}|{eb0w}"),
            format!("s0t,b0t={es0t}|{eb0t}"),
            format!("s0w,b0t={es0w}|{eb0t}"),
            format!("s0t,b0w={es0t}|{eb0w}"),
            format!("s0wt,b0t={es0w}/{es0t}|{eb0t}"),
            format!("s1t,s0t={es1t}|{es0t}"),
            format!("s1t,s0w={es1t}|{es0w}"),
            format!("s1w,s0t={es1w}|{es0t}"),
            format!("s1t,s0t,b0t={es1t}|{es0t}|{eb0t}"),
            format!("s0t,b0t,b1t={es0t}|{eb0t}|{eb1t}"),
            format!("s2t,s1t,s0t={es2t}|{es1t}|{es0t}"),
            format!("s0lct={s0lct}"),
            format!("s0rct={s0rct}"),
            format!("s1lct={s1lct}"),
            format!("s1rct={s1rct}"),
            format!("s0lcl={s0lcl}"),
            format!("s0rcl={s0rcl}"),
            format!("s1rcl={s1rcl}"),
            format!("s1t,s1rct,s0t={es1t}|{es1rct}|{es0t}"),
            format!("s0t,s0lct,b0t={es0t}|{es0lct}|{eb0t}"),
            format!("s0deps={s0deps}"),
            format!("s1deps={s1deps}"),
            format!("dist={dist}"),
            format!("dist,s0t,b0t={dist}|{es0t}|{eb0t}"),
        ])
    }
}
/// `value` with every `\`, `/` and `|` in it preceded by a `\`, so a
/// feature joining several values with those separators reads one way.
fn escape(value: &str) -> Cow<'_, str> {
    if !value.contains(['\\', '/', '|']) {
        return Cow::Borrowed(value);
    }
    let mut out = String::with_capacity(value.len() + 4);
    for c in value.chars() {
        if matches!(c, '\\' | '/' | '|') {
            out.push('\\');
        }
        out.push(c);
    }
    Cow::Owned(out)
}

fn value<'a, A>(values: &[&'a str], p: Position<A>) -> &'a str {
    match p {
        Position::Root => "*ROOT*",
        Position::Missing => "*NULL*",
        Position::Node(i) => values[i.get()],
    }
}
