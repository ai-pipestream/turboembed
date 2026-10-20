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
                let d = b.get() - s.get();
                if d >= 4 { "4+".into() } else { d.to_string() }
            }
            _ => "*NULL*".into(),
        };
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
            format!("s0wt={s0w}/{s0t}"),
            format!("s1wt={s1w}/{s1t}"),
            format!("b0wt={b0w}/{b0t}"),
            format!("s0w,b0w={s0w}|{b0w}"),
            format!("s0t,b0t={s0t}|{b0t}"),
            format!("s0w,b0t={s0w}|{b0t}"),
            format!("s0t,b0w={s0t}|{b0w}"),
            format!("s0wt,b0t={s0w}/{s0t}|{b0t}"),
            format!("s1t,s0t={s1t}|{s0t}"),
            format!("s1t,s0w={s1t}|{s0w}"),
            format!("s1w,s0t={s1w}|{s0t}"),
            format!("s1t,s0t,b0t={s1t}|{s0t}|{b0t}"),
            format!("s0t,b0t,b1t={s0t}|{b0t}|{b1t}"),
            format!("s2t,s1t,s0t={s2t}|{s1t}|{s0t}"),
            format!("s0lct={s0lct}"),
            format!("s0rct={s0rct}"),
            format!("s1lct={s1lct}"),
            format!("s1rct={s1rct}"),
            format!("s0lcl={s0lcl}"),
            format!("s0rcl={s0rcl}"),
            format!("s1rcl={s1rcl}"),
            format!("s1t,s1rct,s0t={s1t}|{s1rct}|{s0t}"),
            format!("s0t,s0lct,b0t={s0t}|{s0lct}|{b0t}"),
            format!("s0deps={s0deps}"),
            format!("s1deps={s1deps}"),
            format!("dist={dist}"),
            format!("dist,s0t,b0t={dist}|{s0t}|{b0t}"),
        ])
    }
}
fn value<'a, A>(values: &[&'a str], p: Position<A>) -> &'a str {
    match p {
        Position::Root => "*ROOT*",
        Position::Missing => "*NULL*",
        Position::Node(i) => values[i.get()],
    }
}
