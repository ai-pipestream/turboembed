use turbo_depparse::{DependencyModel, DependencyParserME};

pub fn raw(outcomes: &[&str], feature: &str, weights: &[f64]) -> Vec<u8> {
    fn utf(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u16).to_be_bytes());
        out.extend_from_slice(s.as_bytes());
    }
    let mut out = Vec::new();
    utf(&mut out, "QN");
    out.extend_from_slice(&(outcomes.len() as i32).to_be_bytes());
    for s in outcomes {
        utf(&mut out, s);
    }
    out.extend_from_slice(&1i32.to_be_bytes());
    utf(&mut out, &format!("1 {}", (0..outcomes.len()).map(|i| i.to_string()).collect::<Vec<_>>().join(" ")));
    out.extend_from_slice(&1i32.to_be_bytes());
    utf(&mut out, feature);
    for w in weights {
        out.extend_from_slice(&w.to_be_bytes());
    }
    out
}
pub fn parser() -> DependencyParserME {
    DependencyParserME::new(
        DependencyModel::from_raw(&raw(&["SHIFT", "LEFT_ARC:dep", "RIGHT_ARC:dep"], "s0t=NOUN", &[0.0, 1.0, 2.0]))
            .unwrap(),
    )
}
