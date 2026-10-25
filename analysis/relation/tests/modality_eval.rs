//! A small honest eval of `ModalityAnnotator`: 16 hand-parsed sentences,
//! 24 target predicates with gold (polarity, epistemic, deontic), scored
//! for an all-asserted baseline, the token-window fallback (the same
//! sentences without dependencies) and dependency scoping. The numbers
//! are in `docs/modality.md`; misses are printed.

mod modality_support;

use modality_support::{at, modality, parsed, tokens_only};
use turbo_analysis_layers::modality::{DeonticKind as D, EpistemicStatus as E, Polarity as P};

type Gold = (P, E, D);
const AS: Gold = (P::Affirmed, E::Asserted, D::None);
const NEG: Gold = (P::Negated, E::Asserted, D::None);

/// (text, gold UD parse, (target token, gold tag)).
type Case = (&'static str, &'static str, &'static [(u32, Gold)]);
const CASES: &[Case] = &[
    (
        "Acme did not buy Beta and sold Gamma .",
        "4:nsubj 4:aux 4:advmod 0:root 4:obj 7:cc 4:conj 7:obj 4:punct",
        &[(3, NEG), (6, AS)],
    ),
    (
        "If the deal closes , Acme will expand .",
        "4:mark 3:det 4:nsubj 8:advcl 8:punct 8:nsubj 8:aux 0:root 8:punct",
        &[(3, (P::Affirmed, E::Conditional, D::None)), (7, AS)],
    ),
    (
        "Vendors must not share customer data .",
        "4:nsubj 4:aux 4:advmod 0:root 6:compound 4:obj 4:punct",
        &[(3, (P::Negated, E::Asserted, D::Prohibition))],
    ),
    (
        "Vendors must encrypt data and may delete logs .",
        "3:nsubj 3:aux 0:root 3:obj 7:cc 7:aux 3:conj 7:obj 3:punct",
        &[(2, (P::Affirmed, E::Asserted, D::Obligation)), (6, (P::Affirmed, E::Asserted, D::Permission))],
    ),
    (
        "A merger is not impossible .",
        "2:det 5:nsubj 5:cop 5:advmod 0:root 5:punct",
        &[(4, (P::Affirmed, E::Hedged, D::None))],
    ),
    (
        "Users may cancel at any time .",
        "3:nsubj 3:aux 0:root 6:case 6:det 3:obl 3:punct",
        &[(2, (P::Affirmed, E::Asserted, D::Permission))],
    ),
    ("The shares may fall .", "2:det 4:nsubj 4:aux 0:root 4:punct", &[(3, (P::Affirmed, E::Hedged, D::None))]),
    (
        "Acme allegedly said it might not sign .",
        "3:nsubj 3:advmod 0:root 7:nsubj 7:aux 7:advmod 3:ccomp 3:punct",
        &[(2, (P::Affirmed, E::Reported, D::None)), (6, (P::Negated, E::Hedged, D::None))],
    ),
    (
        "Beta failed to acquire Gamma , but it bought Delta .",
        "2:nsubj 0:root 4:mark 2:xcomp 4:obj 9:punct 9:cc 9:nsubj 2:conj 9:obj 2:punct",
        &[(3, NEG), (8, AS)],
    ),
    (
        "Regulators confirmed that the merger closed .",
        "2:nsubj 0:root 6:mark 5:det 6:nsubj 2:ccomp 2:punct",
        &[(1, AS), (5, (P::Affirmed, E::Presupposed, D::None))],
    ),
    (
        "Unless Acme pays , the contract ends .",
        "3:mark 3:nsubj 7:advcl 7:punct 6:det 7:nsubj 0:root 7:punct",
        &[(2, (P::Affirmed, E::Conditional, D::None)), (6, AS)],
    ),
    (
        "Tenants are not required to pay rent .",
        "4:nsubj 4:cop 4:advmod 0:root 6:mark 4:xcomp 6:obj 4:punct",
        &[(3, NEG)],
    ),
    (
        "Employees shall not disclose the terms .",
        "4:nsubj 4:aux 4:advmod 0:root 6:det 4:obj 4:punct",
        &[(3, (P::Negated, E::Asserted, D::Prohibition))],
    ),
    (
        "According to Reuters , Acme will buy Beta .",
        "3:case 1:fixed 7:obl 7:punct 7:nsubj 7:aux 0:root 7:obj 7:punct",
        &[(6, (P::Affirmed, E::Reported, D::None))],
    ),
    (
        "Profits will probably rise but costs did not fall .",
        "4:nsubj 4:aux 4:advmod 0:root 9:cc 9:nsubj 9:aux 9:advmod 4:conj 4:punct",
        &[(3, (P::Affirmed, E::Hedged, D::None)), (8, NEG)],
    ),
    (
        "Customers are prohibited from reselling tickets .",
        "3:nsubj:pass 3:aux:pass 0:root 5:mark 3:advcl 5:obj 3:punct",
        &[(4, (P::Affirmed, E::Asserted, D::Prohibition))],
    ),
];

#[derive(Default, Debug)]
struct Score {
    targets: usize,
    polarity: usize,
    epistemic: usize,
    deontic: usize,
    exact: usize,
}

impl Score {
    fn add(&mut self, got: Gold, gold: Gold, label: &str, text: &str, t: u32) {
        self.targets += 1;
        self.polarity += usize::from(got.0 == gold.0);
        self.epistemic += usize::from(got.1 == gold.1);
        self.deontic += usize::from(got.2 == gold.2);
        if got == gold {
            self.exact += 1;
        } else {
            println!("{label} miss: {text:?} token {t}: got {got:?}, gold {gold:?}");
        }
    }

    fn row(&self, label: &str) {
        let n = self.targets;
        println!(
            "| {label} | {}/{n} | {}/{n} | {}/{n} | {}/{n} |",
            self.polarity, self.epistemic, self.deontic, self.exact
        );
    }
}

fn run() -> (Score, Score, Score) {
    let (mut base, mut window, mut deps) = (Score::default(), Score::default(), Score::default());
    for (text, conll, targets) in CASES {
        let d = parsed(text, conll);
        let m = modality(&d);
        let w = tokens_only(text);
        let mw = modality(&w);
        for &(t, gold) in *targets {
            base.add(AS, gold, "baseline", text, t);
            window.add(at(&mw, &w, t), gold, "window", text, t);
            deps.add(at(&m, &d, t), gold, "deps", text, t);
        }
    }
    (base, window, deps)
}

#[test]
fn modality_eval() {
    let (base, window, deps) = run();
    println!("| System | Polarity | Epistemic | Deontic | Exact |");
    base.row("all asserted");
    window.row("window fallback");
    deps.row("dependency scope");
    assert_eq!(deps.targets, 24);
    assert!(window.exact > base.exact);
    assert!(deps.exact > window.exact);
    // Pinned so a regression shows up; see docs/modality.md.
    assert!(deps.exact >= 22, "{deps:?}");
}
