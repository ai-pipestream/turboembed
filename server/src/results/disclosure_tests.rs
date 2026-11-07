use super::*;
use crate::api::{TURBO_E_INVALID_STATE, TURBO_E_UNSUPPORTED_OPTION};

fn opts(rewritten_text: bool, full_text: bool, detail: bool, values: PiiValueForm) -> PiiOutputOptions {
    PiiOutputOptions { rewritten_text, full_text, detail, values: values as i32 }
}

#[test]
fn unset_and_unspecified_options_are_findings_only() {
    let all = PiiOutputPolicy::ALL;
    assert_eq!(Disclosure::decide(None, all).unwrap(), Disclosure::FINDINGS);
    let unset = PiiOutputOptions::default();
    assert_eq!(Disclosure::decide(Some(&unset), all).unwrap(), Disclosure::FINDINGS);
    let withheld = opts(false, false, false, PiiValueForm::Withheld);
    assert_eq!(Disclosure::decide(Some(&withheld), PiiOutputPolicy::NONE).unwrap(), Disclosure::FINDINGS);
}

#[test]
fn every_value_form_but_withheld_is_refused_until_grants_exist() {
    for form in [PiiValueForm::Token, PiiValueForm::Ciphertext, PiiValueForm::Clear] {
        let f = Disclosure::decide(Some(&opts(false, false, false, form)), PiiOutputPolicy::ALL).unwrap_err();
        assert_eq!(f.code, TURBO_E_UNSUPPORTED_OPTION, "{form:?}");
        assert!(f.message.contains("#1909"), "{}", f.message);
    }
    let unknown = PiiOutputOptions { values: 99, ..Default::default() };
    assert_eq!(Disclosure::decide(Some(&unknown), PiiOutputPolicy::ALL).unwrap_err().code, TURBO_E_UNSUPPORTED_OPTION);
}

#[test]
fn an_option_the_policy_refuses_fails_the_call_by_name() {
    let only_rewrite = PiiOutputPolicy { rewritten_text: true, ..PiiOutputPolicy::NONE };
    let asked = opts(true, false, false, PiiValueForm::Unspecified);
    assert_eq!(
        Disclosure::decide(Some(&asked), only_rewrite).unwrap(),
        Disclosure { rewritten_text: true, ..Disclosure::FINDINGS }
    );
    for (o, name) in [
        (opts(false, true, false, PiiValueForm::Unspecified), "full_text"),
        (opts(false, false, true, PiiValueForm::Unspecified), "detail"),
    ] {
        let f = Disclosure::decide(Some(&o), only_rewrite).unwrap_err();
        assert_eq!(f.code, TURBO_E_INVALID_STATE);
        assert!(f.message.contains(name), "{}", f.message);
    }
}

#[test]
fn flag_lists_parse_once_each_and_never_allow_values() {
    assert_eq!(PiiOutputPolicy::parse("none").unwrap(), PiiOutputPolicy::NONE);
    assert_eq!(PiiOutputPolicy::parse("rewritten-text,full-text,detail").unwrap(), PiiOutputPolicy::ALL);
    assert_eq!(Disclosure::parse("full-text").unwrap(), Disclosure { full_text: true, ..Disclosure::FINDINGS });
    assert!(Disclosure::parse("detail,detail").unwrap_err().contains("twice"));
    assert!(Disclosure::parse("values").unwrap_err().contains("#1909"));
    assert!(Disclosure::parse("").is_err());
    assert!(PiiOutputPolicy::NONE.allows(&Disclosure::FINDINGS));
    assert!(!PiiOutputPolicy::NONE.allows(&Disclosure::FULL));
}
