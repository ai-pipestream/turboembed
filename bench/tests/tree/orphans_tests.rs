//! The module walk of `orphans.rs`, on small trees.

use std::collections::{BTreeMap, BTreeSet};

use super::orphans;

fn tree(files: &[(&str, &str)]) -> BTreeMap<String, String> {
    files.iter().map(|(p, t)| ((*p).to_owned(), (*t).to_owned())).collect()
}

/// The orphans of `files` in one crate at `a`, with no manifest targets.
fn found(files: &[(&str, &str)]) -> Vec<String> {
    orphans(&tree(files), &BTreeSet::from(["a".to_owned()]), &BTreeSet::new())
}

#[test]
fn a_directory_left_without_its_root_is_reported() {
    // what #2455 found: the test root is gone, its support modules are not
    let files = [
        ("a/src/lib.rs", "pub mod kept;\n"),
        ("a/src/kept.rs", ""),
        ("a/tests/live/main.rs", "mod helper;\n"),
        ("a/tests/live/helper.rs", ""),
        ("a/tests/gone/check.rs", "use crate::golden::*;\n"),
        ("a/tests/gone/golden.rs", ""),
    ];
    assert_eq!(found(&files), ["a/tests/gone/check.rs", "a/tests/gone/golden.rs"]);
}

#[test]
fn each_way_of_naming_a_file_counts() {
    let files = [
        ("a/src/lib.rs", "mod plain;\npub(crate) mod nested;\nmod r#type;\n#[cfg(test)]\nmod gated;\n"),
        (
            "a/src/plain.rs",
            "#[path = \"elsewhere/by_path.rs\"]\nmod renamed;\ninclude!(\"parts/included.rs\");\nmod child;\n",
        ),
        ("a/src/plain/child.rs", ""),
        ("a/src/nested/mod.rs", "mod leaf;\n"),
        ("a/src/nested/leaf.rs", ""),
        ("a/src/type.rs", ""),
        ("a/src/gated.rs", ""),
        // a file named by #[path] looks for its own modules beside itself
        ("a/src/elsewhere/by_path.rs", "mod beside;\n"),
        ("a/src/elsewhere/beside.rs", ""),
        // an included file's modules belong to the module that includes it
        ("a/src/parts/included.rs", "mod from_include;\n"),
        ("a/src/plain/from_include.rs", ""),
        ("a/tests/t.rs", "#[cfg_attr(test, path = \"../src/shared.rs\")]\nmod shared;\nmod common;\n"),
        ("a/tests/common/mod.rs", ""),
        ("a/src/shared.rs", ""),
        ("a/src/bin/tool.rs", ""),
        ("a/src/bin/big/main.rs", "mod part;\n"),
        ("a/src/bin/big/part.rs", ""),
        ("a/examples/demo/main.rs", ""),
        ("a/benches/speed.rs", ""),
        ("a/build.rs", ""),
        ("a/tools/gen.rs", ""),
    ];
    let crates = BTreeSet::from(["a".to_owned()]);
    let manifest: BTreeSet<String> = ["a/tools/gen.rs".to_owned()].into();
    assert_eq!(orphans(&tree(&files), &crates, &manifest), Vec::<String>::new());
    // without the manifest entry the tool is nobody's
    assert_eq!(found(&files), ["a/tools/gen.rs"]);
}

#[test]
fn modules_inside_inline_modules_are_followed_however_deep() {
    // the codex review's case: two inline levels, then a file
    let files = [
        ("a/src/lib.rs", "mod one { mod two { mod leaf; } }\nmod snow { pub mod alg { #[path = \"x.rs\"] mod y; } }\n"),
        ("a/src/one/two/leaf.rs", "mod under;\n"),
        ("a/src/one/two/leaf/under.rs", ""),
        ("a/src/snow/alg/x.rs", ""),
        // a module declared inside a function body is a file of the enclosing module
        ("a/src/main.rs", "fn main() { if true { mod local; } }\n"),
        ("a/src/local.rs", ""),
        // the same name one level off is a different file, and is not reached
        ("a/src/one/leaf.rs", ""),
        ("a/src/leaf.rs", ""),
    ];
    assert_eq!(found(&files), ["a/src/leaf.rs", "a/src/one/leaf.rs"]);
}

#[test]
fn a_module_directory_named_tests_is_not_a_target_directory() {
    // the cursor review's case: src/**/tests/ is a module, so a stray file in it is an orphan
    let files = [
        ("a/src/lib.rs", "mod emotion;\n"),
        ("a/src/emotion/mod.rs", "#[cfg(test)]\nmod tests;\n"),
        ("a/src/emotion/tests/mod.rs", "mod cases;\n"),
        ("a/src/emotion/tests/cases.rs", ""),
        ("a/src/emotion/tests/stray.rs", ""),
        ("a/src/bin/tests/stray.rs", ""),
        ("a/tests/deep/tests/stray.rs", ""),
    ];
    assert_eq!(
        found(&files),
        ["a/src/bin/tests/stray.rs", "a/src/emotion/tests/stray.rs", "a/tests/deep/tests/stray.rs"]
    );
}

#[test]
fn words_that_only_look_like_a_declaration_do_not_count() {
    let files = [
        ("a/src/lib.rs", "// mod commented;\n/* mod blocked; */\nfn f() { let model = \"mod quoted;\"; }\n"),
        ("a/src/commented.rs", ""),
        ("a/src/blocked.rs", ""),
        ("a/src/quoted.rs", ""),
        // an inline module names no file, and a path in a string or a comment names none
        (
            "a/src/main.rs",
            "mod tests { }\n// #[path = \"stray.rs\"] include!(\"stray.rs\");\nconst S: &str = \"include!(\\\"stray.rs\\\")\";\n",
        ),
        ("a/src/tests.rs", ""),
        ("a/src/stray.rs", ""),
        // a module a macro declares is not seen: DECLARED_BY_MACRO is for these
        ("a/src/bin/m/main.rs", "macro_rules! modules { ($m:ident) => { mod $m; } }\nmodules!(by_macro);\n"),
        ("a/src/bin/m/by_macro.rs", ""),
    ];
    assert_eq!(
        found(&files),
        [
            "a/src/bin/m/by_macro.rs",
            "a/src/blocked.rs",
            "a/src/commented.rs",
            "a/src/quoted.rs",
            "a/src/stray.rs",
            "a/src/tests.rs"
        ]
    );
}

#[test]
fn a_crate_inside_another_crate_has_its_own_targets() {
    let files = [
        ("a/src/lib.rs", ""),
        ("a/inner/src/lib.rs", "mod m;\n"),
        ("a/inner/src/m.rs", ""),
        ("a/inner/tests/t.rs", ""),
        // not a crate: nothing makes these targets
        ("a-extra/src/lib.rs", ""),
        ("a/other/tests/t.rs", ""),
    ];
    let crates = BTreeSet::from(["a".to_owned(), "a/inner".to_owned()]);
    assert_eq!(orphans(&tree(&files), &crates, &BTreeSet::new()), ["a-extra/src/lib.rs", "a/other/tests/t.rs"]);
}
