use std::{
    mem::{offset_of, size_of},
    path::Path,
    process::Command,
};
use turbo::*;

#[test]
fn rerank_header_layouts_and_constants_match_rust() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../target/rerank-abi");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("layout.c");
    let binary = dir.join("layout");
    std::fs::write(
        &source,
        r#"
#include <turbo/turbo.h>
#include <stdio.h>
#include <stddef.h>
int main(void) {
    printf("%zu %zu %zu %zu %zu %zu %u %u %u %u\n",
       sizeof(turbo_rerank_options), sizeof(turbo_text_pair),
       offsetof(turbo_rerank_options, stride), offsetof(turbo_text_pair, passage),
       offsetof(turbo_model_info, rerank_query_tokens), offsetof(turbo_model_info, rerank_pair_template),
       TURBO_TASK_RERANK, TURBO_PAIR_BERT, TURBO_PAIR_ROBERTA, TURBO_RERANK_STAGE_COUNT);
}
"#,
    )
    .unwrap();
    let out = Command::new("cc")
        .args(["-std=c11", "-Werror", "-I"])
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("../include"))
        .arg(&source)
        .arg("-o")
        .arg(&binary)
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let out = Command::new(binary).output().unwrap();
    assert!(out.status.success());
    let got: Vec<usize> =
        String::from_utf8(out.stdout).unwrap().split_whitespace().map(|s| s.parse().unwrap()).collect();
    assert_eq!(
        got,
        vec![
            size_of::<turbo_rerank_options>(),
            size_of::<turbo_text_pair>(),
            offset_of!(turbo_rerank_options, stride),
            offset_of!(turbo_text_pair, passage),
            offset_of!(turbo_model_info, rerank_query_tokens),
            offset_of!(turbo_model_info, rerank_pair_template),
            TURBO_TASK_RERANK as usize,
            TURBO_PAIR_BERT as usize,
            TURBO_PAIR_ROBERTA as usize,
            TURBO_RERANK_STAGE_COUNT
        ]
    );
}
