//! Compile the actual host kernels and check concurrent launch isolation.
#![cfg(feature = "cuda-shim")]

use std::path::PathBuf;
use std::process::Command;

#[test]
fn host_launches_have_independent_thread_state() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let binary = PathBuf::from(env!("OUT_DIR")).join(format!(
        "shim-threads-{}{}",
        std::process::id(),
        std::env::consts::EXE_SUFFIX
    ));
    let cxx = std::env::var("CXX").ok().filter(|v| !v.is_empty()).unwrap_or_else(|| "c++".to_owned());
    let built = Command::new(cxx)
        .args(["-O2", "-std=c++17", "-pthread", "-x", "c++", "-include"])
        .arg(root.join("cuda/shim.h"))
        .arg(root.join("cuda/train.cu"))
        .arg(root.join("tests/shim_threads.cpp"))
        .arg("-o")
        .arg(&binary)
        .output()
        .expect("compile the host kernel test");
    assert!(built.status.success(), "{}", String::from_utf8_lossy(&built.stderr));
    let ran = Command::new(&binary).output().expect("run the host kernel test");
    std::fs::remove_file(binary).expect("remove the test executable");
    assert!(ran.status.success(), "{}", String::from_utf8_lossy(&ran.stderr));
}
