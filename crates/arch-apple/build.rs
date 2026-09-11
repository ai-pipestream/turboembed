use std::path::PathBuf;

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    if let Ok(search) = manifest_dir
        .join("../../native/mlx-engine/.build/release")
        .canonicalize()
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", search.display());
    }
    println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");

    if let Ok(out) = std::env::var("OUT_DIR") {
        if let Some(profile) = PathBuf::from(out).ancestors().nth(3) {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", profile.display());
        }
    }
}
