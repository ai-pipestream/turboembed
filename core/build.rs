//! Compiles the backends written in another language and links their
//! vendor runtimes, for the features this build turns on.

fn main() {
    #[cfg(feature = "metal")]
    metal();
}

/// core/metal/*.mm against the macOS SDK, with ARC, and the frameworks
/// it calls.
#[cfg(feature = "metal")]
fn metal() {
    use std::process::Command;

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        panic!("the metal feature builds only for macOS");
    }
    let sdk = Command::new("xcrun")
        .args(["--sdk", "macosx", "--show-sdk-version"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .filter(|v| !v.is_empty())
        .expect("xcrun --sdk macosx --show-sdk-version: the Xcode command line tools are needed for the metal feature");
    println!("cargo:rerun-if-changed=metal");
    println!("cargo:rerun-if-changed=../include/turbo");
    cc::Build::new()
        .cpp(true)
        .file("metal/backend.mm")
        .include("../include")
        .flag("-std=c++17")
        .flag("-fobjc-arc")
        .flag("-Wall")
        // The oldest macOS that has what backend.mm reads and that the
        // SDK's libc++ still supports; rustc's x86_64 default is older.
        .flag("-mmacosx-version-min=11.0")
        .define("TURBO_METAL_SDK", format!("\"{sdk}\"").as_str())
        .compile("turbo_metal");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
}
