fn main() {
    let map = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("exports.map");
    println!("cargo:rerun-if-changed={}", map.display());
    println!(
        "cargo:rustc-cdylib-link-arg=-Wl,--version-script={}",
        map.display()
    );
    println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,libturboembed.so.1");
}
