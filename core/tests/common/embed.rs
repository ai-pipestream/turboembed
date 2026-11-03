// ---- The small BERT loaded, and the options and texts the embed tests share -

/// The sealed tiny bundle's model, loaded on the CPU.
pub fn tiny() -> Loaded {
    Loaded::load(&tiny_bundle()).unwrap_or_else(|e| panic!("{e:?}"))
}

/// `embed_options()` with one field edited.
pub fn opts(edit: impl FnOnce(&mut turbo_embed_options)) -> turbo_embed_options {
    let mut o = embed_options();
    edit(&mut o);
    o
}

/// Three texts of different lengths.
pub const TEXTS: [&str; 3] = ["The quick brown fox jumps over the lazy dog.", "how do I reset a password", "a"];
