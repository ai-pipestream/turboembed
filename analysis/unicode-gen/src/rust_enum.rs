//! The `#[repr(u8)]` enums the generated files define, and the functions
//! on them that map to and from the names the inputs use.

use std::fmt::Write;

/// One enum variant.
pub struct Variant {
    /// The Rust variant name.
    pub name: &'static str,
    /// The discriminant.
    pub value: u8,
    /// The private constant the generated tables write the variant as.
    pub alias: &'static str,
    /// The name an input or an outside system uses for the variant.
    pub label: &'static str,
    /// One line of documentation.
    pub doc: &'static str,
}

pub const fn variant(
    name: &'static str,
    value: u8,
    alias: &'static str,
    label: &'static str,
    doc: &'static str,
) -> Variant {
    Variant { name, value, alias, label, doc }
}

/// The discriminant of the variant labelled `label`.
pub fn by_label(variants: &[Variant], label: &str) -> Option<u8> {
    variants.iter().find(|v| v.label == label).map(|v| v.value)
}

/// The variant with discriminant `value`.
pub fn by_value(variants: &[Variant], value: u8) -> &Variant {
    variants.iter().find(|v| v.value == value).expect("a value the generator assigned")
}

/// Writes the enum. The variant with discriminant 0 is the default.
pub fn write_enum(out: &mut String, doc: &str, name: &str, variants: &[Variant]) {
    for line in doc.lines() {
        writeln!(out, "/// {line}").unwrap();
    }
    writeln!(out, "#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]").unwrap();
    writeln!(out, "#[repr(u8)]\npub enum {name} {{").unwrap();
    for variant in variants {
        writeln!(out, "    /// {}", variant.doc).unwrap();
        if variant.value == 0 {
            writeln!(out, "    #[default]").unwrap();
        }
        writeln!(out, "    {} = {},", variant.name, variant.value).unwrap();
    }
    writeln!(out, "}}\n").unwrap();
}

/// Writes `pub const ALL`, every variant in discriminant order.
pub fn write_all(out: &mut String, name: &str, variants: &[Variant]) {
    writeln!(out, "    /// Every value, in discriminant order.").unwrap();
    writeln!(out, "    pub const ALL: [{name}; {}] = [", variants.len()).unwrap();
    for variant in variants {
        writeln!(out, "        {name}::{},", variant.name).unwrap();
    }
    writeln!(out, "    ];\n").unwrap();
}

/// Writes `pub const fn <function>(self) -> &'static str` returning the
/// string `text` gives for each variant.
pub fn write_to_str(
    out: &mut String,
    doc: &str,
    function: &str,
    name: &str,
    variants: &[Variant],
    text: &dyn Fn(&Variant) -> String,
) {
    writeln!(out, "    /// {doc}").unwrap();
    writeln!(out, "    pub const fn {function}(self) -> &'static str {{\n        match self {{").unwrap();
    for variant in variants {
        writeln!(out, "            {name}::{} => \"{}\",", variant.name, text(variant)).unwrap();
    }
    writeln!(out, "        }}\n    }}\n").unwrap();
}

/// Writes `pub fn <function>(label: &str) -> Option<Self>`, the inverse
/// of the labels.
pub fn write_from_label(out: &mut String, doc: &str, function: &str, name: &str, variants: &[Variant]) {
    writeln!(out, "    /// {doc}").unwrap();
    writeln!(out, "    pub fn {function}(label: &str) -> Option<{name}> {{\n        match label {{").unwrap();
    for variant in variants {
        writeln!(out, "            \"{}\" => Some({name}::{}),", variant.label, variant.name).unwrap();
    }
    writeln!(out, "            _ => None,\n        }}\n    }}").unwrap();
}

/// Writes the private constants a table module uses for the variants
/// that occur in `values`.
pub fn write_aliases(out: &mut String, name: &str, variants: &[Variant], values: &[u8]) {
    let mut used = [false; 256];
    for &value in values {
        used[value as usize] = true;
    }
    for variant in variants.iter().filter(|v| used[v.value as usize]) {
        writeln!(out, "    const {}: {name} = {name}::{};", variant.alias, variant.name).unwrap();
    }
    writeln!(out).unwrap();
}
