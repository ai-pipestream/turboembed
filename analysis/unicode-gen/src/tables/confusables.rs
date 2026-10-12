//! `confusables.rs`: the prototypes of UTS #39 `confusables.txt`.

use std::collections::BTreeMap;
use std::path::Path;

use super::{Inputs, emit_map};
use crate::emit::{RustFile, str_lit};
use crate::ucd::{UcdFile, to_string};

/// The mappings of `confusables.txt`: source code point to prototype.
pub struct Confusables {
    pub prototypes: BTreeMap<u32, Vec<u32>>,
}

impl Confusables {
    pub fn read(dir: &Path) -> Confusables {
        let file = UcdFile::read(dir, "confusables.txt");
        let mut prototypes = BTreeMap::new();
        for record in file.records() {
            if record.field_count() != 3 {
                record.fail("expected `source ; target ; type`");
            }
            if record.field(2) != "MA" {
                record.fail("the type of every mapping is MA");
            }
            let target = record.code_points(1);
            if target.is_empty() {
                record.fail("empty target");
            }
            if prototypes.insert(record.code_point(0), target).is_some() {
                record.fail("source listed twice");
            }
        }
        Confusables { prototypes }
    }
}

pub fn render(inputs: &Inputs) -> String {
    let mut file = RustFile::new(
        "The confusable prototypes of UTS #39: for each source character, the string it is visually confusable \
         with.",
        &["confusables.txt"],
        inputs.version,
    );
    let entries: Vec<(u32, String)> =
        inputs.confusables.prototypes.iter().map(|(&cp, target)| (cp, str_lit(&to_string(target)))).collect();
    emit_map(
        &mut file,
        "PROTOTYPE",
        ["The source characters of confusables.txt.", "The prototype each source character maps to."],
        "&str",
        &entries,
    );
    file.finish()
}
