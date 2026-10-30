//! Tuple implementations for [`Reads`] and [`Writes`].

use std::any::TypeId;

use crate::doc::Document;
use crate::error::Error;
use crate::layer::{Anchor, Key, KeyId, Layer, Provenance};

use super::{Reads, Slot, Writes};

impl Reads for () {
    type Refs<'d> = ();
    const N: usize = 0;
    fn ids(_: &[&'static str]) -> Vec<KeyId> {
        Vec::new()
    }
    fn required() -> Vec<bool> {
        Vec::new()
    }
    fn fetch(_: &Document, _: &[&'static str]) -> Result<(), Error> {
        Ok(())
    }
}

macro_rules! tuples {
    ($($L:ident $i:tt),+) => {
        impl<$($L: Slot),+> Reads for ($($L,)+) {
            type Refs<'d> = ($($L::Ref<'d>,)+);
            const N: usize = [$($i),+].len();
            const ALL_IDS: &'static [KeyId] = &[$(<$L as Slot>::DEFAULT_ID),+];
            const REQUIRED_FLAGS: &'static [bool] = &[$($L::REQUIRED),+];
            const DEFAULT_REQUIRED_IDS: &'static [KeyId] = {
                let mut prefix = 0;
                while prefix < Self::REQUIRED_FLAGS.len() && Self::REQUIRED_FLAGS[prefix] {
                    prefix += 1;
                }
                let mut all_rest_false = true;
                let mut idx = prefix;
                while idx < Self::REQUIRED_FLAGS.len() {
                    if Self::REQUIRED_FLAGS[idx] {
                        all_rest_false = false;
                    }
                    idx += 1;
                }
                if all_rest_false {
                    Self::ALL_IDS.split_at(prefix).0
                } else {
                    let mut suffix_start = 0;
                    while suffix_start < Self::REQUIRED_FLAGS.len() && !Self::REQUIRED_FLAGS[suffix_start] {
                        suffix_start += 1;
                    }
                    Self::ALL_IDS.split_at(suffix_start).1
                }
            };
            fn ids(names: &[&'static str]) -> Vec<KeyId> {
                vec![$($L::id(names[$i])),+]
            }
            fn required() -> Vec<bool> {
                vec![$($L::REQUIRED),+]
            }
            fn fetch<'d>(doc: &'d Document, names: &[&'static str]) -> Result<Self::Refs<'d>, Error> {
                Ok(($($L::fetch(doc, names[$i])?,)+))
            }
        }
        impl<$($L: Layer),+> Writes for ($($L,)+) {
            const N: usize = [$($i),+].len();
            const DEFAULT_IDS: &'static [KeyId] = &[$(Key::<$L>::DEFAULT.id()),+];
            fn ids(names: &[&'static str]) -> Vec<KeyId> {
                vec![$(Key::<$L>::named(names[$i]).id()),+]
            }
            fn anchors() -> Vec<Anchor> {
                vec![$($L::anchor()),+]
            }
            fn store(self, doc: &mut Document, names: &[&'static str], p: &Provenance) -> Result<(), Error> {
                let ids = <Self as Writes>::ids(names);
                $({
                    let p = made_on_own(&self.$i, &ids[..$i], p);
                    doc.add(&Key::<$L>::named(names[$i]), self.$i, p)?;
                })+
                Ok(())
            }
        }
    };
}

/// The provenance of one output: its step's, with `on` listing the outputs
/// the step wrote before it (`earlier`) that this one is aligned to or
/// indexes, so the document checks it against those instances.
fn made_on_own<L: Layer>(layer: &L, earlier: &[KeyId], p: &Provenance) -> Provenance {
    let mut types: Vec<TypeId> = layer.indexes().into_iter().map(|(k, _)| k.ty).collect();
    if let Anchor::Aligned { ty, .. } = L::anchor() {
        types.push(ty);
    }
    let mut p = p.clone();
    p.on = earlier.iter().copied().filter(|k| types.contains(&k.ty)).collect();
    p
}

tuples!(A 0);
tuples!(A 0, B 1);
tuples!(A 0, B 1, C 2);
tuples!(A 0, B 1, C 2, D 3);
tuples!(A 0, B 1, C 2, D 3, E 4);
tuples!(A 0, B 1, C 2, D 3, E 4, F 5);
tuples!(A 0, B 1, C 2, D 3, E 4, F 5, G 6);
tuples!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7);
tuples!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8);
tuples!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9);
tuples!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10);
tuples!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11);
