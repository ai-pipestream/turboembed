//! Tuple implementations for [`Reads`] and [`Writes`].

use std::any::TypeId;

use crate::doc::{Document, View};
use crate::error::Error;
use crate::layer::{Anchor, Key, KeyId, Layer, Provenance};

use super::{Reads, Slot, Writes};

impl<L: Layer> Slot for L {
    type Ref<'d> = &'d L;
    const REQUIRED: bool = true;
    const DEFAULT_ID: KeyId = Key::<L>::DEFAULT.id();
    fn id(name: &'static str) -> KeyId {
        Key::<L>::named(name).id()
    }
    fn fetch<'d>(doc: &'d Document, name: &'static str) -> Result<&'d L, Error> {
        doc.require(&Key::<L>::named(name))
    }
}

impl<L: Layer> Slot for Option<L> {
    type Ref<'d> = Option<&'d L>;
    const REQUIRED: bool = false;
    const DEFAULT_ID: KeyId = Key::<L>::DEFAULT.id();
    fn id(name: &'static str) -> KeyId {
        Key::<L>::named(name).id()
    }
    fn fetch<'d>(doc: &'d Document, name: &'static str) -> Result<Option<&'d L>, Error> {
        Ok(doc.get_key(&Key::<L>::named(name)))
    }
}

trait TupleMeta<const LEN: usize> {
    const COMPACT: [KeyId; LEN];
    const REQUIRED_COUNT: usize;
}

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
    ($len:literal : $($L:ident $i:tt),+) => {
        impl<$($L: Slot),+> TupleMeta<$len> for ($($L,)+) {
            const REQUIRED_COUNT: usize = 0 $(+ ($L::REQUIRED as usize))+;
            const COMPACT: [KeyId; $len] = {
                let all = [$(<$L as Slot>::DEFAULT_ID),+];
                let flags = [$($L::REQUIRED),+];
                let mut out = [Key::<View>::DEFAULT.id(); $len];
                let mut w = 0;
                let mut idx = 0;
                while idx < $len {
                    if flags[idx] {
                        out[w] = all[idx];
                        w += 1;
                    }
                    idx += 1;
                }
                out
            };
        }

        impl<$($L: Slot),+> Reads for ($($L,)+) {
            type Refs<'d> = ($($L::Ref<'d>,)+);
            const N: usize = $len;
            const ALL_IDS: &'static [KeyId] = &[$(<$L as Slot>::DEFAULT_ID),+];
            const REQUIRED_FLAGS: &'static [bool] = &[$($L::REQUIRED),+];
            const DEFAULT_REQUIRED_IDS: &'static [KeyId] =
                <Self as TupleMeta<$len>>::COMPACT.split_at(<Self as TupleMeta<$len>>::REQUIRED_COUNT).0;
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
            const N: usize = $len;
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

tuples!(1 : A 0);
tuples!(2 : A 0, B 1);
tuples!(3 : A 0, B 1, C 2);
tuples!(4 : A 0, B 1, C 2, D 3);
tuples!(5 : A 0, B 1, C 2, D 3, E 4);
tuples!(6 : A 0, B 1, C 2, D 3, E 4, F 5);
tuples!(7 : A 0, B 1, C 2, D 3, E 4, F 5, G 6);
tuples!(8 : A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7);
tuples!(9 : A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8);
tuples!(10 : A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9);
tuples!(11 : A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10);
tuples!(12 : A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11);
