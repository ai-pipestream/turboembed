//! A union-find forest over mention indexes that carries each cluster's
//! accumulated attributes (`Clusters`). A cluster's attribute sets hold
//! the known values of its members; an unknown value contributes nothing.

use std::collections::HashSet;

use crate::layer::{Animacy, Gender, Number, Person};
use crate::lexicon;
use crate::mention::Mention;

/// A set of known enum values as a bit mask over their discriminants.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Known(u8);

impl Known {
    fn of(bit: u8, unknown: bool) -> Known {
        if unknown { Known(0) } else { Known(1 << bit) }
    }

    /// Whether either set is empty or the sets intersect.
    fn agrees(self, other: Known) -> bool {
        self.0 == 0 || other.0 == 0 || self.0 & other.0 != 0
    }
}

#[derive(Clone, Debug)]
struct Cluster {
    ty: Option<String>,
    numbers: Known,
    genders: Known,
    animacies: Known,
    persons: Known,
    words: HashSet<String>,
    heads: HashSet<String>,
    normalized: HashSet<String>,
    members: Vec<usize>,
}

pub(crate) struct Clusters {
    parent: Vec<usize>,
    clusters: Vec<Cluster>,
}

impl Clusters {
    /// One singleton cluster per mention.
    pub fn new(mentions: &[Mention]) -> Clusters {
        let clusters = mentions
            .iter()
            .enumerate()
            .map(|(i, m)| Cluster {
                ty: m.ty.clone().filter(|t| t != crate::detect::UNTYPED),
                numbers: Known::of(m.number as u8, m.number == Number::Unknown),
                genders: Known::of(m.gender as u8, m.gender == Gender::Unknown),
                animacies: Known::of(m.animacy as u8, m.animacy == Animacy::Unknown),
                persons: Known::of(m.person as u8, m.person == Person::Unknown),
                words: m.words.iter().filter(|w| !lexicon::stop_word(w)).cloned().collect(),
                heads: HashSet::from([m.head.clone()]),
                normalized: HashSet::from([m.normalized.clone()]),
                members: vec![i],
            })
            .collect();
        Clusters { parent: (0..mentions.len()).collect(), clusters }
    }

    /// The root of a mention's cluster, its earliest mention, compressing
    /// the walked path.
    pub fn find(&mut self, i: usize) -> usize {
        let mut root = i;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        let mut node = i;
        while self.parent[node] != root {
            let next = self.parent[node];
            self.parent[node] = root;
            node = next;
        }
        root
    }

    fn get(&mut self, i: usize) -> &Cluster {
        let root = self.find(i);
        &self.clusters[root]
    }

    /// Whether two clusters may merge: unless both know differing types.
    pub fn types_compatible(&mut self, a: usize, b: usize) -> bool {
        let ta = self.get(a).ty.clone();
        let tb = self.get(b).ty.clone();
        ta.is_none() || tb.is_none() || ta == tb
    }

    /// The entity type known for a mention's cluster.
    pub fn ty(&mut self, i: usize) -> Option<String> {
        self.get(i).ty.clone()
    }

    /// Whether no attribute has disjoint known values on the two sides.
    pub fn attributes_agree(&mut self, a: usize, b: usize) -> bool {
        let (x, y) = (self.get(a).clone_attrs(), self.get(b).clone_attrs());
        x.0.agrees(y.0) && x.1.agrees(y.1) && x.2.agrees(y.2) && x.3.agrees(y.3)
    }

    pub fn numbers_agree(&mut self, a: usize, b: usize) -> bool {
        let x = self.get(a).numbers;
        x.agrees(self.get(b).numbers)
    }

    pub fn words(&mut self, i: usize) -> &HashSet<String> {
        &self.get(i).words
    }

    pub fn heads(&mut self, i: usize) -> &HashSet<String> {
        &self.get(i).heads
    }

    pub fn normalized_forms(&mut self, i: usize) -> &HashSet<String> {
        &self.get(i).normalized
    }

    pub fn size(&mut self, i: usize) -> usize {
        self.get(i).members.len()
    }

    /// The mention indexes of a cluster, in merge order.
    pub fn members(&mut self, i: usize) -> &[usize] {
        &self.get(i).members
    }

    /// Merges two clusters unless their known types differ; the earlier
    /// root survives. Returns whether the mentions share a cluster after.
    pub fn union(&mut self, a: usize, b: usize) -> bool {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return true;
        }
        if let (Some(x), Some(y)) = (&self.clusters[ra].ty, &self.clusters[rb].ty)
            && x != y
        {
            return false;
        }
        let (keep, drop) = (ra.min(rb), ra.max(rb));
        self.parent[drop] = keep;
        let dropped = std::mem::replace(&mut self.clusters[drop], Cluster::empty());
        let kept = &mut self.clusters[keep];
        if kept.ty.is_none() {
            kept.ty = dropped.ty;
        }
        kept.numbers.0 |= dropped.numbers.0;
        kept.genders.0 |= dropped.genders.0;
        kept.animacies.0 |= dropped.animacies.0;
        kept.persons.0 |= dropped.persons.0;
        kept.words.extend(dropped.words);
        kept.heads.extend(dropped.heads);
        kept.normalized.extend(dropped.normalized);
        kept.members.extend(dropped.members);
        true
    }

    /// The chains as member lists, ordered by their earliest mention;
    /// every mention appears in exactly one, singletons included.
    pub fn chains(&mut self) -> Vec<Vec<usize>> {
        let mut by_root: Vec<Vec<usize>> = vec![Vec::new(); self.parent.len()];
        for i in 0..self.parent.len() {
            let root = self.find(i);
            by_root[root].push(i);
        }
        by_root.into_iter().filter(|c| !c.is_empty()).collect()
    }
}

impl Cluster {
    fn empty() -> Cluster {
        Cluster {
            ty: None,
            numbers: Known(0),
            genders: Known(0),
            animacies: Known(0),
            persons: Known(0),
            words: HashSet::new(),
            heads: HashSet::new(),
            normalized: HashSet::new(),
            members: Vec::new(),
        }
    }

    fn clone_attrs(&self) -> (Known, Known, Known, Known) {
        (self.numbers, self.genders, self.animacies, self.persons)
    }
}
