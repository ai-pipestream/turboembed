use crate::{DependencyGraph, Result, error};
use turbo_analysis::Ix;
use turbo_analysis_layers::{Tokens, WordOrTokenAnchor};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Transition {
    Shift,
    LeftArc(String),
    RightArc(String),
}
impl Transition {
    pub fn decode(s: &str) -> Result<Self> {
        if s == "SHIFT" {
            return Ok(Self::Shift);
        }
        let (kind, label) = s.split_once(':').ok_or_else(|| error("invalid transition"))?;
        if label.chars().all(turbo_unicode::props::white_space) {
            return Err(error("blank arc label"));
        }
        match kind {
            "LEFT_ARC" => Ok(Self::LeftArc(label.into())),
            "RIGHT_ARC" => Ok(Self::RightArc(label.into())),
            _ => Err(error("invalid transition")),
        }
    }
    pub fn encode(&self) -> String {
        match self {
            Self::Shift => "SHIFT".into(),
            Self::LeftArc(l) => format!("LEFT_ARC:{l}"),
            Self::RightArc(l) => format!("RIGHT_ARC:{l}"),
        }
    }
    fn valid(&self) -> bool {
        match self {
            Self::Shift => true,
            Self::LeftArc(label) | Self::RightArc(label) => !label.chars().all(turbo_unicode::props::white_space),
        }
    }
}
/// Root and missing positions cannot be mistaken for token indexes.
#[derive(Debug)]
pub enum Position<A> {
    Root,
    Missing,
    Node(Ix<A>),
}
impl<A> PartialEq for Position<A> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Root, Self::Root) | (Self::Missing, Self::Missing) => true,
            (Self::Node(a), Self::Node(b)) => a == b,
            _ => false,
        }
    }
}
impl<A> Eq for Position<A> {}
impl<A> Copy for Position<A> {}
impl<A> Clone for Position<A> {
    fn clone(&self) -> Self {
        *self
    }
}

pub struct ArcStandardState<A: WordOrTokenAnchor = Tokens> {
    pub(crate) stack: Vec<Position<A>>,
    pub(crate) front: usize,
    pub(crate) heads: Vec<Option<Ix<A>>>,
    pub(crate) relations: Vec<Option<String>>,
    pub(crate) counts: Vec<usize>,
    pub(crate) left: Vec<Option<Ix<A>>>,
    pub(crate) right: Vec<Option<Ix<A>>>,
    pub(crate) directional_left: Vec<[Option<Ix<A>>; 2]>,
    pub(crate) directional_right: Vec<[Option<Ix<A>>; 2]>,
}
impl<A: WordOrTokenAnchor> Clone for ArcStandardState<A> {
    fn clone(&self) -> Self {
        Self {
            stack: self.stack.clone(),
            front: self.front,
            heads: self.heads.clone(),
            relations: self.relations.clone(),
            counts: self.counts.clone(),
            left: self.left.clone(),
            right: self.right.clone(),
            directional_left: self.directional_left.clone(),
            directional_right: self.directional_right.clone(),
        }
    }
}
impl<A: WordOrTokenAnchor> ArcStandardState<A> {
    pub fn new(n: usize) -> Result<Self> {
        if n == 0 || n > u32::MAX as usize {
            return Err(error("token count must be positive and fit u32"));
        }
        Ok(Self {
            stack: vec![Position::Root],
            front: 0,
            heads: vec![None; n],
            relations: vec![None; n],
            counts: vec![0; n],
            left: vec![None; n],
            right: vec![None; n],
            directional_left: vec![[None; 2]; n],
            directional_right: vec![[None; 2]; n],
        })
    }
    pub fn token_count(&self) -> usize {
        self.heads.len()
    }
    pub fn stack(&self, offset: usize) -> Position<A> {
        self.stack.len().checked_sub(offset).and_then(|n| n.checked_sub(1)).map_or(Position::Missing, |i| self.stack[i])
    }
    pub fn buffer(&self, offset: usize) -> Position<A> {
        self.front
            .checked_add(offset)
            .filter(|&i| i < self.token_count())
            .map_or(Position::Missing, |i| Position::Node(Ix::new(i as u32)))
    }
    pub fn stack_size(&self) -> usize {
        self.stack.len()
    }
    pub fn buffer_size(&self) -> usize {
        self.token_count() - self.front
    }
    pub fn assigned_dependents(&self, node: Ix<A>) -> Result<usize> {
        self.counts.get(node.get()).copied().ok_or_else(|| error("node out of range"))
    }
    pub fn leftmost_dependent(&self, node: Ix<A>) -> Result<Option<Ix<A>>> {
        self.left.get(node.get()).copied().ok_or_else(|| error("node out of range"))
    }
    pub fn rightmost_dependent(&self, node: Ix<A>) -> Result<Option<Ix<A>>> {
        self.right.get(node.get()).copied().ok_or_else(|| error("node out of range"))
    }
    pub fn assigned_relation(&self, node: Ix<A>) -> Result<Option<&str>> {
        self.relations.get(node.get()).map(|r| r.as_deref()).ok_or_else(|| error("node out of range"))
    }
    pub fn is_terminal(&self) -> bool {
        self.front == self.token_count() && self.stack.len() == 1
    }
    pub fn can_apply(&self, t: &Transition) -> bool {
        if !t.valid() {
            return false;
        }
        match t {
            Transition::Shift => self.front < self.token_count(),
            Transition::LeftArc(_) => self.stack.len() >= 3,
            Transition::RightArc(_) => {
                self.stack.len() >= 3 || (self.stack.len() == 2 && self.front == self.token_count())
            }
        }
    }
    pub fn apply(&mut self, t: &Transition) -> Result<()> {
        if !self.can_apply(t) {
            return Err(error("transition not applicable"));
        }
        match t {
            Transition::Shift => {
                self.stack.push(Position::Node(Ix::new(self.front as u32)));
                self.front += 1;
            }
            Transition::LeftArc(label) => {
                let Position::Node(d) = self.stack(1) else { unreachable!() };
                self.attach(self.stack(0), d, label);
                self.stack.remove(self.stack.len() - 2);
            }
            Transition::RightArc(label) => {
                let Position::Node(d) = self.stack(0) else { unreachable!() };
                self.attach(self.stack(1), d, label);
                self.stack.pop();
            }
        }
        Ok(())
    }
    fn attach(&mut self, head: Position<A>, d: Ix<A>, label: &str) {
        self.relations[d.get()] = Some(label.into());
        if let Position::Node(h) = head {
            let left = d.get() < h.get();
            let slots = if left { &mut self.directional_left[h.get()] } else { &mut self.directional_right[h.get()] };
            let precedes = |a: Ix<A>, b: Ix<A>| if left { a.get() < b.get() } else { a.get() > b.get() };
            if slots[0].is_none_or(|n| precedes(d, n)) {
                slots[1] = slots[0];
                slots[0] = Some(d);
            } else if slots[1].is_none_or(|n| precedes(d, n)) {
                slots[1] = Some(d);
            }
            self.heads[d.get()] = Some(h);
            self.counts[h.get()] += 1;
            if self.left[h.get()].is_none_or(|n| d.get() < n.get()) {
                self.left[h.get()] = Some(d);
            }
            if self.right[h.get()].is_none_or(|n| d.get() > n.get()) {
                self.right[h.get()] = Some(d);
            }
        }
    }
    pub fn to_graph(&self) -> Result<DependencyGraph<A>> {
        if !self.is_terminal() {
            return Err(error("parse not terminal"));
        }
        DependencyGraph::new(
            self.heads.clone(),
            self.relations.iter().map(|r| r.clone().expect("terminal node attached")).collect(),
        )
    }
}
pub struct ArcStandardOracle;
impl ArcStandardOracle {
    pub fn transitions<A: WordOrTokenAnchor>(gold: &DependencyGraph<A>) -> Result<Vec<Transition>> {
        gold.check_tree()?;
        if !gold.is_projective() {
            return Err(error("non-projective tree"));
        }
        let mut state = ArcStandardState::<A>::new(gold.len())?;
        let mut result = Vec::new();
        while !state.is_terminal() {
            let s0 = state.stack(0);
            let s1 = state.stack(1);
            let mut next = None;
            if let (Position::Node(top), Position::Node(second)) = (s0, s1) {
                let arc = gold.arc(second)?;
                if arc.head == Some(top) {
                    next = Some(Transition::LeftArc(arc.relation.into()));
                }
            }
            if next.is_none()
                && let Position::Node(top) = s0
            {
                let arc = gold.arc(top)?;
                let matches = match s1 {
                    Position::Root => arc.head.is_none(),
                    Position::Node(h) => arc.head == Some(h),
                    Position::Missing => false,
                };
                if matches && state.counts[top.get()] == gold.children(top)?.len() {
                    let t = Transition::RightArc(arc.relation.into());
                    if state.can_apply(&t) {
                        next = Some(t);
                    }
                }
            }
            let t = next
                .or_else(|| state.can_apply(&Transition::Shift).then_some(Transition::Shift))
                .ok_or_else(|| error("non-projective tree"))?;
            state.apply(&t)?;
            result.push(t);
        }
        Ok(result)
    }
}
