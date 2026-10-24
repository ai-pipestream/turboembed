//! Pike VM execution engine for [`super::JavaRegex`].

use super::compile::Inst;
use super::{JavaRegex, next_char};

/// A Pike VM over the same program: every thread advances one character
/// at a time, at most one thread per instruction, kept in priority order,
/// so the first thread to reach [`Inst::Match`] is the match Java's
/// backtracking finds. It needs memory linear in the pattern only, for
/// searches whose `(pc, pos)` table would be too large.
#[derive(Default)]
pub(super) struct Pike {
    current: Threads,
    next: Threads,
    stack: Vec<u32>,
}

/// A sparse set of instructions, each with the start of its thread, in
/// the order they were added.
#[derive(Default)]
struct Threads {
    dense: Vec<(u32, usize)>,
    sparse: Vec<u32>,
}

impl Threads {
    fn reset(&mut self, insts: usize) {
        self.dense.clear();
        if self.sparse.len() < insts {
            self.sparse.resize(insts, 0);
        }
    }

    #[inline]
    fn contains(&self, pc: u32) -> bool {
        let i = self.sparse[pc as usize] as usize;
        i < self.dense.len() && self.dense[i].0 == pc
    }

    #[inline]
    fn insert(&mut self, pc: u32, start: usize) {
        self.sparse[pc as usize] = self.dense.len() as u32;
        self.dense.push((pc, start));
    }
}

impl Pike {
    pub(super) fn find(&mut self, re: &JavaRegex, text: &str, from: usize, anchored: bool) -> Option<(usize, usize)> {
        let insts = &re.insts;
        self.current.reset(insts.len());
        self.next.reset(insts.len());
        let mut matched = None;
        let mut pos = from;
        loop {
            // A new start has a lower priority than every thread already
            // running, and none is needed once a match is known.
            if matched.is_none() && !(anchored && pos > 0) {
                Pike::add(insts, &mut self.current, &mut self.stack, 0, pos, text, pos);
            }
            if self.current.dense.is_empty() && (matched.is_some() || anchored) {
                break;
            }
            let step = next_char(text, pos);
            for i in 0..self.current.dense.len() {
                let (pc, start) = self.current.dense[i];
                match insts[pc as usize] {
                    // Threads after this one have a lower priority.
                    Inst::Match => {
                        matched = Some((start, pos));
                        break;
                    }
                    inst => {
                        if let Some((c, next_pos)) = step
                            && re.consumes(inst, c)
                        {
                            Pike::add(insts, &mut self.next, &mut self.stack, pc + 1, start, text, next_pos);
                        }
                    }
                }
            }
            let Some((_, next_pos)) = step else { break };
            std::mem::swap(&mut self.current, &mut self.next);
            self.next.dense.clear();
            pos = next_pos;
        }
        matched
    }

    /// Adds the thread at `pc` and everything it reaches without consuming
    /// a character at `pos`, in priority order.
    fn add(insts: &[Inst], list: &mut Threads, stack: &mut Vec<u32>, pc: u32, start: usize, text: &str, pos: usize) {
        stack.clear();
        stack.push(pc);
        while let Some(mut pc) = stack.pop() {
            loop {
                if list.contains(pc) {
                    break;
                }
                list.insert(pc, start);
                match insts[pc as usize] {
                    Inst::Jump(target) => pc = target,
                    Inst::Split(prefer, fallback) => {
                        stack.push(fallback);
                        pc = prefer;
                    }
                    Inst::Assert(a) if a.holds(text, pos) => pc += 1,
                    _ => break,
                }
            }
        }
    }
}
