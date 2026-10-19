//! AST nodes, bytecode instructions, and bytecode compiler for [`super::JavaRegex`].

use crate::error::Error;

use super::MAX_INSTS;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum Assertion {
    StartText,
    EndText,
    AsciiWordBoundary,
    AsciiNonWordBoundary,
}

#[derive(Copy, Clone, Debug)]
pub(super) enum Inst {
    Match,
    Jump(u32),
    Split(u32, u32),
    Assert(Assertion),
    Literal(char),
    AsciiCase(u8),
    AnyChar,
    NotLineTerminator,
    Class(u32),
}

#[derive(Clone, Debug)]
pub(super) enum Node {
    Empty,
    Assert(Assertion),
    Literal(char),
    AsciiCase(u8),
    AnyChar,
    NotLineTerminator,
    Class(u32),
    Seq(Vec<Node>),
    Alt(Vec<Node>),
    Rep { body: Box<Node>, min: u32, max: Option<u32>, greedy: bool },
}

pub(super) struct Compiler<'s> {
    pattern: &'s str,
    insts: Vec<Inst>,
}

impl<'s> Compiler<'s> {
    pub(super) fn compile(pattern: &'s str, root: &Node) -> Result<Box<[Inst]>, Error> {
        let mut c = Compiler { pattern, insts: Vec::new() };
        c.emit(root)?;
        c.push(Inst::Match)?;
        Ok(c.insts.into_boxed_slice())
    }

    fn pc(&self) -> u32 {
        self.insts.len() as u32
    }

    fn push(&mut self, inst: Inst) -> Result<u32, Error> {
        if self.insts.len() >= MAX_INSTS {
            return Err(Error::Compile {
                pattern: self.pattern.into(),
                message: format!("compiled pattern exceeds {MAX_INSTS} instructions"),
            });
        }
        let pc = self.insts.len() as u32;
        self.insts.push(inst);
        Ok(pc)
    }

    fn emit(&mut self, node: &Node) -> Result<(), Error> {
        match node {
            Node::Empty => Ok(()),
            Node::Assert(a) => {
                self.push(Inst::Assert(*a))?;
                Ok(())
            }
            Node::Literal(c) => {
                self.push(Inst::Literal(*c))?;
                Ok(())
            }
            Node::AsciiCase(u) => {
                self.push(Inst::AsciiCase(*u))?;
                Ok(())
            }
            Node::AnyChar => {
                self.push(Inst::AnyChar)?;
                Ok(())
            }
            Node::NotLineTerminator => {
                self.push(Inst::NotLineTerminator)?;
                Ok(())
            }
            Node::Class(idx) => {
                self.push(Inst::Class(*idx))?;
                Ok(())
            }
            Node::Seq(items) => {
                for item in items {
                    self.emit(item)?;
                }
                Ok(())
            }
            Node::Alt(branches) => {
                // `Translator::expr` makes an `Alt` of two or more branches.
                let mut end_jumps = Vec::with_capacity(branches.len() - 1);
                for branch in &branches[..branches.len() - 1] {
                    let split_pc = self.push(Inst::Split(0, 0))?;
                    let branch_start = self.pc();
                    self.emit(branch)?;
                    let jump_pc = self.push(Inst::Jump(0))?;
                    end_jumps.push(jump_pc);
                    let next_pc = self.pc();
                    self.insts[split_pc as usize] = Inst::Split(branch_start, next_pc);
                }
                self.emit(&branches[branches.len() - 1])?;
                let end_pc = self.pc();
                for jp in end_jumps {
                    self.insts[jp as usize] = Inst::Jump(end_pc);
                }
                Ok(())
            }
            Node::Rep { body, min, max, greedy } => {
                if (*min as usize) > MAX_INSTS {
                    return Err(Error::Compile {
                        pattern: self.pattern.into(),
                        message: format!("repetition count {min} exceeds limit"),
                    });
                }
                match *max {
                    Some(0) => Ok(()),
                    Some(m) => {
                        if (m as usize) > MAX_INSTS {
                            return Err(Error::Compile {
                                pattern: self.pattern.into(),
                                message: format!("repetition count {m} exceeds limit"),
                            });
                        }
                        for _ in 0..*min {
                            self.emit(body)?;
                        }
                        if m > *min {
                            let count = (m - *min) as usize;
                            let mut splits = Vec::with_capacity(count);
                            for _ in *min..m {
                                let split_pc = self.push(Inst::Split(0, 0))?;
                                let body_start = self.pc();
                                self.emit(body)?;
                                splits.push((split_pc, body_start));
                            }
                            let end_pc = self.pc();
                            for (split_pc, body_start) in splits {
                                self.insts[split_pc as usize] = if *greedy {
                                    Inst::Split(body_start, end_pc)
                                } else {
                                    Inst::Split(end_pc, body_start)
                                };
                            }
                        }
                        Ok(())
                    }
                    None => {
                        if *min == 0 {
                            let split_pc = self.push(Inst::Split(0, 0))?;
                            let body_start = self.pc();
                            self.emit(body)?;
                            self.push(Inst::Jump(split_pc))?;
                            let end_pc = self.pc();
                            self.insts[split_pc as usize] =
                                if *greedy { Inst::Split(body_start, end_pc) } else { Inst::Split(end_pc, body_start) };
                        } else {
                            for _ in 0..(*min - 1) {
                                self.emit(body)?;
                            }
                            let body_start = self.pc();
                            self.emit(body)?;
                            let split_pc = self.push(Inst::Split(0, 0))?;
                            let end_pc = self.pc();
                            self.insts[split_pc as usize] =
                                if *greedy { Inst::Split(body_start, end_pc) } else { Inst::Split(end_pc, body_start) };
                        }
                        Ok(())
                    }
                }
            }
        }
    }
}
