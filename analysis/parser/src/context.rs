//! Context generators and sequence validators shared across the constituency parsers
//! (`opennlp.tools.parser.Cons`, `opennlp.tools.parser.AbstractContextGenerator`,
//! `opennlp.tools.parser.ChunkContextGenerator`, and
//! `opennlp.tools.parser.ParserChunkerSequenceValidator`).

use std::collections::{HashMap, HashSet, VecDeque};

use turbo_sequence::chunker::TokenTag;
use turbo_sequence::{ContextGenerator, Features, SequenceValidator};

use crate::parse::{CONT, OTHER, Parse, START};

/// End-of-sentence boundary marker (`AbstractContextGenerator.EOS`).
pub const EOS: &str = "eos";

/// Zero probability marker (`AbstractContextGenerator.ZERO_PROB`).
pub const ZERO_PROB: &str = "0.0";

/// Holds feature strings and unigram visibility for a constituent slot
/// (`opennlp.tools.parser.Cons`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cons {
    /// Lexicalized constituent feature string (`cons`).
    pub cons: String,
    /// Backoff constituent feature string (`consbo`).
    pub cons_bo: String,
    /// Relative slot index (`-2..=2`).
    pub index: i32,
    /// Whether the lexicalized head token resides in the training dictionary (`unigram`).
    pub unigram: bool,
}

impl Cons {
    /// Creates a new [`Cons`] descriptor.
    pub fn new(cons: String, cons_bo: String, index: i32, unigram: bool) -> Self {
        Self { cons, cons_bo, index, unigram }
    }
}

/// Builds a lexicalized constituent feature (`AbstractContextGenerator.cons(Parse, int)`).
pub fn cons(node: Option<&Parse>, i: i32) -> String {
    let mut feat = String::with_capacity(24);
    feat.push_str(&i.to_string());
    feat.push('=');
    if let Some(p) = node {
        if i < 0 {
            feat.push_str(p.label().unwrap_or("null"));
            feat.push('|');
        }
        feat.push_str(p.kind());
        feat.push('|');
        feat.push_str(p.head_text());
    } else {
        feat.push_str(EOS);
    }
    feat
}

/// Builds a backoff constituent feature (`AbstractContextGenerator.consbo(Parse, int)`).
pub fn cons_bo(node: Option<&Parse>, i: i32) -> String {
    let mut feat = String::with_capacity(16);
    feat.push_str(&i.to_string());
    feat.push_str("*=");
    if let Some(p) = node {
        if i < 0 {
            feat.push_str(p.label().unwrap_or("null"));
            feat.push('|');
        }
        feat.push_str(p.kind());
    } else {
        feat.push_str(EOS);
    }
    feat
}

/// Formats the grammar production string for `p` (`AbstractContextGenerator.production`).
pub fn production(p: &Parse, include_punctuation: bool) -> String {
    let mut prod = String::with_capacity(32);
    prod.push_str(p.kind());
    prod.push_str("->");
    let children = p.children();
    for (ci, child) in children.iter().enumerate() {
        prod.push_str(child.kind());
        if ci + 1 != children.len() {
            prod.push(',');
            if include_punctuation {
                for punct in child.next_punctuation() {
                    prod.push_str(punct.kind());
                    prod.push(',');
                }
            }
        }
    }
    prod
}

/// Populates bigram constituent features (`AbstractContextGenerator.cons2`).
pub fn cons2(features: &mut Vec<String>, c0: &Cons, c1: &Cons, punct_1s: Option<&[Parse]>, bigram: bool) {
    if let Some(puncts) = punct_1s.filter(|s| !s.is_empty()) {
        for p in puncts {
            let punct = p.kind();
            let punct_bo = format!("*={punct},");
            let punct_lex = format!("={punct}|{},", p.covered_text());

            //consbo(0)punctbo(1)consbo(1)
            features.push(format!("{}{}{}", c0.cons_bo, punct_bo, c1.cons_bo));
            if c0.unigram {
                //cons(0)punctbo(1)consbo(1)
                features.push(format!("{}{}{}", c0.cons, punct_bo, c1.cons_bo));
            }
            if c1.unigram {
                //consbo(0)punctbo(1)cons(1)
                features.push(format!("{}{}{}", c0.cons_bo, punct_bo, c1.cons));
            }
            if bigram {
                //cons(0)punctbo(1)cons(1)
                features.push(format!("{}{}{}", c0.cons, punct_bo, c1.cons));
            }

            //consbo(0)punct(1)consbo(1)
            features.push(format!("{}{}{}", c0.cons_bo, punct_lex, c1.cons_bo));
            if c0.unigram {
                //cons(0)punct(1)consbo(1)
                features.push(format!("{}{}{}", c0.cons, punct_lex, c1.cons_bo));
            }
            if c1.unigram {
                //consbo(0)punct(1)cons(1)
                features.push(format!("{}{}{}", c0.cons_bo, punct_lex, c1.cons));
            }
            if bigram {
                //cons(0)punct(1)cons(1)
                features.push(format!("{}{}{}", c0.cons, punct_lex, c1.cons));
            }
        }
    } else {
        //consbo(0),consbo(1)
        features.push(format!("{},{}", c0.cons_bo, c1.cons_bo));
        if c0.unigram {
            //cons(0),consbo(1)
            features.push(format!("{},{}", c0.cons, c1.cons_bo));
        }
        if c1.unigram {
            //consbo(0),cons(1)
            features.push(format!("{},{}", c0.cons_bo, c1.cons));
        }
        if bigram {
            //cons(0),cons(1)
            features.push(format!("{},{}", c0.cons, c1.cons));
        }
    }
}

/// Populates trigram constituent features (`AbstractContextGenerator.cons3`).
#[allow(clippy::too_many_arguments)]
pub fn cons3(
    features: &mut Vec<String>,
    c0: &Cons,
    c1: &Cons,
    c2: &Cons,
    punct_1s: Option<&[Parse]>,
    punct_2s: Option<&[Parse]>,
    trigram: bool,
    bo1: bool,
    bo2: bool,
) {
    let p1 = punct_1s.filter(|s| !s.is_empty());
    let p2 = punct_2s.filter(|s| !s.is_empty());
    if let Some(punct1s) = p1 {
        if c0.index == -2 {
            for p in punct1s {
                let punct_bo = format!("*={},", p.kind());
                let punct = format!("={}|{},", p.kind(), p.covered_text());
                // -2,p,-1,0
                if let Some(punct2s) = p2 {
                    for p2_node in punct2s {
                        let punct2_bo = format!("*={},", p2_node.kind());
                        let punct2 = format!("={}|{},", p2_node.kind(), p2_node.covered_text());

                        //consbo(-2),punctbo(-2),consbo(-1),punctbo(-1),consbo(0)
                        features.push(format!("{}{}{}{}{}", c0.cons_bo, punct_bo, c1.cons_bo, punct2_bo, c2.cons_bo));
                        if c0.unigram {
                            features.push(format!("{}{}{}{}{}", c0.cons, punct_bo, c1.cons_bo, punct2_bo, c2.cons_bo));
                        }
                        if c1.unigram {
                            features.push(format!("{}{}{}{}{}", c0.cons_bo, punct_bo, c1.cons, punct2_bo, c2.cons_bo));
                        }
                        if c2.unigram {
                            features.push(format!("{}{}{}{}{}", c0.cons_bo, punct_bo, c1.cons_bo, punct2_bo, c2.cons));
                        }
                        if bo1 {
                            features.push(format!("{}{}{}{}{}", c0.cons, punct_bo, c1.cons, punct2_bo, c2.cons_bo));
                        }
                        if bo2 {
                            features.push(format!("{}{}{}{}{}", c0.cons_bo, punct_bo, c1.cons, punct2_bo, c2.cons));
                        }
                        if trigram {
                            features.push(format!("{}{}{}{}{}", c0.cons, punct_bo, c1.cons, punct2_bo, c2.cons));
                        }

                        //consbo(-2),punct(-2),consbo(-1),punctbo(-1),consbo(0)
                        features.push(format!("{}{}{}{}{}", c0.cons_bo, punct, c1.cons_bo, punct2_bo, c2.cons_bo));
                        if c0.unigram {
                            features.push(format!("{}{}{}{}{}", c0.cons, punct, c1.cons_bo, punct2_bo, c2.cons_bo));
                        }
                        if c1.unigram {
                            features.push(format!("{}{}{}{}{}", c0.cons_bo, punct, c1.cons, punct2_bo, c2.cons_bo));
                        }
                        if c2.unigram {
                            features.push(format!("{}{}{}{}{}", c0.cons_bo, punct, c1.cons_bo, punct2_bo, c2.cons));
                        }
                        if bo1 {
                            features.push(format!("{}{}{}{}{}", c0.cons, punct, c1.cons, punct2_bo, c2.cons_bo));
                        }
                        if bo2 {
                            features.push(format!("{}{}{}{}{}", c0.cons_bo, punct, c1.cons, punct2_bo, c2.cons));
                        }
                        if trigram {
                            features.push(format!("{}{}{}{}{}", c0.cons, punct, c1.cons, punct2_bo, c2.cons));
                        }

                        //consbo(-2),punctbo(-2),consbo(-1),punct(-1),consbo(0)
                        features.push(format!("{}{}{}{}{}", c0.cons_bo, punct_bo, c1.cons_bo, punct2, c2.cons_bo));
                        if c0.unigram {
                            features.push(format!("{}{}{}{}{}", c0.cons, punct_bo, c1.cons_bo, punct2, c2.cons_bo));
                        }
                        if c1.unigram {
                            features.push(format!("{}{}{}{}{}", c0.cons_bo, punct_bo, c1.cons, punct2, c2.cons_bo));
                        }
                        if c2.unigram {
                            features.push(format!("{}{}{}{}{}", c0.cons_bo, punct_bo, c1.cons_bo, punct2, c2.cons));
                        }
                        if bo1 {
                            features.push(format!("{}{}{}{}{}", c0.cons, punct_bo, c1.cons, punct2, c2.cons_bo));
                        }
                        if bo2 {
                            features.push(format!("{}{}{}{}{}", c0.cons_bo, punct_bo, c1.cons, punct2, c2.cons));
                        }
                        if trigram {
                            features.push(format!("{}{}{}{}{}", c0.cons, punct_bo, c1.cons, punct2, c2.cons));
                        }

                        //consbo(-2),punct(-2),consbo(-1),punct(-1),consbo(0)
                        features.push(format!("{}{}{}{}{}", c0.cons_bo, punct, c1.cons_bo, punct2, c2.cons_bo));
                        if c0.unigram {
                            features.push(format!("{}{}{}{}{}", c0.cons, punct, c1.cons_bo, punct2, c2.cons_bo));
                        }
                        if c1.unigram {
                            features.push(format!("{}{}{}{}{}", c0.cons_bo, punct, c1.cons, punct2, c2.cons_bo));
                        }
                        if c2.unigram {
                            features.push(format!("{}{}{}{}{}", c0.cons_bo, punct, c1.cons_bo, punct2, c2.cons));
                        }
                        if bo1 {
                            features.push(format!("{}{}{}{}{}", c0.cons, punct, c1.cons, punct2, c2.cons_bo));
                        }
                        if bo2 {
                            features.push(format!("{}{}{}{}{}", c0.cons_bo, punct, c1.cons, punct2, c2.cons));
                        }
                        if trigram {
                            features.push(format!("{}{}{}{}{}", c0.cons, punct, c1.cons, punct2, c2.cons));
                        }
                    }
                } else {
                    //consbo(-2),punctbo(-2),consbo(-1),consbo(0)
                    features.push(format!("{}{}{},{}", c0.cons_bo, punct_bo, c1.cons_bo, c2.cons_bo));
                    if c0.unigram {
                        features.push(format!("{}{}{},{}", c0.cons, punct_bo, c1.cons_bo, c2.cons_bo));
                    }
                    if c1.unigram {
                        features.push(format!("{}{}{},{}", c0.cons_bo, punct_bo, c1.cons, c2.cons_bo));
                    }
                    if c2.unigram {
                        features.push(format!("{}{}{},{}", c0.cons_bo, punct_bo, c1.cons_bo, c2.cons));
                    }
                    if bo1 {
                        features.push(format!("{}{}{},{}", c0.cons, punct_bo, c1.cons, c2.cons_bo));
                    }
                    if bo2 {
                        features.push(format!("{}{}{},{}", c0.cons_bo, punct_bo, c1.cons, c2.cons));
                    }
                    if trigram {
                        features.push(format!("{}{}{},{}", c0.cons, punct_bo, c1.cons, c2.cons));
                    }

                    //consbo(-2),punct(-2),consbo(-1),consbo(0)
                    features.push(format!("{}{}{},{}", c0.cons_bo, punct, c1.cons_bo, c2.cons_bo));
                    if c0.unigram {
                        features.push(format!("{}{}{},{}", c0.cons, punct, c1.cons_bo, c2.cons_bo));
                    }
                    if c1.unigram {
                        features.push(format!("{}{}{},{}", c0.cons_bo, punct, c1.cons, c2.cons_bo));
                    }
                    if c2.unigram {
                        features.push(format!("{}{}{},{}", c0.cons_bo, punct, c1.cons_bo, c2.cons));
                    }
                    if bo1 {
                        features.push(format!("{}{}{},{}", c0.cons, punct, c1.cons, c2.cons_bo));
                    }
                    if bo2 {
                        features.push(format!("{}{}{},{}", c0.cons_bo, punct, c1.cons, c2.cons));
                    }
                    if trigram {
                        features.push(format!("{}{}{},{}", c0.cons, punct, c1.cons, c2.cons));
                    }
                }
            }
        } else {
            // punct1s != null && c0.index != -2
            for p in punct1s {
                let punct_bo = format!("*={},", p.kind());
                let punct = format!("={}|{},", p.kind(), p.covered_text());
                // -1,p,0,1
                if let Some(punct2s) = p2 {
                    for p2_node in punct2s {
                        let punct2_bo = format!("*={},", p2_node.kind());
                        let punct2 = format!("={}|{},", p2_node.kind(), p2_node.covered_text());

                        //consbo(-1),punctbo(-1),consbo(0),punctbo(0),consbo(1)
                        features.push(format!("{}{}{}{}{}", c0.cons_bo, punct_bo, c1.cons_bo, punct2_bo, c2.cons_bo));
                        if c0.unigram {
                            features.push(format!("{}{}{}{}{}", c0.cons, punct_bo, c1.cons_bo, punct2_bo, c2.cons_bo));
                        }
                        if c1.unigram {
                            features.push(format!("{}{}{}{}{}", c0.cons_bo, punct_bo, c1.cons, punct2_bo, c2.cons_bo));
                        }
                        if c2.unigram {
                            features.push(format!("{}{}{}{}{}", c0.cons_bo, punct_bo, c1.cons_bo, punct2_bo, c2.cons));
                        }
                        if bo1 {
                            features.push(format!("{}{}{}{}{}", c0.cons, punct_bo, c1.cons, punct2_bo, c2.cons_bo));
                        }
                        if bo2 {
                            features.push(format!("{}{}{}{}{}", c0.cons_bo, punct_bo, c1.cons, punct2_bo, c2.cons));
                        }
                        if trigram {
                            features.push(format!("{}{}{}{}{}", c0.cons, punct_bo, c1.cons, punct2_bo, c2.cons));
                        }

                        //consbo(-1),punct(-1),consbo(0),punctbo(0),consbo(1)
                        features.push(format!("{}{}{}{}{}", c0.cons_bo, punct, c1.cons_bo, punct2_bo, c2.cons_bo));
                        if c0.unigram {
                            features.push(format!("{}{}{}{}{}", c0.cons, punct, c1.cons_bo, punct2_bo, c2.cons_bo));
                        }
                        if c1.unigram {
                            features.push(format!("{}{}{}{}{}", c0.cons_bo, punct, c1.cons, punct2_bo, c2.cons_bo));
                        }
                        if c2.unigram {
                            features.push(format!("{}{}{}{}{}", c0.cons_bo, punct, c1.cons_bo, punct2_bo, c2.cons));
                        }
                        if bo1 {
                            features.push(format!("{}{}{}{}{}", c0.cons, punct, c1.cons, punct2_bo, c2.cons_bo));
                        }
                        if bo2 {
                            features.push(format!("{}{}{}{}{}", c0.cons_bo, punct, c1.cons, punct2_bo, c2.cons));
                        }
                        if trigram {
                            features.push(format!("{}{}{}{}{}", c0.cons, punct, c1.cons, punct2_bo, c2.cons));
                        }

                        //consbo(-1),punctbo(-1),consbo(0),punct(0),consbo(1)
                        features.push(format!("{}{}{}{}{}", c0.cons_bo, punct_bo, c1.cons_bo, punct2, c2.cons_bo));
                        if c0.unigram {
                            features.push(format!("{}{}{}{}{}", c0.cons, punct_bo, c1.cons_bo, punct2, c2.cons_bo));
                        }
                        if c1.unigram {
                            features.push(format!("{}{}{}{}{}", c0.cons_bo, punct_bo, c1.cons, punct2, c2.cons_bo));
                        }
                        if c2.unigram {
                            features.push(format!("{}{}{}{}{}", c0.cons_bo, punct_bo, c1.cons_bo, punct2, c2.cons));
                        }
                        if bo1 {
                            features.push(format!("{}{}{}{}{}", c0.cons, punct_bo, c1.cons, punct2, c2.cons_bo));
                        }
                        if bo2 {
                            features.push(format!("{}{}{}{}{}", c0.cons_bo, punct_bo, c1.cons, punct2, c2.cons));
                        }
                        if trigram {
                            features.push(format!("{}{}{}{}{}", c0.cons, punct_bo, c1.cons, punct2, c2.cons));
                        }

                        //consbo(-1),punct(-1),consbo(0),punct(0),consbo(1)
                        features.push(format!("{}{}{}{}{}", c0.cons_bo, punct, c1.cons_bo, punct2, c2.cons_bo));
                        if c0.unigram {
                            features.push(format!("{}{}{}{}{}", c0.cons, punct, c1.cons_bo, punct2, c2.cons_bo));
                        }
                        if c1.unigram {
                            features.push(format!("{}{}{}{}{}", c0.cons_bo, punct, c1.cons, punct2, c2.cons_bo));
                        }
                        if c2.unigram {
                            features.push(format!("{}{}{}{}{}", c0.cons_bo, punct, c1.cons_bo, punct2, c2.cons));
                        }
                        if bo1 {
                            features.push(format!("{}{}{}{}{}", c0.cons, punct, c1.cons, punct2, c2.cons_bo));
                        }
                        if bo2 {
                            features.push(format!("{}{}{}{}{}", c0.cons_bo, punct, c1.cons, punct2, c2.cons));
                        }
                        if trigram {
                            features.push(format!("{}{}{}{}{}", c0.cons, punct, c1.cons, punct2, c2.cons));
                        }
                    }
                } else {
                    //consbo(-1),punctbo(-1),consbo(0),consbo(1)
                    features.push(format!("{}{}{},{}", c0.cons_bo, punct_bo, c1.cons_bo, c2.cons_bo));
                    if c0.unigram {
                        features.push(format!("{}{}{},{}", c0.cons, punct_bo, c1.cons_bo, c2.cons_bo));
                    }
                    if c1.unigram {
                        features.push(format!("{}{}{},{}", c0.cons_bo, punct_bo, c1.cons, c2.cons_bo));
                    }
                    if c2.unigram {
                        features.push(format!("{}{}{},{}", c0.cons_bo, punct_bo, c1.cons_bo, c2.cons));
                    }
                    if bo1 {
                        features.push(format!("{}{}{},{}", c0.cons, punct_bo, c1.cons, c2.cons_bo));
                    }
                    if bo2 {
                        features.push(format!("{}{}{},{}", c0.cons_bo, punct_bo, c1.cons, c2.cons));
                    }
                    if trigram {
                        features.push(format!("{}{}{},{}", c0.cons, punct_bo, c1.cons, c2.cons));
                    }

                    //consbo(-1),punct(-1),consbo(0),consbo(1)
                    features.push(format!("{}{}{},{}", c0.cons_bo, punct, c1.cons_bo, c2.cons_bo));
                    if c0.unigram {
                        features.push(format!("{}{}{},{}", c0.cons, punct, c1.cons_bo, c2.cons_bo));
                    }
                    if c1.unigram {
                        features.push(format!("{}{}{},{}", c0.cons_bo, punct, c1.cons, c2.cons_bo));
                    }
                    if c2.unigram {
                        features.push(format!("{}{}{},{}", c0.cons_bo, punct, c1.cons_bo, c2.cons));
                    }
                    if bo1 {
                        features.push(format!("{}{}{},{}", c0.cons, punct, c1.cons, c2.cons_bo));
                    }
                    if bo2 {
                        features.push(format!("{}{}{},{}", c0.cons_bo, punct, c1.cons, c2.cons));
                    }
                    if trigram {
                        features.push(format!("{}{}{},{}", c0.cons, punct, c1.cons, c2.cons));
                    }
                }
            }
        }
    } else if let Some(punct2s) = p2 {
        // punct1s == null && punct2s != null
        for p2_node in punct2s {
            let punct2_bo = format!("*={},", p2_node.kind());
            let punct2 = format!("={}|{},", p2_node.kind(), p2_node.covered_text());

            //consbo(-2),consbo(-1),punctbo(-1),consbo(0)
            features.push(format!("{},{}{}{}", c0.cons_bo, c1.cons_bo, punct2_bo, c2.cons_bo));
            if c0.unigram {
                features.push(format!("{},{}{}{}", c0.cons, c1.cons_bo, punct2_bo, c2.cons_bo));
            }
            if c1.unigram {
                features.push(format!("{},{}{}{}", c0.cons_bo, c1.cons, punct2_bo, c2.cons_bo));
            }
            if c2.unigram {
                features.push(format!("{},{}{}{}", c0.cons_bo, c1.cons_bo, punct2_bo, c2.cons));
            }
            if bo1 {
                features.push(format!("{},{}{}{}", c0.cons, c1.cons, punct2_bo, c2.cons_bo));
            }
            if bo2 {
                features.push(format!("{},{}{}{}", c0.cons_bo, c1.cons, punct2_bo, c2.cons));
            }
            if trigram {
                features.push(format!("{},{}{}{}", c0.cons, c1.cons, punct2_bo, c2.cons));
            }

            //consbo(-2),consbo(-1),punct(-1),consbo(0)
            features.push(format!("{},{}{}{}", c0.cons_bo, c1.cons_bo, punct2, c2.cons_bo));
            if c0.unigram {
                features.push(format!("{},{}{}{}", c0.cons, c1.cons_bo, punct2, c2.cons_bo));
            }
            if c1.unigram {
                features.push(format!("{},{}{}{}", c0.cons_bo, c1.cons, punct2, c2.cons_bo));
            }
            if c2.unigram {
                features.push(format!("{},{}{}{}", c0.cons_bo, c1.cons_bo, punct2, c2.cons));
            }
            if bo1 {
                features.push(format!("{},{}{}{}", c0.cons, c1.cons, punct2, c2.cons_bo));
            }
            if bo2 {
                features.push(format!("{},{}{}{}", c0.cons_bo, c1.cons, punct2, c2.cons));
            }
            if trigram {
                features.push(format!("{},{}{}{}", c0.cons, c1.cons, punct2, c2.cons));
            }
        }
    } else {
        //consbo(-2),consbo(-1),consbo(0)
        features.push(format!("{},{},{}", c0.cons_bo, c1.cons_bo, c2.cons_bo));
        if c0.unigram {
            features.push(format!("{},{},{}", c0.cons, c1.cons_bo, c2.cons_bo));
        }
        if c1.unigram {
            features.push(format!("{},{},{}", c0.cons_bo, c1.cons, c2.cons_bo));
        }
        if c2.unigram {
            features.push(format!("{},{},{}", c0.cons_bo, c1.cons_bo, c2.cons));
        }
        if bo1 {
            features.push(format!("{},{},{}", c0.cons, c1.cons, c2.cons_bo));
        }
        if bo2 {
            features.push(format!("{},{},{}", c0.cons_bo, c1.cons, c2.cons));
        }
        if trigram {
            features.push(format!("{},{},{}", c0.cons, c1.cons, c2.cons));
        }
    }
}

/// Appends surrounding constituent features (`AbstractContextGenerator.surround`).
pub fn surround(node: Option<&Parse>, i: i32, kind: &str, punctuation: Option<&[Parse]>, features: &mut Vec<String>) {
    let mut feat = String::with_capacity(32);
    feat.push('s');
    feat.push_str(&i.to_string());
    feat.push('=');
    if let Some(puncts) = punctuation.filter(|s| !s.is_empty()) {
        for punct in puncts {
            if let Some(n) = node {
                write_surround_lex(&mut feat, i, kind, n, Some(punct));
            } else {
                write_surround_eos(&mut feat, i, kind, Some(punct));
            }
            features.push(feat.clone());

            feat.clear();
            feat.push('s');
            feat.push_str(&i.to_string());
            feat.push_str("*=");
            if let Some(n) = node {
                write_surround_bo(&mut feat, i, kind, n.kind(), Some(punct.kind()));
            } else {
                write_surround_bo(&mut feat, i, kind, EOS, Some(punct.kind()));
            }
            features.push(feat.clone());

            feat.clear();
            feat.push('s');
            feat.push_str(&i.to_string());
            feat.push_str("*=");
            if let Some(n) = node {
                write_surround_punct_lex(&mut feat, i, kind, n.kind(), punct);
            } else {
                write_surround_punct_lex(&mut feat, i, kind, EOS, punct);
            }
            features.push(feat.clone());
        }
    } else {
        if let Some(n) = node {
            write_surround_lex(&mut feat, i, kind, n, None);
        } else {
            write_surround_eos(&mut feat, i, kind, None);
        }
        features.push(feat.clone());

        feat.clear();
        feat.push('s');
        feat.push_str(&i.to_string());
        feat.push_str("*=");
        if let Some(n) = node {
            write_surround_bo(&mut feat, i, kind, n.kind(), None);
        } else {
            write_surround_bo(&mut feat, i, kind, EOS, None);
        }
        features.push(feat);
    }
}

fn write_surround_lex(feat: &mut String, _i: i32, kind: &str, node: &Parse, punct: Option<&Parse>) {
    feat.push_str(node.head_text());
    feat.push('|');
    feat.push_str(kind);
    feat.push('|');
    feat.push_str(node.kind());
    if let Some(p) = punct {
        feat.push('|');
        feat.push_str(p.covered_text());
        feat.push('|');
        feat.push_str(p.kind());
    }
}

fn write_surround_eos(feat: &mut String, _i: i32, kind: &str, punct: Option<&Parse>) {
    feat.push_str(kind);
    feat.push('|');
    feat.push_str(EOS);
    if let Some(p) = punct {
        feat.push('|');
        feat.push_str(p.covered_text());
        feat.push('|');
        feat.push_str(p.kind());
    }
}

fn write_surround_bo(feat: &mut String, _i: i32, kind: &str, node_kind: &str, punct_kind: Option<&str>) {
    feat.push_str(kind);
    feat.push('|');
    feat.push_str(node_kind);
    if let Some(pk) = punct_kind {
        feat.push('|');
        feat.push_str(pk);
    }
}

fn write_surround_punct_lex(feat: &mut String, _i: i32, kind: &str, node_kind: &str, punct: &Parse) {
    feat.push_str(kind);
    feat.push('|');
    feat.push_str(node_kind);
    feat.push('|');
    feat.push_str(punct.covered_text());
    feat.push('|');
    feat.push_str(punct.kind());
}

/// Appends check constituent features for a single child (`AbstractContextGenerator.checkCons(Parse, String, String, List)`).
pub fn check_cons(child: &Parse, index: &str, kind: &str, features: &mut Vec<String>) {
    features.push(format!("c{index}={}|{}|{kind}", child.kind(), child.head_text()));
    features.push(format!("c{index}*={}|{kind}", child.kind()));
}

/// Appends check constituent features for an adjacent child pair (`AbstractContextGenerator.checkCons(Parse, Parse, String, List)`).
pub fn check_cons_pair(p1: &Parse, p2: &Parse, kind: &str, features: &mut Vec<String>) {
    let mut feat = String::with_capacity(48);
    feat.push_str("cil=");
    feat.push_str(kind);
    feat.push(',');
    feat.push_str(p1.kind());
    feat.push('|');
    feat.push_str(p1.head_text());
    for punct in p1.next_punctuation() {
        feat.push(',');
        feat.push_str(punct.kind());
        feat.push('|');
        feat.push_str(punct.covered_text());
    }
    feat.push(',');
    feat.push_str(p2.kind());
    feat.push('|');
    feat.push_str(p2.head_text());
    features.push(feat.clone());

    feat.clear();
    feat.push_str("ci*=");
    feat.push_str(kind);
    feat.push(',');
    feat.push_str(p1.kind());
    for punct in p1.next_punctuation() {
        feat.push(',');
        feat.push_str(punct.kind());
    }
    feat.push(',');
    feat.push_str(p2.kind());
    features.push(feat.clone());

    feat.clear();
    feat.push_str("cil*=");
    feat.push_str(kind);
    feat.push(',');
    feat.push_str(p1.kind());
    feat.push('|');
    feat.push_str(p1.head_text());
    for punct in p1.next_punctuation() {
        feat.push(',');
        feat.push_str(punct.kind());
    }
    feat.push(',');
    feat.push_str(p2.kind());
    features.push(feat.clone());

    feat.clear();
    feat.push_str("ci*l=");
    feat.push_str(kind);
    feat.push(',');
    feat.push_str(p1.kind());
    for punct in p1.next_punctuation() {
        feat.push(',');
        feat.push_str(punct.kind());
    }
    feat.push(',');
    feat.push_str(p2.kind());
    feat.push('|');
    feat.push_str(p2.head_text());
    features.push(feat);
}

/// Extracts the 5-constituent window `[c_2, c_1, c0, c1, c2]` around `index` in `chunks`
/// (`AbstractContextGenerator.getNodes`).
pub fn get_nodes(chunks: &[Parse], index: usize) -> [Option<&Parse>; 5] {
    [
        index.checked_sub(2).and_then(|j| chunks.get(j)),
        index.checked_sub(1).and_then(|j| chunks.get(j)),
        chunks.get(index),
        chunks.get(index + 1),
        chunks.get(index + 2),
    ]
}

/// Builds the Cons window `(c_2, c_1, c0, c1, c2)` and unigram/bigram/trigram dictionary flags.
pub struct ConsWindow {
    pub c_2: Cons,
    pub c_1: Cons,
    pub c0: Cons,
    pub c1: Cons,
    pub c2: Cons,
    pub u_2: bool,
    pub u_1: bool,
    pub u0: bool,
    pub u1: bool,
    pub u2: bool,
    pub b_2_1: bool,
    pub b_10: bool,
    pub b01: bool,
    pub b12: bool,
    pub t_2_10: bool,
    pub t_101: bool,
    pub t012: bool,
}

impl ConsWindow {
    /// Computes the constituent window and dictionary membership flags around `nodes`
    /// (`[c_2, c_1, c0, c1, c2]`). When `dict` is `None`, all unigram/bigram/trigram flags are
    /// `true`.
    pub fn build<'a>(nodes: [Option<&'a Parse>; 5], dict: Option<&HashSet<String>>) -> Self {
        let [cf_2, cf_1, cf0, cf1, cf2] = nodes;
        let head_word = |n: Option<&'a Parse>| -> &'a str { n.map_or(EOS, Parse::head_text) };
        let w_2 = head_word(cf_2);
        let w_1 = head_word(cf_1);
        let w0 = head_word(cf0);
        let w1 = head_word(cf1);
        let w2 = head_word(cf2);

        let (u_2, u_1, u0, u1, u2, b_2_1, b_10, b01, b12, t_2_10, t_101, t012) = match dict {
            Some(words) => (
                words.contains(w_2),
                words.contains(w_1),
                words.contains(w0),
                words.contains(w1),
                words.contains(w2),
                words.contains(&format!("{w_2},{w_1}")),
                words.contains(&format!("{w_1},{w0}")),
                words.contains(&format!("{w0},{w1}")),
                words.contains(&format!("{w1},{w2}")),
                words.contains(&format!("{w_2},{w_1},{w0}")),
                words.contains(&format!("{w_1},{w0},{w1}")),
                words.contains(&format!("{w0},{w1},{w2}")),
            ),
            None => (true, true, true, true, true, true, true, true, true, true, true, true),
        };

        let c_2 = Cons::new(cons(cf_2, -2), cons_bo(cf_2, -2), -2, u_2);
        let c_1 = Cons::new(cons(cf_1, -1), cons_bo(cf_1, -1), -1, u_1);
        let c0 = Cons::new(cons(cf0, 0), cons_bo(cf0, 0), 0, u0);
        let c1 = Cons::new(cons(cf1, 1), cons_bo(cf1, 1), 1, u1);
        let c2 = Cons::new(cons(cf2, 2), cons_bo(cf2, 2), 2, u2);

        Self { c_2, c_1, c0, c1, c2, u_2, u_1, u0, u1, u2, b_2_1, b_10, b01, b12, t_2_10, t_101, t012 }
    }
}

/// Context generator for the parser's basal chunking phase (`opennlp.tools.parser.ChunkContextGenerator`).
///
/// Reproduces `ChunkContextGenerator` including its optional cache (`HashFMap` keyed by
/// `i + t_2 + t1 + t0 + t1 + t2 + p_2 + p_1` when `words == wordsKey`) and its exact 45 features.
#[derive(Clone, Debug, Default)]
pub struct ChunkContextGenerator {
    cache_size: usize,
    words_identity: Option<(usize, usize)>,
    cache: VecDeque<(String, Vec<String>)>,
}

impl ChunkContextGenerator {
    /// Creates an uncached [`ChunkContextGenerator`].
    pub fn new() -> Self {
        Self::with_cache(0)
    }

    /// Creates a [`ChunkContextGenerator`] with `cache_size` cached contexts.
    pub fn with_cache(cache_size: usize) -> Self {
        Self { cache_size, words_identity: None, cache: VecDeque::with_capacity(cache_size) }
    }

    /// Generates the 45 chunking features at token `i` (`ChunkContextGenerator.getContext`).
    pub fn get_context(&mut self, i: usize, words: &[&str], tags: &[&str], preds: &[&str]) -> Vec<String> {
        let w_2 = i.checked_sub(2).map_or("w_2=bos", |j| words[j]);
        let t_2 = i.checked_sub(2).map_or("t_2=bos", |j| tags[j]);
        let p_2 = i.checked_sub(2).map_or("p_2=bos", |j| preds[j]);

        let w_1 = i.checked_sub(1).map_or("w_1=bos", |j| words[j]);
        let t_1 = i.checked_sub(1).map_or("t_1=bos", |j| tags[j]);
        let p_1 = i.checked_sub(1).map_or("p_1=bos", |j| preds[j]);

        let w0 = words[i];
        let t0 = tags[i];

        let (w1, t1) = if i + 1 < words.len() { (words[i + 1], tags[i + 1]) } else { ("w1=eos", "t1=eos") };
        let (w2, t2) = if i + 2 < words.len() { (words[i + 2], tags[i + 2]) } else { ("w2=eos", "t2=eos") };

        let cache_key = if self.cache_size > 0 {
            let ident = (words.as_ptr() as usize, words.len());
            if self.words_identity != Some(ident) {
                self.cache.clear();
                self.words_identity = Some(ident);
            }
            // Note: OpenNLP uses `i + t_2 + t1 + t0 + t1 + t2 + p_2 + p_1` (`t1` twice, no `t_1`).
            let key = format!("{i}{t_2}{t1}{t0}{t1}{t2}{p_2}{p_1}");
            if let Some(pos) = self.cache.iter().position(|(k, _)| k == &key) {
                let entry = self.cache.remove(pos).expect("pos in bounds");
                let out = entry.1.clone();
                self.cache.push_front(entry);
                return out;
            }
            Some(key)
        } else {
            None
        };

        let mut features = Vec::with_capacity(45);
        let ct_2 = Self::chunkandpostag(-2, w_2, t_2, p_2);
        let ctbo_2 = Self::chunkandpostagbo(-2, t_2, p_2);
        let ct_1 = Self::chunkandpostag(-1, w_1, t_1, p_1);
        let ctbo_1 = Self::chunkandpostagbo(-1, t_1, p_1);
        let ct0 = Self::chunkandpostag(0, w0, t0, "");
        let ctbo0 = Self::chunkandpostagbo(0, t0, "");
        let ct1 = Self::chunkandpostag(1, w1, t1, "");
        let ctbo1 = Self::chunkandpostagbo(1, t1, "");
        let ct2 = Self::chunkandpostag(2, w2, t2, "");
        let ctbo2 = Self::chunkandpostagbo(2, t2, "");

        // postag(-2)
        features.push(ctbo_2.clone());
        features.push(ct_2.clone());

        // postag(-1)
        features.push(ctbo_1.clone());
        features.push(ct_1.clone());

        // postag(0)
        features.push(ctbo0.clone());
        features.push(ct0.clone());

        // postag(1)
        features.push(ctbo1.clone());
        features.push(ct1.clone());

        // postag(2)
        features.push(ctbo2.clone());
        features.push(ct2.clone());

        // postag(-2, -1)
        features.push(format!("{ctbo_2},{ctbo_1}"));
        features.push(format!("{ct_2},{ctbo_1}"));
        features.push(format!("{ctbo_2},{ct_1}"));
        features.push(format!("{ct_2},{ct_1}"));

        // postag(-1, 0)
        features.push(format!("{ctbo_1},{ctbo0}"));
        features.push(format!("{ct_1},{ctbo0}"));
        features.push(format!("{ctbo_1},{ct0}"));
        features.push(format!("{ct_1},{ct0}"));

        // postag(0, 1)
        features.push(format!("{ctbo0},{ctbo1}"));
        features.push(format!("{ct0},{ctbo1}"));
        features.push(format!("{ctbo0},{ct1}"));
        features.push(format!("{ct0},{ct1}"));

        // postag(1, 2)
        features.push(format!("{ctbo1},{ctbo2}"));
        features.push(format!("{ct1},{ctbo2}"));
        features.push(format!("{ctbo1},{ct2}"));
        features.push(format!("{ct1},{ct2}"));

        // postag(-2, -1, 0)
        features.push(format!("{ctbo_2},{ctbo_1},{ctbo0}"));
        features.push(format!("{ct_2},{ctbo_1},{ctbo0}"));
        features.push(format!("{ctbo_2},{ct_1},{ctbo0}"));
        features.push(format!("{ctbo_2},{ctbo_1},{ct0}"));
        features.push(format!("{ct_2},{ct_1},{ctbo0}"));
        features.push(format!("{ctbo_2},{ct_1},{ct0}"));
        features.push(format!("{ct_2},{ct_1},{ct0}"));

        // postag(-1, 0, 1)
        features.push(format!("{ctbo_1},{ctbo0},{ctbo1}"));
        features.push(format!("{ct_1},{ctbo0},{ctbo1}"));
        features.push(format!("{ctbo_1},{ct0},{ctbo1}"));
        features.push(format!("{ctbo_1},{ctbo0},{ct1}"));
        features.push(format!("{ct_1},{ct0},{ctbo1}"));
        features.push(format!("{ctbo_1},{ct0},{ct1}"));
        features.push(format!("{ct_1},{ct0},{ct1}"));

        // postag(0, 1, 2)
        features.push(format!("{ctbo0},{ctbo1},{ctbo2}"));
        features.push(format!("{ct0},{ctbo1},{ctbo2}"));
        features.push(format!("{ctbo0},{ct1},{ctbo2}"));
        features.push(format!("{ctbo0},{ctbo1},{ct2}"));
        features.push(format!("{ct0},{ct1},{ctbo2}"));
        features.push(format!("{ctbo0},{ct1},{ct2}"));
        features.push(format!("{ct0},{ct1},{ct2}"));

        if let Some(key) = cache_key {
            if self.cache.len() >= self.cache_size {
                self.cache.pop_back();
            }
            self.cache.push_front((key, features.clone()));
        }
        features
    }

    fn chunkandpostag(i: i32, tok: &str, tag: &str, chunk: &str) -> String {
        if i < 0 { format!("{i}={tok}|{tag}|{chunk}") } else { format!("{i}={tok}|{tag}") }
    }

    fn chunkandpostagbo(i: i32, tag: &str, chunk: &str) -> String {
        if i < 0 { format!("{i}*={tag}|{chunk}") } else { format!("{i}*={tag}") }
    }
}

impl<'a> ContextGenerator<TokenTag<'a>> for ChunkContextGenerator {
    fn start_sequence(&mut self) {
        self.cache.clear();
        self.words_identity = None;
    }

    fn context(&mut self, index: usize, sequence: &[TokenTag<'a>], prior: &[&str], out: &mut Features) {
        let words: Vec<&str> = sequence.iter().map(|t| t.token).collect();
        let tags: Vec<&str> = sequence.iter().map(|t| t.tag).collect();
        let feats = self.get_context(index, &words, &tags, prior);
        for f in &feats {
            out.push_str(f);
            out.finish();
        }
    }
}

/// Sequence validator for the parser's basal chunker (`opennlp.tools.parser.ParserChunkerSequenceValidator`).
///
/// Outcomes use `S-<type>`, `C-<type>`, and `O`. A continuation outcome `C-<type>` is valid only
/// after `S-<type>` or `C-<type>`.
#[derive(Clone, Debug, Default)]
pub struct ParserChunkerSequenceValidator {
    continue_start_map: HashMap<String, String>,
}

impl ParserChunkerSequenceValidator {
    /// Creates a validator from the chunker model's outcome labels.
    pub fn new<S: AsRef<str>>(outcomes: &[S]) -> Self {
        let mut continue_start_map = HashMap::with_capacity(outcomes.len());
        for outcome in outcomes {
            let outcome = outcome.as_ref();
            if let Some(rest) = outcome.strip_prefix(CONT) {
                continue_start_map.insert(outcome.to_string(), format!("{START}{rest}"));
            }
        }
        Self { continue_start_map }
    }

    /// Validates whether `outcome` can occur at `i` after `outcomes`.
    pub fn valid_outcome(&self, outcomes: &[&str], outcome: &str) -> bool {
        if let Some(start_tag) = self.continue_start_map.get(outcome) {
            if outcomes.is_empty() {
                return false;
            }
            let last_tag = outcomes[outcomes.len() - 1];
            if last_tag == outcome || last_tag == start_tag {
                return true;
            }
            if last_tag == OTHER {
                return false;
            }
            return false;
        }
        true
    }
}

impl<'a> SequenceValidator<TokenTag<'a>> for ParserChunkerSequenceValidator {
    fn valid_sequence(&self, _index: usize, _sequence: &[TokenTag<'a>], prior: &[&str], outcome: &str) -> bool {
        self.valid_outcome(prior, outcome)
    }
}
