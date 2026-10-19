//! Context generators and sequence validators shared across the constituency parsers
//! (`opennlp.tools.parser.Cons`, `opennlp.tools.parser.AbstractContextGenerator`,
//! `opennlp.tools.parser.ChunkContextGenerator`, and
//! `opennlp.tools.parser.ParserChunkerSequenceValidator`).

mod chunk;
mod cons;
mod sink;

pub use chunk::{ChunkContextGenerator, ConsWindow, ParserChunkerSequenceValidator};
pub use cons::{
    Cons, EOS, check_cons, check_cons_pair, cons, cons_bo, cons_bo_labeled, cons_labeled, cons2, cons3, get_nodes,
    production, production_in, surround,
};
pub(crate) use cons::{check_cons_into, check_cons_pair_into, cons2_into, cons3_into, surround_into};
pub(crate) use sink::{ConsRef, DisplayFn, FeatureSink};
