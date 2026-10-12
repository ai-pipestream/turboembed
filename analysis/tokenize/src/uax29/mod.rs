//! The UAX #29 word tokenizer: the segmenter ([`segmenter`]), the word
//! type of a segment ([`word_type`]) and the annotators ([`tokenizer`]).

pub mod segmenter;
pub mod tokenizer;
pub mod word_type;

pub use segmenter::{for_each_segment, word_boundaries};
pub use tokenizer::{UntypedWordTokenizer, WordTokenizer, word_tokens};
pub use word_type::word_type;
