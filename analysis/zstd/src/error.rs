//! What can be wrong with a Zstandard input, one variant per rule of the
//! format the decoder checks.

use std::fmt;

/// The part of a frame the input ends inside.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Part {
    Magic,
    FrameHeader,
    BlockHeader,
    Block,
    Checksum,
    SkippableFrame,
    LiteralsHeader,
    Literals,
    HuffmanTable,
    JumpTable,
    SequencesHeader,
    FseTable,
}

impl fmt::Display for Part {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Part::Magic => "a magic number",
            Part::FrameHeader => "a frame header",
            Part::BlockHeader => "a block header",
            Part::Block => "a block",
            Part::Checksum => "a content checksum",
            Part::SkippableFrame => "a skippable frame",
            Part::LiteralsHeader => "a literals section header",
            Part::Literals => "a literals section",
            Part::HuffmanTable => "a Huffman table description",
            Part::JumpTable => "a jump table",
            Part::SequencesHeader => "a sequences section header",
            Part::FseTable => "an FSE table description",
        })
    }
}

/// One of the three kinds of sequence symbol, each with its own table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Symbol {
    LiteralLength,
    Offset,
    MatchLength,
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Symbol::LiteralLength => "literal length",
            Symbol::Offset => "offset",
            Symbol::MatchLength => "match length",
        })
    }
}

/// Why an input could not be decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The input is empty; Zstandard data is one or more frames.
    NoFrames,
    /// The caller asks for more than the frames can decode to, as their
    /// frame and block headers bound it.
    CannotExpand { max: u64, want: usize },
    /// The input ends inside a frame.
    Truncated(Part),
    /// A frame starts with this number, which is not a Zstandard frame's
    /// magic number nor a skippable frame's.
    BadMagic(u32),
    /// The frame header's reserved bit is set.
    ReservedBit,
    /// The frame needs the dictionary with this id, and dictionaries are
    /// not supported.
    Dictionary(u32),
    /// The frame's window is larger than the decoder's limit.
    WindowTooLarge { window: u64, limit: u64 },
    /// The output is longer than the caller allows.
    TooLong { want: usize },
    /// The output is shorter than the caller asks for.
    TooShort { got: usize, want: usize },
    /// A frame's content is longer than its header says.
    ContentTooLong { header: u64 },
    /// A frame's content is shorter than its header says.
    ContentTooShort { header: u64, got: u64 },
    /// The content checksum stored in the frame differs from the one
    /// computed over its content.
    Checksum { stored: u32, computed: u32 },
    /// A block of the reserved type 3.
    ReservedBlockType,
    /// A block, compressed or decompressed, larger than the frame allows.
    BlockTooLarge { size: usize, max: usize },
    /// A literals section with more literals than a block of its frame
    /// holds.
    TooManyLiterals { len: usize, max: usize },
    /// A treeless literals section with no earlier Huffman table in its
    /// frame.
    NoHuffmanTable,
    /// The four Huffman streams' sizes do not fit in the literals section.
    JumpTable,
    /// Four Huffman streams for fewer than 6 literals.
    TooFewLiteralsForFourStreams(usize),
    /// A Huffman table whose weights are all 0.
    NoWeights,
    /// More Huffman weights than the format allows (255).
    TooManyWeights,
    /// Huffman weights that no last weight completes to a power of 2.
    WeightsIncomplete,
    /// A Huffman table deeper than 11 bits.
    HuffmanTooDeep(u32),
    /// A Huffman table with no symbol of weight 1.
    NoWeightOne,
    /// A Huffman or FSE stream that is empty or whose last byte is 0, so
    /// it has no end mark.
    NoEndMark,
    /// A Huffman or FSE stream that does not end where its symbols do.
    StreamNotConsumed,
    /// An FSE table's accuracy log is over the limit for its use.
    AccuracyLog { log: u32, max: u32 },
    /// An FSE table gives a probability to a symbol past the last one its
    /// use allows.
    SymbolPastMax { symbol: usize, max: usize },
    /// The reserved bits of the sequence compression modes are set.
    ReservedModeBits,
    /// A table in repeat mode with no earlier table to repeat.
    NoTableToRepeat(Symbol),
    /// A code past the last one of its kind (35 for literal lengths, 31
    /// for offsets, 52 for match lengths).
    CodePastMax { symbol: Symbol, code: u8 },
    /// A sequences section whose 1-byte count is 0, so with no
    /// sequences, has bytes after it.
    BytesAfterNoSequences,
    /// The sequences use more literals than the literals section holds.
    LiteralsOverrun,
    /// A match reaches before the start of its frame's content.
    OffsetBeforeStart { offset: usize, available: usize },
    /// A match reaches further back than the frame's window.
    OffsetPastWindow { offset: usize, window: u64 },
    /// A repeated offset of 1, less 1, which is 0.
    ZeroOffset,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NoFrames => f.write_str("no frames: the input is empty"),
            Error::CannotExpand { max, want } => {
                write!(f, "the frames decode to at most {max} bytes, fewer than the {want} asked for")
            }
            Error::Truncated(part) => write!(f, "the input ends inside {part}"),
            Error::BadMagic(m) => write!(f, "bad magic number {m:#010x}"),
            Error::ReservedBit => f.write_str("the frame header's reserved bit is set"),
            Error::Dictionary(id) => write!(f, "the frame needs dictionary {id}; dictionaries are not supported"),
            Error::WindowTooLarge { window, limit } => {
                write!(f, "a window of {window} bytes, over the limit of {limit}")
            }
            Error::TooLong { want } => write!(f, "the content is longer than the {want} bytes asked for"),
            Error::TooShort { got, want } => write!(f, "the content is {got} bytes, not the {want} asked for"),
            Error::ContentTooLong { header } => {
                write!(f, "a frame's content is longer than the {header} bytes its header says")
            }
            Error::ContentTooShort { header, got } => {
                write!(f, "a frame's content is {got} bytes, its header says {header}")
            }
            Error::Checksum { stored, computed } => {
                write!(f, "content checksum {computed:#010x}, the frame says {stored:#010x}")
            }
            Error::ReservedBlockType => f.write_str("a block of the reserved type"),
            Error::BlockTooLarge { size, max } => write!(f, "a block of {size} bytes, over the frame's {max}"),
            Error::TooManyLiterals { len, max } => {
                write!(f, "a literals section of {len} bytes, over the frame's block size of {max}")
            }
            Error::NoHuffmanTable => f.write_str("treeless literals with no earlier Huffman table"),
            Error::JumpTable => f.write_str("the jump table's stream sizes do not fit the literals section"),
            Error::TooFewLiteralsForFourStreams(n) => write!(f, "four Huffman streams for {n} literals"),
            Error::NoWeights => f.write_str("a Huffman table whose weights are all 0"),
            Error::TooManyWeights => f.write_str("more than 255 Huffman weights"),
            Error::WeightsIncomplete => f.write_str("Huffman weights that do not complete to a power of 2"),
            Error::HuffmanTooDeep(bits) => write!(f, "a Huffman table {bits} bits deep, over 11"),
            Error::NoWeightOne => f.write_str("a Huffman table with no weight 1"),
            Error::NoEndMark => f.write_str("a bitstream that is empty or whose last byte is 0, with no end mark"),
            Error::StreamNotConsumed => f.write_str("a bitstream that does not end with its last symbol"),
            Error::AccuracyLog { log, max } => write!(f, "an FSE accuracy log of {log}, over {max}"),
            Error::SymbolPastMax { symbol, max } => {
                write!(f, "an FSE table with symbol {symbol}, past the last allowed, {max}")
            }
            Error::ReservedModeBits => f.write_str("the sequence compression modes' reserved bits are set"),
            Error::NoTableToRepeat(s) => write!(f, "the {s} table is in repeat mode with no earlier table"),
            Error::CodePastMax { symbol, code } => write!(f, "{symbol} code {code}, past the last one"),
            Error::BytesAfterNoSequences => f.write_str("bytes after a sequences section of no sequences"),
            Error::LiteralsOverrun => f.write_str("the sequences use more literals than the block holds"),
            Error::OffsetBeforeStart { offset, available } => {
                write!(f, "a match from {offset} bytes back, where the frame's content is {available} bytes")
            }
            Error::OffsetPastWindow { offset, window } => {
                write!(f, "a match from {offset} bytes back, past the window of {window}")
            }
            Error::ZeroOffset => f.write_str("a repeated offset of 0"),
        }
    }
}

impl std::error::Error for Error {}
