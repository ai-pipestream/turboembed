//! Allocation checks count only the current thread after warming the output buffers.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use turbo_parser::context::ChunkContextGenerator;
use turbo_sequence::chunker::TokenTag;
use turbo_sequence::{ContextGenerator, Features};

struct CountingAllocator;
thread_local! {
    static COUNT: Cell<Option<usize>> = const { Cell::new(None) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        COUNT.with(|count| count.set(count.get().map(|n| n + 1)));
        // SAFETY: forward the caller's valid layout to the system allocator.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forward the allocation and its original layout unchanged.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        COUNT.with(|count| count.set(count.get().map(|n| n + 1)));
        // SAFETY: forward the allocation, layout, and requested size unchanged.
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn allocations(mut f: impl FnMut()) -> usize {
    f();
    COUNT.with(|count| count.set(Some(0)));
    f();
    COUNT.with(|count| count.replace(None).unwrap())
}

#[test]
fn chunk_beam_features_allocate_nothing_after_warmup() {
    let sequence = [
        TokenTag { token: "café", tag: "NN" },
        TokenTag { token: "\0🦀", tag: "NN" },
        TokenTag { token: "runs", tag: "VBZ" },
    ];
    let mut generator = ChunkContextGenerator::new();
    let mut features = Features::new();
    let count = allocations(|| {
        for i in 0..sequence.len() {
            features.clear();
            generator.context(i, &sequence, &["S-NP", "C-NP"], &mut features);
        }
    });
    assert_eq!(count, 0, "warmed chunk feature allocations");
}

fn assert_buffered(name: &str, expected: Vec<String>, mut populate: impl FnMut(&mut Features)) {
    let mut wanted = Features::new();
    for feature in &expected {
        wanted.push_str(feature);
        wanted.finish();
    }
    let mut actual = Features::new();
    let count = allocations(|| populate(&mut actual));
    assert_eq!(actual, wanted, "{name}: feature bytes and order");
    assert_eq!(count, 0, "{name}: warmed allocations");
    println!("{name}: {} features, {count} warmed allocations", actual.len());
}

#[test]
fn all_parser_feature_buffers_match_string_contexts_without_allocations() {
    use std::collections::HashSet;
    use turbo_parser::{Parse, bottom_up::dictionary_key, chunking, treeinsert};

    let tree = Parse::parse("(TOP (S (NP (NN café\0🦀)) (VP (VB sees) (NP (NN sky)))) (NP (NN stars)) (VP (VB glow)))")
        .unwrap();
    let mut children = tree.children().to_vec();
    // Distinct probabilities retain multiple marks in OpenNLP's probability-ordered sets.
    for (kind, prob) in [("``", 0.9), (",", 0.7), ("''", 0.5)] {
        let mark = Parse::new(tree.text_arc().clone(), turbo_analysis::Span::new(0, 0), kind, prob, 0);
        for child in &mut children {
            child.add_previous_punctuation(mark.clone());
            child.add_next_punctuation(mark.clone());
        }
    }
    children[0].set_label(Some("S-S".into()));
    let refs: Vec<&Parse> = children.iter().collect();
    let punctuation: HashSet<String> = [",", "``", "''"].map(str::to_owned).into();
    let frontier = treeinsert::Parser::get_right_frontier(&children[0], &punctuation);
    let build = chunking::BuildContextGenerator::new();
    let check = chunking::CheckContextGenerator::new();
    let tb = treeinsert::BuildContextGenerator::new();
    let ta = treeinsert::AttachContextGenerator::new(punctuation.clone());
    let tc = treeinsert::CheckContextGenerator::new(punctuation);
    let mut dict = HashSet::new();
    for start in 0..children.len() {
        for end in start + 1..=children.len() {
            dict.insert(dictionary_key(&children[start..end].iter().map(Parse::head_text).collect::<Vec<_>>()));
        }
    }
    let db = chunking::BuildContextGenerator::with_dictionary(dict);
    for index in 0..children.len() {
        assert_buffered("chunking build", build.get_context(&children, index), |out| {
            build.get_features(&children, index, out)
        });
        assert_buffered("dictionary build", db.get_context(&children, index), |out| {
            db.get_features(&children, index, out)
        });
        assert_buffered("chunking check", check.get_context(&children, "S", 0, index), |out| {
            check.get_features(&children, "S", 0, index, out)
        });
        assert_buffered("tree build", tb.get_context(&refs, index), |out| tb.get_features(&refs, index, out));
        for rfi in 0..frontier.len() {
            assert_buffered("tree attach", ta.get_context(&refs, index, &frontier, rfi), |out| {
                ta.get_features(&refs, index, &frontier, rfi, out)
            });
            for trim in [false, true] {
                assert_buffered("tree check", tc.get_context(frontier[rfi], &refs, index, trim).unwrap(), |out| {
                    tc.get_features(frontier[rfi], &refs, index, trim, out).unwrap()
                });
            }
        }
    }
}

#[test]
fn chunk_buffer_preserves_unicode_and_cached_context_behavior() {
    let words = ["café", "\0🦀", "runs"];
    let tags = ["NN", "NN", "VBZ"];
    let sequence: Vec<TokenTag<'_>> = words.iter().zip(tags).map(|(&token, tag)| TokenTag { token, tag }).collect();
    let mut generator = ChunkContextGenerator::new();
    for i in 0..words.len() {
        assert_buffered("chunk", generator.get_context(i, &words, &tags, &["S-NP", "C-NP"]), |out| {
            generator.get_features(i, &words, &tags, &["S-NP", "C-NP"], out);
        });
    }
    // Concatenated cache keys can collide across different prior outcomes; keep the
    // established opt-in cache behavior in both public entry points.
    let mut strings = ChunkContextGenerator::with_cache(2);
    let mut buffered = ChunkContextGenerator::with_cache(2);
    for prior in [["S-NP", "C-NP"], ["S-N", "PC-NP"], ["S-NP", "O"]] {
        let expected = strings.get_context(2, &words, &tags, &prior);
        let mut wanted = Features::new();
        for feature in expected {
            wanted.push_str(&feature);
            wanted.finish();
        }
        let mut actual = Features::new();
        buffered.context(2, &sequence, &prior, &mut actual);
        assert_eq!(actual, wanted);
    }
}

#[test]
fn oversized_unicode_pieces_keep_exact_features_without_allocations() {
    use turbo_parser::{Parse, treeinsert};
    for length in [31, 32, 33, 127, 128, 129, 1024] {
        let word = "🦀".repeat(length);
        let words = [word.as_str(), "next"];
        let tags = ["NN", "VB"];
        let mut chunk = ChunkContextGenerator::new();
        assert_buffered("long chunk", chunk.get_context(0, &words, &tags, &[]), |out| {
            chunk.get_features(0, &words, &tags, &[], out);
        });
        let tree = Parse::parse(&format!("(TOP (NP (NN {word})) (VP (VB next)))")).unwrap();
        let refs: Vec<&Parse> = tree.children().iter().collect();
        let build = treeinsert::BuildContextGenerator::new();
        assert_buffered("long constituent", build.get_context(&refs, 0), |out| {
            build.get_features(&refs, 0, out);
        });
    }
}
