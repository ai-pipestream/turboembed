//! Left-to-right walks over the nodes of a parse (`Parse.getTagNodes` and kin).

use super::{Parse, TOK_NODE};

impl Parse {
    /// Returns all POS-tag nodes under this parse in left-to-right order (`Parse.getTagNodes`).
    pub fn tag_nodes(&self) -> Vec<&Parse> {
        self.collect_nodes(Parse::is_pos_tag)
    }

    /// The nodes under this parse for which `wanted` holds, left to right, without looking under
    /// them. The walk keeps its own stack, so a tree as deep as a hostile input can make it does
    /// not overflow the thread's.
    fn collect_nodes(&self, wanted: impl Fn(&Parse) -> bool) -> Vec<&Parse> {
        let mut out = Vec::new();
        let mut pending: Vec<&Parse> = self.parts.iter().rev().collect();
        while let Some(node) = pending.pop() {
            if wanted(node) {
                out.push(node);
            } else {
                pending.extend(node.parts.iter().rev());
            }
        }
        out
    }

    /// Returns all `TK` token leaf nodes under this parse in left-to-right order
    /// (`Parse.getTokenNodes`).
    pub fn token_nodes(&self) -> Vec<&Parse> {
        self.collect_nodes(|p| p.kind == TOK_NODE)
    }

    /// Returns the basal phrase chunks and unchunked POS nodes under this parse in left-to-right
    /// order (`ChunkSampleStream.getInitialChunks`). A node without children counts as a node
    /// whose children are all tags, as in Java.
    pub fn initial_chunks(&self) -> Vec<&Parse> {
        let mut chunks = Vec::new();
        self.collect_initial_chunks(&mut chunks);
        chunks
    }

    fn collect_initial_chunks<'a>(&'a self, out: &mut Vec<&'a Parse>) {
        if self.is_pos_tag() || self.parts.iter().all(Parse::is_pos_tag) {
            out.push(self);
        } else {
            for kid in &self.parts {
                kid.collect_initial_chunks(out);
            }
        }
    }

    /// The paths ([`Parse::node_at_path`]) of the nodes [`Parse::initial_chunks`] returns, in
    /// the same order.
    pub fn initial_chunk_paths(&self) -> Vec<Vec<usize>> {
        let mut paths = Vec::new();
        let mut cur = Vec::new();
        self.collect_initial_chunk_paths(&mut cur, &mut paths);
        paths
    }

    fn collect_initial_chunk_paths(&self, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if self.is_pos_tag() || self.parts.iter().all(Parse::is_pos_tag) {
            out.push(cur.clone());
        } else {
            for (i, kid) in self.parts.iter().enumerate() {
                cur.push(i);
                kid.collect_initial_chunk_paths(cur, out);
                cur.pop();
            }
        }
    }
}
