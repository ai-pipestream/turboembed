use std::borrow::Borrow;
use std::cmp::Ordering;

use super::map::{Links, NIL, NO_LINKS};
use super::{JavaHash, JavaHashMap, MIN_TREEIFY_CAPACITY};

impl<K: JavaHash + Eq, V> JavaHashMap<K, V> {
    /// `TreeNode.find` from the root of a tree bin: descends by hash as a
    /// signed `int` and then by [`JavaHash::tree_order`], and searches both
    /// sides of a node whose key the order cannot place, so a bin of many
    /// keys with one hash code is searched in logarithmic time when the
    /// keys are ordered, as the JDK searches it.
    pub(super) fn find_in_tree<Q>(&self, root: u32, h: u32, key: &Q) -> Option<u32>
    where
        K: Borrow<Q>,
        Q: JavaHash + Eq + ?Sized,
    {
        // The right subtrees still to search after an unordered node.
        let mut rest = Vec::new();
        let mut p = root;
        loop {
            if p == NIL {
                p = rest.pop()?;
                continue;
            }
            let (pk, ph) = self.key(p);
            let Links { left, right, .. } = self.links[p as usize];
            p = match (ph as i32).cmp(&(h as i32)) {
                Ordering::Greater => left,
                Ordering::Less => right,
                Ordering::Equal if pk.borrow() == key => return Some(p),
                Ordering::Equal => match key.tree_order(pk.borrow()) {
                    Some(Ordering::Less) => left,
                    Some(Ordering::Greater) => right,
                    _ => {
                        rest.push(right);
                        left
                    }
                },
            };
        }
    }

    /// The side of tree node `p` that the new node `x` goes to: `-1` (left)
    /// or `1` (right). Nodes are ordered by their hash as a signed `int`,
    /// then by [`JavaHash::tree_order`], and the rest by `tieBreakOrder`,
    /// whose `System.identityHashCode` comparison is not reproducible
    /// (`-1` is used and the map is marked not exact).
    fn dir(&mut self, x: u32, p: u32) -> i32 {
        let (xk, xh) = self.key(x);
        let (pk, ph) = self.key(p);
        let (xh, ph) = (xh as i32, ph as i32);
        if ph > xh {
            return -1;
        }
        if ph < xh {
            return 1;
        }
        match xk.tree_order(pk) {
            Some(Ordering::Less) => -1,
            Some(Ordering::Greater) => 1,
            _ => {
                self.exact = false;
                -1
            }
        }
    }

    /// `treeifyBin`: doubles a table of fewer than `MIN_TREEIFY_CAPACITY`
    /// bins, or turns the bin into a tree.
    pub(super) fn treeify_bin(&mut self, bin: usize) {
        if self.capacity < MIN_TREEIFY_CAPACITY {
            self.resize();
        } else {
            self.bins[bin].tree = true;
            self.treeify(bin);
        }
    }

    /// `TreeNode.treeify`: builds a red-black tree by inserting the bin's
    /// nodes in `next` order, then moves the root to the front.
    pub(super) fn treeify(&mut self, bin: usize) {
        let mut root = NIL;
        for i in 0..self.bins[bin].nodes.len() {
            let x = self.bins[bin].nodes[i];
            self.links[x as usize].left = NIL;
            self.links[x as usize].right = NIL;
            if root == NIL {
                self.links[x as usize].parent = NIL;
                self.links[x as usize].red = false;
                root = x;
                continue;
            }
            let mut p = root;
            loop {
                let dir = self.dir(x, p);
                let xp = p;
                p = if dir <= 0 { self.links[xp as usize].left } else { self.links[xp as usize].right };
                if p == NIL {
                    self.links[x as usize].parent = xp;
                    if dir <= 0 {
                        self.links[xp as usize].left = x;
                    } else {
                        self.links[xp as usize].right = x;
                    }
                    root = self.balance_insertion(root, x);
                    break;
                }
            }
        }
        self.move_root_to_front(bin, root);
    }

    /// `TreeNode.putTreeVal` for a key not in the tree: links `x` under the
    /// leaf it sorts to and right after that leaf in `next` order, then
    /// rebalances and moves the root to the front.
    pub(super) fn put_tree_val(&mut self, bin: usize, x: u32) {
        let root = self.bins[bin].nodes[0];
        let mut p = root;
        loop {
            let dir = self.dir(x, p);
            let xp = p;
            p = if dir <= 0 { self.links[xp as usize].left } else { self.links[xp as usize].right };
            if p == NIL {
                let at = self.bins[bin].nodes.iter().position(|&n| n == xp).expect("parent in its bin");
                self.bins[bin].nodes.insert(at + 1, x);
                self.links[x as usize] = Links { parent: xp, ..NO_LINKS };
                if dir <= 0 {
                    self.links[xp as usize].left = x;
                } else {
                    self.links[xp as usize].right = x;
                }
                let root = self.balance_insertion(root, x);
                self.move_root_to_front(bin, root);
                return;
            }
        }
    }

    /// `moveRootToFront`: makes `root` the first node of its bin.
    fn move_root_to_front(&mut self, bin: usize, root: u32) {
        let nodes = &mut self.bins[bin].nodes;
        if nodes[0] != root {
            let at = nodes.iter().position(|&n| n == root).expect("root in its bin");
            nodes.remove(at);
            nodes.insert(0, root);
        }
    }

    /// `TreeNode.rotateLeft`.
    fn rotate_left(&mut self, mut root: u32, p: u32) -> u32 {
        let l = &mut self.links;
        let r = if p == NIL { NIL } else { l[p as usize].right };
        if r != NIL {
            let rl = l[r as usize].left;
            l[p as usize].right = rl;
            if rl != NIL {
                l[rl as usize].parent = p;
            }
            let pp = l[p as usize].parent;
            l[r as usize].parent = pp;
            if pp == NIL {
                root = r;
                l[r as usize].red = false;
            } else if l[pp as usize].left == p {
                l[pp as usize].left = r;
            } else {
                l[pp as usize].right = r;
            }
            l[r as usize].left = p;
            l[p as usize].parent = r;
        }
        root
    }

    /// `TreeNode.rotateRight`.
    fn rotate_right(&mut self, mut root: u32, p: u32) -> u32 {
        let l = &mut self.links;
        let left = if p == NIL { NIL } else { l[p as usize].left };
        if left != NIL {
            let lr = l[left as usize].right;
            l[p as usize].left = lr;
            if lr != NIL {
                l[lr as usize].parent = p;
            }
            let pp = l[p as usize].parent;
            l[left as usize].parent = pp;
            if pp == NIL {
                root = left;
                l[left as usize].red = false;
            } else if l[pp as usize].right == p {
                l[pp as usize].right = left;
            } else {
                l[pp as usize].left = left;
            }
            l[left as usize].right = p;
            l[p as usize].parent = left;
        }
        root
    }

    /// `TreeNode.balanceInsertion`: restores the red-black properties after
    /// `x` was linked as a leaf, and returns the root.
    fn balance_insertion(&mut self, mut root: u32, mut x: u32) -> u32 {
        self.links[x as usize].red = true;
        loop {
            let mut xp = self.links[x as usize].parent;
            if xp == NIL {
                self.links[x as usize].red = false;
                return x;
            }
            let mut xpp = self.links[xp as usize].parent;
            if !self.links[xp as usize].red || xpp == NIL {
                return root;
            }
            let xppl = self.links[xpp as usize].left;
            if xp == xppl {
                let xppr = self.links[xpp as usize].right;
                if xppr != NIL && self.links[xppr as usize].red {
                    self.links[xppr as usize].red = false;
                    self.links[xp as usize].red = false;
                    self.links[xpp as usize].red = true;
                    x = xpp;
                } else {
                    if x == self.links[xp as usize].right {
                        x = xp;
                        root = self.rotate_left(root, x);
                        xp = self.links[x as usize].parent;
                        xpp = if xp == NIL { NIL } else { self.links[xp as usize].parent };
                    }
                    if xp != NIL {
                        self.links[xp as usize].red = false;
                        if xpp != NIL {
                            self.links[xpp as usize].red = true;
                            root = self.rotate_right(root, xpp);
                        }
                    }
                }
            } else if xppl != NIL && self.links[xppl as usize].red {
                self.links[xppl as usize].red = false;
                self.links[xp as usize].red = false;
                self.links[xpp as usize].red = true;
                x = xpp;
            } else {
                if x == self.links[xp as usize].left {
                    x = xp;
                    root = self.rotate_right(root, x);
                    xp = self.links[x as usize].parent;
                    xpp = if xp == NIL { NIL } else { self.links[xp as usize].parent };
                }
                if xp != NIL {
                    self.links[xp as usize].red = false;
                    if xpp != NIL {
                        self.links[xpp as usize].red = true;
                        root = self.rotate_left(root, xpp);
                    }
                }
            }
        }
    }
}
