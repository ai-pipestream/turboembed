//! The pickled state dict of a `torch.save` checkpoint, read by a small
//! stack machine that knows only the opcodes and globals such a state dict
//! uses. Nothing is imported or called: a global is one of three names
//! (`collections.OrderedDict`, `torch.FloatStorage`,
//! `torch._utils._rebuild_tensor_v2`), and `REDUCE` and `BINPERSID` build
//! data from them. Any other opcode or global is an error.

use crate::Result;

/// A tensor the pickle describes: where its values are and how they lie.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorRef {
    /// The storage's file under `data/` in the archive.
    pub key: String,
    /// The storage's length in values.
    pub numel: u64,
    pub offset: u64,
    pub size: Vec<u64>,
    pub stride: Vec<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Global {
    OrderedDict,
    FloatStorage,
    RebuildTensor,
}

#[derive(Clone, Debug, PartialEq)]
enum Value {
    Mark,
    None,
    Bool(bool),
    Int(i64),
    Str(String),
    Tuple(Vec<Value>, Size),
    Dict(Vec<(Value, Value)>),
    Global(Global),
    Storage { key: String, numel: u64 },
    Tensor(TensorRef),
}

/// How many values a value holds, itself included, and how deeply they
/// nest. A tuple keeps its own, so a copy from the memo is charged without
/// walking it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Size {
    nodes: usize,
    depth: usize,
}

/// The most values the memo may copy in one pickle. A state dict's memo
/// copies are small tuples and tensors: the GLiREL checkpoint's 420 tensors
/// take under 10,000. Copying a tuple that holds itself twice doubles at
/// every level, and is refused here before the copy is made.
const MAX_COPIED: usize = 1 << 16;

/// The deepest nesting read; a state dict nests four deep.
const MAX_DEPTH: usize = 32;

/// A string, and a tensor's size and stride, are charged by what a copy
/// allocates: one value per this many bytes of a string, one per dimension.
const STR_BYTES: usize = 64;

fn chars(s: &str) -> usize {
    s.len() / STR_BYTES
}

fn size(v: &Value) -> Size {
    match v {
        Value::Tuple(_, s) => *s,
        Value::Dict(items) => combine(items.iter().flat_map(|(k, v)| [k, v])),
        Value::Str(s) | Value::Storage { key: s, .. } => Size { nodes: 1 + chars(s), depth: 0 },
        Value::Tensor(t) => Size { nodes: 1 + chars(&t.key) + t.size.len() + t.stride.len(), depth: 0 },
        _ => Size { nodes: 1, depth: 0 },
    }
}

fn combine<'a>(items: impl IntoIterator<Item = &'a Value>) -> Size {
    items.into_iter().map(size).fold(Size { nodes: 1, depth: 1 }, |a, b| Size {
        nodes: a.nodes.saturating_add(b.nodes),
        depth: a.depth.max(b.depth + 1),
    })
}

fn tuple(items: Vec<Value>) -> Result<Value> {
    let s = combine(&items);
    if s.depth > MAX_DEPTH {
        return Err(format!("pickle: values nest deeper than {MAX_DEPTH}"));
    }
    Ok(Value::Tuple(items, s))
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        let s = self.at.checked_add(n).and_then(|e| self.b.get(self.at..e)).ok_or("pickle: truncated")?;
        self.at += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        let s = self.take(2)?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }
    fn u32(&mut self) -> Result<u32> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    /// A `GLOBAL` argument: a line, as Python writes module and name.
    fn line(&mut self) -> Result<&str> {
        let rest = &self.b[self.at..];
        let n = rest.iter().position(|&c| c == b'\n').ok_or("pickle: truncated")?;
        let s = std::str::from_utf8(&rest[..n]).map_err(|_| "pickle: a global name is not UTF-8")?;
        self.at += n + 1;
        Ok(s)
    }
}

fn pop(stack: &mut Vec<Value>) -> Result<Value> {
    stack.pop().ok_or_else(|| "pickle: stack underflow".into())
}

/// Everything above the topmost mark, which is removed.
fn pop_mark(stack: &mut Vec<Value>) -> Result<Vec<Value>> {
    let m = stack.iter().rposition(|v| *v == Value::Mark).ok_or("pickle: no mark")?;
    let items = stack.split_off(m + 1);
    stack.pop();
    Ok(items)
}

fn nat(v: &Value, what: &str) -> Result<u64> {
    match v {
        Value::Int(i) if *i >= 0 => Ok(*i as u64),
        _ => Err(format!("pickle: {what} is not a non-negative integer")),
    }
}

fn nats(v: &Value, what: &str) -> Result<Vec<u64>> {
    match v {
        Value::Tuple(t, _) => t.iter().map(|x| nat(x, what)).collect(),
        _ => Err(format!("pickle: {what} is not a tuple")),
    }
}

/// `callable(*args)` for the two callables a state dict uses.
fn reduce(callable: Value, args: Value) -> Result<Value> {
    let Value::Tuple(args, _) = args else { return Err("pickle: REDUCE arguments are not a tuple".into()) };
    match (callable, args.as_slice()) {
        (Value::Global(Global::OrderedDict), []) => Ok(Value::Dict(Vec::new())),
        (Value::Global(Global::RebuildTensor), [storage, offset, size, stride, Value::Bool(_), hooks, rest @ ..])
            if rest.len() <= 1 =>
        {
            let Value::Storage { key, numel } = storage else {
                return Err("pickle: a tensor without a storage".into());
            };
            if !matches!(hooks, Value::Dict(h) if h.is_empty()) {
                return Err("pickle: a tensor with backward hooks".into());
            }
            Ok(Value::Tensor(TensorRef {
                key: key.clone(),
                numel: *numel,
                offset: nat(offset, "a storage offset")?,
                size: nats(size, "a tensor size")?,
                stride: nats(stride, "a tensor stride")?,
            }))
        }
        (c, _) => Err(format!("pickle: REDUCE of {c:?} is not read")),
    }
}

/// A persistent id: `("storage", torch.FloatStorage, key, location, numel)`.
fn persistent(pid: Value) -> Result<Value> {
    match pid {
        Value::Tuple(t, _) => match t.as_slice() {
            [Value::Str(kind), Value::Global(Global::FloatStorage), Value::Str(key), Value::Str(_), numel]
                if kind == "storage" =>
            {
                if key.is_empty() || !key.bytes().all(|c| c.is_ascii_alphanumeric()) {
                    return Err(format!("pickle: storage key {key:?}"));
                }
                Ok(Value::Storage { key: key.clone(), numel: nat(numel, "a storage length")? })
            }
            _ => Err("pickle: a persistent id other than an F32 storage".into()),
        },
        _ => Err("pickle: a persistent id that is not a tuple".into()),
    }
}

/// The state dict a checkpoint's `data.pkl` holds: each name with its tensor,
/// in the pickle's order.
pub fn state_dict(bytes: &[u8]) -> Result<Vec<(String, TensorRef)>> {
    let mut r = Reader { b: bytes, at: 0 };
    let mut stack: Vec<Value> = Vec::new();
    let mut memo: std::collections::HashMap<u32, Value> = std::collections::HashMap::new();
    // Values copied into and out of the memo, which a pickle can make grow
    // by doubling; every other value costs an opcode.
    let mut copied = 0usize;
    let mut copy = |v: &Value| -> Result<Value> {
        copied = copied.saturating_add(size(v).nodes);
        if copied > MAX_COPIED {
            return Err(format!("pickle: the memo copies more than {MAX_COPIED} values"));
        }
        Ok(v.clone())
    };
    loop {
        let op = r.u8()?;
        match op {
            0x80 => {
                let p = r.u8()?;
                if !(2..=5).contains(&p) {
                    return Err(format!("pickle: protocol {p}"));
                }
            }
            b'c' => {
                let module = r.line()?.to_owned();
                let name = r.line()?;
                stack.push(Value::Global(match (module.as_str(), name) {
                    ("collections", "OrderedDict") => Global::OrderedDict,
                    ("torch", "FloatStorage") => Global::FloatStorage,
                    ("torch._utils", "_rebuild_tensor_v2") => Global::RebuildTensor,
                    _ => return Err(format!("pickle: global {module}.{name} is not read")),
                }));
            }
            b'q' | b'r' => {
                let k = if op == b'q' { u32::from(r.u8()?) } else { r.u32()? };
                let top = copy(stack.last().ok_or("pickle: stack underflow")?)?;
                memo.insert(k, top);
            }
            b'h' | b'j' => {
                let k = if op == b'h' { u32::from(r.u8()?) } else { r.u32()? };
                let v = memo.get(&k).ok_or("pickle: memo key not set")?;
                // A dict is built in place on the stack; a copy from the memo
                // would miss what was set after it was stored.
                if matches!(v, Value::Dict(_)) {
                    return Err("pickle: a dict referenced twice".into());
                }
                stack.push(copy(v)?);
            }
            b'(' => stack.push(Value::Mark),
            b')' => stack.push(tuple(Vec::new())?),
            b't' => {
                let items = pop_mark(&mut stack)?;
                stack.push(tuple(items)?);
            }
            0x85..=0x87 => {
                let n = usize::from(op - 0x84);
                let at = stack.len().checked_sub(n).ok_or("pickle: stack underflow")?;
                let items = stack.split_off(at);
                stack.push(tuple(items)?);
            }
            b'X' => {
                let n = r.u32()? as usize;
                let s = std::str::from_utf8(r.take(n)?).map_err(|_| "pickle: a string is not UTF-8")?;
                stack.push(Value::Str(s.to_owned()));
            }
            b'J' => stack.push(Value::Int(i64::from(r.u32()? as i32))),
            b'K' => stack.push(Value::Int(i64::from(r.u8()?))),
            b'M' => stack.push(Value::Int(i64::from(r.u16()?))),
            0x88 => stack.push(Value::Bool(true)),
            0x89 => stack.push(Value::Bool(false)),
            b'N' => stack.push(Value::None),
            b'}' => stack.push(Value::Dict(Vec::new())),
            b's' | b'u' => {
                let items = if op == b's' {
                    let v = pop(&mut stack)?;
                    let k = pop(&mut stack)?;
                    vec![k, v]
                } else {
                    pop_mark(&mut stack)?
                };
                if !items.len().is_multiple_of(2) {
                    return Err("pickle: SETITEMS with an odd count".into());
                }
                if items.iter().any(|v| size(v).depth >= MAX_DEPTH) {
                    return Err(format!("pickle: values nest deeper than {MAX_DEPTH}"));
                }
                let Some(Value::Dict(d)) = stack.last_mut() else {
                    return Err("pickle: SETITEM on something not a dict".into());
                };
                let mut it = items.into_iter();
                while let (Some(k), Some(v)) = (it.next(), it.next()) {
                    d.push((k, v));
                }
            }
            b'R' => {
                let args = pop(&mut stack)?;
                let callable = pop(&mut stack)?;
                stack.push(reduce(callable, args)?);
            }
            b'Q' => {
                let pid = pop(&mut stack)?;
                stack.push(persistent(pid)?);
            }
            b'b' => {
                // The state of an OrderedDict: its `_metadata`, not tensors.
                pop(&mut stack)?;
                if !matches!(stack.last(), Some(Value::Dict(_))) {
                    return Err("pickle: BUILD on something not a dict".into());
                }
            }
            b'.' => break,
            _ => return Err(format!("pickle: opcode 0x{op:02x} is not read")),
        }
    }
    let Some(Value::Dict(items)) = stack.pop() else { return Err("pickle: the result is not a dict".into()) };
    items
        .into_iter()
        .map(|(k, v)| match (k, v) {
            (Value::Str(k), Value::Tensor(t)) => Ok((k, t)),
            (k, _) => Err(format!("pickle: state dict entry {k:?} is not a named tensor")),
        })
        .collect()
}
