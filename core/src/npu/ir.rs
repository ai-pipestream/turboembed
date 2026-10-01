//! The OpenVINO IR as the driver's compiler takes it, with no OpenVINO in
//! the library.
//!
//! zeGraphCreate2 with ZE_GRAPH_FORMAT_NGRAPH_LITE takes one buffer whose
//! layout is a contract between OpenVINO's NPU plugin and the compiler in
//! the driver (openvino, intel_npu/src/compiler_adapter, serializeIR):
//! the compiler's version, the count of blocks (2), then the xml and the
//! weights, each with its u64 size in front. `container` builds it from
//! the bundle's verified files.
//!
//! The compiler also takes the graph's boundary in pBuildFlags:
//! --inputs_precisions, --inputs_layouts, --outputs_precisions and
//! --outputs_layouts, each argument named by its index (compilers 5.9 and
//! later; this backend refuses older ones rather than guessing names).
//! `interface` reads what those flags need, and nothing else, from the
//! IR's xml: each Parameter layer's element type and rank, and each
//! Result layer's port precision and rank, in document order, which is
//! the order the IR front end numbers them in.

/// One Parameter or Result of the IR: what the build flags say about it.
#[derive(Debug, PartialEq, Eq)]
pub struct Port {
    pub name: String,
    /// The compiler's legacy precision name: FP32, FP16, I64...
    pub precision: String,
    pub rank: usize,
    /// Each dimension, None where the xml leaves it dynamic.
    pub dims: Vec<Option<u64>>,
}

/// The IR's boundary: its Parameters and Results, in document order,
/// and the highest opset any of its layers names, for the check against
/// what the driver's compiler supports.
#[derive(Debug, Default)]
pub struct Interface {
    pub inputs: Vec<Port>,
    pub outputs: Vec<Port>,
    pub max_opset: u32,
}

/// The ov element type names an IR's Parameter carries, as the compiler's
/// legacy precision names them (--inputs_precisions takes the legacy
/// ones).
fn legacy_precision(element_type: &str) -> Option<&'static str> {
    Some(match element_type {
        "f64" => "FP64",
        "f32" => "FP32",
        "f16" => "FP16",
        "bf16" => "BF16",
        "u64" => "U64",
        "u32" => "U32",
        "u16" => "U16",
        "u8" => "U8",
        "u4" => "U4",
        "i64" => "I64",
        "i32" => "I32",
        "i16" => "I16",
        "i8" => "I8",
        "i4" => "I4",
        _ => return None,
    })
}

/// The precisions a Result port may carry, already in legacy spelling.
const PORT_PRECISIONS: [&str; 14] =
    ["FP64", "FP32", "FP16", "BF16", "U64", "U32", "U16", "U8", "U4", "I64", "I32", "I16", "I8", "I4"];

/// The layout the compiler files a rank under, as OpenVINO's adapter maps
/// it (rankToLegacyLayoutString).
pub fn layout(rank: usize) -> Result<&'static str, String> {
    Ok(match rank {
        1 => "C",
        2 => "NC",
        3 => "CHW",
        4 => "NCHW",
        5 => "NCDHW",
        _ => return Err(format!("npu: an IR argument of rank {rank} has no layout the compiler takes")),
    })
}

/// One xml element's attributes, as name to value, read with a quote
/// state so a quoted '>' does not end the tag.
fn attributes(tag: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let b = tag.as_bytes();
    let mut i = 0;
    while i < b.len() {
        // The attribute name, up to '='.
        while i < b.len() && (b[i].is_ascii_whitespace() || b[i] == b'/' || b[i] == b'?') {
            i += 1;
        }
        let start = i;
        while i < b.len() && b[i] != b'=' && !b[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= b.len() || b[i] != b'=' {
            break;
        }
        let name = &tag[start..i];
        i += 1;
        if i >= b.len() || (b[i] != b'"' && b[i] != b'\'') {
            break;
        }
        let quote = b[i];
        i += 1;
        let vstart = i;
        while i < b.len() && b[i] != quote {
            i += 1;
        }
        if i >= b.len() {
            break;
        }
        out.push((name.to_owned(), tag[vstart..i].to_owned()));
        i += 1;
    }
    out
}

fn attr<'a>(attrs: &'a [(String, String)], name: &str) -> Option<&'a str> {
    attrs.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_str())
}

/// The next element opening with `name` at or after `from`, and where its
/// tag's attribute text starts and the tag ends. None when there is none
/// before `until`.
fn find_tag(xml: &str, name: &str, from: usize, until: usize) -> Option<(usize, usize, usize)> {
    let open = format!("<{name}");
    let mut at = from;
    while at < until {
        let rel = xml[at..until].find(&open)?;
        let start = at + rel;
        let after = start + open.len();
        // "<layer" must not match "<layers": the next byte ends the name.
        match xml.as_bytes().get(after) {
            Some(b) if b.is_ascii_whitespace() || *b == b'>' || *b == b'/' => {}
            _ => {
                at = after;
                continue;
            }
        }
        // The tag's end, '>' outside quotes.
        let b = xml.as_bytes();
        let mut i = after;
        let mut quote = 0u8;
        while i < until {
            match b[i] {
                q @ (b'"' | b'\'') if quote == 0 => quote = q,
                q if q == quote => quote = 0,
                b'>' if quote == 0 => return Some((start, after, i)),
                _ => {}
            }
            i += 1;
        }
        return None;
    }
    None
}

/// `shape="1,128"` as dims: "?" and "-1" and ".." spans are dynamic.
fn shape_dims(shape: &str) -> Vec<Option<u64>> {
    if shape.trim().is_empty() {
        return Vec::new();
    }
    shape.split(',').map(|d| d.trim().parse::<u64>().ok()).collect()
}

/// The `<dim>` values of the first `<port>` under the element between
/// `from` and `until`: the Result's input port.
fn port_dims(xml: &str, from: usize, until: usize) -> Option<(Vec<Option<u64>>, Option<String>)> {
    let (_, pstart, pend) = find_tag(xml, "port", from, until)?;
    let attrs = attributes(&xml[pstart..pend]);
    let precision = attr(&attrs, "precision").map(str::to_owned);
    let port_end = xml[pend..until].find("</port>").map_or(until, |r| pend + r);
    let mut dims = Vec::new();
    let mut at = pend;
    while let Some((_, dstart, dend)) = find_tag(xml, "dim", at, port_end) {
        let text_end = xml[dend + 1..port_end].find('<').map_or(port_end, |r| dend + 1 + r);
        dims.push(xml[dend + 1..text_end].trim().parse::<u64>().ok());
        let _ = dstart;
        at = text_end;
    }
    Some((dims, precision))
}

/// The IR's Parameters and Results, document order, from its xml. Refuses
/// an IR it cannot read exactly, naming what is missing: the build flags
/// built from a guess would compile a wrong boundary.
pub fn interface(xml: &[u8]) -> Result<Interface, String> {
    let xml = std::str::from_utf8(xml).map_err(|_| "npu: the IR's xml is not UTF-8".to_owned())?;
    let mut io = Interface::default();
    let mut at = 0;
    while let Some((start, astart, aend)) = find_tag(xml, "layer", at, xml.len()) {
        let attrs = attributes(&xml[astart..aend]);
        let kind = attr(&attrs, "type").unwrap_or("");
        let name = attr(&attrs, "name").unwrap_or("").to_owned();
        if let Some(opset) = attr(&attrs, "version").and_then(|v| v.strip_prefix("opset")).and_then(|v| v.parse().ok())
        {
            io.max_opset = io.max_opset.max(opset);
        }
        let body_end = match xml[aend..].find("</layer>") {
            Some(r) => aend + r,
            None => xml.len(),
        };
        match kind {
            "Parameter" => {
                let (_, dstart, dend) = find_tag(xml, "data", aend, body_end)
                    .ok_or_else(|| format!("npu: Parameter {name:?} has no <data> in the IR's xml"))?;
                let dattrs = attributes(&xml[dstart..dend]);
                let et = attr(&dattrs, "element_type")
                    .ok_or_else(|| format!("npu: Parameter {name:?} has no element_type"))?;
                let precision = legacy_precision(et)
                    .ok_or_else(|| format!("npu: Parameter {name:?} is {et}, which the compiler flags cannot name"))?;
                let dims = shape_dims(attr(&dattrs, "shape").unwrap_or(""));
                io.inputs.push(Port { name, precision: precision.to_owned(), rank: dims.len(), dims });
            }
            "Result" => {
                let (dims, precision) = port_dims(xml, aend, body_end)
                    .ok_or_else(|| format!("npu: Result {name:?} has no <port> in the IR's xml"))?;
                let precision = precision.ok_or_else(|| format!("npu: Result {name:?}'s port has no precision"))?;
                if !PORT_PRECISIONS.contains(&precision.as_str()) {
                    return Err(format!("npu: Result {name:?} is {precision}, which the compiler flags cannot name"));
                }
                io.outputs.push(Port { name, precision, rank: dims.len(), dims });
            }
            _ => {}
        }
        at = body_end.max(start + 1);
    }
    if io.inputs.is_empty() || io.outputs.is_empty() {
        return Err(format!(
            "npu: the IR's xml has {} Parameter and {} Result layers; an encoder needs at least one of each",
            io.inputs.len(),
            io.outputs.len()
        ));
    }
    Ok(io)
}

/// The compiler's build flags for the boundary, each argument named by
/// its index, exactly as OpenVINO's adapter writes them for compilers 5.9
/// and later.
pub fn build_flags(io: &Interface) -> Result<String, String> {
    let group = |ports: &[Port], value: &dyn Fn(&Port) -> Result<String, String>| -> Result<String, String> {
        let mut s = String::new();
        for (i, p) in ports.iter().enumerate() {
            if i > 0 {
                s.push(' ');
            }
            s.push_str(&format!("{i}:{}", value(p)?));
        }
        Ok(s)
    };
    let precisions_in = group(&io.inputs, &|p| Ok(p.precision.clone()))?;
    let layouts_in = group(&io.inputs, &|p| layout(p.rank).map(str::to_owned))?;
    let precisions_out = group(&io.outputs, &|p| Ok(p.precision.clone()))?;
    let layouts_out = group(&io.outputs, &|p| layout(p.rank).map(str::to_owned))?;
    Ok(format!(
        "--inputs_precisions=\"{precisions_in}\" --inputs_layouts=\"{layouts_in}\" \
         --outputs_precisions=\"{precisions_out}\" --outputs_layouts=\"{layouts_out}\""
    ))
}

/// The NGRAPH_LITE buffer: the compiler's version, the block count (2),
/// then the xml and the weights, each behind its u64 size. Little-endian,
/// as the plugin's memcpy writes it on the x86 hosts an Intel NPU sits
/// in.
pub fn container(compiler: (u16, u16), xml: &[u8], weights: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + 4 + 8 + xml.len() + 8 + weights.len());
    out.extend_from_slice(&compiler.0.to_le_bytes());
    out.extend_from_slice(&compiler.1.to_le_bytes());
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(xml.len() as u64).to_le_bytes());
    out.extend_from_slice(xml);
    out.extend_from_slice(&(weights.len() as u64).to_le_bytes());
    out.extend_from_slice(weights);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A MiniLM-shaped IR boundary, cut down to what interface() reads.
    const XML: &str = r#"<?xml version="1.0"?>
<net name="torch_jit" version="11">
    <layers>
        <layer id="0" name="input_ids" type="Parameter" version="opset1">
            <data shape="1,128" element_type="i64" />
            <output>
                <port id="0" precision="I64" names="input_ids"><dim>1</dim><dim>128</dim></port>
            </output>
        </layer>
        <layer id="1" name="attention_mask" type="Parameter" version="opset1">
            <data shape="1,128" element_type="i64" />
            <output>
                <port id="0" precision="I64" names="attention_mask"><dim>1</dim><dim>128</dim></port>
            </output>
        </layer>
        <layer id="2" name="token_type_ids" type="Parameter" version="opset1">
            <data shape="1,128" element_type="i64" />
            <output>
                <port id="0" precision="I64" names="token_type_ids"><dim>1</dim><dim>128</dim></port>
            </output>
        </layer>
        <layer id="3" name="Constant_1" type="Const" version="opset8">
            <data element_type="f16" shape="30522,384" offset="0" size="23440896" />
            <output><port id="0" precision="FP16"><dim>30522</dim><dim>384</dim></port></output>
        </layer>
        <layer id="400" name="last_hidden_state" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32"><dim>1</dim><dim>128</dim><dim>384</dim></port>
            </input>
        </layer>
    </layers>
</net>"#;

    #[test]
    fn the_boundary_is_read_from_the_xml() {
        let io = interface(XML.as_bytes()).unwrap();
        assert_eq!(io.inputs.len(), 3, "the Const layer is not a Parameter");
        assert_eq!(io.inputs[0].name, "input_ids");
        assert_eq!(io.inputs[0].precision, "I64");
        assert_eq!(io.inputs[0].dims, vec![Some(1), Some(128)]);
        assert_eq!(io.inputs[2].name, "token_type_ids");
        assert_eq!(io.outputs.len(), 1);
        assert_eq!(io.outputs[0].name, "last_hidden_state");
        assert_eq!(io.outputs[0].precision, "FP32");
        assert_eq!(io.outputs[0].dims, vec![Some(1), Some(128), Some(384)]);
        assert_eq!(io.max_opset, 8, "the highest opset any layer names");
    }

    #[test]
    fn a_dynamic_shape_is_read_as_unknown_dims() {
        let xml = XML.replace("shape=\"1,128\"", "shape=\"?,?\"");
        let io = interface(xml.as_bytes()).unwrap();
        assert_eq!(io.inputs[0].dims, vec![None, None]);
        assert_eq!(io.inputs[0].rank, 2, "the rank is still known, for the layout flag");
    }

    #[test]
    fn the_build_flags_are_the_adapters_exactly() {
        let io = interface(XML.as_bytes()).unwrap();
        assert_eq!(
            build_flags(&io).unwrap(),
            "--inputs_precisions=\"0:I64 1:I64 2:I64\" --inputs_layouts=\"0:NC 1:NC 2:NC\" \
             --outputs_precisions=\"0:FP32\" --outputs_layouts=\"0:CHW\""
        );
    }

    #[test]
    fn an_unreadable_boundary_is_refused_with_its_reason() {
        let e = interface(b"<net></net>").unwrap_err();
        assert!(e.contains("0 Parameter and 0 Result"), "{e}");
        let xml = XML.replace("element_type=\"i64\" ", "");
        // The Parameter's <data> loses element_type; the Const keeps its own.
        let e = interface(xml.as_bytes()).unwrap_err();
        assert!(e.contains("input_ids") && e.contains("element_type"), "{e}");
        let xml = XML.replace("element_type=\"i64\"", "element_type=\"string\"");
        let e = interface(xml.as_bytes()).unwrap_err();
        assert!(e.contains("string"), "{e}");
    }

    #[test]
    fn the_container_is_the_plugins_serialization() {
        let b = container((7, 20), b"<xml/>", b"\x01\x02\x03");
        let mut want = Vec::new();
        want.extend_from_slice(&7u16.to_le_bytes());
        want.extend_from_slice(&20u16.to_le_bytes());
        want.extend_from_slice(&2u32.to_le_bytes());
        want.extend_from_slice(&6u64.to_le_bytes());
        want.extend_from_slice(b"<xml/>");
        want.extend_from_slice(&3u64.to_le_bytes());
        want.extend_from_slice(b"\x01\x02\x03");
        assert_eq!(b, want);
        assert_eq!(b.len(), 4 + 4 + 8 + 6 + 8 + 3);
    }

    #[test]
    fn ranks_map_to_the_legacy_layouts() {
        assert_eq!(layout(2).unwrap(), "NC");
        assert_eq!(layout(3).unwrap(), "CHW");
        assert_eq!(layout(4).unwrap(), "NCHW");
        assert!(layout(0).is_err() && layout(6).is_err());
    }
}
