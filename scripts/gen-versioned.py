#!/usr/bin/env python3
"""Regenerate crates/turbo-abi/src/versioned.rs from the ABI struct field lists."""
import re, subprocess, sys, pathlib
root = pathlib.Path(__file__).resolve().parent.parent
structs = []
for f in ['crates/turbo-abi/src/lib.rs', 'crates/turbo-abi/src/provider.rs']:
    s = (root / f).read_text()
    for m in re.finditer(r'pub struct (turbo_\w+) \{(.*?)\n\}', s, re.S):
        name, body = m.group(1), m.group(2)
        fields = re.findall(r'^\s*pub (\w+):', body, re.M)
        if fields and fields[0] == 'struct_size':
            structs.append((name, fields))
out = ['//! Known `struct_size` values per ABI struct.',
'//!',
'//! A caller declares the layout it was compiled against through',
'//! `struct_size`. The library accepts a size only when it is the end of a',
'//! field the struct has ever had, so every accepted prefix is a layout that',
'//! could have shipped: fields are only ever appended, and a prefix that ends',
'//! inside a field (half a pointer, half a count) is never a layout.',
'//! Everything else is `TURBO_E_INVALID_STRUCT_SIZE`.',
'//!',
'//! Generated from the field lists of `lib.rs` and `provider.rs` by',
'//! `scripts/gen-versioned.py`; regenerate when a struct changes.',
'',
'use core::mem::{offset_of, size_of};',
'',
'/// An ABI struct that starts with `uint32_t struct_size`.',
'pub trait Versioned {',
'    /// Every accepted `struct_size`, ascending, ending with the current size.',
"    const SIZES: &'static [usize];",
'',
'    /// True when `size` is a layout this library understands.',
'    fn size_is_known(size: u32) -> bool {',
'        Self::SIZES.contains(&(size as usize))',
'    }',
'}',
'']
for name, fields in structs:
    ends = [f'offset_of!(super::{name}, {f})' for f in fields[2:]] + [f'size_of::<super::{name}>()']
    out.append(f'impl Versioned for super::{name} {{')
    out.append("    const SIZES: &'static [usize] = &[")
    out += [f'        {e},' for e in ends]
    out.append('    ];')
    out.append('}')
    out.append('')
target = root / 'crates/turbo-abi/src/versioned.rs'
# Format with the repository's rustfmt settings so the committed file is
# byte-identical to what `cargo fmt` would produce.
tmp = target.with_name('versioned.gen.rs')
tmp.write_text('\n'.join(out) + '\n')
subprocess.run(['rustfmt', '--edition', '2021', str(tmp)], check=True)
text = tmp.read_text()
tmp.unlink()
if '--check' in sys.argv:
    sys.exit(0 if target.read_text() == text else 'versioned.rs is stale; run scripts/gen-versioned.py')
target.write_text(text)
