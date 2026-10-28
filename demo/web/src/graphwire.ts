// A reader for the binary DocumentGraph bytes (the analysis export
// boundary, format version 1): little-endian counts, u32-length-prefixed
// bytes and strings, one-byte tags for options and statuses. It names
// every part with its offset and length so the page can show where the
// bytes go. It follows analysis/core/src/export (graph.rs, wire.rs,
// error_wire.rs) and reads, never decodes, a layer body: that is its
// codec's.

export type Part = { name: string; at: number; len: number; value?: string; parts?: Part[] };

export const MAGIC = "TGRF";
export const FORMAT_VERSION = 1;
const LISTED = 64;

export class WireError extends Error {
  constructor(
    message: string,
    readonly at: number,
  ) {
    super(`${message} at byte ${at}`);
  }
}

class Cursor {
  at = 0;
  private readonly dv: DataView;
  private readonly utf8 = new TextDecoder("utf-8", { fatal: true });

  constructor(readonly buf: Uint8Array) {
    this.dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  }

  take(n: number): Uint8Array {
    if (this.at + n > this.buf.length) throw new WireError("truncated", this.at);
    const out = this.buf.subarray(this.at, this.at + n);
    this.at += n;
    return out;
  }

  u8(): number {
    return this.take(1)[0];
  }

  u16(): number {
    const v = this.dv.getUint16(this.at, true);
    this.take(2);
    return v;
  }

  u32(): number {
    if (this.at + 4 > this.buf.length) throw new WireError("truncated", this.at);
    const v = this.dv.getUint32(this.at, true);
    this.at += 4;
    return v;
  }

  u64(): bigint {
    if (this.at + 8 > this.buf.length) throw new WireError("truncated", this.at);
    const v = this.dv.getBigUint64(this.at, true);
    this.at += 8;
    return v;
  }

  str(): string {
    const start = this.at;
    const b = this.take(this.u32());
    try {
      return this.utf8.decode(b);
    } catch {
      throw new WireError("a string that is not UTF-8", start);
    }
  }
}

type Read = (c: Cursor) => { value?: string; parts?: Part[] };

/** Runs `read` and records the bytes it took as one named part. */
function part(c: Cursor, name: string, read: Read): Part {
  const at = c.at;
  const r = read(c);
  return { name, at, len: c.at - at, ...r };
}

function str(c: Cursor, name: string): Part {
  return part(c, name, (c) => ({ value: c.str() }));
}

function tagError(c: Cursor, what: string, tag: number): WireError {
  return new WireError(`${what} tag of ${tag}`, c.at - 1);
}

function layerRef(c: Cursor, name: string): Part {
  return part(c, name, (c) => ({ value: `${c.str()}#${c.str()}` }));
}

function edits(c: Cursor, name: string): Part {
  return part(c, name, (c) => {
    const n = c.u32();
    const parts: Part[] = [];
    for (let i = 0; i < n; i++) {
      const at = c.at;
      const [ps, pe, cs, ce] = [c.u32(), c.u32(), c.u32(), c.u32()];
      if (i < LISTED) parts.push({ name: `edit ${i}`, at, len: 16, value: `${ps}..${pe} \u2192 ${cs}..${ce}` });
    }
    return { value: `${n} edits${n > LISTED ? `, first ${LISTED} listed` : ""}`, parts };
  });
}

function error(c: Cursor): Part {
  return part(c, "error", (c) => {
    const tag = c.u8();
    const one = (kind: string, parts: Part[]) => ({ value: kind, parts });
    switch (tag) {
      case 0:
        return one("missing", [layerRef(c, "layer")]);
      case 1:
        return one("duplicate", [layerRef(c, "layer")]);
      case 2:
        return one("clash", [layerRef(c, "layer")]);
      case 3:
        return one("bounds", [layerRef(c, "layer"), str(c, "view"), str(c, "detail")]);
      case 4:
        return one("misaligned", [layerRef(c, "layer"), layerRef(c, "anchor"), u32(c, "len"), u32(c, "anchor len")]);
      case 5:
        return one("unaligned", [layerRef(c, "layer"), layerRef(c, "on"), str(c, "read")]);
      case 6:
        return one("invalid", [layerRef(c, "layer"), str(c, "detail")]);
      case 7:
        return one("pipeline", [str(c, "detail")]);
      case 8:
        return one("annotator", [str(c, "annotator"), str(c, "detail")]);
      case 9:
        return one("skipped", [str(c, "reason")]);
      case 10:
        return one("offset", [
          part(c, "unit", (c) => {
            const u = c.u8();
            if (u > 1) throw tagError(c, "an offset unit", u);
            return { value: u === 0 ? "utf16" : "code point" };
          }),
          u32(c, "at"),
          part(c, "reason", (c) => {
            const r = c.u8();
            if (r === 0) return { value: `past end, len ${c.u32()}` };
            if (r === 1) return { value: "inside a character" };
            throw tagError(c, "an offset reason", r);
          }),
        ]);
      case 11:
        return one("source", [
          part(c, "reason", (c) => {
            const r = c.u8();
            if (r === 0) return { value: `too long, ${c.u64()} bytes` };
            if (r === 1) return { value: `misaligned: ${c.str()}` };
            throw tagError(c, "a source error", r);
          }),
        ]);
      default:
        throw tagError(c, "an error", tag);
    }
  });
}

function u32(c: Cursor, name: string): Part {
  return part(c, name, (c) => ({ value: String(c.u32()) }));
}

function status(c: Cursor): Part {
  return part(c, "status", (c) => {
    const tag = c.u8();
    if (tag === 0) return { value: "complete" };
    if (tag === 1) return { value: `skipped: ${c.str()}` };
    if (tag === 2) return { value: "failed", parts: [error(c)] };
    throw tagError(c, "a status", tag);
  });
}

function option(c: Cursor, name: string, what: string, read: Read): Part {
  return part(c, name, (c) => {
    const tag = c.u8();
    if (tag === 0) return { value: "none" };
    if (tag === 1) return read(c);
    throw tagError(c, what, tag);
  });
}

function refs(c: Cursor, name: string): Part {
  return part(c, name, (c) => {
    const n = c.u32();
    const parts: Part[] = [];
    for (let i = 0; i < n; i++) parts.push(layerRef(c, `${name} ${i}`));
    return { value: parts.map((p) => p.value).join(", ") || "none", parts };
  });
}

function provenance(c: Cursor): Part {
  return part(c, "provenance", (c) => {
    const parts = [
      str(c, "annotator"),
      str(c, "version"),
      part(c, "config", (c) => ({ value: c.u64().toString(16).padStart(16, "0") })),
      option(c, "model", "a model", (c) => ({ parts: [str(c, "id"), str(c, "revision"), str(c, "sha256")] })),
      option(c, "execution", "an execution", (c) => ({ parts: [str(c, "backend"), str(c, "device"), str(c, "precision")] })),
      refs(c, "reads"),
      refs(c, "on"),
    ];
    return { value: `${parts[0].value} ${parts[1].value}`, parts };
  });
}

function sized(c: Cursor, name: string): Part {
  return part(c, name, (c) => {
    const n = c.u32();
    c.take(n);
    return { value: `${n} bytes` };
  });
}

/** The parts of a whole graph, in byte order; throws WireError on anything it cannot read. */
export function walkGraph(bytes: Uint8Array): Part {
  const c = new Cursor(bytes);
  const parts: Part[] = [];
  parts.push(
    part(c, "magic", (c) => {
      const m = String.fromCharCode(...c.take(4));
      if (m !== MAGIC) throw new WireError("not a document graph (bad magic)", 0);
      return { value: m };
    }),
  );
  parts.push(
    part(c, "format version", (c) => {
      const v = c.u16();
      if (v !== FORMAT_VERSION) throw new WireError(`format version ${v}, not ${FORMAT_VERSION}`, 4);
      return { value: String(v) };
    }),
  );
  parts.push(sized(c, "source bytes"));
  parts.push(part(c, "source text", (c) => ({ value: `${c.str().length} UTF-16 units` })));
  parts.push(option(c, "source alignment", "an alignment", (c) => ({ parts: [edits(c, "edits")] })));
  parts.push(
    part(c, "views", (c) => {
      const n = c.u32();
      const views: Part[] = [];
      for (let i = 0; i < n; i++) {
        views.push(
          part(c, `view ${i}`, (c) => {
            const p = [
              str(c, "instance"),
              status(c),
              provenance(c),
              option(c, "parent", "a parent", (c) => ({ value: c.str() })),
              option(c, "text", "a text", (c) => ({ value: `${c.str().length} UTF-16 units` })),
              edits(c, "edits"),
            ];
            return { value: p[0].value, parts: p };
          }),
        );
      }
      return { value: `${n} views`, parts: views };
    }),
  );
  parts.push(
    part(c, "layers", (c) => {
      const n = c.u32();
      const layers: Part[] = [];
      for (let i = 0; i < n; i++) {
        layers.push(
          part(c, `layer ${i}`, (c) => {
            const p = [
              str(c, "layer"),
              str(c, "instance"),
              part(c, "schema", (c) => ({ value: String(c.u16()) })),
              u32(c, "items"),
              status(c),
              provenance(c),
              option(c, "body", "a body", (c) => {
                const n = c.u32();
                c.take(n);
                return { value: `${n} bytes` };
              }),
            ];
            return { value: `${p[0].value}#${p[1].value}`, parts: p };
          }),
        );
      }
      return { value: `${n} layers`, parts: layers };
    }),
  );
  if (c.at !== bytes.length) throw new WireError(`${bytes.length - c.at} trailing bytes`, c.at);
  return { name: "document graph", at: 0, len: bytes.length, value: `format ${FORMAT_VERSION}`, parts };
}

/** The first `n` bytes of a part as hex, for a tooltip. */
export function hex(bytes: Uint8Array, p: Part, n = 32): string {
  const b = bytes.subarray(p.at, p.at + Math.min(p.len, n));
  let s = "";
  for (const x of b) s += (s ? " " : "") + x.toString(16).padStart(2, "0");
  return p.len > n ? `${s} \u2026` : s;
}
