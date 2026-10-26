// A one-line reading of any generated message, from its descriptor: the
// fields that are set, enum values by name, nested messages to a few
// levels. The annotation drawer shows every kind of typed annotation
// this way without a hand-written view per kind.

import type { DescEnum, DescField, DescMessage } from "@bufbuild/protobuf";

type Fields = Record<string, unknown>;

const LISTED = 6;
const TEXT = 48;

function scalar(v: unknown): string {
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : String(Number(v.toPrecision(4)));
  if (typeof v === "bigint") return v.toString();
  if (typeof v === "boolean") return v ? "true" : "";
  if (typeof v === "string") return v.length > TEXT ? `"${v.slice(0, TEXT - 1)}\u2026"` : `"${v}"`;
  if (v instanceof Uint8Array) return `${v.length} bytes`;
  return "";
}

function unset(v: unknown): boolean {
  return v === undefined || v === 0 || v === "" || v === false || v === 0n || (v instanceof Uint8Array && v.length === 0);
}

function enumName(e: DescEnum, n: number): string {
  const v = e.value[n];
  if (!v) return String(n);
  const pre = e.sharedPrefix ?? "";
  return v.name.toLowerCase().startsWith(pre) ? v.name.slice(pre.length) : v.name;
}

function list(f: DescField & { fieldKind: "list" }, v: unknown[], depth: number): string {
  if (f.listKind === "message") {
    const shown = v.slice(0, depth >= 2 ? 0 : 3).map((m) => `{${brief(f.message, m as Fields, depth + 1)}}`);
    return `[${v.length}]${shown.length ? ` ${shown.join(" ")}${v.length > shown.length ? " \u2026" : ""}` : ""}`;
  }
  const one = (x: unknown) => (f.listKind === "enum" ? enumName(f.enum, x as number) : scalar(x));
  return `[${v.slice(0, LISTED).map(one).join(", ")}${v.length > LISTED ? `, \u2026 ${v.length} in all` : ""}]`;
}

/** `v` as field `f` holds it, or "" when it is unset. */
function field(f: DescField, v: unknown, depth: number): string {
  switch (f.fieldKind) {
    case "scalar":
      return unset(v) && !f.proto.proto3Optional ? "" : scalar(v);
    case "enum":
      return v === undefined ? "" : enumName(f.enum, v as number);
    case "message":
      if (v === undefined) return "";
      return depth >= 2 ? "{\u2026}" : `{${brief(f.message, v as Fields, depth + 1)}}`;
    case "list":
      return (v as unknown[]).length ? list(f, v as unknown[], depth) : "";
    case "map": {
      const n = Object.keys(v as object).length;
      return n ? `{${n} entries}` : "";
    }
  }
}

/** The set fields of `m`, a message of `desc`, as `name value` pairs. */
export function brief(desc: DescMessage, m: Fields, depth = 0, skip: string[] = []): string {
  const out: string[] = [];
  for (const member of desc.members) {
    if (skip.includes(member.localName)) continue;
    if (member.kind === "oneof") {
      const o = m[member.localName] as { case?: string; value?: unknown } | undefined;
      const f = o?.case ? member.fields.find((x) => x.localName === o.case) : undefined;
      if (f) out.push(`${f.localName} ${field(f, o?.value, depth) || "{}"}`);
      continue;
    }
    const s = field(member, m[member.localName], depth);
    if (s) out.push(`${member.localName} ${s}`);
  }
  return out.join(", ");
}
