// Narrowing the model catalog in the page: by kind and by words typed.
// Nothing here calls the server; the catalog is read once and filtered
// as shown.

/** What the filter reads of a catalog entry. */
export interface Filterable {
  id: string;
  kind: string;
  language: string;
  licence: string;
}

/** A catalog kind in plain words: `KIND_MODEL` is "model". */
export function kindName(kind: string): string {
  const name = kind.startsWith("KIND_") ? kind.slice(5) : kind;
  return name.toLowerCase().split("_").join(" ");
}

/** An enum value without its type's prefix, in lower case: `TASK_EMBED` with `TASK_` is "embed". */
export function plain(value: string, prefix: string): string {
  const name = value.startsWith(prefix) ? value.slice(prefix.length) : value;
  return name.toLowerCase().split("_").join(" ");
}

/**
 * Text folded for comparing whatever the case: upper case then lower
 * case, so `ß` and `SS` both read "ss", and the dot that lower casing
 * a dotted capital I leaves on the `i` dropped, so "istanbul" finds it.
 */
export function fold(text: string): string {
  return text.toUpperCase().toLowerCase().split("i\u0307").join("i");
}

/**
 * The words of a typed filter, folded. Any white space parts two
 * words: a tab, a no-break space and an ideographic space as much as
 * a space.
 */
export function words(text: string): string[] {
  const out: string[] = [];
  let word = "";
  for (const ch of text) {
    if (ch.trim() !== "") {
      word += ch;
      continue;
    }
    if (word) out.push(fold(word));
    word = "";
  }
  if (word) out.push(fold(word));
  return out;
}

/**
 * Whether `e` is of `kind` (any when empty) and every word typed is in
 * its id, kind, language or licence, whatever the case. The kind
 * matches in plain words and in its enum spelling.
 */
export function matches(e: Filterable, kind: string, text: string): boolean {
  if (kind && e.kind !== kind) return false;
  const hay = fold(`${e.id} ${kindName(e.kind)} ${e.kind} ${e.language} ${e.licence}`);
  return words(text).every((w) => hay.includes(w));
}

/** The kinds in `entries` with how many of each, in the order first seen. */
export function kinds(entries: Filterable[]): [string, number][] {
  const count = new Map<string, number>();
  for (const e of entries) count.set(e.kind, (count.get(e.kind) ?? 0) + 1);
  return [...count];
}
