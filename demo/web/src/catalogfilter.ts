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

/** The words of a typed filter, lower case. */
export function words(text: string): string[] {
  return text.toLowerCase().split(" ").map((w) => w.trim()).filter((w) => w.length > 0);
}

/**
 * Whether `e` is of `kind` (any when empty) and every word typed is in
 * its id, kind, language or licence, whatever the case.
 */
export function matches(e: Filterable, kind: string, text: string): boolean {
  if (kind && e.kind !== kind) return false;
  const hay = `${e.id} ${kindName(e.kind)} ${e.language} ${e.licence}`.toLowerCase();
  return words(text).every((w) => hay.includes(w));
}

/** The kinds in `entries` with how many of each, in the order first seen. */
export function kinds(entries: Filterable[]): [string, number][] {
  const count = new Map<string, number>();
  for (const e of entries) count.set(e.kind, (count.get(e.kind) ?? 0) + 1);
  return [...count];
}
