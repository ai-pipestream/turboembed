// What the document views point at together: one chunk and one term.
// Pointing at a chunk in any view lights it in every other view and in
// the text; a term picked in the river marks its occurrences.

export type Focus = { chunk: number | null; term: string | null };

export type Link = {
  /** Calls `l` now and on every change. */
  on(l: (f: Focus) => void): void;
  chunk(i: number | null): void;
  term(t: string | null): void;
};

export function linker(): Link {
  const focus: Focus = { chunk: null, term: null };
  const listeners: ((f: Focus) => void)[] = [];
  const tell = (): void => listeners.forEach((l) => l(focus));
  return {
    on(l) {
      listeners.push(l);
      l(focus);
    },
    chunk(i) {
      if (focus.chunk === i) return;
      focus.chunk = i;
      tell();
    },
    term(t) {
      if (focus.term === t) return;
      focus.term = t;
      tell();
    },
  };
}
