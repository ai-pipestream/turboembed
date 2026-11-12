// A search hit's text cut at its matched spans, so each match can be
// marked in place. Spans are UTF-8 byte offsets in the document; the
// hit's text starts at the chunk's own start. A cut never splits a
// character: an offset inside one moves back to its first byte.

/** A matched span: which signal matched, what it matched, and where. */
export interface Match {
  source: string;
  label: string;
  start: number;
  end: number;
}

/** A piece of the text and the signals that matched it; none for plain text. */
export interface Piece {
  text: string;
  sources: string[];
  labels: string[];
}

function boundary(bytes: Uint8Array, at: number): number {
  let i = Math.max(0, Math.min(bytes.length, at));
  while (i > 0 && i < bytes.length && (bytes[i] & 0xc0) === 0x80) i--;
  return i;
}

/**
 * `text` in pieces, cut where a match starts or ends. `chunkStart` is
 * the document offset of the text's first byte. A match that lies
 * outside the text, or covers all of it, marks nothing: it says
 * nothing about where in the text the match is.
 */
export function pieces(text: string, chunkStart: number, matches: Match[]): Piece[] {
  const bytes = new TextEncoder().encode(text);
  const inside = matches
    .map((m) => ({ ...m, start: boundary(bytes, m.start - chunkStart), end: boundary(bytes, m.end - chunkStart) }))
    .filter((m, i) => matches[i].start >= chunkStart && matches[i].end - chunkStart <= bytes.length && m.end > m.start && !(m.start === 0 && m.end === bytes.length));
  const cuts = [...new Set([0, bytes.length, ...inside.flatMap((m) => [m.start, m.end])])].sort((a, b) => a - b);
  const dec = new TextDecoder();
  const out: Piece[] = [];
  for (let i = 0; i + 1 < cuts.length; i++) {
    const [a, b] = [cuts[i], cuts[i + 1]];
    const on = inside.filter((m) => m.start <= a && m.end >= b);
    const piece = { text: dec.decode(bytes.subarray(a, b)), sources: [...new Set(on.map((m) => m.source))], labels: [...new Set(on.map((m) => m.label).filter(Boolean))] };
    const last = out[out.length - 1];
    // Neighbouring plain pieces are one piece.
    if (last && !last.sources.length && !piece.sources.length) last.text += piece.text;
    else out.push(piece);
  }
  return out;
}

/** The matches `pieces` cannot place in the text: outside it, or covering all of it. */
export function unplaced(text: string, chunkStart: number, matches: Match[]): Match[] {
  const length = new TextEncoder().encode(text).length;
  return matches.filter((m) => m.start < chunkStart || m.end - chunkStart > length || m.end <= m.start || (m.start === chunkStart && m.end - chunkStart === length));
}
