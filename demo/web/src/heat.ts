// Colour for a similarity matrix: where a value sits in the matrix's own
// range, so cosines that all fall in a narrow band still show contrast,
// or on the absolute 0..1 scale; and each row's best match.

/** The least and greatest values off the diagonal; [0, 1] with fewer than two rows. */
export function range(m: number[][]): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  m.forEach((row, i) =>
    row.forEach((v, j) => {
      if (i === j || !Number.isFinite(v)) return;
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }),
  );
  return lo <= hi ? [lo, hi] : [0, 1];
}

/** Where `v` sits between `lo` and `hi`, from 0 to 1; 1 when the range is empty. */
export function place(v: number, lo: number, hi: number): number {
  if (!(hi > lo)) return 1;
  return Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
}

/** The column of the greatest value in row `i`, the diagonal aside; -1 with nothing to compare. */
export function best(m: number[][], i: number): number {
  let at = -1;
  m[i].forEach((v, j) => {
    if (j !== i && Number.isFinite(v) && (at < 0 || v > m[i][at])) at = j;
  });
  return at;
}

/** The opacity of a cell's colour for a place from 0 to 1. */
export function strength(t: number): number {
  return 0.06 + 0.8 * t;
}
