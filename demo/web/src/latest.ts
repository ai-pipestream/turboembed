// An answer is drawn only when it belongs to the latest call of its
// kind: a slow answer to an older call must not paint over a newer one.

/**
 * A counter for one kind of call. Each call takes a ticket before it
 * asks; the ticket says, when the answer is in, whether a later call
 * has started since.
 *
 *     const fresh = latest();
 *     const mine = fresh();
 *     const r = await ask();
 *     if (!mine()) return;
 */
export function latest(): () => () => boolean {
  let n = 0;
  return () => {
    const mine = ++n;
    return () => mine === n;
  };
}
