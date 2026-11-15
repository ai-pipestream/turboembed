// What a status check's outcome means for the lists the page shows.
// Kept apart from the calls so it can be tested.

/** One status check: whether the server answered, and whether it said ready. */
export type Status = { up: boolean; ready: boolean };

/**
 * Whether the lists read from the server are worth reading again after
 * a check went from `was` (null before the first check) to `now`:
 * - the first outcome of any kind, so a list is first read once it is
 *   known whether the server answers;
 * - the server stopped answering, so each list says so in its place;
 * - the server answers again;
 * - the server turned ready, since one that answers while it loads its
 *   models lists none of them yet.
 */
export function stale(was: Status | null, now: Status): boolean {
  if (was === null) return true;
  if (was.up !== now.up) return true;
  return now.up && now.ready && !was.ready;
}
