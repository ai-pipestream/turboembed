// Waiting for a page to stop calling the server. The panels under
// Analyze each make their own call for a new document and can push the
// ones below them down when they draw, so a mark pointed at before then
// can move out from under the pointer.

import type { Page } from "playwright";

const inFlight = new WeakMap<Page, number>();

/** Counts the calls `page` has made that have not answered yet; call before the page loads. */
export function track(page: Page): void {
  const add = (n: number) => inFlight.set(page, (inFlight.get(page) ?? 0) + n);
  page.on("request", () => add(1));
  page.on("requestfinished", () => add(-1));
  page.on("requestfailed", () => add(-1));
}

/**
 * Resolves once no call from `page` has been open for half a second, and
 * rejects if that has not happened within `ms`, so a call that never
 * answers fails the test instead of hanging it.
 */
export async function settled(page: Page, ms = 30_000): Promise<void> {
  const deadline = Date.now() + ms;
  let quietSince = Date.now();
  for (;;) {
    if ((inFlight.get(page) ?? 0) > 0) quietSince = Date.now();
    else if (Date.now() - quietSince >= 500) return;
    if (Date.now() > deadline) throw new Error(`the page still had ${inFlight.get(page)} calls open after ${ms} ms`);
    await page.waitForTimeout(50);
  }
}
