// Waiting for a page to stop calling the server. Panels are drawn when
// their own calls answer and can push the ones below them down, so a
// mark pointed at before then can move out from under the pointer.

import type { Page } from "playwright";

const inFlight = new WeakMap<Page, number>();

/** Counts the calls `page` has made that have not answered yet; call before the page loads. */
export function track(page: Page): void {
  const add = (n: number) => inFlight.set(page, (inFlight.get(page) ?? 0) + n);
  page.on("request", () => add(1));
  page.on("requestfinished", () => add(-1));
  page.on("requestfailed", () => add(-1));
}

/** Resolves once no call from `page` has been open for half a second. */
export async function settled(page: Page): Promise<void> {
  let quietSince = Date.now();
  for (;;) {
    if ((inFlight.get(page) ?? 0) > 0) quietSince = Date.now();
    else if (Date.now() - quietSince >= 500) return;
    await page.waitForTimeout(50);
  }
}
