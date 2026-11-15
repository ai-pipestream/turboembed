// Holds real answers back so a test can make one arrive late. A held
// call reaches the server at once and its own answer is handed to the
// page on release; nothing is stubbed.

import assert from "node:assert/strict";
import type { BrowserContext, Page, Route } from "playwright";

/** Holds the answers to the calls of `method` made between start() and release(). */
export class Hold {
  private on = false;
  private waiting: Array<() => void> = [];

  static async of(context: BrowserContext, method: string): Promise<Hold> {
    const h = new Hold();
    await context.route((u) => u.pathname.endsWith(method), (route) => h.handle(route));
    return h;
  }

  private async handle(route: Route): Promise<void> {
    if (!this.on) return route.continue();
    const response = await route.fetch();
    await new Promise<void>((r) => this.waiting.push(r));
    await route.fulfill({ response });
  }

  start(): void {
    this.on = true;
  }

  /** Waits until `n` answers are held, then holds no further call. */
  async held(page: Page, n: number): Promise<void> {
    for (let i = 0; this.waiting.length < n; i++) {
      assert.equal(i < 100, true, `${this.waiting.length} of ${n} calls were held in 10 s`);
      await page.waitForTimeout(100);
    }
    assert.equal(this.waiting.length, n);
    this.on = false;
  }

  /** Hands the held answers to the page and gives it time to draw them. */
  async release(page: Page): Promise<void> {
    this.waiting.splice(0).forEach((r) => r());
    await page.waitForTimeout(700);
  }
}
