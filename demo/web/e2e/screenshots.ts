// Screenshots of the demo page for docs/demo.md, taken in a real browser
// against a real turbo-kserve-demo in front of a real turbo-kserve
// (scripts/demo-screenshots.sh starts both with the bundles it is given).
// Each tab is used the way a visitor would use it first and shot once in
// the light scheme and once in the dark. This is not a test and the
// browser tests do not run it: its file name does not end in .test.ts.

import { chromium, type Browser, type Page } from "playwright";

const BASE = process.env.TURBO_E2E_URL ?? "http://127.0.0.1:8080";
const OUT = process.env.TURBO_SHOTS_DIR ?? "";
const MODEL = process.env.TURBO_SHOTS_MODEL ?? "";
const WIDTH = 1280;
const HEIGHT = 860;

const DOCUMENT = `Ada Lovelace wrote the first published program in 1843. She worked with Charles Babbage in London on the Analytical Engine, a mechanical computer that was never finished. Babbage had designed the machine to read instructions from punched cards, an idea borrowed from the looms of the textile trade.

Her notes on the engine are longer than the paper they accompany. They describe how it could compute Bernoulli numbers step by step, and they suggest that such a machine might one day compose music or draw graphics if its symbols stood for notes or shapes rather than numbers.

Historians still argue over how much of the program was hers and how much came from Babbage, but the notes show a clear grasp of loops, of variables that change as a calculation proceeds, and of the difference between a machine that calculates and one that follows a general plan.

The central bank in Frankfurt raised interest rates again this spring. Inflation had pushed borrowing costs higher across Europe, and lenders in Paris and Madrid followed within a week. Mortgage rates rose first, then the rates on business loans.

Markets fell on the news, then recovered as traders read the statement more closely and decided that further rises were unlikely this year. Bond yields settled a little higher than before, and the euro gained against the dollar for the first time in a month.

Economists expect the effect on prices to take a year or more to show. Firms that borrowed heavily when money was cheap will feel it soonest, while households with fixed rate loans may not notice until their terms come up for renewal.

A long walk along the coast near Lisbon ends at a lighthouse. The path climbs over low cliffs, past fishing boats pulled up on the sand and small cafes that open only in summer, and the light turns gold in the late afternoon.

Walkers start early to avoid the heat. Water and a hat matter more than good boots, since the trail is dry and mostly flat, and the last bus back to the city leaves before the sun goes down.

Sourdough bread needs only flour, water, salt and a starter, but it asks for patience. The dough rises slowly overnight in a cool kitchen, and the long fermentation gives the crust its colour and the crumb its sour taste.

Bakers judge the dough by touch rather than by the clock. A loaf that has risen enough springs back slowly when pressed, and one baked too soon tears open along the side instead of the cut across the top.`;

const PASSAGES = `Rust is a systems programming language focused on safety and speed.

Python is popular for data science and machine learning.

The Analytical Engine was a proposed mechanical general purpose computer.

Garbage collected languages trade some speed for freedom from manual memory management.

The borrow checker rejects programs that could use memory after it is freed.

Sourdough bread needs a starter, flour, water, salt and a long slow rise.`;

/** A shot: a file name, how to get the page there, and what to capture. */
interface Shot {
  name: string;
  tab: string;
  ready: (page: Page) => Promise<void>;
  /** An element to capture instead of the viewport. */
  element?: string;
  fullPage?: boolean;
}

const SHOTS: Shot[] = [
  {
    name: "embed",
    tab: "embed",
    fullPage: true,
    ready: async (page) => {
      await page.click("#embed-run");
      await page.waitForSelector("#embed-results .card table.sim");
    },
  },
  {
    name: "semantic",
    tab: "semantic",
    ready: async (page) => {
      await page.click("#sem-run");
      await page.waitForSelector("#sem-sim table");
    },
  },
  {
    name: "analyze",
    tab: "analyze",
    fullPage: true,
    ready: async (page) => {
      await page.fill("#analyze-text", DOCUMENT);
      await page.selectOption("#analyze-profile", "default-v1");
      await page.waitForSelector(`#analyze-model option[value="${MODEL}"]`, { state: "attached" });
      await page.selectOption("#analyze-model", MODEL);
      await page.click("#analyze-run");
      await page.waitForSelector("#analyze-window .doc");
    },
  },
  {
    name: "analyze-map",
    tab: "analyze",
    element: "#analyze-projection",
    ready: async (page) => {
      await page.waitForSelector("#analyze-projection[open]");
      await page.waitForSelector("#projection-out .map-and-text");
    },
  },
  {
    name: "analyze-wire",
    tab: "analyze",
    element: "#analyze-wire",
    ready: async (page) => {
      await page.click("#analyze-wire summary");
      await page.waitForSelector("#wire-out details, #wire-out table, #wire-out li");
      await page.click("#wire-roundtrip");
      await page.waitForSelector("#wire-note:not([hidden])");
    },
  },
  {
    name: "search",
    tab: "search",
    fullPage: true,
    ready: async (page) => {
      await page.waitForSelector(`#search-create-bundle option[value="${MODEL}"]`, { state: "attached" });
      await page.selectOption("#search-create-bundle", MODEL);
      await page.fill("#search-create-name", "notes");
      await page.fill("#search-create-collection", "notes");
      await page.click("#search-create button[type=submit]");
      await page.waitForSelector('#search-index option[value="notes"]', { state: "attached" });
      await page.selectOption("#search-index", "notes");
      await page.fill("#search-docs", PASSAGES);
      await page.click("#search-add");
      await page.waitForSelector("#search-add-note:not([hidden])");
      await page.waitForFunction(() => (document.getElementById("search-add-stop") as HTMLButtonElement).disabled);
      await page.click("#search-refresh");
      await page.click("#coll-refresh");
      await page.waitForSelector("#coll-list td");
      await page.click("#search-form button[type=submit]");
      await page.waitForSelector("#search-results .hit");
    },
  },
  {
    name: "vocabulary",
    tab: "vocabulary",
    ready: async (page) => {
      await page.fill("#vocab-docs", PASSAGES);
      await page.fill("#vocab-language", "en");
      await page.check("#vocab-stopwords");
      await page.click("#vocab-learn button[type=submit]");
      await page.waitForSelector("#vocab-note:not([hidden])");
      await page.waitForFunction(() => (document.getElementById("vocab-stop") as HTMLButtonElement).disabled);
      const note = await page.textContent("#vocab-note");
      await page.click("#vocab-refresh");
      await page.waitForSelector("#vocab-lists td", { timeout: 10000 }).catch(() => {
        throw new Error(`no vocabulary was listed after learning: ${note}`);
      });
    },
  },
  {
    name: "models",
    tab: "models",
    ready: async (page) => {
      await page.click("#models-catalog-load");
      await page.waitForSelector("#models-filter");
    },
  },
  {
    name: "server",
    tab: "server",
    fullPage: true,
    ready: async (page) => {
      await page.waitForSelector("#server-caps table");
    },
  },
];

async function run(browser: Browser, scheme: "light" | "dark"): Promise<void> {
  const context = await browser.newContext({ viewport: { width: WIDTH, height: HEIGHT }, colorScheme: scheme });
  const page = await context.newPage();
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(`${BASE}/#embed`);
  await page.waitForSelector("#server.status.up");
  await page.waitForSelector("#embed-models label.model");
  for (const shot of SHOTS) {
    await page.click(`nav.tabs button[data-tab=${shot.tab}]`);
    await shot.ready(page);
    // Let the last answer draw and any transition settle.
    await page.waitForTimeout(500);
    const path = `${OUT}/${shot.name}-${scheme}.png`;
    if (shot.element) {
      await page.locator(shot.element).screenshot({ path });
    } else {
      await page.evaluate(() => {
        for (const t of document.querySelectorAll("textarea")) {
          t.scrollTop = 0;
          t.scrollLeft = 0;
          t.setSelectionRange(0, 0);
        }
        (document.activeElement as HTMLElement | null)?.blur();
        window.scrollTo(0, 0);
      });
      await page.screenshot({ path, fullPage: shot.fullPage ?? false });
    }
    if (errors.length) throw new Error(`the page failed on ${shot.name}: ${errors.join("; ")}`);
  }
  await context.close();
}

if (!OUT || !MODEL) throw new Error("TURBO_SHOTS_DIR and TURBO_SHOTS_MODEL must be set");
const browser = await chromium.launch();
try {
  // Each scheme gets a fresh server state: the script that starts the
  // stack runs this once per scheme.
  await run(browser, (process.env.TURBO_SHOTS_SCHEME ?? "light") as "light" | "dark");
} finally {
  await browser.close();
}
