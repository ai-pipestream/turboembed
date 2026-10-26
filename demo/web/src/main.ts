// The demo page: one tab per area, each over the generated Connect-ES
// clients with binary protobuf. Leaving a tab cancels its streams and
// uploads.

import * as analyze from "./tabs/analyze.js";
import * as embed from "./tabs/embed.js";
import * as searchTab from "./tabs/search.js";
import * as server from "./tabs/server.js";
import * as vocabularyTab from "./tabs/vocabulary.js";

const stops: Record<string, () => void> = { analyze: analyze.stop, search: searchTab.stop, vocabulary: vocabularyTab.stop };

function show(name: string): void {
  for (const [tab, stop] of Object.entries(stops)) if (tab !== name) stop();
  for (const b of document.querySelectorAll<HTMLButtonElement>("nav.tabs button")) b.setAttribute("aria-selected", String(b.dataset.tab === name));
  for (const p of document.querySelectorAll<HTMLElement>("section.panel")) p.hidden = p.id !== `tab-${name}`;
  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
}

function start(): void {
  for (const b of document.querySelectorAll<HTMLButtonElement>("nav.tabs button")) b.addEventListener("click", () => show(b.dataset.tab ?? "embed"));
  server.init();
  embed.init();
  analyze.init();
  searchTab.init();
  vocabularyTab.init();
  const want = location.hash.slice(1);
  show(document.getElementById(`tab-${want}`) ? want : "embed");
}

start();
