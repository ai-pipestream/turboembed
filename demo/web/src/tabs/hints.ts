// Shows each tab's "needs" hint (about.ts) as Capabilities answers.

import { HINTED, needs } from "../about.js";
import { $ } from "../dom.js";
import { onCapabilities, onReachable } from "../state.js";

export function init(): void {
  const show = (tasks: string[] | null): void => {
    for (const tab of HINTED) {
      const line = $(`needs-${tab}`);
      const text = tasks ? needs(tab, tasks) : "";
      line.textContent = text;
      line.hidden = text === "";
    }
  };
  onCapabilities((c) => show(c.bundles.map((b) => b.task)));
  // With no server the notice under the tabs says so; this hint is about what a server serves.
  onReachable((up) => {
    if (!up) show(null);
  });
}
