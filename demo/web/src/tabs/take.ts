// "Copy" and "Download CSV" for a result, built from the values the
// page already has. The rows are made when a control is used, so a
// result that is never taken costs nothing.

import { csv, fileName, tsv } from "../csv.js";
import { download, el } from "../dom.js";

async function copy(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // No clipboard here (a page not served over TLS or localhost): the old way.
    const area = el("textarea", { class: "offscreen", "aria-hidden": "true" }) as HTMLTextAreaElement;
    area.value = text;
    document.body.append(area);
    area.select();
    const ok = document.execCommand("copy");
    area.remove();
    return ok;
  }
}

/**
 * The two controls for a result called `name` (it names the file).
 * `rows` gives the header row and then the values.
 */
export function take(name: string, rows: () => string[][]): HTMLElement {
  const copyButton = el("button", { type: "button", class: "small", text: "Copy", title: "Copy as tab-separated rows, which a spreadsheet pastes into cells" });
  const said = el("span", { class: "muted", role: "status" });
  copyButton.addEventListener("click", async () => {
    const all = rows();
    said.textContent = (await copy(tsv(all))) ? `Copied ${all.length - 1} rows.` : "The browser did not allow the copy.";
    setTimeout(() => (said.textContent = ""), 4000);
  });
  const save = el("button", { type: "button", class: "small", text: "Download CSV" });
  save.addEventListener("click", () => download(fileName(name), [csv(rows())], "text/csv;charset=utf-8"));
  return el("div", { class: "take", "data-take": name }, copyButton, save, said);
}
