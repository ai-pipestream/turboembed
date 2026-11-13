// What a visitor typed or picked on a tab, kept in the browser so a
// reload does not lose it, with a "Reset to sample" control per tab.
// Only the tabs that take text keep anything; a password, a file, a
// control marked data-unkept and the bearer token are never kept (the token stays in sessionStorage,
// rpc.ts). Every read and write is guarded: the page works the same
// with storage blocked, it just forgets.

const PREFIX = "turbo-demo.input.";

/** The tabs whose inputs are kept. Models and Server hold addresses and keys, and keep nothing. */
export const TABS = ["embed", "semantic", "analyze", "search", "vocabulary"];

type Field = HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement;

/** Whether a control of this tag and type is kept. */
export function storable(tag: string, type: string): boolean {
  const t = tag.toLowerCase();
  if (t === "textarea" || t === "select") return true;
  if (t !== "input") return false;
  return !["password", "file", "hidden", "submit", "button", "reset", "image"].includes(type.toLowerCase());
}

/** What is stored for a control: a checkbox as "1" or "0", anything else as its text. */
export function stored(type: string, value: string, checked: boolean): string {
  return type === "checkbox" || type === "radio" ? (checked ? "1" : "0") : value;
}

/** The kept value under `id`, or null. */
export function recall(id: string): string | null {
  try {
    return localStorage.getItem(PREFIX + id);
  } catch {
    return null;
  }
}

/** Keeps `value` under `id`. */
export function remember(id: string, value: string): void {
  if (resetting) return;
  try {
    localStorage.setItem(PREFIX + id, value);
  } catch {
    // Storage is blocked or full: the page goes on without it.
  }
}

function forget(id: string): void {
  try {
    localStorage.removeItem(PREFIX + id);
  } catch {
    // Nothing was kept.
  }
}

function fields(panel: Element): Field[] {
  return [...panel.querySelectorAll<Field>("input[id], textarea[id], select[id]")].filter((f) => storable(f.tagName, f.type) && !f.hasAttribute("data-unkept"));
}

function put(f: Field, v: string): void {
  if (f instanceof HTMLInputElement && (f.type === "checkbox" || f.type === "radio")) f.checked = v === "1";
  else if (f instanceof HTMLSelectElement) {
    if ([...f.options].some((o) => o.value === v)) f.value = v;
  } else f.value = v;
}

// The selects whose kept pick has been put back since the page loaded.
const restored = new WeakSet<HTMLSelectElement>();

/**
 * After a select's options are filled from the server: picks what was
 * kept, the first time it is offered. Only once per page load, so a
 * later refill of the list keeps what is picked now, whether the
 * visitor or the page (a new index, say) picked it.
 */
export function restore(select: HTMLSelectElement): void {
  if (restored.has(select) || !select.closest(TABS.map((t) => `#tab-${t}`).join(", "))) return;
  const v = kept(select.id, select.hasAttribute("data-unkept"), select.id ? recall(select.id) : null, [...select.options].map((o) => o.value));
  if (v === null) return;
  select.value = v;
  restored.add(select);
}

/** The kept pick `v` to put back in a select `id` offering `values`, or null: none for a select marked `unkept`, with no id, or not offering it. */
export function kept(id: string, unkept: boolean, v: string | null, values: string[]): string | null {
  return id !== "" && !unkept && offered(v, values) ? v : null;
}

/** Whether a kept pick `v` is among `values`, the ones a select offers now. */
export function offered(v: string | null, values: string[]): v is string {
  return v !== null && values.includes(v);
}

let resetting = false;
const resets = new Map<string, (() => void)[]>();

/** Runs `again` when `tab` is reset, for state a tab keeps under its own key. */
export function onReset(tab: string, again: () => void): void {
  resets.set(tab, [...(resets.get(tab) ?? []), again]);
}

function reset(tab: string, panel: Element): void {
  resetting = true;
  try {
    for (const f of fields(panel)) {
      forget(f.id);
      if (f instanceof HTMLInputElement && (f.type === "checkbox" || f.type === "radio")) f.checked = f.defaultChecked;
      else if (f instanceof HTMLSelectElement) f.selectedIndex = Math.max(0, [...f.options].findIndex((o) => o.defaultSelected));
      else f.value = f.defaultValue;
      // The tab's own listeners (options that show for a kind of model) follow the change.
      f.dispatchEvent(new Event("change", { bubbles: true }));
    }
    for (const again of resets.get(tab) ?? []) again();
  } finally {
    resetting = false;
  }
}

/** Puts back what was kept, keeps what is typed from here on, and adds each tab's reset control. */
export function init(): void {
  for (const tab of TABS) {
    const panel = document.getElementById(`tab-${tab}`);
    if (!panel) continue;
    for (const f of fields(panel)) {
      const v = recall(f.id);
      if (v !== null) put(f, v);
    }
    const keep = (ev: Event): void => {
      const f = ev.target as Field;
      if (f.id && storable(f.tagName ?? "", f.type ?? "") && !f.hasAttribute("data-unkept")) remember(f.id, stored(f.type, f.value, (f as HTMLInputElement).checked === true));
    };
    panel.addEventListener("input", keep);
    panel.addEventListener("change", keep);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "reset";
    button.id = `${tab}-reset`;
    button.textContent = "Reset to sample";
    button.title = "Put this tab's inputs back to the sample and forget what was typed";
    button.addEventListener("click", () => reset(tab, panel));
    panel.querySelector(".box")?.append(button);
  }
}

/** The kept list under `id` (one name per line), or null when none was kept. */
export function recallList(id: string): string[] | null {
  const v = recall(id);
  return v === null ? null : v.split("\n").filter((x) => x.length);
}

/** Forgets the value a tab keeps under its own key. */
export function forgetOwn(id: string): void {
  forget(id);
}
