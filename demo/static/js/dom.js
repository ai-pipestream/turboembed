// Small DOM helpers: element and SVG builders, formatting and a status line.

const SVG = "http://www.w3.org/2000/svg";

function apply(el, attrs) {
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : String(v));
  }
}

function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

// h("div", {class: "x"}, child, "text", [more]) builds an element.
export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  apply(el, attrs);
  return append(el, children);
}

export function svg(tag, attrs, ...children) {
  const el = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined) continue;
    if (k === "text") el.textContent = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, String(v));
  }
  return append(el, children);
}

export function $(selector, root = document) {
  return root.querySelector(selector);
}

export function clear(el, ...children) {
  el.replaceChildren();
  return append(el, children);
}

export function fmt(n, digits = 3) {
  if (n === null || n === undefined || Number.isNaN(n)) return "";
  if (Number.isInteger(n)) return n.toLocaleString("en-US");
  return Number(n).toFixed(digits);
}

export function bytes(n) {
  if (!n) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i += 1;
  }
  return `${i ? n.toFixed(1) : n} ${units[i]}`;
}

// A status line under a form: busy, done or failed.
export function status(el, kind, text) {
  el.hidden = !text;
  el.className = `status ${kind || ""}`;
  el.textContent = text || "";
}

// Runs `fn` with `button` disabled and reports a thrown error on `line`.
export async function busy(button, line, fn) {
  if (button) button.disabled = true;
  if (line) status(line, "busy", "Working");
  try {
    const out = await fn();
    if (line && line.classList.contains("busy")) status(line, "", "");
    return out;
  } catch (e) {
    if (line) status(line, "error", e.message || String(e));
    return undefined;
  } finally {
    if (button) button.disabled = false;
  }
}

// A table from rows of plain values; `cols` is [[title, row => value]].
export function table(rows, cols, empty = "Nothing yet.") {
  if (!rows || !rows.length) return h("p", { class: "muted" }, empty);
  return h(
    "table",
    { class: "grid" },
    h("thead", {}, h("tr", {}, cols.map(([t]) => h("th", {}, t)))),
    h("tbody", {}, rows.map((r) => h("tr", {}, cols.map(([, f]) => h("td", {}, f(r)))))),
  );
}

export function download(name, bytesOrText, type = "application/octet-stream") {
  const blob = new Blob([bytesOrText], { type });
  const url = URL.createObjectURL(blob);
  const a = h("a", { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// A stable hue for a layer or label name.
export function hue(name) {
  let x = 0;
  for (let i = 0; i < name.length; i++) x = (x * 31 + name.charCodeAt(i)) >>> 0;
  return x % 360;
}
