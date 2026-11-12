// Which of the server's write methods the demo forwards, as the demo
// says in DemoService/Describe. A form whose write would be refused is
// turned off with a line naming the --allow that enables it. Until the
// demo has answered, nothing is turned off.

/** A method as the generated services describe it. */
export interface Named {
  name: string;
  parent: { typeName: string };
}

let writes: Set<string> | null = null;
let user = true;

// The service whose writes act for the demo's user (--user).
const FOR_A_USER = "turbo.training.v1.TurboCatalogService";
const locked: [HTMLElement, Named][] = [];

/** `/<package>.<Service>/<Method>`. */
export function path(m: Named): string {
  return `/${m.parent.typeName}/${m.name}`;
}

/**
 * Takes the demo's answer (the writes it forwards and whether it has a
 * user), or null when it gave none, and applies it to the forms locked
 * so far.
 */
export function set(forwarded: string[] | null, hasUser = true): void {
  writes = forwarded ? new Set(forwarded) : null;
  user = forwarded === null || hasUser;
  for (const [box, m] of locked) apply(box, m);
}

/** Why the write `m` cannot be made from this page, or "" when it can. */
export function off(m: Named): string {
  if (!writes) return "";
  if (!writes.has(path(m))) return `Off: the demo does not forward this write. Start it with --allow ${path(m).slice(1)}`;
  if (!user && m.parent.typeName === FOR_A_USER) return "Off: this acts for a user. Start the demo with --user NAME";
  return "";
}
function apply(box: HTMLElement, m: Named): void {
  const why = off(m);
  box.inert = why !== "";
  box.classList.toggle("off", why !== "");
  const id = `${box.id}-off`;
  const old = document.getElementById(id);
  if (!why) return void old?.remove();
  const line = old ?? document.createElement("p");
  line.id = id;
  line.className = "muted off-why";
  line.textContent = why;
  if (!old) box.after(line);
}

/** Turns `box` (a form, or the controls of one write) off when the demo does not forward `m`, now and after each `set`. */
export function lock(box: HTMLElement, m: Named): void {
  locked.push([box, m]);
  apply(box, m);
}
