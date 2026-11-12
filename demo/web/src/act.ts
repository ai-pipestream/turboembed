// One shape for every action on the page: a run button is off and counts
// the time while its call is in flight, a Stop button is on only then,
// and a delete asks in place before it calls the server.

import { el, elapsed } from "./dom.js";
import { describe } from "./rpc.js";

const running = new WeakSet<HTMLButtonElement>();
// How many jobs each Stop button stands for: one button may serve
// several actions, and it stays on until the last of them ends.
const stoppable = new WeakMap<HTMLButtonElement, number>();

function hold(stop: HTMLButtonElement, by: number): void {
  const n = Math.max(0, (stoppable.get(stop) ?? 0) + by);
  stoppable.set(stop, n);
  stop.disabled = n === 0;
}

/**
 * Runs `job` with `button` off and showing the elapsed time, and `stop`
 * (when given) on. Both go back when the job ends, however it ends,
 * except that a `stop` shared with another job still running stays on.
 * A second call for the same button while the first runs does nothing.
 */
export async function busy(button: HTMLButtonElement, job: () => Promise<void>, stop?: HTMLButtonElement): Promise<void> {
  if (running.has(button)) return;
  running.add(button);
  const t0 = performance.now();
  const clock = el("span", { class: "elapsed", text: ` ${elapsed(0)}` });
  button.disabled = true;
  button.setAttribute("aria-busy", "true");
  button.append(clock);
  if (stop) hold(stop, 1);
  const tick = setInterval(() => (clock.textContent = ` ${elapsed(performance.now() - t0)}`), 100);
  try {
    await job();
  } finally {
    clearInterval(tick);
    clock.remove();
    button.disabled = false;
    button.removeAttribute("aria-busy");
    running.delete(button);
    if (stop) hold(stop, -1);
  }
}

/** The button that sent a form, or the form's first submit button. */
export function submitter(ev: Event): HTMLButtonElement {
  const by = (ev as SubmitEvent).submitter;
  if (by instanceof HTMLButtonElement) return by;
  return (ev.target as HTMLFormElement).querySelector("button[type=submit]") as HTMLButtonElement;
}

/**
 * A control for something that cannot be undone. The first click turns it
 * into `question` with Yes and Cancel; only Yes runs `act`. A failure
 * shows beside the control, which goes back to its first state. With an
 * `off` reason the control is off and says why when pointed at.
 */
export function confirmed(label: string, question: string, act: () => Promise<void>, off = ""): HTMLElement {
  const box = el("span", { class: "confirm" });
  const first = (why = ""): HTMLButtonElement => {
    const b = el("button", { type: "button", text: label, title: off || undefined });
    b.disabled = off !== "";
    b.addEventListener("click", ask);
    box.replaceChildren(b, why ? el("span", { class: "error", role: "alert", text: ` ${why}` }) : "");
    return b;
  };
  const ask = (): void => {
    const yes = el("button", { type: "button", class: "danger", text: "Yes" });
    const no = el("button", { type: "button", text: "Cancel" });
    no.addEventListener("click", () => first().focus());
    yes.addEventListener("click", () => {
      no.disabled = true;
      void busy(yes, async () => {
        try {
          await act();
          if (box.isConnected) first();
        } catch (e) {
          first(describe(e)).focus();
        }
      });
    });
    box.replaceChildren(el("span", { class: "ask", text: question }), " ", yes, " ", no);
    no.focus();
  };
  box.addEventListener("keydown", (ev) => {
    // Not while Yes is running: the question would come back over a
    // call still in flight, and a second Yes would send it again.
    const no = box.querySelector<HTMLButtonElement>(".ask ~ button:not(.danger)");
    if (ev.key === "Escape" && no && !no.disabled) first().focus();
  });
  first();
  return box;
}
