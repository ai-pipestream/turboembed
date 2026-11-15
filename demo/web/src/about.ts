// The hint under a tab's description when the server serves nothing
// that tab can use, worked out from Capabilities (tabs/hints.ts shows it). A tab that needs no
// model (Vocabulary, Models, Server, and Analyze, whose default profile
// runs without one) never shows it.

const WHERE = "Load a bundle on the Server tab, or install a model on the Models tab.";

/** What `tab` needs that `tasks` (the tasks of the served bundles) does not give; "" when it has what it needs. */
export function needs(tab: string, tasks: string[]): string {
  const has = (t: string): boolean => tasks.includes(t);
  if (tab === "embed" && !has("TASK_EMBED")) return `Needs: an embedding model. The server serves none. ${WHERE}`;
  if (tab === "semantic" && !has("TASK_EMBED") && !has("TASK_RERANK") && !has("TASK_EXTRACTIVE_QA")) {
    return `Needs: an embedding model, a reranker or a QA model. The server serves none. ${WHERE}`;
  }
  if (tab === "search" && !has("TASK_EMBED")) return `Needs: an embedding model to create an index. The server serves none. ${WHERE}`;
  return "";
}

/** The tabs that can show the hint. */
export const HINTED = ["embed", "semantic", "search"];
