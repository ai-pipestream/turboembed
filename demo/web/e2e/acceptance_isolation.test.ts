// Per-user licence acceptance isolation, anti-spoofing, and
// authenticated write controls in a real browser against a real server
// and per-user demos (stack.ts). Nothing is stubbed.

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { test } from "node:test";
import { create, fromBinary, toBinary } from "@bufbuild/protobuf";
import { chromium, type Page } from "playwright";
import {
  ListLicenceAcceptancesRequestSchema,
  ListLicenceAcceptancesResponseSchema,
  RecordLicenceAcceptanceRequestSchema,
  RecordLicenceAcceptanceResponseSchema,
} from "../src/gen/turbo_training_pb.js";
import { Stack } from "./stack.js";

const INDEX = process.env.TURBO_E2E_INSTALL_INDEX ?? "";
const RECORD = "/turbo.training.v1.TurboCatalogService/RecordLicenceAcceptance";
const LIST = "/turbo.training.v1.TurboCatalogService/ListLicenceAcceptances";
const INSTALL = "/turbo.training.v1.TurboCatalogService/InstallModel";
const LOAD = "/turbo.result.v1.AnalysisService/LoadBundle";
const ID = "tiny-bert-sa";
const TERMS = "https://creativecommons.org/licenses/by-sa/4.0/legalcode";
const TOKEN = "isolation-test-token";

/** Returns the catalog index and sibling file:// store built by scripts/e2e-web.sh. */
function shareAlikeCatalog(): { index: string; store: string } {
  assert.notEqual(INDEX, "", "TURBO_E2E_INSTALL_INDEX names the share-alike catalog index (scripts/e2e-web.sh)");
  return { index: INDEX, store: `file://${INDEX.replace(/\/[^/]+$/, "")}/store` };
}

/** Sends a binary protobuf POST from inside `page` and returns status and raw bytes. */
async function postProto(page: Page, path: string, payload: Uint8Array): Promise<{ status: number; bytes: Uint8Array }> {
  const res = await page.evaluate(
    async ([p, body]) => {
      const r = await fetch(p, {
        method: "POST",
        headers: { "Content-Type": "application/proto", "Connect-Protocol-Version": "1" },
        body: new Uint8Array(body),
      });
      return { status: r.status, bytes: Array.from(new Uint8Array(await r.arrayBuffer())) };
    },
    [path, Array.from(payload)] as const,
  );
  return { status: res.status, bytes: new Uint8Array(res.bytes) };
}

test("per-user acceptance isolation and anti-spoofing in the browser", async () => {
  const cat = shareAlikeCatalog();
  const stack = await Stack.start({
    admin: true,
    catalogIndex: cat.index,
    catalogStore: cat.store,
    user: "alice",
    allow: [RECORD, INSTALL, LOAD],
  });
  const browser = await chromium.launch();
  try {
    const bobUrl = await stack.addDemo({ user: "bob", allow: [RECORD, INSTALL, LOAD] });
    const errors: string[] = [];

    // Alice accepts the terms and installs the share-alike bundle.
    const aliceCtx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    const alice = await aliceCtx.newPage();
    alice.on("pageerror", (e) => errors.push(e.message));
    await alice.goto(`${stack.url}/#models`);
    await alice.click("#models-catalog-load");
    const aliceCard = alice.locator(".entry", { has: alice.locator(".id", { hasText: new RegExp(`^${ID}$`) }) });
    await aliceCard.waitFor();
    assert.equal(await aliceCard.getByRole("button", { name: "Install" }).isDisabled(), true);

    await aliceCard.locator("summary", { hasText: "CC-BY-SA-4.0 terms to accept" }).click();
    const terms = aliceCard.locator("details", { has: alice.locator("summary", { hasText: "terms to accept" }) });
    await terms.locator("input[type=checkbox]").check();
    await terms.getByRole("button", { name: "Accept CC-BY-SA-4.0" }).click();
    await alice.waitForSelector("#models-note:has-text('alice accepted CC-BY-SA-4.0')");

    await alice.locator(".entry", { has: alice.locator(".id", { hasText: new RegExp(`^${ID}$`) }) }).getByRole("button", { name: "Install" }).click();
    await alice.waitForFunction(() => /^Installed /.test(document.querySelector("#models-install p")?.textContent ?? ""), null, { timeout: 60_000 });
    const done = (await alice.locator("#models-install p").textContent()) ?? "";
    const match = done.match(new RegExp(`^Installed ${ID} 1: \\d+ files, .* in (.+)\\.$`));
    if (!match) throw new Error(`unexpected install status: ${done}`);
    const installedDir = match[1];

    // Bob on the same server still sees missing terms, disabled Install, and cannot load the bundle.
    const bobCtx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    const bob = await bobCtx.newPage();
    bob.on("pageerror", (e) => errors.push(e.message));
    await bob.goto(`${bobUrl}/#models`);
    await bob.click("#models-catalog-load");
    const bobCard = bob.locator(".entry", { has: bob.locator(".id", { hasText: new RegExp(`^${ID}$`) }) });
    await bobCard.waitFor();
    const bobInstall = bobCard.getByRole("button", { name: "Install" });
    assert.equal(await bobInstall.isDisabled(), true);
    assert.equal(await bobInstall.getAttribute("title"), "Accept the licence terms first");
    assert.match(await bob.locator("#models-catalog").innerText(), /bob has accepted no terms\./);

    await bob.click("#tabbtn-server");
    await bob.fill("#server-load-name", "bob-sa");
    await bob.fill("#server-load-path", installedDir);
    await bob.click("#server-load button[type=submit]");
    await bob.waitForSelector("#server-load-note.error");
    assert.match((await bob.textContent("#server-load-note")) ?? "", /CC-BY-SA-4\.0/);

    // Alice's browser attempts to spoof Bob in RecordLicenceAcceptance and ListLicenceAcceptances:
    // the demo proxy replaces field 1 with Alice's identity.
    const termsDigest = createHash("sha256").update(new TextEncoder().encode(TERMS)).digest("hex");
    const spoofRecord = toBinary(
      RecordLicenceAcceptanceRequestSchema,
      create(RecordLicenceAcceptanceRequestSchema, {
        user: "bob",
        licence: "CC-BY-SA-4.0",
        terms: "",
        artifactId: ID,
        textSha256: termsDigest,
      }),
    );
    const recRes = await postProto(alice, RECORD, spoofRecord);
    assert.equal(recRes.status, 200);
    const recMsg = fromBinary(RecordLicenceAcceptanceResponseSchema, recRes.bytes);
    assert.equal(recMsg.user, "alice", "proxy overwrites spoofed user on write");

    const spoofList = toBinary(
      ListLicenceAcceptancesRequestSchema,
      create(ListLicenceAcceptancesRequestSchema, { user: "bob" }),
    );
    const listRes = await postProto(alice, LIST, spoofList);
    assert.equal(listRes.status, 200);
    const listMsg = fromBinary(ListLicenceAcceptancesResponseSchema, listRes.bytes);
    assert.equal(listMsg.user, "alice", "proxy overwrites spoofed user on read");

    // Reloading Bob's page confirms Bob still has no acceptances.
    await bob.goto(`${bobUrl}/#models`);
    await bob.click("#models-catalog-load");
    await bob.waitForSelector("#models-catalog");
    assert.match(await bob.locator("#models-catalog").innerText(), /bob has accepted no terms\./);
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
    await stack.close();
  }
});

test("authenticated write controls enforce --user, bearer token, and --allow in the browser", async () => {
  const cat = shareAlikeCatalog();
  const stack = await Stack.start({
    admin: true,
    catalogIndex: cat.index,
    catalogStore: cat.store,
    allow: [RECORD, INSTALL, LOAD],
  });
  const browser = await chromium.launch();
  try {
    const errors: string[] = [];

    // 1. Anonymous demo (no --user): Models tab is read-only and direct writes are refused 403.
    const anonCtx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    const anon = await anonCtx.newPage();
    anon.on("pageerror", (e) => errors.push(e.message));
    await anon.goto(`${stack.url}/#models`);
    await anon.click("#models-catalog-load");
    const anonCard = anon.locator(".entry", { has: anon.locator(".id", { hasText: new RegExp(`^${ID}$`) }) });
    await anonCard.waitFor();
    assert.equal(await anonCard.getByRole("button", { name: "Install" }).count(), 0);
    assert.equal(await anonCard.locator("summary", { hasText: "terms to accept" }).count(), 0);
    assert.equal(await anonCard.getByRole("button", { name: "Accept CC-BY-SA-4.0" }).count(), 0);
    assert.match(await anon.locator("#models-catalog").innerText(), /read-only: the demo was started without --user/);

    const spoof = toBinary(
      RecordLicenceAcceptanceRequestSchema,
      create(RecordLicenceAcceptanceRequestSchema, {
        user: "alice",
        licence: "CC-BY-SA-4.0",
        terms: "",
        artifactId: ID,
        textSha256: createHash("sha256").update(new TextEncoder().encode(TERMS)).digest("hex"),
      }),
    );
    const anonWrite = await postProto(anon, RECORD, spoof);
    assert.equal(anonWrite.status, 403);
    const anonText = new TextDecoder().decode(anonWrite.bytes);
    assert.match(anonText, /permission_denied/);
    assert.match(anonText, /--user/);

    // 2. Token-protected demo with --user carol and --allow RecordLicenceAcceptance only.
    const carolUrl = await stack.addDemo({ user: "carol", token: TOKEN, allow: [RECORD] });
    const carolCtx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    const carol = await carolCtx.newPage();
    carol.on("pageerror", (e) => errors.push(e.message));
    await carol.goto(`${carolUrl}/#models`);
    await carol.waitForSelector("#server:has-text('token')");

    // Without the bearer token, direct writes fail with 401 unauthenticated.
    const noToken = await postProto(carol, RECORD, spoof);
    assert.equal(noToken.status, 401);
    assert.match(new TextDecoder().decode(noToken.bytes), /unauthenticated/);

    // Typing the token unlocks the page; RecordLicenceAcceptance works, while InstallModel and LoadBundle stay off.
    await carol.fill("#token", TOKEN);
    await carol.locator("#token").dispatchEvent("change");
    await carol.waitForSelector("#server.status.up");
    await carol.click("#models-catalog-load");

    const carolCard = carol.locator(".entry", { has: carol.locator(".id", { hasText: new RegExp(`^${ID}$`) }) });
    await carolCard.waitFor();
    await carolCard.locator("summary", { hasText: "CC-BY-SA-4.0 terms to accept" }).click();
    const terms = carolCard.locator("details", { has: carol.locator("summary", { hasText: "terms to accept" }) });
    await terms.locator("input[type=checkbox]").check();
    await terms.getByRole("button", { name: "Accept CC-BY-SA-4.0" }).click();
    await carol.waitForSelector("#models-note:has-text('carol accepted CC-BY-SA-4.0')");

    const carolInstall = carol.locator(".entry", { has: carol.locator(".id", { hasText: new RegExp(`^${ID}$`) }) }).getByRole("button", { name: "Install" });
    assert.equal(await carolInstall.isDisabled(), true);
    assert.equal(
      await carolInstall.getAttribute("title"),
      "Off: the demo does not forward this write. Start it with --allow turbo.training.v1.TurboCatalogService/InstallModel",
    );

    await carol.click("#tabbtn-server");
    await carol.waitForFunction(() => (document.querySelector("#server-load") as HTMLElement).inert);
    assert.equal(
      await carol.textContent("#server-load-off"),
      "Off: the demo does not forward this write. Start it with --allow turbo.result.v1.AnalysisService/LoadBundle",
    );
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
    await stack.close();
  }
});
