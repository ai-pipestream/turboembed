// Results as text a spreadsheet takes: CSV for a file (RFC 4180: a
// field with a comma, a quote or a line break is quoted, a quote is
// doubled, rows end CR LF) and tab-separated for the clipboard, which
// is what a sheet pastes into cells.

/** One CSV field. */
export function field(v: string): string {
  const quote = v.includes(",") || v.includes('"') || v.includes("\n") || v.includes("\r") || v.startsWith(" ") || v.endsWith(" ");
  return quote ? `"${v.replaceAll('"', '""')}"` : v;
}

/** Rows as a CSV file's text. */
export function csv(rows: string[][]): string {
  return rows.map((r) => r.map(field).join(",") + "\r\n").join("");
}

/** Rows as tab-separated lines; a tab or a line break inside a value becomes a space. */
export function tsv(rows: string[][]): string {
  const flat = (v: string): string => v.replaceAll("\t", " ").replaceAll("\r\n", " ").replaceAll("\n", " ").replaceAll("\r", " ");
  return rows.map((r) => r.map(flat).join("\t")).join("\n");
}

/** A file name from a result's name: letters and digits kept (any script), the rest a dash. */
export function fileName(name: string): string {
  let out = "";
  for (const ch of name.normalize("NFC")) {
    const keep = ch.toLowerCase() !== ch.toUpperCase() || (ch >= "0" && ch <= "9") || ch > "\u007f";
    out += keep ? ch : out.endsWith("-") || !out ? "" : "-";
  }
  while (out.endsWith("-")) out = out.slice(0, -1);
  return `${out || "result"}.csv`;
}
