"""Convert FLORES+ devtest jsonl files to the bench's language gold form.

Usage: convert_gold.py DEVTEST_DIR OUT.tsv [--short-out SHORT.tsv]

With --short-out, a second gold file is written from the same rows
for the short-text half of the language gate: each text cut to its
first four whitespace-separated words, or to its first 20
characters when it has fewer words to cut (the scripts without
spaces). The labels do not change; only the evidence per text
shrinks. Both files' digests are printed.

DEVTEST_DIR holds the .jsonl files of the devtest split of
openlanguagedata/flores_plus at the revision pinned in the records
README, downloaded after accepting the dataset terms (CC BY-SA 4.0,
evaluation only; the files are never committed). Output is
`code<TAB>text` lines, exactly the --lang-gold form, sorted by gold
code then numeric sentence id, so the converted bytes are stable for
a given upstream revision. The converter prints the SHA-256 of the
output and of every input file it read; the record pins the output
digest through --sha256.

Gold codes are ISO 639-3 at macrolanguage grain where FLORES+ and
the compared systems split codes differently. FILES below maps each
gold code to the one devtest languoid that represents it: Mandarin
cmn_Hans is gold zho, Standard Arabic arb_Arab is gold arb, Western
Persian pes is fas, Standard Estonian ekk is est, Standard Latvian
lvs is lav, Halh Mongolian khk_Cyrl is mon, Standard Malay zsm is
msa, Swahili swh is swa, North Azerbaijani azj is aze and Tosk
Albanian als is sqi. Romanized, variant and second-script languoids
of the same gold code (arb_Latn, cmn_Hant, khk_Mong,
cat_Latn_vale1252, nob_Latn_radical, tso_Latn_MZ) are left out so
each gold code is one languoid of at most 1012 sentences.

The 73 gold codes are the languages both FLORES+ devtest and lingua
2.2.0 cover. FLORES+ has no Latin file, and its fil_Latn is Filipino,
not lingua's Tagalog tgl, so lat and tgl have no gold row. Tabs and
newlines inside a text are replaced with a space: the gold form is
one text per line.
"""

import hashlib
import json
import os
import sys

FILES = {
    "afr": "afr_Latn.jsonl",
    "arb": "arb_Arab.jsonl",
    "aze": "azj_Latn.jsonl",
    "bel": "bel_Cyrl.jsonl",
    "ben": "ben_Beng.jsonl",
    "bos": "bos_Latn.jsonl",
    "bul": "bul_Cyrl.jsonl",
    "cat": "cat_Latn.jsonl",
    "ces": "ces_Latn.jsonl",
    "cym": "cym_Latn.jsonl",
    "dan": "dan_Latn.jsonl",
    "deu": "deu_Latn.jsonl",
    "ell": "ell_Grek.jsonl",
    "eng": "eng_Latn.jsonl",
    "epo": "epo_Latn.jsonl",
    "est": "ekk_Latn.jsonl",
    "eus": "eus_Latn.jsonl",
    "fas": "pes_Arab.jsonl",
    "fin": "fin_Latn.jsonl",
    "fra": "fra_Latn.jsonl",
    "gle": "gle_Latn.jsonl",
    "guj": "guj_Gujr.jsonl",
    "heb": "heb_Hebr.jsonl",
    "hin": "hin_Deva.jsonl",
    "hrv": "hrv_Latn.jsonl",
    "hun": "hun_Latn.jsonl",
    "hye": "hye_Armn.jsonl",
    "ind": "ind_Latn.jsonl",
    "isl": "isl_Latn.jsonl",
    "ita": "ita_Latn.jsonl",
    "jpn": "jpn_Jpan.jsonl",
    "kat": "kat_Geor.jsonl",
    "kaz": "kaz_Cyrl.jsonl",
    "kor": "kor_Hang.jsonl",
    "lav": "lvs_Latn.jsonl",
    "lit": "lit_Latn.jsonl",
    "lug": "lug_Latn.jsonl",
    "mar": "mar_Deva.jsonl",
    "mkd": "mkd_Cyrl.jsonl",
    "mon": "khk_Cyrl.jsonl",
    "mri": "mri_Latn.jsonl",
    "msa": "zsm_Latn.jsonl",
    "nld": "nld_Latn.jsonl",
    "nno": "nno_Latn.jsonl",
    "nob": "nob_Latn.jsonl",
    "pan": "pan_Guru.jsonl",
    "pol": "pol_Latn.jsonl",
    "por": "por_Latn.jsonl",
    "ron": "ron_Latn.jsonl",
    "rus": "rus_Cyrl.jsonl",
    "slk": "slk_Latn.jsonl",
    "slv": "slv_Latn.jsonl",
    "sna": "sna_Latn.jsonl",
    "som": "som_Latn.jsonl",
    "sot": "sot_Latn.jsonl",
    "spa": "spa_Latn.jsonl",
    "sqi": "als_Latn.jsonl",
    "srp": "srp_Cyrl.jsonl",
    "swa": "swh_Latn.jsonl",
    "swe": "swe_Latn.jsonl",
    "tam": "tam_Taml.jsonl",
    "tel": "tel_Telu.jsonl",
    "tha": "tha_Thai.jsonl",
    "tsn": "tsn_Latn.jsonl",
    "tso": "tso_Latn.jsonl",
    "tur": "tur_Latn.jsonl",
    "ukr": "ukr_Cyrl.jsonl",
    "urd": "urd_Arab.jsonl",
    "vie": "vie_Latn.jsonl",
    "xho": "xho_Latn.jsonl",
    "yor": "yor_Latn.jsonl",
    "zho": "cmn_Hans.jsonl",
    "zul": "zul_Latn.jsonl",
}


def rows_of(path, gold):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "devtest":
                raise SystemExit(f"{path}: row split is {row.get('split')!r}, not devtest")
            try:
                ident = int(row["id"])
            except (KeyError, TypeError, ValueError):
                raise SystemExit(f"{path}: row without a numeric id: {line[:80]!r}")
            text = row.get("text", "")
            if not text.strip():
                raise SystemExit(f"{path}: empty text at id {ident}")
            text = text.replace("\t", " ").replace("\r", " ").replace("\n", " ")
            rows.append((ident, gold, text))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    rows.sort(key=lambda r: r[0])
    if len({r[0] for r in rows}) != len(rows):
        raise SystemExit(f"{path}: duplicate sentence ids")
    return rows


def short_of(text):
    # Split on runs of Unicode whitespace (str.split with no
    # separator): leading or repeated spaces make no empty words,
    # and a nonbreaking or ideographic space is a separator like
    # any other. With fewer than four words to cut (the scripts
    # without spaces), the fallback is the first 20 characters of
    # the stripped text, so the short form is never empty.
    words = text.split()
    if len(words) >= 4:
        return " ".join(words[:4])
    return text.strip()[:20]


def main():
    src_dir, out = sys.argv[1], sys.argv[2]
    short_out = None
    if sys.argv[3:] and sys.argv[3] == "--short-out" and len(sys.argv) == 5:
        short_out = sys.argv[4]
    elif len(sys.argv) != 3:
        raise SystemExit(__doc__)
    lines, short_lines = [], []
    for gold in sorted(FILES):
        name = FILES[gold]
        path = os.path.join(src_dir, name)
        if not os.path.isfile(path):
            raise SystemExit(f"missing devtest file {path}")
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        rows = rows_of(path, gold)
        lines.extend(f"{gold}\t{text}\n" for _, _, text in rows)
        short_lines.extend(f"{gold}\t{short_of(text)}\n" for _, _, text in rows)
        print(f"{name} sha256 {digest} rows {len(rows)}", file=sys.stderr)
    body = "".join(lines).encode("utf-8")
    with open(out, "wb") as f:
        f.write(body)
    print(f"{out} sha256 {hashlib.sha256(body).hexdigest()} texts {len(lines)}", file=sys.stderr)
    if short_out is not None:
        short_body = "".join(short_lines).encode("utf-8")
        with open(short_out, "wb") as f:
            f.write(short_body)
        print(
            f"{short_out} sha256 {hashlib.sha256(short_body).hexdigest()} texts {len(short_lines)} (short)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
