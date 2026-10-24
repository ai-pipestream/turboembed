"""Regenerate static inventories from the pinned sources documented in README.md.

Usage: python3 generate.py PRAGMATIC_ROOT SLOVAK_PY
Run from the repository root. Source files are parsed as data, never executed.
"""
import ast
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

# Project classifications refine the source lists.
mapping = {'am': 'amharic',
 'ar': 'arabic',
 'bg': 'bulgarian',
 'da': 'danish',
 'de': 'deutsch',
 'el': 'greek',
 'es': 'spanish',
 'fa': 'persian',
 'fr': 'french',
 'hi': 'hindi',
 'hy': 'armenian',
 'it': 'italian',
 'kk': 'kazakh',
 'my': 'burmese',
 'nl': 'dutch',
 'pl': 'polish',
 'ru': 'russian',
 'ur': 'urdu'}

titles = {'da': 'hr fr dr prof',
 'de': 'dr prof hr hrn fr frl ing mag dipl.-ing',
 'es': 'sr sra sres srta dr dra prof lic ing',
 'fr': 'm mm mme mlle dr pr prof',
 'it': 'sig sigg sig.ra sig.na dott dott.ssa prof ing avv',
 'nl': 'dhr mevr mw dr drs mr prof ir ds',
 'pl': 'dr prof mgr inż',
 'pt': 'sr sra srs sras dr dra prof eng',
 'ru': 'г-н г-жа тов',
 'sk': 'ing mgr bc doc prof mudr judr rndr phdr drsc'}

connectors = {'de': 'z.b d.h u.a',
 'es': 'p.ej',
 'fr': 'c.-à-d i.e e.g',
 'it': 'p.es',
 'nl': 'bijv b.v d.w.z m.a.w t.a.v',
 'pl': 'np tj tzn',
 'pt': 'p.ex',
 'sk': 'napr t.j tzn tzv'}

numerics = {'de': 'nr abb s abs art',
 'es': 'núm pág págs art',
 'fr': 'fig p pp art vol',
 'it': 'art pag pagg vol',
 'nl': 'nr nrs blz art',
 'pl': 'str nr rys',
 'pt': 'n.º núm pág art',
 'ru': 'стр рис',
 'sk': 'č str obr ods'}

legal = {'a. s',
 'a.s',
 'b.v',
 'co',
 'corp',
 'gmbh',
 'inc',
 'ltd',
 'n.v',
 's. r. o',
 's.a',
 's.p.a',
 's.r.l',
 's.r.o'}

def arrays(path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        for key in ("ABBREVIATIONS", "PREPOSITIVE_ABBREVIATIONS", "NUMBER_ABBREVIATIONS"):
            if line.startswith(key + " ="):
                # Both source languages use literal lists on one line.
                expr = line[line.index("["):line.rindex("]") + 1]
                result[key] = ast.literal_eval(expr)
    return result


for lang in list(mapping) + ["pt", "sk"]:
    rows = {}
    xml = Path(f"analysis/sentdetect/data/rules/cldr/{lang}.xml")
    if xml.exists():
        for entry in ET.parse(xml).iter("suppression"):
            if entry.text.endswith("."):
                rows[entry.text] = ("General", "Trailing")
    if lang == "pt":
        source = {}
    elif lang == "sk":
        source = arrays(Path(sys.argv[2]))
    else:
        source = arrays(Path(sys.argv[1]) / f"lib/pragmatic_segmenter/languages/{mapping[lang]}.rb")
    for key, category, rule in (
        ("ABBREVIATIONS", "General", "Trailing"),
        ("PREPOSITIVE_ABBREVIATIONS", "General", "Prepositive"),
        ("NUMBER_ABBREVIATIONS", "Academic", "Numeric"),
    ):
        for raw in source.get(key, []):
            word = " ".join(raw.strip().split()).rstrip(".") + "."
            if len(word.encode("utf-8")) <= 64 and len(word) > 1:
                rows[word] = (category, rule)
    for word in rows:
        root = word[:-1].lower()
        if root in legal:
            rows[word] = ("LegalTechnical", "Trailing")
        if root in numerics.get(lang, "").split():
            rows[word] = ("Academic", "Numeric")
        if root in titles.get(lang, "").split():
            rows[word] = ("Titles", "Prepositive")
        if root in connectors.get(lang, "").split():
            rows[word] = ("General", "Prepositive")
    if rows:
        data = "".join(f"{category}\t{rule}\t{word}\n" for word, (category, rule) in sorted(rows.items()))
        Path(f"analysis/sentdetect/data/inventory/{lang}.tsv").write_text(data, encoding="utf-8")
        print(lang, len(rows))
