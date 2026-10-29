"""Convert a MultiCoNER v2 .conll split to the bench's entity gold form.

Usage: convert_gold.py IN.conll OUT.tsv

MultiCoNER v2 labels 33 fine-grained kinds; the benchmark compares
peers at the dataset's six coarse groups (PER, LOC, GRP, PROD, CW,
MED), because the peer label sets are coarse OntoNotes types. The
fine-to-coarse map below is the MultiCoNER 2 taxonomy grouping;
every label in the input must be covered or conversion refuses.

Output: token<TAB>BIO-coarse lines, blank line between sentences,
exactly the --ner-gold form. The record pins both the source .conll
SHA-256 and this converted file's SHA-256.
"""

import sys

COARSE = {
    # PER
    "Artist": "PER", "Athlete": "PER", "Cleric": "PER", "OtherPER": "PER",
    "Politician": "PER", "Scientist": "PER", "SportsManager": "PER",
    # LOC
    "Facility": "LOC", "HumanSettlement": "LOC", "OtherLOC": "LOC",
    "Station": "LOC",
    # GRP
    "AerospaceManufacturer": "GRP", "CarManufacturer": "GRP",
    "MusicalGRP": "GRP", "ORG": "GRP", "PrivateCorp": "GRP",
    "PublicCorp": "GRP", "SportsGRP": "GRP",
    # PROD
    "Clothing": "PROD", "Drink": "PROD", "Food": "PROD",
    "OtherPROD": "PROD", "Software": "PROD", "Vehicle": "PROD",
    # CW
    "ArtWork": "CW", "MusicalWork": "CW", "VisualWork": "CW",
    "WrittenWork": "CW",
    # MED
    "AnatomicalStructure": "MED", "Disease": "MED",
    "MedicalProcedure": "MED", "Medication/Vaccine": "MED",
    "Symptom": "MED",
}


def main():
    src, out = sys.argv[1], sys.argv[2]
    sentences, current = [], []
    for line in open(src, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip() or line.startswith("#"):
            if current:
                sentences.append(current)
                current = []
            continue
        fields = line.split()
        if len(fields) != 4:
            raise SystemExit(f"bad conll line: {line!r}")
        token, label = fields[0], fields[3]
        if label == "O":
            current.append((token, "O"))
            continue
        prefix, kind = label.split("-", 1)
        if kind not in COARSE:
            raise SystemExit(f"unmapped MultiCoNER kind: {kind}")
        current.append((token, f"{prefix}-{COARSE[kind]}"))
    if current:
        sentences.append(current)
    with open(out, "w", encoding="utf-8") as f:
        for i, sentence in enumerate(sentences):
            if i:
                f.write("\n")
            for token, label in sentence:
                f.write(f"{token}\t{label}\n")
    print(f"{len(sentences)} sentences", file=sys.stderr)


if __name__ == "__main__":
    main()
