"""Produce a block-tree oracle from unmodified CommonMark 0.31.2 examples.
Usage: python3 commonmark.py SPEC_JSON CMARK OUTPUT_TSV
cmark 0.31.1 must render every example's expected HTML before it is used.
No Markdown parser or reference implementation is a runtime dependency.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

spec, cmark, output = sys.argv[1:]
data = Path(spec).read_bytes()
assert hashlib.sha256(data).hexdigest() == 'd431b29d97b6f73e69d547109cf5081578fac931e72afe95639ebe766c1b2a20'
assert subprocess.check_output([cmark, '--version'], text=True).split()[:2] == ['cmark', '0.31.1']
print('spec sha256:', hashlib.sha256(data).hexdigest())
rows = []
for example in json.loads(data):
    text = example['markdown']
    html = subprocess.check_output([cmark, '--unsafe'], input=text.encode()).decode()
    assert html == example['html'], (example['example'], html, example['html'])
    xml = subprocess.check_output([cmark, '--to', 'xml'], input=text.encode())
    root = ET.fromstring(xml)
    nodes = []
    def visit(node, parent=-1):
        name = node.tag.split('}')[-1]
        kind = {'paragraph': 'Paragraph', 'heading': 'Heading', 'code_block': 'Code',
                'block_quote': 'Quote', 'item': 'ListItem', 'html_block': 'Html'}.get(name)
        if name == 'list':
            kind = 'List' if node.get('tight') == 'true' else 'LooseList'
        if kind:
            i = len(nodes)
            nodes.append(f"{kind}:{parent}:{node.get('level', '0')}")
            parent = i
        for child in node:
            visit(child, parent)
    visit(root)
    rows.append(f"{example['example']}\t{example['section']}\t{text.encode().hex()}\t{','.join(nodes)}")
Path(output).write_text('\n'.join(rows) + '\n')
print('verified HTML and wrote block trees:', len(rows))
