import re,glob,statistics as st,collections,sys
d=sys.argv[1]; R=collections.defaultdict(list); L=[];G=[];bad=[]
for f in sorted(glob.glob(f'{d}/r*-*.log')):
    t=open(f).read(); r,sz=re.search(r'r(\d+)-(\w+)\.log',f).groups()
    m=re.search(r'library .*?p50 ([\d.]+) ms, p99 ([\d.]+)',t); x=re.search(r'mlx .*?p50 ([\d.]+) ms, p99 ([\d.]+)',t)
    c=re.search(r'min cosine ([\d.e-]+)',t); b=re.search(r'before: load ([\d.]+) ([\d.]+) ([\d.]+) gpu (\d+)',t)
    if 'exit: 0' not in t or not m or not x: bad.append(f); continue
    L.append(float(b.group(1)));G.append(int(b.group(4)))
    R[sz].append((float(m.group(1)),float(x.group(1)),float(m.group(2)),float(x.group(2)),float(c.group(1))))
print('bad',bad,'| 1-min load before',min(L),max(L),'| gpu before',min(G),max(G))
print('| Rows | turbo p50 median (least) | MLX p50 median (least) | ratio of medians | per-round ratios | min cosine |\n|---|---|---|---|---|---|')
for sz in ['32x256','32x128','16x128','8x128','4x128','2x128','1x128','1x32','1x512','8x512','32x512']:
    v=R[sz]
    if not v: print(f"| {sz.replace('x',' x ')} | refused: no reference case has 512 tokens | | | | |"); continue
    t=[a[0] for a in v]; m=[a[1] for a in v]
    print(f"| {sz.replace('x',' x ')} | {st.median(t):.2f} ({min(t):.2f}) | {st.median(m):.2f} ({min(m):.2f}) | {st.median(t)/st.median(m):.3f} | {' '.join(f'{a[0]/a[1]:.3f}' for a in v)} | {min(a[4] for a in v):.7f} |")
