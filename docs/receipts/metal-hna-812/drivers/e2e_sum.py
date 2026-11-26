import re,glob,statistics as st,collections,sys
d=sys.argv[1]; R=collections.defaultdict(lambda: collections.defaultdict(list)); L=[]; G=[]; bad=[]
for f in sorted(glob.glob(f'{d}/r*-*-*.log')):
    t=open(f).read(); r,sz,side=re.search(r'r(\d+)-(\w+)-(base|head)\.log',f).groups()
    m=re.search(r'library .*?p50 ([\d.]+) ms',t); x=re.search(r'mlx .*?p50 ([\d.]+) ms',t); c=re.search(r'min cosine ([\d.e-]+)',t)
    b=re.search(r'before: load ([\d.]+) ([\d.]+) ([\d.]+) gpu (\d+)',t)
    if 'exit: 0' not in t or not m or not x: bad.append(f); continue
    L.append(float(b.group(1))); G.append(int(b.group(4)))
    R[sz][side].append((int(r),float(m.group(1)),float(x.group(1)),float(c.group(1))))
print('bad',bad,'| 1-min load before',min(L),max(L),'| gpu before',min(G),max(G))
print('| Rows | base p50 median (least) | head p50 median (least) | head / base | MLX f16 p50 median (least) | base / MLX | head / MLX | head < base rounds | min cosine head |')
print('|---|---|---|---|---|---|---|---|---|')
for sz in ['32x256','32x128','16x128','8x128','4x128','2x128','1x128','1x32']:
    B=sorted(R[sz]['base']); H=sorted(R[sz]['head'])
    if not B or not H: continue
    bt=[a[1] for a in B]; ht=[a[1] for a in H]; mx=[a[2] for a in B+H]
    wins=sum(1 for a,b in zip(B,H) if b[1]<a[1])
    print(f"| {sz.replace('x',' x ')} | {st.median(bt):.2f} ({min(bt):.2f}) | {st.median(ht):.2f} ({min(ht):.2f}) | {st.median(ht)/st.median(bt):.3f} | {st.median(mx):.2f} ({min(mx):.2f}) | {st.median(bt)/st.median(mx):.3f} | {st.median(ht)/st.median(mx):.3f} | {wins}/{len(H)} | {min(a[3] for a in H):.7f} |")
