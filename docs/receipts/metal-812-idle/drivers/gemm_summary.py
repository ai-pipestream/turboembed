import json,statistics as st,sys,collections
d=sys.argv[1]
G=collections.defaultdict(list)
for f in ['turbo.jsonl','mlx.jsonl']:
    for l in open(f'{d}/{f}'):
        r=json.loads(l)
        if 'shape' not in r: continue
        side='mlx' if r['side']=='mlx' else 'turbo/'+r['tile']
        for clk in ('host','gpu'):
            s=r.get('single_'+clk)
            if s: G[(r['hidden'],r['m'],r['shape'],side,clk)].append(s)
out=[]
for k in sorted(G):
    v=G[k]
    out.append(dict(hidden=k[0],m=k[1],shape=k[2],side=k[3],clock=k[4],rounds=len(v),least=min(x['least'] for x in v),
      median_of_medians=st.median(x['median'] for x in v),mean_of_means=st.mean(x['mean'] for x in v),max_p99=max(x['p99'] for x in v)))
json.dump(out,open(f'{d}/summary.json','w'),indent=1)
S={(o['hidden'],o['m'],o['shape'],o['side'],o['clock']):o for o in out}
f=lambda o:f"{o['least']:.3f} / {o['median_of_medians']:.3f} / {o['mean_of_means']:.3f} / {o['max_p99']:.3f}"
print('| h | M | shape | ours | MLX | ours / MLX | our fastest tile |\n|---|---|---|---|---|---|---|')
for h in (384,768):
  for m in (128,512,1024,4096,16384):
    for sh in ('qkv','attn_out','ffn_in','ffn_out'):
      a=S[(h,m,sh,'turbo/auto','host')]; x=S[(h,m,sh,'mlx','host')]
      best=min((o for kk,o in S.items() if kk[:3]==(h,m,sh) and kk[3].startswith('turbo/') and kk[3]!='turbo/auto' and kk[4]=='host'),key=lambda o:o['median_of_medians'])
      print(f"| {h} | {m} | {sh} | {f(a)} | {f(x)} | {a['median_of_medians']/x['median_of_medians']:.2f} | {best['side'][6:]} {best['median_of_medians']:.3f} |")
