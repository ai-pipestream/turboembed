import json,statistics as st,collections,sys
def load(f):
    R=collections.defaultdict(list)
    for l in open(f):
        d=json.loads(l); R[(d['hidden'],d['m'],d['shape'],d.get('tile','mlx'))].append(d)
    return R
B=load('base.jsonl'); H=load('head.jsonl'); M=load('mlx.jsonl')
med=lambda v,k: st.median(x[k]['median'] for x in v)
print('| h | M | shape | base auto GPU | head auto GPU | head / base GPU | base auto host | head auto host | MLX host | base / MLX host | head / MLX host |')
print('|---|---|---|---|---|---|---|---|---|---|---|')
for (h,m,s,t),v in sorted(B.items(), key=lambda kv:(kv[0][0],kv[0][1],['qkv','attn_out','ffn_in','ffn_out'].index(kv[0][2]))):
    hv=H[(h,m,s,'auto')]; mv=M[(h,m,s,'mlx')]
    bg,hg=med(v,'single_gpu'),med(hv,'single_gpu'); bh,hh,mh=med(v,'single_host'),med(hv,'single_host'),med(mv,'single_host')
    print(f'| {h} | {m} | {s} | {bg:.3f} | {hg:.3f} | {hg/bg:.3f} | {bh:.3f} | {hh:.3f} | {mh:.3f} | {bh/mh:.2f} | {hh/mh:.2f} |')
