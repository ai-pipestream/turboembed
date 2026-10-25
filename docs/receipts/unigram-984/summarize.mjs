import fs from 'node:fs';
const [dir, through = '12'] = process.argv.slice(2);
const median = a => {const b=[...a].sort((x,y)=>x-y); const n=b.length; return n%2?b[n>>1]:(b[n/2-1]+b[n/2])/2;};
const rows={}; const scores={}; const rounds={};
for(let round=1;round<=+through;round++)for(const v of ['main','blocked','fixed']){
  const raw=fs.readFileSync(`${dir}/${round}-${v}.log`,'utf8');
  for(const line of raw.trim().split('\n')){
    const fields=line.split(' ');
    if(fields[0]==='score'){
      const k=`${v}/${fields[1]}/${fields[2]}`, value=fields.slice(3).join(' ');
      if(scores[k] && scores[k]!==value)throw new Error(`unstable output ${k}`);
      scores[k]=value;
    }
    if(fields[0]==='timing'){
      const k=`${v}/${fields[1]}/${fields[2]}`, value=+fields[4].slice(3);
      if(!Number.isFinite(value))throw new Error(`invalid timing ${line}`);
      (rows[k]??=[]).push(value); (rounds[`${round}/${k}`]??=[]).push(value);
    }
  }
}
for(const c of ['pku','msr']){
  for(const mode of ['opennlp','dictionary','chinese','jieba-dictionary','jieba-hmm']){
    if(scores[`blocked/${c}/${mode}`]!==scores[`fixed/${c}/${mode}`])throw new Error(`changed full output ${c}/${mode}`);
  }
  if(scores[`main/${c}/main`]!==scores[`fixed/${c}/opennlp`])throw new Error(`main output mismatch ${c}`);
  if(scores[`fixed/${c}/main-source`] && scores[`fixed/${c}/main-source`]!==scores[`fixed/${c}/opennlp`])throw new Error(`embedded main output mismatch ${c}`);
}
const comparisons={};
for(const c of ['pku','msr'])for(const [name,a,b] of [
  ['fixed-default/main',`fixed/${c}/dictionary`,rows[`fixed/${c}/main-source`]?`fixed/${c}/main-source`:`main/${c}/main`],
  ['fixed-default/separate-main',`fixed/${c}/dictionary`,`main/${c}/main`],
  ['blocked-default/main',`blocked/${c}/dictionary`,`main/${c}/main`],
  ['fixed-default/blocked-default',`fixed/${c}/dictionary`,`blocked/${c}/dictionary`],
  ['fixed-default/fixed-opennlp',`fixed/${c}/dictionary`,`fixed/${c}/opennlp`],
]){
  const pairs=[];
  for(let r=1;r<=+through;r++){
    const candidate=median(rounds[`${r}/${a}`]), baseline=median(rounds[`${r}/${b}`]);
    pairs.push({round:r,candidate_ms:candidate,baseline_ms:baseline,ratio:candidate/baseline});
  }
  comparisons[`${c}/${name}`]={median_paired_ratio:median(pairs.map(p=>p.ratio)),pairs};
}
const runs=JSON.parse(fs.readFileSync(`${dir}/runs.json`));
for(const k of Object.keys(scores)) if(rows[k]?.length!==+through*2)throw new Error(`missing timing samples ${k}`);
const summary={samples:Object.fromEntries(Object.entries(rows).map(([k,a])=>[k,{n:a.length,median_ms:median(a),min_ms:Math.min(...a),max_ms:Math.max(...a)}])),scores,comparisons,load_1m:{min:Math.min(...runs.map(r=>+r.loadBefore.split(' ')[0])),max:Math.max(...runs.map(r=>+r.loadAfter.split(' ')[0]))}};
fs.writeFileSync(`${dir}/summary.json`,JSON.stringify(summary,null,2)+'\n');
console.log(JSON.stringify({samples:summary.samples,comparisons:Object.fromEntries(Object.entries(comparisons).map(([k,v])=>[k,v.median_paired_ratio])),load:summary.load_1m},null,2));
