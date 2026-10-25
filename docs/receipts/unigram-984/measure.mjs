import fs from 'node:fs';
import cp from 'node:child_process';
import crypto from 'node:crypto';
const [fixed, main, blocked, dict, data, out, from = '1', through = '12'] = process.argv.slice(2);
fs.mkdirSync(out, {recursive:true});
const bins = {fixed, main, blocked};
const hash = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const identities = Object.fromEntries(Object.entries(bins).map(([k,p])=>[k,hash(p)]));
const inputs = Object.fromEntries([dict,...['pku','msr'].flatMap(c=>[`${data}/testing/${c}_test.utf8`,`${data}/gold/${c}_test_gold.utf8`,`${data}/gold/${c}_training_words.utf8`])].map(p=>[p.split('/').pop(),hash(p)]));
const records=fs.existsSync(`${out}/runs.json`) ? JSON.parse(fs.readFileSync(`${out}/runs.json`)) : [];
fs.writeFileSync(`${out}/identities.json`,JSON.stringify({binaries:identities,inputs,core:16,rounds:12,measured_passes:2,repetitions:50},null,2)+'\n');
for(let round=+from;round<=+through;round++){
  if(records.some(r=>r.round===round)) throw new Error(`round ${round} already recorded`);
  const order=round%2 ? ['main','blocked','fixed'] : ['fixed','blocked','main'];
  for(const version of order){
    const start=new Date().toISOString();
    const loadBefore=fs.readFileSync('/proc/loadavg','utf8').trim();
    const r=cp.spawnSync('taskset',['-c','16',bins[version],dict,data,'2','50'],{encoding:'utf8',env:{...process.env,TURBO_CPU_THREADS:'1'},maxBuffer:10*1024*1024});
    const loadAfter=fs.readFileSync('/proc/loadavg','utf8').trim();
    fs.writeFileSync(`${out}/${round}-${version}.log`,r.stdout+r.stderr);
    records.push({round,version,start,end:new Date().toISOString(),loadBefore,loadAfter,status:r.status});
    fs.writeFileSync(`${out}/runs.json`,JSON.stringify(records,null,2)+'\n');
    if(r.status!==0)throw new Error(`${round} ${version}: ${r.stderr}`);
  }
  console.log(`round ${round}/12 complete; load ${records.at(-1).loadAfter}`);
}
