// Six alternating, single-CPU rounds, each policy in a fresh process.
// Run only after the shared timing slot is handed off.
// node quant_search_compare.mjs BINARY DATA_DIRECTORY OUTPUT.json CPU
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import os from 'node:os';
import {spawnSync} from 'node:child_process';
const [binary, data, output, cpu] = process.argv.slice(2);
if (Object.keys(process.env).some(k=>k.startsWith('TURBOVEC_'))) throw new Error('unset kernel environment overrides');
const hash = b => crypto.createHash('sha256').update(b).digest('hex');
const command = (program, args, options={}) => {
  const r = spawnSync(program, args, {encoding:'utf8', maxBuffer:16*1024*1024, ...options});
  if (r.error || r.status !== 0) throw new Error(`${program} failed: ${r.error ?? r.stderr}`);
  return r.stdout.trim();
};
const commit = command('git', ['rev-parse', 'HEAD']);
if (command('git', ['status','--porcelain']).length) throw new Error('commit the measurement code first');
const prepared = JSON.parse(fs.readFileSync(path.join(data,'prepared.json')));
if (prepared.rows < 32768 || prepared.dim !== 512 || prepared.k !== 10) throw new Error('wrong real-data split');
const source = JSON.parse(fs.readFileSync(path.join(data,'source.json')));
const configs = [2,4].flatMap(bits => [false,true].map(calibrate => ({bits,calibrate})));
const samples = [], first = new Map();
const ambient = () => ({load_average:os.loadavg(), processes:command('ps',['-eo','pcpu,comm','--sort=-pcpu']).split('\n').slice(0,7)});
const start = new Date().toISOString(), ambientStart = ambient();
for (let round=0;round<6;round++) {
  for (const config of configs) {
    for (const mode of round%2===0 ? ['staged','exhaustive'] : ['exhaustive','staged']) {
      const raw = command('taskset',['-c',cpu,binary,'run',data,'512',String(config.bits),String(config.calibrate),mode], {
        env:{...process.env,RAYON_NUM_THREADS:'1',OMP_NUM_THREADS:'1',OPENBLAS_NUM_THREADS:'1'}, timeout:180000,
      });
      const run = JSON.parse(raw), key = `${config.bits}/${config.calibrate}/${mode}`;
      if (run.rows !== prepared.rows || run.queries !== prepared.queries) throw new Error('split changed');
      if (run.bits!==config.bits || run.calibrate!==config.calibrate || run.mode!==mode || run.ids.length!==prepared.queries*10) throw new Error('wrong configuration or hit count');
      const idsHash = hash(JSON.stringify(run.ids));
      if (!first.has(key)) first.set(key,{ids:run.ids});
      else if (hash(JSON.stringify(first.get(key).ids)) !== idsHash) throw new Error('IDs changed between rounds');
      fs.writeFileSync(path.join(data,`raw-${round}-${config.bits}-${config.calibrate}-${mode}.json`), raw+'\n');
      delete run.ids;
      samples.push({...run,round:round+1,ids_sha256:idsHash});
      console.log(JSON.stringify({round:round+1,bits:config.bits,calibrate:config.calibrate,mode,query_ms:run.query_ms,batch_ms_per_query:run.batch_ms_per_query}));
    }
  }
}
const agreement = configs.map(config => {
  const a = first.get(`${config.bits}/${config.calibrate}/staged`).ids;
  const b = first.get(`${config.bits}/${config.calibrate}/exhaustive`).ids;
  let ordered=0, sets=0, overlap=0;
  for (let q=0;q<prepared.queries;q++) {
    const x=a.slice(q*10,q*10+10), y=b.slice(q*10,q*10+10);
    if (String(x)===String(y)) ordered++;
    const common=x.filter(id=>y.includes(id)).length;
    if (common===10) sets++;
    overlap+=common;
  }
  return {...config,queries:prepared.queries,ordered_identical_queries:ordered,same_set_queries:sets,top_10_overlap:overlap/(prepared.queries*10)};
});
const files = ['vectors.f32','rows.f32','queries.f32','truth.u32'].map(name=>({file:name,bytes:fs.statSync(path.join(data,name)).size,sha256:hash(fs.readFileSync(path.join(data,name)))}));
const receipt = {
  measured_commit:commit,binary_sha256:hash(fs.readFileSync(binary)),source,prepared,files,
  rustc:command('rustc',['--version']),node:process.version,
  build_command:'cargo build --locked --release -p turbo-vector-index --features turboquant --example quant_search_compare',
  kernel_environment_overrides:[],
  cpu:os.cpus()[0].model,cpu_core:Number(cpu),threads:1,started_at:start,finished_at:new Date().toISOString(),
  method:'Rust release; six alternating staged/exhaustive rounds per configuration; fresh process per sample; three warm-up single-query and batch passes; median of three passes of individual queries; median of five batch passes; k=10; no concurrent measured sweep',
  memory:'rss_* are whole-process resident bytes from /proc/self/status, including inputs, query scratch, metadata and allocator-retained arenas; fresh processes per policy; not code-only allocation or compact storage sizes',
  ambient_start:ambientStart,ambient_end:ambient(),agreement,samples,
};
if (command('git',['rev-parse','HEAD'])!==commit || command('git',['status','--porcelain']).length) throw new Error('tree changed during measurement');
fs.writeFileSync(output,JSON.stringify(receipt,null,2)+'\n');
