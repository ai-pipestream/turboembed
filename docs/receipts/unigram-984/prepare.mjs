// node prepare.mjs FIXED_CHECKOUT MAIN_CHECKOUT BLOCKED_CHECKOUT NEW_OUTPUT_DIR
import fs from 'node:fs';
import path from 'node:path';
import cp from 'node:child_process';
const [fixed, main, blocked, output] = process.argv.slice(2).map(p => path.resolve(p));
if (!output || fs.existsSync(output)) throw new Error('provide three checkouts and a new output directory');
const receipt = path.join(fixed, 'docs/receipts/unigram-984');
const mainLibrary = path.join(output, 'main-library');
fs.mkdirSync(mainLibrary, {recursive: true});
for (const dir of ['src', 'data']) fs.symlinkSync(path.join(main, 'analysis/tokenize', dir), path.join(mainLibrary, dir));
let baseline = '[package]\nname="turbo-tokenize-main-baseline"\nversion="0.1.0"\nedition="2024"\nlicense="Apache-2.0"\npublish=false\n[lib]\nname="turbo_tokenize_main"\n[workspace]\n[dependencies]\n';
for (const [name, dir] of [['turbo-analysis','core'],['turbo-analysis-layers','layers'],['turbo-maxent','maxent'],['turbo-jvm','jvm'],['turbo-tokenize-layers','tokenize-layers'],['turbo-unicode','unicode']]) {
  baseline += `${name}={path=${JSON.stringify(path.join(fixed, 'analysis', dir))}}\n`;
}
fs.writeFileSync(path.join(mainLibrary,'Cargo.toml'), baseline);
for (const [name, source] of Object.entries({fixed, main, blocked})) {
  const dir = path.join(output, name);
  fs.mkdirSync(path.join(dir, 'src'), {recursive: true});
  fs.copyFileSync(path.join(fixed,'analysis/tokenize/comparisons/unigram_overhead.rs'), path.join(dir,'src/main.rs'));
  fs.copyFileSync(path.join(receipt,name==='fixed'?'paired.Cargo.lock':'base.Cargo.lock'), path.join(dir,'Cargo.lock'));
  let manifest = '[package]\nname="unigram-comparison"\nversion="0.1.0"\nedition="2024"\n[workspace]\n[features]\npolicies=[]\n';
  if (name==='fixed') manifest += 'main-baseline=["policies","dep:turbo-tokenize-main-baseline"]\n';
  manifest += `[dependencies]\njieba-rs="=0.11.0"\nturbo-tokenize={path=${JSON.stringify(path.join(source,'analysis/tokenize'))}}\n`;
  if (name==='fixed') manifest += `turbo-tokenize-main-baseline={path=${JSON.stringify(mainLibrary)},optional=true}\n`;
  fs.writeFileSync(path.join(dir,'Cargo.toml'),manifest);
  const args=['build','--locked','--release','--manifest-path',path.join(dir,'Cargo.toml')];
  if (name!=='main') args.push('--features',name==='fixed'?'main-baseline':'policies');
  const r=cp.spawnSync('cargo',args,{stdio:'inherit',env:{...process.env,CARGO_BUILD_JOBS:'2',CARGO_PROFILE_DEV_DEBUG:'0'}});
  if(r.status!==0) throw new Error(`${name} build failed`);
}
