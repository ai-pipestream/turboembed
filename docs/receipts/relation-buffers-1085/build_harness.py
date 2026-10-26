from pathlib import Path
import subprocess
source='docs/receipts/relation-buffers-1085/head_timing.rs'
for variant,target in [('main','target/baseline-build'),('fixed','target/build')]:
 d=Path(target)/'release/deps'
 args=['rustc','--edition=2024','-O',source,'-L',f'dependency={d}','-o',f'target/proof/head-timing-{variant}']
 if variant=='fixed':args+=['--cfg','reuse']
 for crate in ['turbo','turbo_relation','serde_json']:
  matches=list(d.glob(f'lib{crate}-*.rlib'))
  if not matches and (d/f'lib{crate}.rlib').exists(): matches=[d/f'lib{crate}.rlib']
  if crate=='serde_json': matches=sorted(matches)[:1]
  assert len(matches)==1,(crate,matches)
  args+=['--extern',f'{crate}={matches[0]}']
 subprocess.run(args,check=True)
