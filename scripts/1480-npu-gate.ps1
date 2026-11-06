<# #1480: chunker v1 gate records on Intel NPU. Non-elevated PowerShell on turboembed-npu.
   Writes under C:\te\out-1480-<stamp>\ then copies JSON into records\npu\ for packaging.
   Needs an NPU MiniLM bundle with fixed_seq >= 254 (febfed530b66 is 128 and will fail --max 254).
   Does not push. #>
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
$Root = "C:\te"
$Src = Join-Path $Root "turboembed"
$Stamp = Get-Date -Format "yyyyMMdd-HHmm"
$Out = Join-Path $Root ("out-1480-" + $Stamp)
$Scratch = Join-Path $Out "scratch"
$Records = Join-Path $Out "records\npu"
$Dumps = Join-Path $Out "dumps"
$Logs = Join-Path $Records "logs"
$null = New-Item -ItemType Directory -Force -Path $Out, $Scratch, $Records, $Dumps, $Logs, "$Root\bundles"
"step,exit,minutes" | Set-Content "$Out\summary.csv"
function Step([string]$Name, [scriptblock]$Body) {
  Write-Host "=== $Name" -ForegroundColor Cyan
  $t = Get-Date; $global:LASTEXITCODE = 0
  & $Body *>&1 | Tee-Object -FilePath "$Out\$Name.log"
  $code = if ($null -ne $global:LASTEXITCODE) { $global:LASTEXITCODE } else { 0 }
  "$Name,$code,{0:N1}" -f ((Get-Date) - $t).TotalMinutes | Add-Content "$Out\summary.csv"
  if ($code -ne 0) { Write-Host "--- $Name exit $code" -ForegroundColor Red }
}
function Have([string]$c) { [bool](Get-Command $c -ErrorAction SilentlyContinue) }
function RefreshPath {
  $env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
              [Environment]::GetEnvironmentVariable("Path","User") + ";$env:USERPROFILE\.cargo\bin"
  foreach ($d in @("$env:ProgramFiles\7-Zip", "$env:ProgramFiles\Git\cmd")) {
    if (Test-Path $d) { $env:Path += ";$d" }
  }
}
RefreshPath
Get-ChildItem Env: | Where-Object { $_.Name -like "TURBO_*" } | ForEach-Object { Remove-Item "Env:$($_.Name)" }

# Machine facts
& {
  "date: $(Get-Date -Format o)"
  (Get-CimInstance Win32_Processor).Name
  (Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber | Format-List | Out-String).Trim()
  Get-PnpDevice -PresentOnly | Where-Object { $_.FriendlyName -match "AI Boost|NPU" } | Format-Table -Auto FriendlyName, Status | Out-String
  Get-CimInstance Win32_PnPSignedDriver | Where-Object { $_.DeviceName -match "AI Boost|NPU" } | Format-List DeviceName, DriverVersion, DriverDate | Out-String
  "ze_loader: $((Get-Item $env:SystemRoot\System32\ze_loader.dll -ErrorAction SilentlyContinue).VersionInfo.FileVersion)"
  rustc -V; cargo -V; git --version
} *>&1 | Set-Content "$Out\machine.txt"

git config --global core.autocrlf false
if (-not (Test-Path $Src)) {
  git clone --branch main https://github.com/ai-pipestream/turboembed.git $Src
} else {
  git -C $Src fetch origin
  git -C $Src checkout -B main origin/main
}
Set-Location $Src
git log -1 --format="%H %s" | Tee-Object "$Out\head.txt"
$env:CARGO_TERM_COLOR = "never"
$env:TURBO_TEST_REQUIRE_NPU = "1"

Step "npu-device-list" {
  cargo test -p turbo --features npu --test npu -- --nocapture the_devices_listed the_table_is_whole
}

function Get-Bundle([string]$Tag, [string]$Prefix) {
  $dir = Join-Path $Root "bundles\$Prefix"; $x = Join-Path $dir "x"
  if ((Test-Path "$x\.ok") -and (Test-Path "$x\manifest.json")) { return [string]$x }
  $null = New-Item -ItemType Directory -Force $dir
  try { $rel = Invoke-RestMethod "https://api.github.com/repos/ai-pipestream/turboembed/releases/tags/$Tag" -UseBasicParsing }
  catch { "bundle $Prefix`: release lookup failed: $_" | Add-Content "$Out\machine.txt"; return "" }
  $assets = @($rel.assets | Where-Object { $_.name -like "$Prefix.*" })
  foreach ($a in $assets) {
    $dst = Join-Path $dir $a.name
    for ($try = 1; $try -le 3; $try++) {
      if ((Test-Path $dst) -and (Get-Item $dst).Length -eq $a.size) { break }
      Remove-Item $dst -Force -ErrorAction SilentlyContinue
      try { Invoke-WebRequest $a.browser_download_url -OutFile $dst -UseBasicParsing } catch { "download $($a.name) try $try`: $_" | Add-Content "$Out\machine.txt" }
    }
    if (-not ((Test-Path $dst) -and (Get-Item $dst).Length -eq $a.size)) { return "" }
  }
  $sz = if (Have "7z") { (Get-Command 7z).Source } else { "$env:ProgramFiles\7-Zip\7z.exe" }
  Remove-Item $x -Recurse -Force -ErrorAction SilentlyContinue
  & $sz x "$dir\$Prefix.zip" "-o$x" -y *>&1 | Add-Content "$Out\extract-$Prefix.log"
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path "$x\manifest.json")) { return "" }
  $null = New-Item -ItemType File -Force "$x\.ok"
  return [string]$x
}

$minilm = [string](Get-Bundle "minilm-npu-token-ids-f16-febfed530b66" "minilm-npu-token-ids-f16-febfed530b66")
"minilm bundle: $minilm" | Tee-Object -Append "$Out\machine.txt"
if (-not $minilm) {
  Write-Host "NO NPU BUNDLE; aborting eval" -ForegroundColor Red
  Compress-Archive -Path "$Out\*" -DestinationPath "$Out.zip" -Force
  Write-Host "Send back: $Out.zip"
  exit 2
}

$Cache = Join-Path $Root "eval-cache"
$null = New-Item -ItemType Directory -Force $Cache
$Sets = "scifact,nfcorpus,arguana,qasper"

# NPU-sealed MiniLM (febfed530b66) is fixed_seq 128; #1480 / #172 need max 254.
# Refuse early unless the loaded bundle reports max_seq >= 254.
Step "fetch-sets" {
  cargo run --release -p turbo-eval --features turbo/npu -- fetch --sets $Sets --cache $Cache
}

# Pipelines matching v1-172 metal/cpu (requires NPU IR fixed_seq >= 254)
$pipelines = @(
  @{ Name = "turbo-chunker-254-200-hybrid"; Flags = @("--chunker","turbo-chunker","--max","254","--target","200"); Dump = $true },
  @{ Name = "turbo-chunker-254-200-cohesion-1000-hybrid"; Flags = @("--chunker","turbo-chunker","--max","254","--target","200","--cohesion","1000"); Dump = $true },
  @{ Name = "sentence-windows-5-hybrid"; Flags = @("--chunker","sentence-windows","--window","5"); Dump = $false },
  @{ Name = "fixed-tokens-200-hybrid"; Flags = @("--chunker","fixed-tokens","--tokens","200","--overlap","0"); Dump = $false },
  @{ Name = "opennlp-sandbox-semantic-hybrid"; Flags = @("--chunker","opennlp-sandbox-semantic"); Dump = $false }
)

foreach ($p in $pipelines) {
  $name = $p.Name
  $outDir = Join-Path $Scratch $name
  $dumpArgs = @()
  if ($p.Dump) { $dumpArgs = @("--dump-cuts", (Join-Path $Dumps $name)) }
  Step "eval-$name" {
    cargo run --release -p turbo-eval --features turbo/npu -- run `
      --bundle $minilm --sets $Sets --cache $Cache --device npu --retrieval hybrid `
      --out $outDir @dumpArgs @($p.Flags)
  }
  if (Test-Path $outDir) {
    $dest = Join-Path $Records $name
    $null = New-Item -ItemType Directory -Force $dest
    Copy-Item "$outDir\*.json" $dest -Force -ErrorAction SilentlyContinue
    if (Test-Path "$Out\eval-$name.log") { Copy-Item "$Out\eval-$name.log" "$Logs\$name.log" -Force }
  }
}

$cand = Join-Path $Scratch "turbo-chunker-254-200-hybrid"
foreach ($base in @("turbo-chunker-254-200-cohesion-1000-hybrid","sentence-windows-5-hybrid","fixed-tokens-200-hybrid","opennlp-sandbox-semantic-hybrid")) {
  $bdir = Join-Path $Scratch $base
  if ((Test-Path $cand) -and (Test-Path $bdir)) {
    Step "gate-vs-$base" {
      cargo run --release -p turbo-eval --features turbo/npu -- gate $cand $bdir
    }
    if (Test-Path "$Out\gate-vs-$base.log") { Copy-Item "$Out\gate-vs-$base.log" "$Logs\gate-vs-$base.log" -Force }
  }
}

# Cut report vs CPU dumps if present on disk; otherwise note missing path.
$CpuDumps = Join-Path $Root "eval-cuts-172-main\cpu"
$CutReport = Join-Path $Out "cuts-vs-cpu.txt"
if (Test-Path $CpuDumps) {
  foreach ($name in @("turbo-chunker-254-200-hybrid","turbo-chunker-254-200-cohesion-1000-hybrid")) {
    $npuDump = Join-Path $Dumps $name
    $cpuDump = Join-Path $CpuDumps $name
    if ((Test-Path $npuDump) -and (Test-Path $cpuDump)) {
      Step "cuts-$name" {
        cargo run --release -p turbo-eval --features turbo/npu -- cuts $npuDump $cpuDump
      }
    }
  }
} else {
  "MISSING CPU cut dumps at $CpuDumps (exact path). NPU dumps are under $Dumps. Cut report deferred." | Set-Content $CutReport
}

# README stub (host fields must be scrubbed before commit; do not invent metrics).
@"
# #1480 chunker v1 records: Intel NPU (Arrow Lake)

See head.txt and machine.txt in this out package. provenance.device.backend must be npu on every JSON.
Bundle: NPU-sealed minilm-npu-token-ids-f16-febfed530b66 under $minilm
"@ | Set-Content "$Records\README.md"

Compress-Archive -Path "$Out\*" -DestinationPath "$Out.zip" -Force
Get-Content "$Out\summary.csv"
Write-Host "Send back: $Out.zip"
