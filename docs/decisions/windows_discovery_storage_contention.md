# Windows discovery storage contention fix

This is a filesystem reliability correction, not a scientific protocol change.
No production campaign was started, resumed, stopped or edited during this work.

## Diagnosis

Seed 17101 failed in `DiscoveryEngine._exact_stage` while finalizing
`attempts/000150.json`. The numerical stage had already been persisted to
`cycle-0006/base-0001.json`; the attempt ledger still says `started` because its
replacement failed. The original traceback records `PermissionError` with
Windows error 5 at `os.replace(temporary, destination)`.

`toposc-live` uses `Path.read_bytes()` in `ArtifactCache.read`. Python's ordinary
Windows file open can deny rename/delete sharing while its read handle is open.
The handle closes before JSON decoding/checksum checks, but a writer can reach
`os.replace` during the read itself. The previous writer made one replacement
attempt and propagated the error immediately.

A real Windows fixture on this machine reproduces **WinError 5** by holding
`target.open('rb')` while calling `os.replace`. Closing that reader permits the
same temporary file to replace the target. This confirms the mechanism and the
missing contention handling; the traceback alone cannot prove which process
held the historical handle. Antivirus/indexing handles and permanent ACL
denials can produce related failures. See Microsoft's
[sharing rules](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew)
and the [CPython replacement discussion](https://github.com/python/cpython/issues/90161).

A delete-sharing read-handle experiment did not reliably permit destination
replacement on this machine. No custom Win32 reader is included in the fix.
Monitoring retains its existing short-lived read-only handles and cached reads.

## Change and exact error policy

Changed for this fix:

- `src/toposc_lab/_file_io.py`: shared filesystem-only replacement/cleanup helper.
- `src/toposc_lab/discovery/storage.py`: use the helper at atomic JSON publication
  and cleanup, keeping serialization, checksums, fsync and reader semantics.
- `src/toposc_live/processes.py`: use the same helper for application process
  metadata, which is also replaced while monitored by another process.
- `tests/test_windows_atomic_io.py`: filesystem and exact crash-boundary fixtures.
- This report, `docs/toposc_live.md`, and the tooling execution-state notes.

For a failed `os.replace`, retry **only on Windows** and **only winerror
5 (access denied), 32 (sharing violation), or 33 (lock violation)**. There are
eight total attempts, with seven delays of **10, 20, 40, 80, 160, 320, 500 ms**:
at most **1.13 seconds of sleeping** plus syscall time. The same serialized,
fsynced and closed temporary is reused. No numerical evaluation, serialization,
cycle, seed, RNG action, or budget debit is repeated by this retry.

Error 5 cannot by itself distinguish transient contention from permanent ACL
denial. A known non-regular destination or Windows read-only target fails
immediately. Other access denials receive only the bounded retries; exhaustion
re-raises the original OS exception, including winerror/paths, with a note naming
the failed replacement and its retry limit. Other OS errors (including missing
paths, cross-device moves, invalid parameters, disk full, and POSIX EACCES)
propagate immediately. Serialization, file creation, writing, flushing and
fsync errors are not retried. No ACLs or attributes are changed.

Cleanup removes only the unique temporary created by that call, never the
destination or unrelated stale files. Transient cleanup errors have the same
bounded policy. If cleanup cannot succeed, its error propagates, or is attached
to an already propagating primary error with the retained temporary path; it
does not mask the original write failure or silently pretend cleanup succeeded.

## Atomicity, integrity, and scope

The complete checksum envelope is still serialized first, then written to a
unique temporary **in the destination directory**, flushed, fsynced and closed.
Only `os.replace` publishes it. There is no destination unlink, truncation,
copy fallback or alternate directory. Readers observe the complete old or new
artifact, never the partially written temporary. If replacement never succeeds,
the old destination remains unchanged.

This helper does not authorize overwrites: existing caller overwrite policies,
checksums, single-writer lease, config/source/runtime checks, cycle inventories,
attempt accounting and recovery logic remain unchanged. Corrupt or incompatible
state is not repaired, migrated or granted an exception. The existing engine's
read-before-reuse checks still fail on corruption.

Inspected related paths: discovery attempt/stage/plan/checkpoint/report writes
all use `atomic_json`; application requests/state/preferences use
`write_metadata`. The live reader consumes cycle journals and reports rather
than `dataset.json`. Dataset publication has its own atomic serializer; source
ZIP publication occurs before the manifest, and its reader is resume preflight,
not repeated live refresh. Those writers are unchanged. Historical search
checkpoint/candidate writers are outside this discovery-monitor contention fix.
`writer.lock` and the application launcher lease are unchanged.

## Seed 17101: read-only audit

Directory:
`results/gui-smoke-test/discovery-1a-patch/new-campaign/seed-17101`

- Config: Patch, seed 17101, ten cycles, batch four, pool eight, attempt cap 248.
- Checkpoint: six completed cycles. There are 151 charged attempt files.
- Attempt 150: `started`, exact seed 2203577106,
  `cycle-0006/base-0001.json` exists with status `complete`.
- All 342 stored checksum envelopes and all six committed-cycle inventories
  checked successfully. The dataset loads with 25 records; exclusions are empty.
  The saved base-stage record passes the existing record validator.
- Source ZIP checksum:
  `077c5c575037597b0e5dcb41388898e9a5f7664d90abc10a7dfd1962de3b2322`.
- Source bytes reconstructed directly from that ZIP match frozen source SHA-256
  `d8be4164e285f4dfb2b4a5bdeacb28c0b450a61b0753616713d9261bdb32c8ba`.
- Original base commit: `d3180d9fa1c538fd8a084ba3c80a5099a0c23c16` plus the exact
  dirty source archived in the ZIP. The commit alone is insufficient.

This is consistent with a recoverable persistence interruption. The existing
engine reuses the complete stage file without recalculating it. The `started`
attempt remains charged and is not manually rewritten. The fixture regression
tests this precise behavior. The audit does not rerun physics or certify the
outcome of future stages; final authoritative recovery checks still run on
resume.

## How to resume, without weakening provenance

**Do not resume this seed directly from the patched checkout.** The unchanged
engine fingerprints every Python file under `src`, so it correctly rejects this
storage fix as a source mismatch. Do not edit the manifest fingerprint, replace
the source ZIP, delete attempt 150, or monkeypatch the provenance check.

Under the current strict provenance contract, use the verified original source
in an isolated recovery worktree with monitoring closed. This uses the original
writer, so it avoids the known competing monitor but does not include the new
retry protection against unrelated external file handles. Resuming with the
patched writer would require a separate, explicit, auditable storage-version
amendment; such an amendment is not silently introduced here.

The following commands are **for the user to execute**. They were not executed
against a production campaign during this fix. Close `toposc-live` and any
editor preview of the failed seed's JSON files first. Confirm no process is
already running this seed. Preserve a backup of the failed seed directory and
its original application job logs before resuming.

Open a **new PowerShell session**, then prepare the original source:

```powershell
$repo = 'C:\Users\Leonard\Documents\GitHub\toposc-lab'
$seed = Join-Path $repo 'results\gui-smoke-test\discovery-1a-patch\new-campaign\seed-17101'
$recovery = Join-Path (Split-Path $repo -Parent) 'toposc-lab-17101-frozen-recovery'
$python = Join-Path $repo '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $recovery) { throw 'Choose a new, unused recovery worktree path.' }
git -C $repo worktree add --detach $recovery d3180d9fa1c538fd8a084ba3c80a5099a0c23c16
if ($LASTEXITCODE -ne 0) { throw 'Worktree creation failed.' }
$zip = Join-Path $seed 'source.zip'
if ((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant() -ne '077c5c575037597b0e5dcb41388898e9a5f7664d90abc10a7dfd1962de3b2322') {
    throw 'Original source archive mismatch; stop.'
}
Expand-Archive -LiteralPath $zip -DestinationPath $recovery -Force
$env:PYTHONPATH = Join-Path $recovery 'src'
$env:PYTHONDONTWRITEBYTECODE = '1'
# All four settings were unset in this campaign's original manifest.
Remove-Item Env:OMP_NUM_THREADS, Env:OPENBLAS_NUM_THREADS, Env:MKL_NUM_THREADS, Env:BLIS_NUM_THREADS -ErrorAction SilentlyContinue
Set-Location -LiteralPath $recovery
```

Next run **read-only** configuration/source/runtime/archive preflight using the
original frontend adapter from the original source ZIP:

```powershell
@'
import sys
from pathlib import Path
from toposc_live.configuration import ConfigurationAdapter
plan = ConfigurationAdapter().resume_plan(Path(sys.argv[1]))
print('Resume preflight passed; frozen seed:', plan.config.seed)
print('Frozen source loaded from:', __import__('toposc_lab.discovery.engine', fromlist=['x']).__file__)
'@ | & $python -B - $seed
if ($LASTEXITCODE -ne 0) { throw 'Preflight failed; do not resume or change fingerprints.' }
```

It requires Python 3.14.7, NumPy 2.5.3, SciPy 1.18.1 and the original platform
metadata, as well as the four unset thread variables. Do not use an editable
reinstall that changes the working application's environment. The temporary
`PYTHONPATH` chooses the isolated archived source with the existing interpreter.

Only after this succeeds, explicitly resume the existing seed:

```powershell
& $python -B -u -m toposc_lab.discovery $seed
if ($LASTEXITCODE -ne 0) { throw 'Resume failed; preserve its error and all campaign files.' }
```

The directory's stored config is loaded automatically. There is no `--resume`
flag and no new seed or changed budget. This command continues remaining work
from the journals/checkpoint. Keep the GUI closed until it exits. Close this
PowerShell session afterward to discard its temporary environment settings.
The original GUI supervisor job still records the old failed invocation;
an externally resumed CLI is not tracked by that supervisor. Use the CLI exit
status and stored final summary/checkpoint to establish its new completion.

## Other seeds

At inspection, seeds **17102–17110 were already complete: 10/10 cycles and 240
attempts each**. They need no restart, rerun or migration. If another process is
still running elsewhere, leave it running and close monitoring until it exits;
already imported code does not acquire this writer fix just because files on
disk changed. Do not hot-patch or restart a healthy scientific process. Future
campaigns launched after loading the corrected code receive the retry policy;
no future campaign was launched as part of this task.

## Validation

The focused discovery/configuration/validation/live regression passed **98
tests in 59.34 seconds**. Two additional tests then brought the new filesystem
module to **24 passing tests**, including real Windows contention/release,
bounded synthetic winerror 5/32/33, exhaustion, permanent/read-only failures,
same-directory publication, complete old/new JSON, temporary cleanup failures,
concurrent `CampaignReader` refresh and checkpoint writes, and the exact saved
stage / unfinalized ledger recovery boundary. All use temporary fixtures.

Full repository regression: **2920 passed in 448.39 seconds**, recorded in
`results/windows-storage-full.log`. Ruff and isolated strict Mypy pass for the
three changed/new runtime modules (`--python-version 3.14 --follow-imports
silent --ignore-missing-imports`). The full suite uses isolated test directories;
its existing small numerical tests are not production discovery campaigns.

The final preservation audit matches **all 6007 files** across seeds 17101–17110:
no modified, missing or added campaign files. The report is
`results/windows-storage-preservation.json`; pre-fix hashes are preserved in
`results/windows-storage-all-seeds-before.json` and
`results/windows-storage-seed17101-before.json`. No campaign was launched,
resumed or stopped during this fix. Pre-existing user edits and bytecode changes
were preserved.
