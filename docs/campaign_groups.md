# Campaign Group / Multi-Seed Summary

Choose **New Group…** in `toposc-live` to add individual campaign directories or
the immediate seed directories of a parent folder. Each member has an explicit
role: `included`, `technical_dropout`, or `excluded`; exclusions require a
reason. Choose **Validate and open group** to read evidence in the background.
Save the definition outside campaign directories and reopen it with
**Open Group…**. Definitions support relative paths and optional expected seed
IDs, so a renamed directory cannot silently become a different replicate.

For the current Campaign 1A, open the supplied definition:

```powershell
.venv/Scripts/python.exe -B -m toposc_live --group docs/campaign_groups/campaign_1a.json
```

This opens an analysis view; it does not launch or resume a campaign. The
definition includes seeds 17102–17110 under the original parent and seed 17111
at `results/new-campaign`. Seed 17101 is shown as `technical_dropout`, with its
Windows persistence failure stated, and is excluded from every scientific
aggregate, curve, denominator and top-candidate ranking.

## Compatibility and read-only integrity

Compatibility requires a complete supported official `DiscoveryConfig` and
matching stored config/checkpoint fingerprints. Every configuration field
except campaign seed and the figure-output `plots` option must agree. This
includes stratum, generator, model, threshold, tolerances, batch/pool/cycle
allocation, proposal limits, selection controls, disorder settings and retry
reserve. Stored protocol identifiers, runtime environments and external
exclusion cohorts must also agree. Generator `auto` and `patch` are not silently
aliased for group compatibility. The engine's stored random first-cycle warmup
remains labeled `random` in the candidate table.

The existing reader now supplies group preflight: checksum envelopes,
committed-cycle file inventories, candidate record identities/validation and
agreement with the stored model configuration. It never repairs data or invokes
an evaluator. Unsupported, missing or corrupt required evidence rejects the
group refresh. The last display is marked stale and cannot be exported until a
valid refresh arrives. Groups use the same bounded background refresh worker as
single campaigns; only one read operation runs at a time. Large groups can take
several seconds per refresh because inventory validation traverses stored
artifacts. This is a read-only observation, not a transaction across independent
seeds, and does not replace the engine's authoritative resume checks.

Source fingerprints are retained per included member. A source difference is
prominently reported: matching configuration does not establish identical
implementation or causal comparability. Group inspection never bypasses
source/runtime resume checks. For the current group, archived Python source
differences are confined to the three files of the documented Windows
persistence fix. Historical/report-only and future configurations that cannot
pass group validation remain inspectable in the single-campaign view.

## Summary definitions

The view shows completion/failure/incomplete counts, exclusions, campaign IDs,
proposal counts, exact evaluations and charged attempts, per-seed best quality,
global best quality, mean/median/sample standard deviation (`n-1`) and min/max.
`Valid pool` means accepted unique proposals; invalid and duplicate/near-duplicate
proposals are separate. Missing totals stay unavailable. Success is copied
exact quality compared inclusively with the stored frozen threshold. Candidate
success fraction uses observed exact candidate records; seed success fraction
uses seeds with an observed best. Both denominators are displayed. Unfinished
included seeds make observed-best summaries provisional. Repeated geometries
across seeds are not silently pooled or counted as additional independent seeds.

The top-N table has explicit **Campaign Seed** and **Exact Eval Seed** columns,
plus generator, zero-based cycle, topology, indices, minimum |E|, confirmation,
OOD and stored robustness evidence. Selection opens the existing geometry and
diagnostic details. Single-seed views retain their behavior and now use the same
unambiguous seed column names.

Each group curve is the running best stored exact quality through a completed
cycle. The white group curve is the equal-seed mean through the common completed
horizon, with no extrapolation or changing contributor set. Matching cycle and
batch allocations makes this a search-progress axis rather than wall time or
retry count. Current uncommitted-cycle outcomes may appear in the leaderboard
but do not enter the completed-cycle plot. Last-improvement cycle and last-cycle
gain help inspect flat sections; no saturation claim is made.

## Exports

**Export JSON…** includes configuration, members/exclusion reasons, statistics,
curves, warnings and top-N diagnostic records. **Export CSV…** is one long-form
table with explicit `row_type` values `summary`, `member`, and `candidate`;
nested diagnostic evidence is encoded as JSON cells. Exports use exclusive
creation outside all member directories and reject existing files. They do not
alter campaign records. See the [Campaign 1A results](decisions/campaign_1a_group_summary.md).

No physics, scoring, generator, threshold, tolerance, scientific record,
checkpoint or engine behavior is changed by this feature. No Majorana or
thermodynamic interpretation is inferred by the group layer.
