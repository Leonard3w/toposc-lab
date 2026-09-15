# Phase 16B: independent falsification protocol v1

Authorized by the user's 2026-09-15 request to continue with 16B. This protocol
is written before generating validation geometries or inspecting their physics.
The frozen Phase-16A report supplies five questions, not established mechanisms.
Block G stops at Gate 16; no Phase 17 or large search is included.

## Population, separation and budget

Use seeds 16201-16216, one parent per seed, alternating Random and Patch.
Take the first valid separated proposal, without quality selection, with a cap
of 2000 raw proposals per seed. Preserve all proposal decisions. Fixed 36-site,
60-edge space, model, basis, localizer probes/kappas/tolerance and Q >= 0.20
success definition remain unchanged. No high-quartile surrogate success label.

Freeze a structural exclusion snapshot from all Phase-16A analysis, intervention
and simplification plans and all existing production discovery cycle plans
(including Campaign 1A, technical dropouts and subsequent campaigns). Include
even rejected proposal edge sets conservatively. Record paths and SHA-256s;
later campaign growth does not change this cutoff. No old outcomes enter selection.
Every parent and selected edit must have D4 edge-Jaccard distance >0.15 from
this snapshot. Every member of a new seed family must also be >0.15 from every
member of earlier seed families. Within-parent related edits are intentional;
the independent statistical unit is the seed/parent, never an edit or model point.
An unavailable strict match stays unavailable; do not relax matching or replace
a parent based on edit availability or physical outcomes.

Freeze all parents, edits and deletion choices before the first numerical call.
At most two arms for each of H1/H2/H4/H5 and two deletions per parent. Evaluate
each main-model geometry twice with distinct exact seeds; require identical
scientific records and diagnostics. Also evaluate every 60-edge geometry at
mu=1.9 and 2.1 once. Two regular-grid references (mu=2 and 8), each repeated.
Maximum planned calls: 16*(11*2 + 9*2)+4 = 644. Hard cap 700 charged attempts,
including failed/interrupted attempts. Four numerical thread limits are one.
Use the existing exact journal, checksum storage and single-writer lease.
Resume requires frozen source, runtime, protocol, plans and exclusions.

## Five predeclared tests

Enumerate existing valid degree-preserving double-edge swaps. Count K3, induced
C4, boundary/bulk cut edges, central diagonals and total diagonals as in 16A.
Boundary K3 means at least one fixed perimeter vertex. Spatial matching cost is
the squared distance between means of the four changed vertices plus the squared
difference of their mean radius from the center. Choose minimum cost, seeded
ties, with no physical labels. Record all motif deltas and spatial cost.

* **H1 cut coupling:** cut-decreasing versus cut-increasing swaps. Match triangle,
  square, central-diagonal and total-diagonal deltas exactly. Cut reduction is
  beneficial if Q(decrease)-Q(increase)>0. Report both changes from the parent;
  support additionally requires mean decrease-parent>0 and increase-parent<0.
* **H2 triangle placement:** both arms remove exactly one total triangle; the
  boundary arm removes one boundary K3 and no bulk K3, the bulk arm removes one
  bulk K3 and no boundary K3. Match square, cut and both diagonal deltas exactly.
  Boundary triangles are beneficial if Q(bulk removal)-Q(boundary removal)>0.
  Instance-level losses/gains must establish an actual removal without replacing
  other triangles; net counts alone are insufficient.
* **H3 deletion tolerance:** choose two distinct valid non-perimeter deletions
  uniformly without replacement using the parent seed, never quality. Retention
  requires positive parent Q, deleted Q >=90% parent Q, and the same nonzero
  resolved indices at all three kappas. Zero-quality parents count as failures.
  Each parent contributes its mean retention; test the claim that mean retention
  exceeds 0.75. Report every parent, quality distribution and topology changes.
* **H4 squares:** C4 destruction versus creation, matching triangle, cut,
  central-diagonal and total-diagonal deltas exactly. Report Q(create)-Q(destroy).
  This is a falsification test for a direct beneficial effect. Failure to reject
  zero is inconclusive, never evidence of equivalence or proof of the proxy claim.
* **H5 diagonal placement:** central-decreasing versus central-increasing swaps,
  both with zero total-diagonal delta and equal triangle, square and cut deltas.
  Report Q(increase)-Q(decrease); use a two-sided test. A detected location effect
  challenges the provisional explanation; absence is inconclusive.

For each test use exact two-sided sign flips of independent parent values
(H3: parent retention minus 0.75). Validity assumes independent symmetric errors;
this is not a randomized treatment-assignment test. Require at least eight
available parents. Unavailable tests receive p=1 in the fixed five-test Holm
family; alpha=0.05. Show seed means, bootstrap 95% intervals (4000 draws, seed
16299), availability and sign counts. H1/H2 support needs the predicted sign,
Holm significance, >=75% positive parents and the same mean sign at mu=1.9/2.1.
H3 support needs positive centered mean and Holm significance. H4/H5 significant
findings are reported as challenges, not retroactively selected mechanisms.
No motif is promoted on a main-model scalar score alone.

## Cross-model scope and Gate 16

The existing exact wiring adapter supports mu=1.9,2,2.1,8 in the same chiral
p-wave model. Nearby mu runs are parameter sensitivity, **not cross-model
validation**. Existing QWZ/BHZ models generate their own square lattice and do
not accept these diagonal wiring graphs. A new extension would require choosing
additional physical coupling rules absent from the frozen evidence. Task 16.15
therefore records cross-model generality as unavailable; do not invent a model
to complete a checkbox. No size-extension map or new disorder claim is made.

Gate 16 distinguishes completed falsification work from a demonstrated mechanism.
Report null/inconclusive results honestly. A mechanism remains unestablished
unless recurrent evidence, controlled intervention, unseen-family replication
and justified physical interpretation all hold. In particular 16A's failed
intervention criterion cannot be silently rewritten by this validation run.
Archive source/config/plans/raw eigensystems and verify checksums, exact repeats,
known positive/negative references, zero-call resume and the full test suite.
