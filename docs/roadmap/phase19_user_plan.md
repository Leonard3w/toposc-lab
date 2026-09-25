We are starting the next development stage of TOPOSC.

IMPORTANT CONTEXT
=================

The previous validation stage is complete.

Results:

- 2,500 new disorder realizations + 10 references were evaluated successfully.
- No numerical failures.
- The existing pipeline is reproducible.
- The previously identified `historical_best` geometry did NOT outperform the regular lattice robustly.
- For W = 6...9, the regular lattice had the higher mean score in all corrected comparisons.
- Example:
    W = 7.5:
        regular ≈ 76 %
        historical_best ≈ 54 %

    W = 9:
        regular ≈ 16 %
        historical_best ≈ 4 %

- Bulk/interior gap, edge-localization and Chern-related diagnostics also predominantly favor the regular baseline.
- Some `historical_best` realizations additionally expose spatial/boundary limitations.
- Therefore we currently have NO evidence that small connectivity rewiring of the present lattice improves disorder robustness.

This is an important negative result.

DO NOT try to rescue the old candidate.
DO NOT tune parameters until the old candidate wins.
DO NOT change historical results.
DO NOT delete or overwrite previous runs.

The new scientific direction is broader:

    Which geometric and connectivity principles can generate
    two-dimensional topological superconducting structures
    whose disorder robustness exceeds that of conventional
    regular lattices?

The goal of TOPOSC is therefore changing from

    "optimize small rewiring around one square lattice"

to

    "explore a broad space of 2D geometries and graph connectivities
     and discover structural principles associated with robust
     topological superconductivity."

==================================================
FIRST: PROTECT THE CURRENT STATE
==================================================

Before implementing anything:

1. Inspect the repository and current architecture.
2. Run the full existing test suite.
3. Confirm that the current validated results can still be reproduced.
4. Create a clear checkpoint/tag/branch if appropriate.
5. Do not modify historical result files.
6. Record the current git commit hash in the new research documentation.

All subsequent changes must remain backwards compatible where reasonably possible.

==================================================
CORE ARCHITECTURAL GOAL
==================================================

Introduce a general graph-based geometry representation.

A physical system should conceptually be represented as

    G = (V, E, r_i)

where

    V   = sites
    E   = bonds/connectivity
    r_i = physical 2D coordinates of each site

The code must clearly distinguish:

    physical coordinates
    connectivity
    Hamiltonian parameters

Do NOT assume that nearest neighbors are defined by a square lattice.

The same BdG physics pipeline should eventually work on arbitrary valid
2D embedded graphs.

==================================================
STEP 1 — GENERAL GRAPH GEOMETRY
==================================================

Implement a clean general geometry abstraction.

It should support at least:

- site IDs
- x/y position of each site
- list of bonds
- optional bond metadata
- boundary-site information if needed
- graph connectivity checks
- bond length
- bond direction / angle
- degree of every node

Useful API ideas are welcome, but adapt them to the existing repository
instead of rewriting everything.

For every directed bond i -> j it should be easy to calculate

    dx = x_j - x_i
    dy = y_j - y_i

    r_ij = sqrt(dx^2 + dy^2)

    theta_ij = atan2(dy, dx)

This representation must NOT be tied to one specific lattice.

==================================================
STEP 2 — REPRODUCE THE OLD SQUARE LATTICE
==================================================

This is a critical validation test.

Reconstruct the current regular square lattice using the new generic
graph representation.

Run the old Hamiltonian and diagnostics through the new representation.

For identical physical parameters, the old implementation and the new
graph implementation must agree numerically within an appropriate
floating-point tolerance for quantities such as:

- Hamiltonian matrix
- eigenvalues
- gap diagnostics
- localization diagnostics
- score
- Chern/topological diagnostics where applicable

Do NOT proceed to new graph families until this equivalence test passes.

Add automated regression tests.

==================================================
STEP 3 — PHYSICALLY CONSISTENT GRAPH BdG MODEL
==================================================

Make sure the superconducting model remains physically consistent on
arbitrary bonds.

In particular inspect the existing p-wave pairing convention.

For a bond i -> j, the pairing must respect the correct antisymmetry,
schematically

    Delta_ij = -Delta_ji

and directional dependence must be consistently related to the physical
bond direction.

Do NOT invent new physics if the existing model already defines this.

Instead:

1. document the current convention,
2. generalize it correctly to arbitrary 2D bonds,
3. add tests verifying Hermiticity / BdG symmetry / pairing antisymmetry.

Add automated numerical checks for:

    H = H^\dagger

and the relevant particle-hole symmetry of the implemented BdG model.

If there is ambiguity in how the current p+ip convention should be
generalized to arbitrary embedded graphs, STOP and explain the issue
before silently choosing a new model.

==================================================
STEP 4 — INITIAL STRUCTURE GENERATORS
==================================================

After the generic graph model is validated, add generators for several
structure classes.

For this FIRST extension, implement only:

A. regular square lattice
B. connectivity-rewired square lattice
C. random geometric graph / amorphous point geometry
D. constrained random graph embedded in 2D

Design the API so that future generators can later include:

- triangular
- honeycomb
- quasiperiodic
- fractal-like
- optimized/free geometries

but do NOT implement all of them yet.

==================================================
FAIR-COMPARISON CONSTRAINTS
==================================================

This is extremely important scientifically.

Different graph families must not be compared unfairly.

Implement a constraint/configuration system allowing us to control things
such as:

- number of sites N
- approximate / exact number of bonds
- mean degree
- maximum degree
- connectivity of the graph
- maximum allowed physical bond length
- minimum site separation
- spatial system size / density
- boundary definition

A candidate should be rejected if it violates required constraints.

Track rejection reasons.

Do not allow the search to win simply by adding arbitrarily many
long-range bonds or changing the amount of material.

==================================================
STEP 5 — GRAPH FEATURES
==================================================

For every geometry automatically calculate and store descriptive graph
features.

Start with:

- N
- number of edges
- mean degree
- degree variance
- min/max degree
- mean bond length
- bond length variance
- maximum bond length
- clustering coefficient
- connected components
- shortest-path statistics
- boundary-site fraction
- local orientation / bond-angle statistics if straightforward

Architecture should make it easy to add more features later.

These features are NOT optimization targets yet.

They are stored because eventually we want to determine WHY a geometry
is good.

==================================================
STEP 6 — PHYSICS OUTPUT
==================================================

Do NOT collapse all information into one score.

Every evaluated geometry should store the individual physical
observables used by TOPOSC.

Use the existing diagnostics and preserve them individually.

For example, if currently available:

- minimum / interior / bulk gap
- edge localization
- center weight
- low-energy states
- Chern/topological diagnostic
- robustness success/failure
- final legacy score

Do not rename quantities in a way that breaks old analyses.

The legacy score can remain for comparison, but the raw diagnostics must
always be available separately.

==================================================
STEP 7 — SMALL EXPLORATORY RANDOM SEARCH
==================================================

Do NOT implement deep learning.

Do NOT implement MAP-Elites yet.

Do NOT launch a huge autonomous search.

Once all previous steps and tests pass, create a SMALL exploratory search
using ordinary random sampling.

Purpose:

    Determine whether the broadened geometry space actually produces a
    broad distribution of physical behavior.

Suggested initial scale:

    roughly 50–100 valid geometries per structure family

Use a modest number of disorder seeds initially so that this remains an
exploratory run rather than a final statistical study.

Use the exact same physical parameter set when comparing structure
families.

Include the regular lattice as the reference.

==================================================
ANALYSIS OF THE EXPLORATORY RUN
==================================================

Generate a research report containing:

1. Distribution of each raw physics diagnostic by graph family.
2. Distribution of legacy score.
3. Best interesting candidates, but do NOT call them discoveries.
4. Comparison with the regular baseline.
5. Graph-feature vs physics correlations.
6. Candidate visualizations.
7. Energy spectra for selected examples.
8. Localization plots for selected examples.
9. Clear documentation of constraint violations / rejected geometries.
10. Any indication that the existing score disagrees with the raw
    physics diagnostics.

Important:

We are looking for SIGNAL and STRUCTURE, not merely a maximum score.

Examples of useful observations:

    - gap appears correlated with mean degree
    - long bonds destroy edge localization
    - a certain coordination distribution appears unusually robust
    - random geometries perform universally worse
    - no graph family beats the regular lattice

A negative result is acceptable.

==================================================
TESTING
==================================================

Add tests throughout this implementation.

At minimum test:

- old square lattice -> new graph representation equivalence
- graph serialization/deserialization
- bond length and angle calculations
- connectivity checks
- constraint enforcement
- Hamiltonian Hermiticity
- BdG particle-hole symmetry
- pairing antisymmetry
- reproducibility under fixed seeds
- deterministic generation when seed is fixed
- old score pipeline still functions
- no existing regression tests break

Do not remove old tests just to make new code pass.

==================================================
PERFORMANCE
==================================================

Do not prematurely optimize everything.

But structure the code so that later we can use sparse matrices for
larger systems.

Measure approximate evaluation runtime and record it.

Do not introduce GPU/deep-learning dependencies.

==================================================
REPRODUCIBILITY
==================================================

Every geometry must be reproducible from stored information.

Save at least:

- graph family
- generator parameters
- random seed
- coordinates
- edge list
- physical parameters
- disorder seed(s)
- software/git version
- graph features
- raw physical diagnostics
- legacy score

A candidate ID should uniquely refer to a reproducible geometry.

==================================================
NON-GOALS FOR THIS STAGE
==================================================

Do NOT:

- add neural networks
- add GNNs
- add reinforcement learning
- add Bayesian optimization
- add MAP-Elites
- launch multi-hour autonomous searches
- modify the score to make new candidates look better
- introduce arbitrary long-range hopping without constraints
- claim a new topological phase
- claim improved robustness from a tiny sample
- delete old results
- rewrite the entire repository without necessity

==================================================
STOP CONDITIONS
==================================================

Stop and report to me before proceeding if:

- the new generic graph model cannot reproduce the old square-lattice
  physics,
- the p-wave pairing generalization is physically ambiguous,
- particle-hole symmetry fails,
- graph constraints create an unfair comparison,
- existing tests break in a nontrivial way,
- the exploratory run reveals a serious flaw in the current score.

==================================================
FINAL DELIVERABLE
==================================================

When this stage is complete, stop.

Give me:

1. concise summary of what was changed,
2. files/modules added or modified,
3. test results,
4. confirmation that old results remain reproducible,
5. structure of the new graph representation,
6. graph families currently supported,
7. exploratory-search configuration,
8. plots/results of the exploratory run,
9. whether any graph family shows an interesting signal relative to the
   regular lattice,
10. known physical/numerical limitations,
11. exact recommendation for the NEXT experiment.

Do not automatically begin the next development stage.