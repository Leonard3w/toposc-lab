# PHASE 20 — TOPOSC RESEARCH STUDIO
# Unified experiment control and extensible research interface

We are now starting the next major development stage of TOPOSC-Lab.

This is NOT a new physics experiment.
Do NOT launch a large geometry search.
Do NOT add deep learning yet.
Do NOT change the validated physical model.

The goal of this stage is to turn the existing TOPOSC-Lab codebase into
a unified research application where future experiments can be configured,
started, monitored, inspected and reproduced WITHOUT editing Python code
and WITHOUT needing a new Codex prompt for every experiment.

============================================================
VERY IMPORTANT: EXISTING CODEBASE AND EXISTING INTERFACE
============================================================

TOPOSC-Lab already contains substantial functionality.

DO NOT rebuild functionality that already exists.

There is ALREADY AN INTERFACE / LIVE UI in the repository.

Absolutely do NOT create a second parallel interface.

Before changing anything:

1. Inspect the complete repository.
2. Inspect the current UI/interface and all of its entry points.
3. Inspect the current experiment runners.
4. Inspect configuration mechanisms.
5. Inspect checkpoint/resume.
6. Inspect database/result storage.
7. Inspect plotting/live monitoring.
8. Inspect the existing geometry/physics/search modules.

Then decide between:

A) extend/refactor the existing interface

or, ONLY if clearly cleaner and easier:

B) replace the existing interface with a unified new implementation.

If B is chosen:
- migrate all useful functionality from the old interface,
- keep the existing backend functionality,
- make sure there is only ONE supported interface afterward,
- do not leave two competing UI implementations / entry points.

Do not perform a rewrite simply because it is easier for you.
Prefer incremental reuse where the current architecture is sound.

Before implementation, make an internal audit:

ALREADY IMPLEMENTED:
- ...

PARTIALLY IMPLEMENTED:
- ...

MISSING:
- ...

UI DECISION:
- extend current UI OR replace current UI
- short technical reason

Then implement.

============================================================
CURRENT VALIDATED PROJECT STATE
============================================================

Phase 19 is complete.

Important facts:

- existing Geometry / GeometryEdge already support free 2D coordinates,
  edges, boundary information and metadata.
- no second generic graph representation is needed.
- current Hamiltonian is validated.
- directional antisymmetric p+ip pairing is validated.
- disorder pipeline is validated.
- Q / Localizer / Chern / localization diagnostics already exist.
- checkpoint/resume already exists.
- experiment storage and reporting already exist.
- old square-lattice path and generalized path agree.
- historical results must remain untouched.
- thousands of existing tests already protect the codebase.

Do not reimplement any of these without a concrete reason.

============================================================
MAIN DESIGN PRINCIPLE
============================================================

TOPOSC must become a scientific control center.

The user must be able to decide virtually every meaningful parameter
that affects:

- geometry
- physics
- disorder
- search space
- search algorithm
- objectives
- statistics
- numerical solver
- runtime
- reproducibility
- output/storage

Nothing scientifically important should be silently hard-coded.

However:

Do NOT duplicate parameters in the UI and backend.

There must be ONE canonical configuration representation.

The interface should edit that configuration.

Conceptually:

    UI
     ↓
    ExperimentConfig
     ↓
    validation
     ↓
    existing experiment/search runner
     ↓
    existing physics backend
     ↓
    results database
     ↓
    live monitor / result explorer

The GUI must NOT contain an independent physics implementation.

============================================================
SINGLE SOURCE OF TRUTH FOR CONFIGURATION
============================================================

Create or consolidate a canonical ExperimentConfig system.

Reuse existing configuration classes if possible.

Every scientifically relevant setting should have:

- value
- type
- valid range / validation
- default
- description
- category
- whether fixed or searchable/variable
- serialization support

Prefer a structured schema / dataclass / Pydantic-style architecture
if compatible with the existing repository.

Avoid scattering UI constants across multiple files.

The configuration must be serializable to a human-readable format
such as YAML or JSON.

Every run must save the exact resolved configuration.

============================================================
FULL CONTROL — FIXED VS VARIABLE PARAMETERS
============================================================

This is critical.

For every parameter that can reasonably become a search variable, the
user must be able to explicitly decide:

    FIXED / LOCKED
or
    VARIABLE / SEARCHABLE

Conceptually:

    mu              2.0       LOCKED
    Delta           1.0       LOCKED
    connectivity              VARIABLE
    coordinates               VARIABLE
    max bond length 1.75      LOCKED

Search algorithms must NEVER silently change locked values.

Provide clear UI controls for this.

Use a lock/toggle metaphor if appropriate.

The resulting run configuration must explicitly store which variables
were allowed to change.

============================================================
INTERFACE SECTIONS
============================================================

The unified interface should expose at least the following sections.

------------------------------------------------------------
1. EXPERIMENT
------------------------------------------------------------

- experiment name
- description / notes
- preset
- result directory
- random master seed
- tags
- load configuration
- save configuration
- clone previous experiment

------------------------------------------------------------
2. GEOMETRY
------------------------------------------------------------

Expose existing supported geometry families.

Currently relevant examples include:

- regular square
- rewired square
- amorphous planar
- constrained embedded random

Automatically use the actual repository capabilities rather than
inventing duplicate generators.

Expose all meaningful generator/geometry parameters, including where
supported:

- N / number of sites
- E / edge count
- box dimensions
- fixed vs variable coordinates
- fixed vs variable connectivity
- min degree
- max degree
- target/mean degree if applicable
- minimum site distance
- maximum bond length
- edge crossings allowed / forbidden
- connectivity requirement
- boundary definition
- boundary width
- generator seed
- family-specific settings

If different geometry families require different parameters,
show the appropriate controls dynamically.

Do not show meaningless controls for a family.

------------------------------------------------------------
3. PHYSICS
------------------------------------------------------------

Expose existing physical parameters, including where available:

- t
- mu
- Delta
- pairing settings
- hopping settings
- boundary conditions
- vector potential / magnetic terms if implemented
- energy windows
- kappa values
- Localizer parameters
- relevant physical tolerances

Do not add new physics in this phase.

Only expose capabilities already implemented.

If a parameter currently exists in code but is hidden/hard-coded,
move it into the canonical config when scientifically appropriate.

------------------------------------------------------------
4. DISORDER / ENSEMBLE
------------------------------------------------------------

Expose:

- disorder enabled
- disorder model/type
- distribution
- W values
- W range / explicit list
- number of disorder seeds
- explicit seed list if desired
- paired vs independent seeds
- disorder target terms if supported
- clean-reference evaluation

Allow:

    W = [3, 6, 9]

as well as range-style creation if useful.

------------------------------------------------------------
5. SEARCH SPACE
------------------------------------------------------------

This section defines WHAT the search algorithm is allowed to change.

Example:

    [x] connectivity
    [x] coordinates
    [ ] mu
    [ ] Delta
    [ ] t

For every searchable quantity support appropriate ranges/constraints.

Examples:

- coordinate displacement limits
- allowed bond additions/removals
- bond-length range
- degree limits
- parameter min/max
- mutation bounds

Search space and search method must be separate concepts.

============================================================
6. SEARCH METHOD
============================================================

Build this as an extensible plugin/strategy architecture.

For NOW, only expose search methods that genuinely exist.

Examples may currently include:

- random sampling
- parameter/grid scan
- existing autonomous/evolutionary methods if already implemented

Do NOT pretend future methods are implemented.

However design the interface/API so future methods can be added cleanly:

- evolutionary search
- MAP-Elites
- Bayesian optimization
- surrogate models
- GNN
- active learning
- reinforcement learning

A future search algorithm should be addable as one module implementing
a common interface rather than requiring a UI rewrite.

Something conceptually like:

    SearchStrategy
        propose(...)
        update(...)
        state_dict(...)
        load_state(...)
        metadata(...)

Adapt this to the current architecture.

============================================================
7. SEARCH SETTINGS
============================================================

Dynamically expose settings relevant to the selected algorithm.

Examples:

- evaluation budget
- candidate count
- generations
- population size
- mutation rate
- crossover rate
- exploration strength
- stopping criterion
- runtime limit
- search seed

Do not show irrelevant parameters.

============================================================
8. OBJECTIVES
============================================================

Do not force the entire physics into Q.

Phase 19 demonstrated that higher Q does not automatically imply
stronger edge localization.

Allow raw observables to remain separate.

Expose currently implemented observables such as:

- Q
- Interior Localizer gap
- edge weight
- center weight
- Chern/topological diagnostics
- spectrum-related diagnostics
- Majorana diagnostics where already available

Provide architecture for:

A) no optimization / pure sampling

B) single objective

C) weighted multi-objective score

D) Pareto/multi-objective optimization

Only enable optimization modes that the active search implementation
actually supports.

The interface should clearly distinguish:

    observable
    constraint
    objective
    diagnostic

Do not silently create a new scientific score.

============================================================
9. NUMERICAL SETTINGS
============================================================

Expose relevant existing numerical controls where safe:

- dense/sparse solver if supported
- eigensolver settings
- eigenvalue count
- tolerances
- worker count
- CPU settings
- checkpoint interval
- runtime limit
- evaluation limit

Do not expose low-level settings that cannot safely be changed without
validation.

Advanced settings belong in Expert Mode.

============================================================
10. OUTPUT / STORAGE
============================================================

Allow control of:

- save every candidate
- save rejected candidates
- save eigenvalues
- save eigenvectors where supported
- save plots
- save raw realizations
- checkpoint frequency
- report generation
- result directory

Every run must retain enough information for reproducibility.

============================================================
STANDARD MODE AND EXPERT MODE
============================================================

Provide two presentation levels if useful:

STANDARD MODE
- sensible common settings
- easy to start
- fewer visible controls

EXPERT MODE
- ALL scientifically meaningful controls
- no hidden important parameters
- full search-space definition
- advanced numerics
- full storage options

Expert Mode is essential.

Standard Mode must never secretly change scientific settings.
It may only hide controls while using explicitly documented defaults.

============================================================
RUN PREVIEW
============================================================

Before a run begins, show a clear RUN PREVIEW.

Example information:

Experiment:
    geometry_discovery_01

Geometry families:
    rewired_square
    amorphous_planar

Fixed:
    t = 1
    mu = 2
    Delta = 1

Search variables:
    connectivity
    coordinates

Disorder:
    W = 0,3,6,9
    20 seeds

Search:
    random sampling

Expected evaluations:
    8000

Estimated runtime:
    ...

Constraints:
    max bond = ...
    crossings = forbidden
    degree = ...

Require explicit START confirmation from this screen.

The resolved configuration shown here must be exactly the configuration
stored with the run.

============================================================
RUN VALIDATION
============================================================

Before starting, automatically validate configuration consistency.

Examples:

- impossible graph constraints
- no searchable variables selected for an optimization search
- invalid W range
- duplicate seeds
- unsupported physics/search combination
- invalid solver options
- unreasonable memory request
- output path conflict
- incompatible geometry settings

Warnings should be clearly separated from hard errors.

============================================================
LIVE RUN DASHBOARD
============================================================

Reuse and improve the EXISTING live dashboard.

Do not build a second independent dashboard.

Show at least:

- experiment ID/name
- state: queued/running/paused/completed/failed
- progress
- completed / planned evaluations
- elapsed time
- estimated remaining time if reliable
- current/best candidate statistics where meaningful
- baseline/reference values
- selected live plots
- rejection counts
- error count

Controls:

- Pause
- Resume
- Graceful Stop

Reuse the existing checkpoint/resume implementation.

Do not create a competing state system.

============================================================
RESULTS BROWSER
============================================================

Add or extend a unified results browser.

The user should be able to inspect old and new experiments.

Allow:

- sort
- filter
- search
- compare
- open candidate
- inspect configuration
- inspect raw observables
- inspect geometry features

Filters should include relevant quantities such as:

- family
- Q
- gap
- edge weight
- Chern diagnostics
- W
- seed
- success status
- graph features

============================================================
CANDIDATE EXPLORER
============================================================

When selecting a geometry/candidate, show a complete scientific view.

Where data exist, show:

- geometry plot
- edge structure
- coordinates
- graph features
- energy spectrum
- low-energy states
- localization map
- Q
- Interior Localizer gap
- edge weight
- center weight
- topological/Chern diagnostics
- Majorana diagnostics
- disorder performance
- provenance

Also show:

    which experiment created it
    generator/search settings
    parent/mutation information if available
    exact candidate ID
    geometry hash

============================================================
FOLLOW-UP EXPERIMENTS
============================================================

One major goal:

The user should NOT need Codex for normal validation experiments.

From the results/candidate view support workflows such as:

    Select candidate(s)
        ↓
    "Create follow-up experiment"
        ↓
    choose W values
        ↓
    choose new seeds
        ↓
    choose observables
        ↓
    Preview
        ↓
    Start

This should create a new ExperimentConfig and use the SAME experiment
runner.

Do not create a special validation backend.

============================================================
PRESETS
============================================================

Support save/load presets.

Useful examples:

- quick_test
- phase19_like
- disorder_scan
- candidate_validation
- broad_random_search

Presets must be normal configs, not special hidden code paths.

Users must be able to duplicate and modify them.

============================================================
CONFIG IMPORT / EXPORT
============================================================

Support:

- Save config
- Load config
- Export config
- Clone run config

Prefer YAML for human readability if consistent with current project
dependencies.

A configuration created in the UI should also be runnable from CLI.

A configuration created manually should also load in the UI.

============================================================
CLI + UI MUST USE SAME RUNNER
============================================================

Critical architecture rule:

    UI -> ExperimentConfig -> run_experiment()

and

    CLI -> ExperimentConfig -> run_experiment()

must reach the SAME backend.

No duplicated experimental logic.

============================================================
SIMPLE STARTUP — VERY IMPORTANT
============================================================

The interface must be easy to start.

The user is on Windows and uses the project locally.

First inspect how the current interface is started.

Preserve the current command if it is already good.

Additionally provide ONE obvious supported startup path.

Preferred user experience:

    double-click start_toposc.bat

and/or a simple command such as:

    python -m <existing_toposc_interface_module>

Choose the appropriate module name from the real repository.
Do NOT invent a disconnected launcher.

Add a root-level Windows launcher if appropriate:

    start_toposc.bat

It should:

- use the project environment correctly
- launch the ONE supported interface
- print a helpful error if dependencies/environment are missing
- not silently install packages
- not create another environment
- keep the terminal open on fatal error so the message can be read

If a PowerShell launcher is useful, it may also be included, but there
must still be one canonical UI application.

Document startup clearly in the README:

    1. activate/install environment as already documented
    2. run start_toposc.bat

Ideally after initial environment setup, starting TOPOSC should require
one double click.

============================================================
FUTURE AI / DEEP LEARNING ARCHITECTURE
============================================================

DO NOT implement deep learning in Phase 20.

But make sure future search methods can be added without redesigning the
application.

Future architecture should be able to support:

    Random
    Evolution
    MAP-Elites
    Bayesian optimization
    Surrogate model
    Graph Neural Network
    Active Learning

Future ML must distinguish clearly between:

    predicted quantities
and
    exact TOPOSC physics evaluation.

For example future data fields may contain:

    Q_predicted
    prediction_uncertainty
    Q_exact

Only exact physics evaluations are authoritative.

Do not implement those models now.

============================================================
REPRODUCIBILITY
============================================================

Each run must automatically save:

- full resolved config
- git commit
- source/version metadata
- start/end time
- seeds
- geometry definitions
- search settings
- physics settings
- constraints
- observables
- raw results
- run status
- checkpoints
- software version
- config hash

A user should be able to reopen an old experiment months later and see
exactly what was used.

============================================================
BACKWARD COMPATIBILITY
============================================================

Phase 19 and all historical experiments must remain readable.

Do not modify historical files.

Do not migrate old databases destructively.

If a new schema is required:
- use versioning
- provide compatibility adapters
- preserve read access to old runs.

Current validated physics must remain unchanged.

============================================================
TESTING
============================================================

Before changes:
- run relevant existing tests
- record baseline

During implementation add tests for at least:

- ExperimentConfig serialization
- config validation
- config round-trip
- UI config -> backend config consistency
- CLI config -> backend config consistency
- lock/fixed-variable behavior
- unsupported combinations
- preset loading
- old run loading
- run creation
- checkpoint/resume integration
- candidate loading
- follow-up experiment creation
- startup entry point
- existing backend regression

Do not remove old tests just to make Phase 20 pass.

Use lightweight mocked/small runs for UI/integration tests.
Do NOT launch a large research campaign during development.

============================================================
NO DUPLICATION
============================================================

Before adding any new component, check whether an equivalent already
exists.

Particularly avoid creating duplicate:

- Geometry models
- Hamiltonians
- disorder evaluators
- diagnostic implementations
- databases
- run managers
- checkpoint systems
- live dashboards
- plotting utilities
- interfaces
- result stores

Extend or adapt the existing implementation.

============================================================
NON-GOALS
============================================================

DO NOT during Phase 20:

- run the proposed 1255-realization confirmation study
- start a new large search
- implement MAP-Elites unless already present and only needs UI exposure
- implement GNNs
- implement deep learning
- implement reinforcement learning
- change Q
- change Hamiltonian physics
- change p+ip conventions
- introduce new physical claims
- delete Phase 19
- optimize new geometries
- create a second UI

============================================================
ACCEPTANCE CRITERIA
============================================================

Phase 20 is complete when:

1. There is exactly ONE supported TOPOSC interface.

2. Existing interface functionality has been preserved or migrated.

3. The interface starts easily from Windows.

4. Normal experiment setup no longer requires editing Python files.

5. Every scientifically meaningful experiment setting can be controlled,
   directly or through Expert Mode.

6. Fixed vs searchable variables are explicit.

7. Configurations can be saved and loaded.

8. UI and CLI use the same ExperimentConfig and experiment runner.

9. Existing live monitoring and checkpoint/resume are integrated.

10. Old experiments can be browsed.

11. Candidates can be inspected.

12. Follow-up experiments can be created from selected candidates.

13. Search methods are modular/extensible.

14. Existing Phase-19 physics remains numerically unchanged.

15. Existing tests continue passing, plus new Phase-20 tests.

16. No large research run has been automatically started.

============================================================
FINAL DELIVERABLE
============================================================

When finished, STOP.

Give me a concise report containing:

1. whether the old interface was refactored or replaced and why
2. the single canonical command / launcher to start TOPOSC
3. files added
4. files modified
5. architecture of ExperimentConfig
6. architecture of experiment runner
7. currently selectable geometry families
8. currently selectable search algorithms
9. all major UI sections
10. Expert Mode capabilities
11. save/load preset behavior
12. Candidate Explorer behavior
13. follow-up experiment workflow
14. backward-compatibility status
15. tests before / after
16. any remaining hard-coded scientific parameters
17. known limitations
18. recommended next Phase after Phase 20

Do NOT automatically begin the next phase.