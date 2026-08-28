# TopoSC Lab

TopoSC Lab is a modular Python toolkit for studying superconducting and topological superconducting systems.

The goal is to provide a clean and extensible framework for:

- building superconducting Hamiltonians
- solving Bogoliubov-de Gennes systems
- visualizing spectra and wavefunctions
- exploring topological phases
- testing models such as the Kitaev chain, nanowires, Josephson junctions, and BCS systems

## Current status

Version 0.1 focuses on the Kitaev chain.

## Command line interface

After installing the project, scan the Kitaev-chain spectrum with:

```bash
toposc kitaev-scan --L 60 --mu-min -4 --mu-max 4
```

The same command is available without installation as:

```bash
python -m toposc_lab kitaev-scan --L 60 --mu-min -4 --mu-max 4
```

Use `--num-points`, `--t`, `--delta`, and `--periodic` to adjust the scan.

## Research workspace

The optional graphical workspace exposes every currently registered model and
its Pydantic parameters. It uses the same solver, observables and plotting
code as the Python API.

```bash
pip install -e ".[app]"
toposc-ui
```

The first workspace provides single-model simulations, spectra,
geometry-aware localization plots, core observables, parameter metadata and
downloadable `.npz` results. Scan and study-comparison workspaces follow as
the model library grows.

The workspace also includes a **Quantum gas laboratory**. It has separate,
fixed-particle-number calculators for a classical Maxwell--Boltzmann gas and
a Bose--Einstein gas with the standard 3D ideal-gas condensation transition.
Every calculator uses the same external
conditions, solves its chemical potential from the number equation and shows
momentum-state occupations on fixed, parameter-independent colour scales.
Start `toposc-ui` and select *Quantum-gas laboratory*.

For explicit statistical mechanics, the UI also contains **Ensembles and
dynamics**.  It separates canonical, grand-canonical and microcanonical
calculations rather than treating their control variables as interchangeable:

- classical ideal gas: fixed-variable dashboard, Poisson number fluctuations
  in the grand-canonical case, and a ballistic real-space motion sample;
- ideal Bose gas: fixed-N condensation, fixed-mu normal Bose statistics and
  an exact finite microcanonical Fock-state enumerator.

The quantum microcanonical calculation is deliberately limited to a small
one-dimensional mode set, where every allowed Fock state can be counted
exactly.  This is preferable to presenting an uncontrolled approximation as
an exact microcanonical quantum result.

## Landau-level learning laboratory

The Streamlit interface now contains a complete interactive treatment of
Section 1.4 of David Tong's *The Quantum Hall Effect*
(arXiv:1606.06687v2). Select **Quantum Hall / Landau levels** in the sidebar
to explore:

- algebraic quantization and the equally spaced Landau-level spectrum;
- a browser-native live material with continuously ramped B and E fields,
  coherent wavepackets, stationary eigenstate densities, probability-current
  arrows and simultaneous level occupations;
- a nine-step in-app physics tutorial that prepares each experiment and explains
  what to do, what to observe, the physical picture and the expected result;
- strip-like Landau-gauge wavefunctions and their guiding centres;
- electric-field dispersion, wavefunction displacement and a playable
  cyclotron / E-cross-B drift simulation;
- lowest-Landau-level ring states in symmetric gauge;
- flux-controlled degeneracy, guiding-centre noncommutativity and optional
  Zeeman splitting.

Magnetic and electric fields, effective mass, sample dimensions, quantum
numbers, g-factor and all display/simulation parameters are adjustable. Every
static scientific figure can be exported as high-resolution PNG or vector PDF,
and the full parameter set can be downloaded as JSON.

## Integer quantum Hall learning laboratory

The same Quantum Hall workspace now starts Chapter 2 of Tong's notes with a
separate **2. Integer quantum Hall effect** laboratory. Its first two roadmap
stages provide:

- explicit electron-charge and Hall-tensor sign conventions;
- the filling factor `nu = n_e h/(e B)` and finite-temperature Landau-level
  occupations obtained from the particle-number equation;
- a synchronized four-panel dashboard for occupations, broadened density of
  states, Hall plateaux and longitudinal transition peaks;
- spinless, unresolved spin-degenerate and Zeeman-resolved modes;
- a direct comparison between the classical Hall line and quantized response;
- high-resolution PNG, vector PDF and reproducible JSON parameter exports.

The next completed stage adds the edge-mode physics of Section 2.1.1:

- a smooth adjustable confinement potential and the dispersion
  `E_n(k) = E_n + V(-k l_B^2)`;
- Fermi-level crossings, group velocities and opposite chirality on the two
  sample edges;
- a browser-native skipping-orbit animation that explicitly distinguishes its
  semiclassical particles from stationary quantum eigenstates;
- the quantized edge-current relation `I_y = N (e^2/h) V_H`, including signed
  Hall bias, channel counting and publication-quality exports.

For the charge, field and tensor convention used by Tong, the signed response
is `rho_xy = -h/(nu e^2)` and `sigma_xy = +nu e^2/h`. Reversing the field
reverses both Hall signs.

The finite plateau width, Gaussian DOS broadening and longitudinal peaks are
clearly labelled as phenomenological controls in this first version. A later
roadmap stage will replace them with an explicit disorder and localization
calculation rather than presenting a visual interpolation as microscopic
physics.

## Planned features

- Kitaev chain
- BdG Hamiltonian builder
- energy spectra
- Majorana edge modes
- phase diagrams
- density of states
- Streamlit interface
