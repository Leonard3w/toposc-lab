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

## Planned features

- Kitaev chain
- BdG Hamiltonian builder
- energy spectra
- Majorana edge modes
- phase diagrams
- density of states
- Streamlit interface
