"""Curated, model-specific explanations for the research workspace.

The texts here are intentionally written by hand rather than inferred from
source code.  A formula shown in a scientific interface must state the
conventions, basis and assumptions used by the implementation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParameterGuide:
    """Meaning and numerical role of one model parameter."""

    name: str
    symbol: str
    meaning: str
    numerical_role: str


@dataclass(frozen=True)
class ModelGuide:
    """Scientific overview belonging to one registered model key."""

    title: str
    summary: str
    hamiltonian_latex: str
    basis_description: str
    construction_steps: tuple[str, ...]
    parameters: tuple[ParameterGuide, ...]
    assumptions: tuple[str, ...]
    reference: str | None = None


@dataclass(frozen=True)
class ObservableGuide:
    """Definition, implementation rule and interpretation of one observable."""

    name: str
    label: str
    formula_latex: str
    calculation: str
    interpretation: str


_BOUNDARY = ParameterGuide(
    "boundary",
    "boundary",
    "Open boundaries terminate the lattice; periodic boundaries connect its ends.",
    "Changes the bond list before the Hamiltonian is assembled.",
)


MODEL_GUIDES: dict[str, ModelGuide] = {
    "kitaev-chain": ModelGuide(
        title="Kitaev chain",
        summary="A one-dimensional spinless p-wave superconductor in BdG form.",
        hamiltonian_latex=(
            r"H=-\sum_i(\mu+\delta\mu_i)c_i^\dagger c_i"
            r"-t\sum_{\langle i,j\rangle}(c_i^\dagger c_j+\mathrm{h.c.})"
            r"+\Delta\sum_{\langle i,j\rangle}(c_i c_j+\mathrm{h.c.})"
        ),
        basis_description=(
            "BdG basis (c_0, ..., c_{N-1}, c_0^dagger, ..., c_{N-1}^dagger). "
            "The first block contains electrons and the second holes."
        ),
        construction_steps=(
            "Create a chain and its nearest-neighbour bonds from the chosen boundary.",
            "Set the diagonal normal term to -(mu + delta_mu_i).",
            "Insert -t for normal hopping and an antisymmetric +/- Delta pairing matrix.",
            "Assemble the BdG blocks [[h, Delta], [-Delta*, -h*]].",
        ),
        parameters=(
            ParameterGuide("n_sites", "N", "Number of chain sites.", "Sets the BdG dimension to 2N."),
            ParameterGuide("hopping", "t", "Nearest-neighbour kinetic hopping.", "Written as -t in the normal block."),
            ParameterGuide("chemical_potential", "mu", "Chemical potential relative to the band centre.", "Sets every clean onsite term to -mu."),
            ParameterGuide("pairing", "Delta", "Real p-wave pairing amplitude.", "Fills the antisymmetric pairing block."),
            _BOUNDARY,
            ParameterGuide("disorder_strength", "W", "Width of uniform onsite disorder.", "Samples delta_mu_i in [-W/2, W/2]."),
            ParameterGuide("disorder_seed", "seed", "Seed of the random-number generator.", "Makes a disorder realization reproducible."),
        ),
        assumptions=("spinless fermions", "non-interacting mean-field BdG theory", "real p-wave pairing"),
        reference="A. Kitaev, Physics-Uspekhi 44, 131 (2001).",
    ),
    "ssh-chain": ModelGuide(
        title="SSH chain",
        summary="A dimerized, chiral one-dimensional tight-binding insulator.",
        hamiltonian_latex=(
            r"H=\sum_n\left(v\,a_n^\dagger b_n+w\,b_n^\dagger a_{n+1}"
            r"+\mathrm{h.c.}\right)"
        ),
        basis_description="One orbital per lattice site, ordered A_0, B_0, A_1, B_1, ... .",
        construction_steps=(
            "Create 2 n_cells sites and nearest-neighbour bonds.",
            "Use v on each A_n--B_n intracell bond.",
            "Use w on each B_n--A_(n+1) intercell bond.",
            "Write both H_ij and H_ji so the matrix is Hermitian.",
        ),
        parameters=(
            ParameterGuide("n_cells", "N", "Number of A/B unit cells.", "Sets the matrix dimension to 2N."),
            ParameterGuide("v", "v", "Intracell hopping A_n <-> B_n.", "Dominant v gives the conventional trivial termination."),
            ParameterGuide("w", "w", "Intercell hopping B_n <-> A_(n+1).", "For open boundaries, v < w supports edge states."),
            _BOUNDARY,
        ),
        assumptions=("one spinless orbital per site", "nearest-neighbour hopping only", "no onsite potential"),
        reference="W. P. Su, J. R. Schrieffer, A. J. Heeger, Phys. Rev. Lett. 42, 1698 (1979).",
    ),
    "kitaev-ladder": ModelGuide(
        title="Kitaev ladder",
        summary="Several coupled p-wave Kitaev chains: a quasi-one-dimensional superconductor.",
        hamiltonian_latex=(
            r"H=\sum_{\ell}H_{\mathrm{Kitaev},\ell}"
            r"-t_\perp\sum_{\langle\ell,\ell'\rangle,j}(c_{\ell j}^\dagger c_{\ell'j}+\mathrm{h.c.})"
            r"+\Delta_\perp\sum_{\langle\ell,\ell'\rangle,j}(c_{\ell j}c_{\ell'j}+\mathrm{h.c.})"
        ),
        basis_description=(
            "BdG basis with all electron sites first and all hole sites second. "
            "A site is labeled by (leg, position)."
        ),
        construction_steps=(
            "Create longitudinal and rung bonds from LadderLattice.",
            "Apply t and Delta to longitudinal bonds; apply t_perp and Delta_perp to rungs.",
            "Add -mu on every normal onsite element.",
            "Construct the particle-hole-related BdG blocks as for the Kitaev chain.",
        ),
        parameters=(
            ParameterGuide("n_legs", "N_leg", "Number of coupled chains.", "Sets the transverse width."),
            ParameterGuide("length", "L", "Sites along every chain.", "Total spatial sites are N_leg L."),
            ParameterGuide("hopping", "t", "Hopping along a leg.", "Fills longitudinal normal bonds."),
            ParameterGuide("chemical_potential", "mu", "Uniform chemical potential.", "Sets each normal onsite entry to -mu."),
            ParameterGuide("pairing", "Delta", "p-wave pairing along a leg.", "Fills longitudinal antisymmetric pairing bonds."),
            ParameterGuide("rung_hopping", "t_perp", "Normal hopping between adjacent legs.", "Fills normal rung bonds."),
            ParameterGuide("rung_pairing", "Delta_perp", "Pairing between adjacent legs.", "Fills rung pairing bonds."),
            ParameterGuide("boundary_length", "boundary_L", "Boundary along the chains.", "Controls longitudinal wrap-around bonds."),
            ParameterGuide("boundary_legs", "boundary_perp", "Boundary across the ladder.", "Controls outer-leg wrap-around bonds."),
        ),
        assumptions=("spinless fermions", "mean-field p-wave pairing", "nearest-neighbour couplings"),
    ),
    "qwz-model": ModelGuide(
        title="Qi-Wu-Zhang model",
        summary="A two-orbital square-lattice Chern insulator.",
        hamiltonian_latex=(
            r"H(\mathbf{k})=\sin k_x\,\sigma_x+\sin k_y\,\sigma_y"
            r"+(m+\cos k_x+\cos k_y)\sigma_z"
        ),
        basis_description="Site-major basis: (orbital +, orbital -) on every square-lattice site.",
        construction_steps=(
            "Put m sigma_z on every lattice site.",
            "Use (sigma_z - i sigma_x)/2 on x-directed bonds.",
            "Use (sigma_z - i sigma_y)/2 on y-directed bonds.",
            "Add the conjugate transpose on the reversed bond to make H Hermitian.",
        ),
        parameters=(
            ParameterGuide("n_x", "N_x", "Number of sites in x direction.", "Sets the finite-system width."),
            ParameterGuide("n_y", "N_y", "Number of sites in y direction.", "Sets the finite-system height."),
            ParameterGuide("mass", "m", "Dirac mass parameter.", "Changes the onsite orbital splitting and can close the bulk gap."),
            ParameterGuide("boundary_x", "boundary_x", "Boundary in x direction.", "Open edges can reveal chiral edge states."),
            ParameterGuide("boundary_y", "boundary_y", "Boundary in y direction.", "Open edges can reveal chiral edge states."),
        ),
        assumptions=("two orbitals per site", "non-interacting particles", "unit hopping scale"),
    ),
    "bhz-model": ModelGuide(
        title="BHZ model",
        summary="A lattice quantum-spin-Hall model built from two time-reversed orbital blocks.",
        hamiltonian_latex=(
            r"H_{\mathrm{BHZ}}(\mathbf{k})=\mathrm{diag}[h(\mathbf{k}),h^*(-\mathbf{k})],"
            r"\quad h(\mathbf{k})=\sin k_x\sigma_x+\sin k_y\sigma_y"
            r"+(m+\cos k_x+\cos k_y)\sigma_z"
        ),
        basis_description="Site-major basis: (E up, H up, E down, H down) on every square-lattice site.",
        construction_steps=(
            "Build a QWZ-like 2x2 block for spin up.",
            "Build the time-reversed 2x2 block for spin down.",
            "Place both blocks on every site and on every directed square-lattice bond.",
            "Add Hermitian-conjugate reverse bonds; spin blocks remain uncoupled in this simplified model.",
        ),
        parameters=(
            ParameterGuide("n_x", "N_x", "Number of sites in x direction.", "Sets the finite-system width."),
            ParameterGuide("n_y", "N_y", "Number of sites in y direction.", "Sets the finite-system height."),
            ParameterGuide("mass", "m", "Band-inversion mass parameter.", "Controls the gap and topological regime."),
            ParameterGuide("boundary_x", "boundary_x", "Boundary in x direction.", "Open edges can show helical states."),
            ParameterGuide("boundary_y", "boundary_y", "Boundary in y direction.", "Open edges can show helical states."),
        ),
        assumptions=("four single-particle states per site", "non-interacting limit", "spin-conserving simplified BHZ lattice form"),
        reference="B. A. Bernevig, T. L. Hughes, S.-C. Zhang, Science 314, 1757 (2006).",
    ),
    "graphene": ModelGuide(
        title="Graphene",
        summary="Spinless nearest-neighbour tight binding on the honeycomb lattice.",
        hamiltonian_latex=r"H=-t\sum_{\langle i,j\rangle}(c_i^\dagger c_j+\mathrm{h.c.})",
        basis_description="Site-major A/B orbital basis on each honeycomb unit cell.",
        construction_steps=(
            "Generate the honeycomb lattice with its A-B nearest-neighbour bonds.",
            "Put -t on every A-B bond.",
            "Write the conjugate reverse element, which here is also real -t.",
            "No onsite or same-sublattice hopping term is added.",
        ),
        parameters=(
            ParameterGuide("n_x", "N_x", "Honeycomb unit cells in x direction.", "Sets the flake width."),
            ParameterGuide("n_y", "N_y", "Honeycomb unit cells in y direction.", "Sets the flake height."),
            ParameterGuide("hopping", "t", "A-B nearest-neighbour hopping.", "Sets the overall energy scale."),
            ParameterGuide("boundary_x", "boundary_x", "Boundary in x direction.", "Controls edge termination."),
            ParameterGuide("boundary_y", "boundary_y", "Boundary in y direction.", "Controls edge termination."),
        ),
        assumptions=("one spinless p_z orbital per atom", "nearest neighbours only", "no interactions or spin-orbit coupling"),
        reference="A. H. Castro Neto et al., Rev. Mod. Phys. 81, 109 (2009).",
    ),
    "haldane-model": ModelGuide(
        title="Haldane model",
        summary="A honeycomb Chern insulator with complex next-nearest-neighbour hopping and zero net flux.",
        hamiltonian_latex=(
            r"H=-t_1\sum_{\langle i,j\rangle}c_i^\dagger c_j"
            r"+t_2\sum_{\langle\!\langle i,j\rangle\!\rangle}e^{i\nu_{ij}\phi}c_i^\dagger c_j"
            r"+M\sum_i\xi_i c_i^\dagger c_i+\mathrm{h.c.}"
        ),
        basis_description="Site-major A/B orbital basis on each honeycomb unit cell.",
        construction_steps=(
            "Add +M on A sites and -M on B sites.",
            "Add real -t1 hopping to every nearest-neighbour A-B bond.",
            "Add complex t2 exp(i nu_ij phi) hopping to next-nearest neighbours.",
            "Use the conjugate amplitude on the reverse bond so that H is Hermitian.",
        ),
        parameters=(
            ParameterGuide("n_x", "N_x", "Honeycomb unit cells in x direction.", "Sets flake width."),
            ParameterGuide("n_y", "N_y", "Honeycomb unit cells in y direction.", "Sets flake height."),
            ParameterGuide("nearest_neighbor_hopping", "t_1", "Real nearest-neighbour hopping.", "Sets the graphene-like bandwidth."),
            ParameterGuide("next_nearest_neighbor_hopping", "t_2", "Magnitude of complex next-nearest hopping.", "Together with phi opens a topological gap."),
            ParameterGuide("phase", "phi", "Circulation phase of next-nearest hopping.", "Breaks time-reversal symmetry when nonzero."),
            ParameterGuide("sublattice_mass", "M", "Staggered A/B onsite mass.", "Competes with the Haldane topological mass."),
            ParameterGuide("boundary_x", "boundary_x", "Boundary in x direction.", "Open boundaries expose edge localization."),
            ParameterGuide("boundary_y", "boundary_y", "Boundary in y direction.", "Open boundaries expose edge localization."),
        ),
        assumptions=("spinless non-interacting fermions", "one orbital per honeycomb site", "complex hopping has zero net unit-cell flux"),
        reference="F. D. M. Haldane, Phys. Rev. Lett. 61, 2015 (1988).",
    ),
}


OBSERVABLE_GUIDES: tuple[ObservableGuide, ...] = (
    ObservableGuide(
        "eigenvalues",
        "Spectrum and eigenstates",
        r"H\,|\psi_n\rangle=E_n|\psi_n\rangle",
        "ExactDiagonalizationSolver diagonalizes the finite Hermitian matrix with a dense eigensolver.",
        "E_n are allowed single-particle or BdG excitation energies; columns psi_n are normalized eigenstates.",
    ),
    ObservableGuide(
        "minimum_abs_energy",
        "Minimum absolute energy",
        r"E_{\min}=\min_n |E_n|",
        "Take the absolute value of every eigenvalue and select its minimum.",
        "A small value signals a possible zero or near-zero mode; finite-size splitting must be considered.",
    ),
    ObservableGuide(
        "bulk_gap",
        "Bulk-gap estimate",
        r"\Delta_{\mathrm{bulk}}=\min\{E_n:E_n>\varepsilon\}",
        "Discard eigenvalues below the numerical tolerance epsilon and take the smallest remaining positive energy.",
        "For finite open systems it is an excitation-gap estimate; edge modes are deliberately ignored if they are near zero.",
    ),
    ObservableGuide(
        "zero_modes",
        "Zero-mode count",
        r"N_0=\sum_n \mathbf{1}(|E_n|\leq\varepsilon)",
        "Count eigenvalues whose absolute value is at most the chosen tolerance epsilon.",
        "Useful for detecting candidate Majorana or boundary zero modes; always report the tolerance.",
    ),
    ObservableGuide(
        "site_probability",
        "Site probability",
        r"p_i=\sum_\alpha |\psi_{i\alpha}|^2,\qquad \sum_i p_i=1",
        "Square the eigenvector amplitudes, sum over internal components alpha and normalize numerically.",
        "This is the spatial density used by all localization plots and edge observables.",
    ),
    ObservableGuide(
        "ipr",
        "Inverse participation ratio (IPR)",
        r"\mathrm{IPR}=\sum_i p_i^2",
        "Sum the square of the normalized site probabilities.",
        "Large IPR means a state occupies few sites; an evenly extended state over N sites has IPR about 1/N.",
    ),
    ObservableGuide(
        "participation_ratio",
        "Participation ratio",
        r"\mathrm{PR}=1/\mathrm{IPR}",
        "Take the reciprocal of the IPR.",
        "Estimates the effective number of occupied sites; it is large for extended states.",
    ),
    ObservableGuide(
        "edge_weight",
        "Edge weight",
        r"w_{\mathrm{edge}}=\sum_{i\in\partial\Lambda}p_i",
        "Add p_i on all sites within the selected edge width of at least one lattice boundary.",
        "Values near one indicate boundary localization. With periodic boundaries it is only a grid-based diagnostic.",
    ),
    ObservableGuide(
        "bulk_weight",
        "Bulk weight",
        r"w_{\mathrm{bulk}}=1-w_{\mathrm{edge}}",
        "Subtract the edge weight from the normalized total probability.",
        "Separates the interior contribution from the selected boundary region.",
    ),
    ObservableGuide(
        "ldos",
        "Local density of states (LDOS)",
        r"\rho_i(E)=\sum_n |\psi_n(i)|^2\,g_\eta(E-E_n)",
        "Weight every eigenstate density by a finite-width spectral kernel g_eta around the chosen energy.",
        "Shows where states at a selected energy live; the broadening eta controls energy resolution.",
    ),
    ObservableGuide(
        "berry_curvature",
        "Berry curvature",
        r"\Omega(\mathbf{k})=i\left(\langle\partial_{k_x}u|\partial_{k_y}u\rangle-\langle\partial_{k_y}u|\partial_{k_x}u\rangle\right)",
        "The implemented periodic k-grid method evaluates gauge-invariant overlaps around each momentum plaquette.",
        "Berry curvature is the local momentum-space contribution to a Chern number.",
    ),
    ObservableGuide(
        "chern_number",
        "Chern number",
        r"C=\frac{1}{2\pi}\int_{\mathrm{BZ}}\Omega(\mathbf{k})\,d^2k",
        "Sum the gauge-invariant Berry flux on every plaquette of the Brillouin-zone grid and round only after convergence checks.",
        "An integer bulk invariant: nonzero C predicts chiral boundary modes in a gapped Chern insulator.",
    ),
    ObservableGuide(
        "hermiticity",
        "Hermiticity check",
        r"\|H-H^\dagger\|\leq\varepsilon",
        "Compute the residual between the matrix and its conjugate transpose and compare it with tolerance epsilon.",
        "A physical closed-system Hamiltonian must pass this check; otherwise its spectrum is not guaranteed real.",
    ),
)


def model_guide(model_key: str) -> ModelGuide:
    """Return the hand-curated guide belonging to a registered model."""
    try:
        return MODEL_GUIDES[model_key]
    except KeyError as error:
        raise ValueError(f"No model guide is registered for {model_key!r}") from error


def observable_guides() -> tuple[ObservableGuide, ...]:
    """Return all observables currently documented by the platform."""
    return OBSERVABLE_GUIDES
