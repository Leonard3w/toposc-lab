import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Pauli matrices
# ============================================================

s0 = np.eye(2, dtype=complex)

sx = np.array([
    [0, 1],
    [1, 0]
], dtype=complex)

sy = np.array([
    [0, -1j],
    [1j, 0]
], dtype=complex)

sz = np.array([
    [1, 0],
    [0, -1]
], dtype=complex)

# Orbital Pauli matrices
t0 = np.eye(2, dtype=complex)

tx = sx.copy()
ty = sy.copy()
tz = sz.copy()


# ============================================================
# Tensor products
#
# basis:
# |orbital 1, spin up>
# |orbital 1, spin down>
# |orbital 2, spin up>
# |orbital 2, spin down>
# ============================================================

I4 = np.kron(t0, s0)

TZ = np.kron(tz, s0)

TX_SX = np.kron(tx, sx)
TX_SY = np.kron(tx, sy)
TX_SZ = np.kron(tx, sz)

SIGMA_Z = np.kron(t0, sz)


# ============================================================
# Model parameters
# ============================================================

Nz = 30

A = 1.0
B = 1.0

# Strong-TI regime for this lattice convention
M0 = 2.0

# number of outer layers used to define "surface weight"
surface_layers = 3


# ============================================================
# Build slab Hamiltonian
#
# periodic x,y
# open z
# ============================================================

def slab_hamiltonian(kx, ky, zeeman=0.0, magnetic_surface_only=False):
    """
    Returns the 4*Nz x 4*Nz slab Hamiltonian.

    zeeman:
        strength of m sigma_z

    magnetic_surface_only:
        if True, the Zeeman term is applied only on the
        outermost few layers.
    """

    dim = 4 * Nz
    H = np.zeros((dim, dim), dtype=complex)

    # --------------------------------------------------------
    # Terms that do not involve kz
    #
    # M(k) =
    # M0 - 2B(3 - cos kx - cos ky - cos kz)
    #
    # The cos(kz) part becomes inter-layer hopping.
    # --------------------------------------------------------

    mass_xy = M0 - 2 * B * (
        3
        - np.cos(kx)
        - np.cos(ky)
    )

    onsite = (
        mass_xy * TZ
        + A * np.sin(kx) * TX_SX
        + A * np.sin(ky) * TX_SY
    )

    # --------------------------------------------------------
    # z-direction hopping
    #
    # 2B cos(kz) tau_z
    # ->
    # B tau_z hopping
    #
    # A sin(kz) tau_x sigma_z
    # ->
    # -i A/2 tau_x sigma_z forward hopping
    # --------------------------------------------------------

    Tz = (
        B * TZ
        - 1j * A / 2 * TX_SZ
    )

    for z in range(Nz):

        i0 = 4 * z
        i1 = i0 + 4

        H[i0:i1, i0:i1] += onsite

        # Zeeman / magnetic term
        if zeeman != 0:

            if magnetic_surface_only:
                if (
                    z < surface_layers
                    or z >= Nz - surface_layers
                ):
                    H[i0:i1, i0:i1] += zeeman * SIGMA_Z
            else:
                H[i0:i1, i0:i1] += zeeman * SIGMA_Z

        # hopping to next layer
        if z < Nz - 1:

            j0 = 4 * (z + 1)
            j1 = j0 + 4

            H[i0:i1, j0:j1] += Tz
            H[j0:j1, i0:i1] += Tz.conj().T

    return H


# ============================================================
# Diagonalization
# ============================================================

def diagonalize(kx, ky, zeeman=0.0, magnetic_surface_only=False):

    H = slab_hamiltonian(
        kx,
        ky,
        zeeman=zeeman,
        magnetic_surface_only=magnetic_surface_only
    )

    E, V = np.linalg.eigh(H)

    return E, V


# ============================================================
# Layer-resolved probability
# ============================================================

def layer_density(state):
    """
    Converts a 4*Nz eigenvector into
    probability per z-layer.
    """

    density = np.zeros(Nz)

    for z in range(Nz):

        block = state[4*z:4*z+4]

        density[z] = np.sum(np.abs(block)**2)

    return density


# ============================================================
# Surface weight
# ============================================================

def surface_weight(state):

    density = layer_density(state)

    top = np.sum(density[:surface_layers])

    bottom = np.sum(density[-surface_layers:])

    return top + bottom


def top_bottom_weights(state):

    density = layer_density(state)

    top = np.sum(density[:surface_layers])

    bottom = np.sum(density[-surface_layers:])

    return top, bottom


# ============================================================
# 1. Band spectrum E(kx), ky=0
# ============================================================

def plot_bandstructure(
    zeeman=0.0,
    magnetic_surface_only=False,
    title=""
):

    kxs = np.linspace(-np.pi, np.pi, 161)

    all_E = []
    all_W = []

    for kx in kxs:

        E, V = diagonalize(
            kx,
            0.0,
            zeeman=zeeman,
            magnetic_surface_only=magnetic_surface_only
        )

        weights = np.array([
            surface_weight(V[:, n])
            for n in range(len(E))
        ])

        all_E.append(E)
        all_W.append(weights)

    all_E = np.array(all_E)
    all_W = np.array(all_W)

    plt.figure(figsize=(9, 6))

    # plot weakly localized states in gray
    for n in range(all_E.shape[1]):

        plt.scatter(
            kxs,
            all_E[:, n],
            c=all_W[:, n],
            cmap="viridis",
            s=5,
            vmin=0,
            vmax=1
        )

    plt.axhline(0, linewidth=0.8)

    plt.xlabel(r"$k_x$")
    plt.ylabel("Energy")

    plt.title(title)

    cbar = plt.colorbar()
    cbar.set_label("Surface weight")

    plt.ylim(-2.0, 2.0)

    plt.tight_layout()
    plt.show()


# ============================================================
# 2. Find interesting states
# ============================================================

def find_states_near_energy(
    kx,
    ky,
    target_energy=0.0,
    number=6,
    zeeman=0.0
):

    E, V = diagonalize(kx, ky, zeeman=zeeman)

    idx = np.argsort(np.abs(E - target_energy))

    return E, V, idx[:number]


# ============================================================
# 3. Plot wave function in z
# ============================================================

def plot_state_density(
    kx,
    ky,
    state_index,
    zeeman=0.0,
    title=""
):

    E, V = diagonalize(
        kx,
        ky,
        zeeman=zeeman
    )

    psi = V[:, state_index]

    density = layer_density(psi)

    top, bottom = top_bottom_weights(psi)

    z = np.arange(1, Nz + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(
        z,
        density,
        "o-"
    )

    plt.xlabel("z layer")
    plt.ylabel(r"$|\psi(z)|^2$")

    plt.title(
        title
        + "\n"
        + rf"$E={E[state_index]:.4f}$"
        + f", top={top:.3f}, bottom={bottom:.3f}"
    )

    plt.grid(alpha=0.25)

    plt.tight_layout()
    plt.show()


# ============================================================
# 4. Automatically choose interesting examples
# ============================================================

def show_interesting_states(kx=0.15, ky=0.0):

    E, V = diagonalize(kx, ky)

    weights = np.array([
        surface_weight(V[:, n])
        for n in range(len(E))
    ])

    # --------------------------------------------------------
    # Surface state above Dirac point
    # --------------------------------------------------------

    positive = np.where(E > 0)[0]

    surf_pos = positive[
        np.argmax(weights[positive] / (1 + 5*np.abs(E[positive])))
    ]

    # --------------------------------------------------------
    # Surface state below Dirac point
    # --------------------------------------------------------

    negative = np.where(E < 0)[0]

    surf_neg = negative[
        np.argmax(weights[negative] / (1 + 5*np.abs(E[negative])))
    ]

    # --------------------------------------------------------
    # Bulk conduction example
    # choose a positive-energy state with small surface weight
    # --------------------------------------------------------

    bulk_positive = positive[np.argsort(weights[positive])]

    bulk_c = bulk_positive[
        np.argmin(
            np.abs(E[bulk_positive] - 1.0)
        )
    ]

    # --------------------------------------------------------
    # Bulk valence example
    # --------------------------------------------------------

    bulk_negative = negative[np.argsort(weights[negative])]

    bulk_v = bulk_negative[
        np.argmin(
            np.abs(E[bulk_negative] + 1.0)
        )
    ]

    examples = [
        (surf_pos, "Surface state above Dirac point"),
        (surf_neg, "Surface state below Dirac point"),
        (bulk_c, "Bulk conduction state"),
        (bulk_v, "Bulk valence state"),
    ]

    for idx, label in examples:

        plot_state_density(
            kx,
            ky,
            idx,
            title=label
        )


# ============================================================
# 5. 3D Dirac cone
# ============================================================

def dirac_cone_surface(
    zeeman=0.0,
    kmax=0.6,
    nk=31
):

    k_values = np.linspace(-kmax, kmax, nk)

    upper = np.zeros((nk, nk))
    lower = np.zeros((nk, nk))

    for ix, kx in enumerate(k_values):

        for iy, ky in enumerate(k_values):

            E, V = diagonalize(
                kx,
                ky,
                zeeman=zeeman
            )

            W = np.array([
                surface_weight(V[:, n])
                for n in range(len(E))
            ])

            # Only consider strongly surface-localized states
            candidates = np.where(W > 0.45)[0]

            if len(candidates) < 2:
                # fallback to states closest to zero
                candidates = np.argsort(np.abs(E))[:8]

            candidate_E = E[candidates]

            positive = candidate_E[candidate_E >= 0]
            negative = candidate_E[candidate_E <= 0]

            if len(positive) > 0:
                upper[iy, ix] = np.min(positive)
            else:
                upper[iy, ix] = np.nan

            if len(negative) > 0:
                lower[iy, ix] = np.max(negative)
            else:
                lower[iy, ix] = np.nan

    KX, KY = np.meshgrid(
        k_values,
        k_values
    )

    fig = plt.figure(figsize=(9, 7))

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    ax.plot_surface(
        KX,
        KY,
        upper,
        alpha=0.8
    )

    ax.plot_surface(
        KX,
        KY,
        lower,
        alpha=0.8
    )

    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    ax.set_zlabel("Energy")

    if zeeman == 0:
        ax.set_title("Surface Dirac cone")
    else:
        ax.set_title(
            f"Surface Dirac cone with Zeeman field m={zeeman}"
        )

    plt.tight_layout()
    plt.show()


# ============================================================
# 6. Compare gap with magnetic field
# ============================================================

def magnetic_gap_comparison():

    fields = [
        0.0,
        0.15,
        0.30
    ]

    kxs = np.linspace(-0.7, 0.7, 121)

    plt.figure(figsize=(9, 6))

    for m in fields:

        upper = []
        lower = []

        for kx in kxs:

            E, V = diagonalize(
                kx,
                0,
                zeeman=m
            )

            W = np.array([
                surface_weight(V[:, n])
                for n in range(len(E))
            ])

            candidate = np.where(W > 0.45)[0]

            if len(candidate) < 2:
                candidate = np.argsort(np.abs(E))[:8]

            Ec = E[candidate]

            pos = Ec[Ec >= 0]
            neg = Ec[Ec <= 0]

            upper.append(
                np.min(pos)
                if len(pos) else np.nan
            )

            lower.append(
                np.max(neg)
                if len(neg) else np.nan
            )

        plt.plot(
            kxs,
            upper,
            label=f"m={m}"
        )

        plt.plot(
            kxs,
            lower
        )

    plt.xlabel(r"$k_x$")
    plt.ylabel("Energy")

    plt.title(
        "Opening of the surface Dirac gap"
    )

    plt.legend()

    plt.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # A. Full slab band structure
    # --------------------------------------------------------

    plot_bandstructure(
        title="3D TI slab: surface states inside the bulk gap"
    )

    # --------------------------------------------------------
    # B. Wave functions of representative states
    # --------------------------------------------------------

    show_interesting_states(
        kx=0.18,
        ky=0.0
    )

    # --------------------------------------------------------
    # C. 3D surface Dirac cone
    # --------------------------------------------------------

    dirac_cone_surface(
        zeeman=0.0
    )

    # --------------------------------------------------------
    # D. Add magnetic Zeeman field
    # --------------------------------------------------------

    plot_bandstructure(
        zeeman=0.25,
        title="TI slab with Zeeman field"
    )

    # --------------------------------------------------------
    # E. Gapped Dirac cone
    # --------------------------------------------------------

    dirac_cone_surface(
        zeeman=0.25
    )

    # --------------------------------------------------------
    # F. Direct comparison of several magnetic fields
    # --------------------------------------------------------

    magnetic_gap_comparison()