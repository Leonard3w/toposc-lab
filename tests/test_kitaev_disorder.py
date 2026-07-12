from __future__ import annotations

import numpy as np
from pydantic import BaseModel as PydanticBaseModel, Field

from toposc_lab.core.model import BaseModel


class KitaevChainParameters(PydanticBaseModel):
    """Parameter der eindimensionalen Kitaev-Kette."""

    n_sites: int = Field(..., gt=1, description="Number of lattice sites.")
    hopping: float = Field(..., description="Nearest-neighbor hopping amplitude t.")
    chemical_potential: float = Field(..., description="Chemical potential mu.")
    pairing: float = Field(..., description="p-wave superconducting pairing Delta.")
    boundary: str = Field(
        default="open",
        description="Boundary condition: open or periodic.",
    )

    # Stärke des zufälligen lokalen Potentials W.
    # W = 0 entspricht dem bisherigen sauberen Modell.
    disorder_strength: float = Field(
        default=0.0,
        ge=0.0,
        description="Strength W of onsite chemical-potential disorder.",
    )

    # Mit einem festen Seed kann dieselbe Disorder-Realisierung reproduziert werden.
    disorder_seed: int | None = Field(
        default=None,
        description="Optional seed for reproducible disorder.",
    )


class KitaevChain(BaseModel):
    """Kitaev-Kette mit optionalem zufälligem Onsite-Disorder."""

    def __init__(self, params: KitaevChainParameters) -> None:
        self.params = params
        self._validate_boundary()

        # Das Disorder-Profil wird einmal erzeugt und danach nicht verändert.
        self._disorder_profile = self._create_disorder_profile()

    def _validate_boundary(self) -> None:
        if self.params.boundary not in ("open", "periodic"):
            raise ValueError("boundary must be either open or periodic")

    def _create_disorder_profile(self) -> np.ndarray:
        """
        Erzeuge lokale Abweichungen delta_mu_i.

        Jede Stelle erhält einen Wert aus dem Intervall
        [-disorder_strength / 2, disorder_strength / 2].
        """
        if self.params.disorder_strength == 0.0:
            return np.zeros(self.params.n_sites)

        random_generator = np.random.default_rng(self.params.disorder_seed)

        return random_generator.uniform(
            low=-self.params.disorder_strength / 2.0,
            high=self.params.disorder_strength / 2.0,
            size=self.params.n_sites,
        )

    @property
    def disorder_profile(self) -> np.ndarray:
        """
        Gib das verwendete Disorder-Profil zurück.

        Eine Kopie verhindert, dass der gespeicherte Modellzustand von außen
        versehentlich verändert wird.
        """
        return self._disorder_profile.copy()

    def hamiltonian(self) -> np.ndarray:
        """Baue die Bogoliubov-de-Gennes-Hamiltonmatrix der Kitaev-Kette."""
        n = self.params.n_sites
        t = self.params.hopping
        mu = self.params.chemical_potential
        delta = self.params.pairing

        h = np.zeros((n, n), dtype=complex)
        d = np.zeros((n, n), dtype=complex)

        for i in range(n):
            # Lokales chemisches Potential: mu_i = mu + delta_mu_i.
            h[i, i] = -(mu + self._disorder_profile[i])

        for i in range(n - 1):
            h[i, i + 1] = -t
            h[i + 1, i] = -t

            d[i, i + 1] = delta
            d[i + 1, i] = -delta

        if self.params.boundary == "periodic":
            h[0, n - 1] = -t
            h[n - 1, 0] = -t

            d[0, n - 1] = -delta
            d[n - 1, 0] = delta

        upper = np.hstack((h, d))
        lower = np.hstack((-d.conj(), -h.conj()))

        return np.vstack((upper, lower))