from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from toposc_lab.core.results import BasisLayout


class BaseModel(ABC):
    """
    Gemeinsame Schnittstelle aller physikalischen Modelle.

    Jedes Modell liefert:
    - eine Hamiltonmatrix,
    - eine Basisbeschreibung,
    - Pydantic-Parameter über self.params,
    - normalerweise eine Geometrie über self.lattice.
    """

    @property
    def model_name(self) -> str:
        """Stabiler Name des Modells für Ergebnisse und spätere Oberflächen."""
        return type(self).__name__

    @property
    def parameters(self) -> dict[str, Any]:
        """Liefere Modellparameter als JSON-freundliches Dictionary."""
        params = getattr(self, "params", None)

        if params is None:
            return {}

        model_dump = getattr(params, "model_dump", None)

        if callable(model_dump):
            return dict(model_dump(mode="json"))

        return dict(vars(params))

    @property
    def basis_layout(self) -> BasisLayout:
        """
        Beschreibe die Reihenfolge der Basiszustände.

        Alte Testmodelle ohne räumliche Basis bleiben weiterhin nutzbar.
        Ein Solver-Aufruf über solve_model() verlangt aber diese Information.
        """
        raise NotImplementedError(
            f"{self.model_name} must define a BasisLayout"
        )

    @abstractmethod
    def hamiltonian(self) -> np.ndarray:
        """Baue die hermitesche Hamiltonmatrix des Modells."""
        raise NotImplementedError