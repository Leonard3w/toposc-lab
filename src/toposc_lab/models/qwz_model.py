from __future__ import annotations
import numpy as np
from pydantic import BaseModel as PdydanticBaseModel, Field
from toposc_lab.core.model import BaseModel

class QWZModelParameters(PdydanticBaseModel):

    n_x: int = Field(..., gt=1, description="Number of lattice sites in x direction.")
    n_y: int = Field(..., gt=1, description="Number of lattice sites in y direction.")

    mass: float = Field(...,description="QWZ mass parameter m.")

    boundary_x: str = Field(default="open", description="Boundary in x direction.")
    boundary_y: str = Field(default="open", description="Boundary in y direction.")

class QWZModel(BaseModel):
    def __init__(self, params: QWZModelParameters)->None:
        self.params = params
        self._validate_boundaries()

    def _validate_boundaries(self)-> None:
        validate_boundaries = ("open", "periodic")

        if self.params.boundary_x not in validate_boundaries:
            raise ValueError("boundary_x must be either open or periodic")
        
        if self.params.boundary_y not in validate_boundaries:
            raise ValueError("boundary_y must be either open or periodic")
        
    def _site_index(self, x:int, y:int)->int:
        return x*self.params.n_y + y
    
    def hamiltonian(self)->np.ndarray:
        n_x = self.params.n_x
        n_y = self.params.n_y
        mass = self.params.mass

        sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
        sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

        dimension = 2 * n_x * n_y
        hamiltonian = np.zeros((dimension, dimension), dtype=complex)

        onsite_term = mass*sigma_z

        hopping_x = (sigma_z - 1.0j * sigma_x) / 2.0
        hopping_y = (sigma_z - 1.0j * sigma_y) / 2.0

        for x in range(n_x):
            for y in range(n_y):
                site = self._site_index(x,y)
                state = slice(2*site, 2*site +2)

                hamiltonian[state,state] += onsite_term

                if x < n_x -1:
                    next_site = self._site_index(x +1,y)
                    next_state = slice(2 * next_site, 2*next_site +2)

                    hamiltonian[state, next_state] += hopping_x
                    hamiltonian[next_state, state] += hopping_x.conj().T

                elif self.params.boundary_x == "periodic":
                    next_site = self._site_index(0, y)
                    next_state = slice(2 * next_site, 2 * next_site + 2)

                    hamiltonian[state, next_state] += hopping_x
                    hamiltonian[next_state, state] += hopping_x.conj().T

                # Kopplung zum nächsten Platz in y-Richtung.
                if y < n_y - 1:
                    next_site = self._site_index(x, y + 1)
                    next_state = slice(2 * next_site, 2 * next_site + 2)

                    hamiltonian[state, next_state] += hopping_y
                    hamiltonian[next_state, state] += hopping_y.conj().T

                # Periodischer Rand in y: letzter Platz verbindet sich mit erstem.
                elif self.params.boundary_y == "periodic":
                    next_site = self._site_index(x, 0)
                    next_state = slice(2 * next_site, 2 * next_site + 2)

                    hamiltonian[state, next_state] += hopping_y
                    hamiltonian[next_state, state] += hopping_y.conj().T

        return hamiltonian
