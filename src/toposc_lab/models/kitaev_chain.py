import numpy as np
from pydantic import BaseModel as PydanticBaseModel, Field

from toposc_lab.core.model import BaseModel

class KitaevChainParameters(PydanticBaseModel):

    n_sites: int = Field(...,gt=1, description="Number of lattice sites.")
    hopping: float = Field(..., description="Nearest-neighbor hopping amplitude t.")
    chemical_potential: float = Field(...,description="Chemical potential mu")
    pairing: float = Field(..., description="p-wave superconducting pairing Delta")
    boundary: str = Field(
        default="open",
        description ="Boundary condition: open or periodic",
    )

class KitaevChain(BaseModel):

    def __init__(self, params: KitaevChainParameter)->None:
        self.params = params
        self._validate_boundary()

    def _validate_boundary(self)->None:
        if self.params.boundary not in ("open", "periodic"):
            raise ValueError("boundary must be either open or periodic")
        
    def hamiltonian(self)->np.ndarray:
        n = self.params.n_sites
        t = self.params.hopping
        mu = self.params.chemical_potential
        delta = self.params.pairing

        h = np.zeros((n,n),dtype=complex)
        d = np.zeros((n,n), dtype=complex)

        for i in range(n):
            h[i,i] = -mu

        for i in range(n-1):
            h[i,i+1] = -t
            h[i+1, i] = -t

            d[i, i+1] = delta
            d[i+1, i] = -delta

        if self.params.boundary == "periodic":
            h[0, n-1] = -t
            h[n-1, 0] = -t

            d[0, n-1] = -delta
            d[n-1, 0] = delta

        upper = np.hstack((h,d))
        lower = np.hstack((-d.conj(), -h.conj()))
        bdg = np.vstack((upper, lower))

        return bdg
    
        
        