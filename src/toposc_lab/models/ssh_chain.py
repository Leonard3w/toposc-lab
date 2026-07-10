from __future__ import annotations

import numpy as np
from pydantic import BaseModel as PydanticBaseModel, Field

from toposc_lab.core.model import BaseModel

class SSHChainParameters(PydanticBaseModel):
        n_cells: int = Field(..., gt=1, description="Number of unit cells.")
        v: float = Field(..., description="Intracell hopping amplitude.")
        w: float = Field(..., description="Intercell hopping amplitude.")
        boundary: str = Field(default="open", description="Boundary: open or periodic.")

class SSHChain(BaseModel):
        def __init__(self, params: SSHChainParameters)->None:
                self.params = params

                if params.boundary not in ("open", "periodic"):
                        raise ValueError("boundary must be either open or periodic")
                
        
        def hamiltonian(self)->np.ndarray:
                n_cells = self.params.n_cells
                v = self.params.v
                w = self.params.w

                hamiltonian = np.zeros((2*n_cells, 2*n_cells), dtype = float)

                for cell in range(n_cells):
                        a = 2*cell
                        b = a + 1

                        hamiltonian[a, b] = v
                        hamiltonian[b,a] = v

                        if cell < n_cells - 1:
                            next_a = 2*(cell +1)
                            hamiltonian[b, next_a] = w
                            hamiltonian[next_a, b] = w

                if self.params.boundary == "periodic":
                       last_b = 2*n_cells -1
                       first_a = 0
                       hamiltonian[last_b, first_a] = w
                       hamiltonian[first_a, last_b] = w

                return hamiltonian
                       
            
                
   