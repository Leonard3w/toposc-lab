from abc import ABC, abstractmethod
import numpy as np

class BaseModel (ABC):

    @abstractmethod
    def hamiltonian(self)->np.ndarray:
        raise NotImplementedError