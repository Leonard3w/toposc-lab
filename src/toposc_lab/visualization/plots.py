import matplotlib.pyplot as plt
import numpy as np

def plot_spectrum(eigenvalues: np.ndarray) -> None:
    """Plot eigenvalues as a simple energy spectrum."""
    indices = np.arange(len(eigenvalues))

    plt.figure()
    plt.scatter(indices, eigenvalues)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("State index")
    plt.ylabel("Energy")
    plt.title("Energy spectrum")
    plt.tight_layout()
    plt.show()