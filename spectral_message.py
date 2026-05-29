"""
SpectralMessage — a message IS a spectral object, not text.

Agents communicate by exchanging Laplacians.
The Laplacian encodes capability structure.
Eigenvalues are the spectral fingerprint.
The Fiedler vector is routing.
The conservation ratio is confidence.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


def build_capability_graph(capabilities: dict, state: dict) -> np.ndarray:
    """
    Build an adjacency matrix from agent capabilities and state.
    Capabilities are nodes; shared domains create edges.
    """
    caps = list(capabilities.keys())
    n = len(caps)
    if n == 0:
        return np.array([[0.0]])

    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            # Edge weight = similarity between capabilities
            ci, cj = capabilities[caps[i]], capabilities[caps[j]]
            # Shared domains strengthen edges
            shared = len(set(ci.get("domains", [])) & set(cj.get("domains", [])))
            strength = ci.get("strength", 0.5) * cj.get("strength", 0.5)
            weight = 0.1 + shared * 0.3 + strength * 0.5
            adj[i][j] = weight
            adj[j][i] = weight

        # State modulates self-connection (active capabilities have higher degree)
        activity = state.get(caps[i], 1.0)
        adj[i][i] = 0.0

    return adj


def build_laplacian(adjacency: np.ndarray) -> np.ndarray:
    """Build graph Laplacian: L = D - A"""
    D = np.diag(adjacency.sum(axis=1))
    return D - adjacency


def project_task(task: Optional[dict], eigenvectors: np.ndarray) -> np.ndarray:
    """
    Project a task onto the eigenvector basis.
    A task is a weighted combination of eigenvectors.
    """
    n = eigenvectors.shape[0]
    if task is None:
        return np.ones(n) / n

    weights = np.zeros(n)
    components = task.get("components", {})
    for i in range(n):
        weights[i] = components.get(str(i), 0.1)
    weights = weights / (np.linalg.norm(weights) + 1e-10)

    # Project onto eigenvectors
    return eigenvectors @ weights


def conservation_ratio(laplacian: np.ndarray, vector: np.ndarray) -> float:
    """
    Conservation ratio: how well the vector is conserved by the Laplacian.
    High conservation = vector lives in low-energy eigenmodes = high confidence.
    """
    n = laplacian.shape[0]
    if vector.shape[0] < n:
        vector = np.pad(vector, (0, n - vector.shape[0]))
    elif vector.shape[0] > n:
        vector = vector[:n]

    # Rayleigh quotient: v^T L v / v^T v
    # Low Rayleigh quotient = low energy = high conservation
    rayleigh = float(vector @ laplacian @ vector) / (float(vector @ vector) + 1e-10)
    # Convert to ratio (0-1): low energy → high conservation
    return 1.0 / (1.0 + rayleigh)


@dataclass
class SpectralMessage:
    """
    A message IS a spectral object, not text.
    Agents communicate by exchanging Laplacians.
    """
    laplacian: np.ndarray
    task_vector: np.ndarray
    eigenvalues: np.ndarray
    conservation_ratio: float
    fiedler_vector: np.ndarray

    @staticmethod
    def from_agent_state(capabilities: dict, task: Optional[dict], state: dict) -> "SpectralMessage":
        """Create a spectral message from agent state."""
        graph = build_capability_graph(capabilities, state)
        L = build_laplacian(graph)
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        task_vec = project_task(task, eigenvectors)
        cr = conservation_ratio(L, task_vec)
        return SpectralMessage(
            laplacian=L,
            task_vector=task_vec,
            eigenvalues=eigenvalues,
            conservation_ratio=cr,
            fiedler_vector=eigenvectors[:, 1] if eigenvectors.shape[1] > 1 else eigenvectors[:, 0],
        )

    def can_handle(self, other: "SpectralMessage") -> float:
        """
        Can this agent handle the other's request?
        Returns alignment coefficient (0-1).
        Cosine similarity of eigenvalue spectra.
        """
        e1 = self.eigenvalues / (np.linalg.norm(self.eigenvalues) + 1e-10)
        e2 = other.eigenvalues / (np.linalg.norm(other.eigenvalues) + 1e-10)
        # Pad to same length
        maxlen = max(len(e1), len(e2))
        e1p = np.pad(e1, (0, maxlen - len(e1)))
        e2p = np.pad(e2, (0, maxlen - len(e2)))
        return float(np.clip(np.dot(e1p, e2p), 0.0, 1.0))

    def compose(self, other: "SpectralMessage") -> "SpectralMessage":
        """
        Compose two agents.
        Block-diagonal Laplacian = parallel composition.
        Cross-coupling encodes interaction strength.
        """
        n1 = self.laplacian.shape[0]
        n2 = other.laplacian.shape[0]
        n = n1 + n2
        L = np.zeros((n, n))

        # Block diagonal: each agent's Laplacian
        L[:n1, :n1] = self.laplacian
        L[n1:, n1:] = other.laplacian

        # Cross-coupling: how the agents connect
        coupling = min(self.conservation_ratio, other.conservation_ratio) * 0.1
        cross_rows = min(n1, n2)
        L[:cross_rows, n1:n1 + cross_rows] = -coupling * np.eye(cross_rows)
        L[n1:n1 + cross_rows, :cross_rows] = -coupling * np.eye(cross_rows)

        eigenvalues, eigenvectors = np.linalg.eigh(L)
        combined_task = np.concatenate([self.task_vector, other.task_vector])
        return SpectralMessage(
            laplacian=L,
            task_vector=combined_task,
            eigenvalues=eigenvalues,
            conservation_ratio=conservation_ratio(L, combined_task),
            fiedler_vector=eigenvectors[:, 1] if eigenvectors.shape[1] > 1 else eigenvectors[:, 0],
        )

    def spectral_summary(self) -> str:
        """Human-readable summary of this spectral message."""
        return (
            f"  Laplacian shape: {self.laplacian.shape}\n"
            f"  Eigenvalues: {np.array2string(self.eigenvalues, precision=4, suppress_small=True)}\n"
            f"  Conservation ratio: {self.conservation_ratio:.4f}\n"
            f"  Fiedler vector (first 5): {np.array2string(self.fiedler_vector[:5], precision=4)}\n"
            f"  Task vector (first 5): {np.array2string(self.task_vector[:5], precision=4)}"
        )
