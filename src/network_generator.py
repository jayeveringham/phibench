"""
Network generation module for PhiBench.

Generates networks across multiple topology classes for IIT Phi computation.
"""

import numpy as np
import networkx as nx
from typing import Tuple, Dict, Any


class NetworkGenerator:
    """Generate networks of various topologies for Phi benchmarking."""

    def __init__(self, seed: int = None):
        """
        Initialize network generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def random_erdos_renyi(self, n: int, p: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate Erdos-Renyi random network.

        Args:
            n: Number of nodes
            p: Connection probability (0-1)

        Returns:
            Tuple of (adjacency matrix, metadata dict)
        """
        G = nx.erdos_renyi_graph(n, p, directed=True, seed=self.seed)
        cm = nx.to_numpy_array(G, dtype=int)

        metadata = {
            'topology': 'erdos_renyi',
            'n_nodes': n,
            'connection_prob': p,
            'n_edges': G.number_of_edges(),
            'density': nx.density(G)
        }

        return cm, metadata

    def small_world_watts_strogatz(self, n: int, k: int, p: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate Watts-Strogatz small-world network.

        Args:
            n: Number of nodes
            k: Each node connected to k nearest neighbors in ring topology
            p: Rewiring probability (0-1)

        Returns:
            Tuple of (adjacency matrix, metadata dict)
        """
        # Watts-Strogatz creates undirected, convert to directed
        G = nx.watts_strogatz_graph(n, k, p, seed=self.seed)
        G_directed = G.to_directed()
        cm = nx.to_numpy_array(G_directed, dtype=int)

        metadata = {
            'topology': 'watts_strogatz',
            'n_nodes': n,
            'k_neighbors': k,
            'rewire_prob': p,
            'n_edges': G_directed.number_of_edges(),
            'density': nx.density(G_directed)
        }

        return cm, metadata

    def scale_free_barabasi_albert(self, n: int, m: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate Barabasi-Albert scale-free network.

        Args:
            n: Number of nodes
            m: Number of edges to attach from new node to existing nodes

        Returns:
            Tuple of (adjacency matrix, metadata dict)
        """
        G = nx.barabasi_albert_graph(n, m, seed=self.seed)
        G_directed = G.to_directed()
        cm = nx.to_numpy_array(G_directed, dtype=int)

        metadata = {
            'topology': 'barabasi_albert',
            'n_nodes': n,
            'm_attach': m,
            'n_edges': G_directed.number_of_edges(),
            'density': nx.density(G_directed)
        }

        return cm, metadata

    def modular_network(self, n: int, n_modules: int, p_in: float, p_out: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate modular network with community structure.

        Args:
            n: Number of nodes (will be rounded to fit modules evenly)
            n_modules: Number of modules/communities
            p_in: Connection probability within modules
            p_out: Connection probability between modules

        Returns:
            Tuple of (adjacency matrix, metadata dict)
        """
        nodes_per_module = n // n_modules
        actual_n = nodes_per_module * n_modules

        # Create sizes list for each module
        sizes = [nodes_per_module] * n_modules

        # Use stochastic block model
        p_matrix = np.full((n_modules, n_modules), p_out)
        np.fill_diagonal(p_matrix, p_in)

        G = nx.stochastic_block_model(sizes, p_matrix, directed=True, seed=self.seed)
        cm = nx.to_numpy_array(G, dtype=int)

        metadata = {
            'topology': 'modular',
            'n_nodes': actual_n,
            'n_modules': n_modules,
            'p_within': p_in,
            'p_between': p_out,
            'n_edges': G.number_of_edges(),
            'density': nx.density(G)
        }

        return cm, metadata

    def regular_lattice_ring(self, n: int, k: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate regular ring lattice.

        Args:
            n: Number of nodes
            k: Each node connected to k nearest neighbors (must be even)

        Returns:
            Tuple of (adjacency matrix, metadata dict)
        """
        # Create cycle and add additional edges
        G = nx.cycle_graph(n, create_using=nx.DiGraph())

        # Add connections to k/2 neighbors on each side
        for node in range(n):
            for offset in range(1, k // 2 + 1):
                if offset > 0:  # Don't double-add direct neighbors
                    target = (node + offset) % n
                    G.add_edge(node, target)

        cm = nx.to_numpy_array(G, dtype=int)

        metadata = {
            'topology': 'ring_lattice',
            'n_nodes': n,
            'k_neighbors': k,
            'n_edges': G.number_of_edges(),
            'density': nx.density(G)
        }

        return cm, metadata

    def fully_connected(self, n: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate fully connected network (all-to-all).

        Args:
            n: Number of nodes

        Returns:
            Tuple of (adjacency matrix, metadata dict)
        """
        G = nx.complete_graph(n, create_using=nx.DiGraph())
        cm = nx.to_numpy_array(G, dtype=int)

        metadata = {
            'topology': 'fully_connected',
            'n_nodes': n,
            'n_edges': G.number_of_edges(),
            'density': 1.0
        }

        return cm, metadata

    def random_geometric(self, n: int, radius: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate random geometric network (spatially embedded).

        Args:
            n: Number of nodes
            radius: Distance threshold for connection

        Returns:
            Tuple of (adjacency matrix, metadata dict)
        """
        G = nx.random_geometric_graph(n, radius, seed=self.seed)
        G_directed = G.to_directed()
        cm = nx.to_numpy_array(G_directed, dtype=int)

        metadata = {
            'topology': 'random_geometric',
            'n_nodes': n,
            'radius': radius,
            'n_edges': G_directed.number_of_edges(),
            'density': nx.density(G_directed)
        }

        return cm, metadata

    def generate_tpm_deterministic(self, n: int, rule: str = 'or') -> np.ndarray:
        """
        Generate deterministic transition probability matrix.

        Args:
            n: Number of nodes
            rule: Update rule ('or', 'and', 'xor', 'majority')

        Returns:
            TPM array of shape (2^n, n)
        """
        tpm = np.zeros((2**n, n))

        for state in range(2**n):
            binary = [(state >> i) & 1 for i in range(n)]

            for i in range(n):
                if rule == 'or':
                    # OR of current node and neighbors
                    tpm[state, i] = min(1, binary[i] + binary[(i-1) % n])
                elif rule == 'and':
                    # AND of current node and neighbors
                    tpm[state, i] = binary[i] * binary[(i-1) % n]
                elif rule == 'xor':
                    # XOR of current node and neighbors
                    tpm[state, i] = (binary[i] + binary[(i-1) % n]) % 2
                elif rule == 'majority':
                    # Majority rule across neighbors
                    neighbor_sum = sum([binary[(i+j) % n] for j in [-1, 0, 1]])
                    tpm[state, i] = 1 if neighbor_sum >= 2 else 0
                else:
                    raise ValueError(f"Unknown rule: {rule}")

        return tpm


def get_default_parameters():
    """
    Get default parameter ranges for each topology type.

    Returns:
        Dictionary mapping topology names to parameter configurations
    """
    params = {
        'erdos_renyi': [
            {'p': 0.2},
            {'p': 0.3},
            {'p': 0.4},
            {'p': 0.5}
        ],
        'watts_strogatz': [
            {'k': 2, 'p': 0.1},
            {'k': 2, 'p': 0.3},
            {'k': 4, 'p': 0.1},
            {'k': 4, 'p': 0.3}
        ],
        'barabasi_albert': [
            {'m': 1},
            {'m': 2},
            {'m': 3}
        ],
        'modular': [
            {'n_modules': 2, 'p_in': 0.6, 'p_out': 0.1},
            {'n_modules': 3, 'p_in': 0.6, 'p_out': 0.1},
            {'n_modules': 2, 'p_in': 0.8, 'p_out': 0.2}
        ],
        'ring_lattice': [
            {'k': 2},
            {'k': 4}
        ],
        'fully_connected': [{}],
        'random_geometric': [
            {'radius': 0.3},
            {'radius': 0.4},
            {'radius': 0.5}
        ]
    }

    return params
