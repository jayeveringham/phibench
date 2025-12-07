"""
Batch processing module for PhiBench.

Handles parallel Phi computation across multiple networks.
"""

import os
import warnings
os.environ['PYPHI_WELCOME_OFF'] = 'yes'
warnings.filterwarnings('ignore')  # Suppress all warnings

import pyphi

# Configure PyPhi for clean batch processing
pyphi.config.PARALLEL_CONCEPT_EVALUATION = False
pyphi.config.PARALLEL_CUT_EVALUATION = False
pyphi.config.PROGRESS_BARS = False  # Disable internal progress bars
import numpy as np
import time
from multiprocessing import Pool, cpu_count
from typing import Dict, Any, Tuple, Optional
from tqdm import tqdm

from .storage import ResultStorage


def compute_phi_for_network(args: Tuple[str, np.ndarray, np.ndarray, tuple]) -> Dict[str, Any]:
    """
    Compute Phi for a single network (designed for multiprocessing).

    Args:
        args: Tuple of (network_id, connectivity_matrix, tpm, state)

    Returns:
        Dictionary with network_id, phi, and computation_time
    """
    network_id, cm, tpm, state = args

    try:
        start_time = time.time()

        # Create PyPhi network
        network = pyphi.Network(tpm, cm)

        # Create subsystem with all nodes
        n_nodes = cm.shape[0]
        subsystem = pyphi.Subsystem(network, state, nodes=range(n_nodes))

        # Compute SIA (System Irreducibility Analysis)
        sia = pyphi.compute.sia(subsystem)

        computation_time = time.time() - start_time

        return {
            'network_id': network_id,
            'phi': sia.phi,
            'computation_time': computation_time,
            'success': True,
            'error': None,
            'cut': str(sia.cut) if hasattr(sia, 'cut') else None
        }

    except Exception as e:
        return {
            'network_id': network_id,
            'phi': None,
            'computation_time': None,
            'success': False,
            'error': str(e),
            'cut': None
        }


class BatchProcessor:
    """Manages batch processing of Phi computations."""

    def __init__(self, storage: ResultStorage, n_workers: Optional[int] = None):
        """
        Initialize batch processor.

        Args:
            storage: ResultStorage instance
            n_workers: Number of parallel workers (default: CPU count - 1)
        """
        self.storage = storage
        self._progress_callback = None

        if n_workers is None:
            # Leave one CPU free
            self.n_workers = max(1, cpu_count() - 1)
        else:
            self.n_workers = n_workers

        print(f"Batch processor initialized with {self.n_workers} workers")

    def set_progress_callback(self, callback):
        """Set callback for progress updates: callback(completed, total)"""
        self._progress_callback = callback

    def process_pending_networks(self, state: Optional[tuple] = None,
                                 max_networks: Optional[int] = None) -> Dict[str, Any]:
        """
        Process all pending networks (saved but not computed).

        Args:
            state: Network state for computation (default: all nodes ON)
            max_networks: Maximum number of networks to process (None = all)

        Returns:
            Dictionary with processing statistics
        """
        pending = self.storage.get_pending_networks()

        if max_networks is not None:
            pending = pending[:max_networks]

        if len(pending) == 0:
            print("No pending networks to process")
            return {'processed': 0, 'success': 0, 'failed': 0}

        print(f"Processing {len(pending)} networks...")

        # Prepare arguments for parallel processing
        tasks = []
        for network_id in pending:
            network_data = self.storage.load_network(network_id)
            cm = network_data['connectivity_matrix']
            tpm = network_data['tpm']

            # Use provided state or default to all OFF (reachable for all TPM rules)
            if state is None:
                network_state = tuple([0] * cm.shape[0])
            else:
                network_state = state

            tasks.append((network_id, cm, tpm, network_state))

        # Process in parallel with progress bar
        results = self._process_parallel(tasks)

        # Save results
        success_count = 0
        failed_count = 0

        for result in results:
            if result['success']:
                self.storage.save_phi_result(
                    result['network_id'],
                    result['phi'],
                    result['computation_time']
                )
                success_count += 1
            else:
                print(f"Failed: {result['network_id']} - {result['error']}")
                failed_count += 1

        return {
            'processed': len(pending),
            'success': success_count,
            'failed': failed_count
        }

    def _process_parallel(self, tasks: list) -> list:
        """
        Process tasks in parallel with progress tracking.

        Args:
            tasks: List of task arguments

        Returns:
            List of results
        """
        if self.n_workers == 1:
            # Single-threaded for debugging
            results = []
            for task in tasks:
                results.append(compute_phi_for_network(task))
            return results
        else:
            # Multi-process with progress callback
            import sys
            with Pool(self.n_workers) as pool:
                results = []
                for result in pool.imap_unordered(compute_phi_for_network, tasks, chunksize=1):
                    results.append(result)
                    if self._progress_callback:
                        self._progress_callback(len(results), len(tasks))
                        sys.stdout.flush()
            return results

    def generate_and_compute_batch(self, generator, topology: str,
                                   n_nodes: int, params: Dict,
                                   n_networks: int, tpm_rule: str = 'or',
                                   state: Optional[tuple] = None) -> Dict[str, Any]:
        """
        Generate networks and compute Phi in one batch.

        Args:
            generator: NetworkGenerator instance
            topology: Topology type
            n_nodes: Number of nodes
            params: Topology parameters
            n_networks: Number of networks to generate
            tpm_rule: TPM update rule
            state: Network state for computation

        Returns:
            Dictionary with batch statistics
        """
        # Silent batch processing

        # Generate networks
        tasks = []
        for i in range(n_networks):
            # Generate network
            if topology == 'erdos_renyi':
                cm, metadata = generator.random_erdos_renyi(n_nodes, **params)
            elif topology == 'watts_strogatz':
                cm, metadata = generator.small_world_watts_strogatz(n_nodes, **params)
            elif topology == 'barabasi_albert':
                cm, metadata = generator.scale_free_barabasi_albert(n_nodes, **params)
            elif topology == 'modular':
                cm, metadata = generator.modular_network(n_nodes, **params)
            elif topology == 'ring_lattice':
                cm, metadata = generator.regular_lattice_ring(n_nodes, **params)
            elif topology == 'fully_connected':
                cm, metadata = generator.fully_connected(n_nodes)
            elif topology == 'random_geometric':
                cm, metadata = generator.random_geometric(n_nodes, **params)
            else:
                raise ValueError(f"Unknown topology: {topology}")

            # Generate TPM
            tpm = generator.generate_tpm_deterministic(cm.shape[0], rule=tpm_rule)

            # Use provided state or default to all OFF (reachable for all TPM rules)
            if state is None:
                network_state = tuple([0] * cm.shape[0])
            else:
                network_state = state

            # Generate network ID and save
            network_id = self.storage.generate_network_id(topology, n_nodes, params)

            # Add TPM rule and state to metadata
            metadata['tpm_rule'] = tpm_rule
            metadata['state'] = network_state

            self.storage.save_network(network_id, cm, tpm, metadata)

            # Add to processing queue
            tasks.append((network_id, cm, tpm, network_state))

        # Process in parallel
        results = self._process_parallel(tasks)

        # Save results
        success_count = 0
        failed_count = 0

        for result in results:
            if result['success']:
                self.storage.save_phi_result(
                    result['network_id'],
                    result['phi'],
                    result['computation_time']
                )
                success_count += 1
            else:
                print(f"Failed: {result['network_id']} - {result['error']}")
                failed_count += 1

        return {
            'generated': n_networks,
            'success': success_count,
            'failed': failed_count
        }
