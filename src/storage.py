"""
Result storage module for PhiBench.

Handles saving network structures, Phi values, and metadata.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class ResultStorage:
    """Manages storage of benchmark results."""

    def __init__(self, base_dir: str = "results"):
        """
        Initialize result storage.

        Args:
            base_dir: Base directory for storing results
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

        # Create subdirectories
        self.networks_dir = self.base_dir / "networks"
        self.results_dir = self.base_dir / "phi_results"
        self.metadata_dir = self.base_dir / "metadata"

        for dir_path in [self.networks_dir, self.results_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)

        # Initialize results dataframe
        self.results_file = self.results_dir / "all_results.csv"
        self.results = self._load_or_create_results()

    def _load_or_create_results(self) -> pd.DataFrame:
        """Load existing results or create new dataframe."""
        if self.results_file.exists():
            return pd.read_csv(self.results_file)
        else:
            return pd.DataFrame(columns=[
                'network_id',
                'topology',
                'n_nodes',
                'n_edges',
                'density',
                'phi',
                'computation_time',
                'timestamp',
                'tpm_rule',
                'state'
            ])

    def generate_network_id(self, topology: str, n_nodes: int, params: Dict) -> str:
        """
        Generate unique network ID.

        Args:
            topology: Topology type
            n_nodes: Number of nodes
            params: Topology parameters

        Returns:
            Unique network identifier
        """
        param_str = "_".join([f"{k}{v}" for k, v in sorted(params.items())])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{topology}_n{n_nodes}_{param_str}_{timestamp}"

    def save_network(self, network_id: str, cm: np.ndarray, tpm: np.ndarray,
                     metadata: Dict[str, Any]) -> None:
        """
        Save network structure to disk.

        Args:
            network_id: Unique network identifier
            cm: Connectivity matrix
            tpm: Transition probability matrix
            metadata: Network metadata
        """
        network_file = self.networks_dir / f"{network_id}.npz"

        np.savez_compressed(
            network_file,
            connectivity_matrix=cm,
            tpm=tpm,
            **metadata
        )

        # Also save metadata as JSON for easy inspection
        metadata_file = self.metadata_dir / f"{network_id}.json"
        with open(metadata_file, 'w') as f:
            # Convert numpy types to native Python types for JSON
            metadata_clean = {k: (v.item() if isinstance(v, np.generic) else v)
                             for k, v in metadata.items()}
            json.dump(metadata_clean, f, indent=2)

    def save_phi_result(self, network_id: str, phi: float, computation_time: float,
                        sia_info: Optional[Dict] = None) -> None:
        """
        Save Phi computation result.

        Args:
            network_id: Network identifier
            phi: Computed Phi value
            computation_time: Time taken to compute (seconds)
            sia_info: Optional additional SIA information
        """
        # Load network metadata to get topology info
        network_file = self.networks_dir / f"{network_id}.npz"
        if not network_file.exists():
            raise ValueError(f"Network file not found: {network_id}")

        data = np.load(network_file, allow_pickle=True)

        # Create result row
        result = {
            'network_id': network_id,
            'topology': str(data['topology']),
            'n_nodes': int(data['n_nodes']),
            'n_edges': int(data['n_edges']),
            'density': float(data['density']),
            'phi': phi,
            'computation_time': computation_time,
            'timestamp': datetime.now().isoformat(),
            'tpm_rule': str(data.get('tpm_rule', 'unknown')),
            'state': str(data.get('state', 'unknown'))
        }

        # Append to results
        self.results = pd.concat([self.results, pd.DataFrame([result])],
                                 ignore_index=True)

        # Save updated results
        self.results.to_csv(self.results_file, index=False)

        # If detailed SIA info provided, save separately
        if sia_info is not None:
            sia_file = self.results_dir / f"{network_id}_sia.json"
            with open(sia_file, 'w') as f:
                json.dump(sia_info, f, indent=2)

    def load_network(self, network_id: str) -> Dict[str, Any]:
        """
        Load network from disk.

        Args:
            network_id: Network identifier

        Returns:
            Dictionary with connectivity matrix, TPM, and metadata
        """
        network_file = self.networks_dir / f"{network_id}.npz"

        if not network_file.exists():
            raise ValueError(f"Network not found: {network_id}")

        data = np.load(network_file, allow_pickle=True)

        return {
            'connectivity_matrix': data['connectivity_matrix'],
            'tpm': data['tpm'],
            'metadata': {k: data[k].item() if data[k].shape == ()
                        else data[k] for k in data.files
                        if k not in ['connectivity_matrix', 'tpm']}
        }

    def get_results_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of results.

        Returns:
            DataFrame with summary statistics grouped by topology and size
        """
        if len(self.results) == 0:
            return pd.DataFrame()

        summary = self.results.groupby(['topology', 'n_nodes']).agg({
            'phi': ['count', 'mean', 'std', 'min', 'max'],
            'computation_time': ['mean', 'std']
        }).round(4)

        return summary

    def get_pending_networks(self) -> list:
        """
        Get list of networks that have been saved but not computed.

        Returns:
            List of network IDs without Phi results
        """
        all_networks = set([f.stem for f in self.networks_dir.glob("*.npz")])
        computed_networks = set(self.results['network_id'].values)

        return sorted(list(all_networks - computed_networks))

    def export_to_parquet(self, filename: Optional[str] = None) -> Path:
        """
        Export results to Parquet format for efficient storage.

        Args:
            filename: Optional output filename

        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"phi_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"

        output_path = self.base_dir / filename
        self.results.to_parquet(output_path, index=False)

        return output_path
