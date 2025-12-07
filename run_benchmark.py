#!/usr/bin/env python3
"""
Main benchmark runner for PhiBench.

Generates networks across topologies and computes exact Phi values.
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import json
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta

from src import NetworkGenerator, ResultStorage, BatchProcessor, get_default_parameters


class LiveProgress:
    """Background thread for live progress updates with weighted ETA."""

    # Expected relative computation time by node count (based on v2 data, excl. fully_connected)
    # n=4: 0.04s, n=5: 0.53s, n=6: 2.47s, n=7: 29.3s, n=8: 20.4s
    NODE_WEIGHTS = {4: 1, 5: 14, 6: 67, 7: 797, 8: 556}

    def __init__(self):
        self.current = 0
        self.total = 1000
        self.topology = ""
        self.n_nodes = 0
        self.start_time = None
        self.running = False
        self._thread = None
        self._lock = threading.Lock()
        self.completed_weight = 0
        self.total_weight = 0
        self.node_counts = {}  # Track networks per node size

    def start(self, start_time, total, node_sizes, networks_per_size):
        self.start_time = start_time
        self.total = total
        self.running = True
        # Calculate total expected weight
        self.total_weight = sum(
            self.NODE_WEIGHTS.get(n, 100) * networks_per_size.get(n, 0)
            for n in node_sizes
        )
        self.node_counts = {n: 0 for n in node_sizes}
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=1)

    def update(self, current, topology, n_nodes):
        with self._lock:
            # Update weighted progress
            if n_nodes in self.node_counts:
                old_count = self.node_counts[n_nodes]
                new_for_size = current - sum(self.node_counts.values()) + old_count
                if new_for_size > old_count:
                    added = new_for_size - old_count
                    self.completed_weight += added * self.NODE_WEIGHTS.get(n_nodes, 100)
                    self.node_counts[n_nodes] = new_for_size
            self.current = current
            self.topology = topology
            self.n_nodes = n_nodes

    def _update_loop(self):
        while self.running:
            self._print_status()
            time.sleep(0.5)

    def _print_status(self):
        with self._lock:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            elapsed_str = str(timedelta(seconds=int(elapsed)))

            # Use weighted progress for ETA
            if elapsed > 0 and self.completed_weight > 0 and self.total_weight > 0:
                weight_rate = self.completed_weight / elapsed
                remaining_weight = self.total_weight - self.completed_weight
                remaining_sec = remaining_weight / weight_rate if weight_rate > 0 else 0
                eta = datetime.now() + timedelta(seconds=remaining_sec)
                eta_str = eta.strftime('%H:%M:%S')
                rate = self.current / elapsed
            else:
                rate = 0
                eta_str = "--:--:--"

            pct = 100 * self.current / self.total if self.total > 0 else 0
            topo = self.topology[:12] if self.topology else "starting"
            print(f"\r[{pct:5.1f}%] {self.current}/{self.total} | {topo:12} n={self.n_nodes} | {rate:5.1f}/s | ETA {eta_str} | {elapsed_str}     ", end="", flush=True)


class BenchmarkRunner:
    """Main benchmark orchestrator with checkpointing."""

    def __init__(self, config_file: str = None, results_dir: str = "results"):
        """
        Initialize benchmark runner.

        Args:
            config_file: Path to configuration JSON file
            results_dir: Directory for storing results
        """
        self.storage = ResultStorage(results_dir)
        self.generator = NetworkGenerator(seed=42)

        # Load or create configuration
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self._default_config()

        # Save config
        config_path = Path(results_dir) / "benchmark_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        print(f"Benchmark configuration saved to: {config_path}")

    def _default_config(self) -> dict:
        """Create default benchmark configuration."""
        return {
            'node_sizes': list(range(4, 9)),  # Start with 4-8 nodes for testing
            'networks_per_config': 10,  # Small test batch
            'tpm_rules': ['or'],  # v2: supports multiple rules
            'state': None,  # All nodes ON
            'n_workers': None,  # Auto-detect
            'topologies': get_default_parameters()
        }

    def run_full_benchmark(self):
        """Run complete benchmark across all configurations."""
        print("=" * 70)
        print("PhiBench - Systematic IIT Approximation Validation")
        print("=" * 70)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Initialize processor
        processor = BatchProcessor(self.storage, n_workers=self.config['n_workers'])

        # Support both old 'tpm_rule' and new 'tpm_rules' config
        tpm_rules = self.config.get('tpm_rules', [self.config.get('tpm_rule', 'or')])
        if isinstance(tpm_rules, str):
            tpm_rules = [tpm_rules]

        total_configs = 0
        completed_configs = 0

        # Count total configurations (rules x sizes x topologies x params)
        for n_nodes in self.config['node_sizes']:
            for topology, param_sets in self.config['topologies'].items():
                # Filter params that work for this size
                valid_params = self._filter_params(topology, param_sets, n_nodes)
                total_configs += len(valid_params) * len(tpm_rules)

        print(f"TPM rules: {tpm_rules}")
        print(f"Total configurations to process: {total_configs}")
        print(f"Networks per configuration: {self.config['networks_per_config']}")
        print(f"Total networks: {total_configs * self.config['networks_per_config']}")
        print()

        start_time = datetime.now()
        total_networks_target = total_configs * self.config['networks_per_config']

        # Calculate networks per node size for weighted ETA
        networks_per_size = {}
        for n_nodes in self.config['node_sizes']:
            count = 0
            for topology, param_sets in self.config['topologies'].items():
                valid_params = self._filter_params(topology, param_sets, n_nodes)
                count += len(valid_params) * len(tpm_rules) * self.config['networks_per_config']
            networks_per_size[n_nodes] = count

        # Start live progress display
        progress = LiveProgress()
        progress.start(start_time, total_networks_target, self.config['node_sizes'], networks_per_size)

        total_networks_so_far = 0

        # Process each configuration: rule x size x topology x params
        for tpm_rule in tpm_rules:
            for n_nodes in self.config['node_sizes']:
                for topology, param_sets in self.config['topologies'].items():
                    # Filter params that work for this size
                    valid_params = self._filter_params(topology, param_sets, n_nodes)

                    if not valid_params:
                        continue

                    for params in valid_params:
                        completed_configs += 1

                        # Update progress display with current task
                        progress.update(total_networks_so_far, f"{tpm_rule}:{topology}", n_nodes)

                        result = processor.generate_and_compute_batch(
                            self.generator,
                            topology,
                            n_nodes,
                            params,
                            self.config['networks_per_config'],
                            tpm_rule=tpm_rule,
                            state=self.config['state']
                        )

                        total_networks_so_far += result['success']
                        progress.update(total_networks_so_far, f"{tpm_rule}:{topology}", n_nodes)

        # Stop progress display
        progress.stop()

        # Final summary
        self._print_final_summary(start_time)

    def _filter_params(self, topology, param_sets, n_nodes):
        """Filter topology parameters that work for given node count."""
        if topology == 'ring_lattice':
            return [p for p in param_sets if p['k'] < n_nodes]
        elif topology == 'barabasi_albert':
            return [p for p in param_sets if p['m'] < n_nodes]
        elif topology == 'modular':
            return [p for p in param_sets if n_nodes >= p['n_modules'] * 2]
        return param_sets

    def _print_final_summary(self, start_time):
        """Print final benchmark summary."""
        print()
        print("=" * 70)
        print("Benchmark Complete")
        print("=" * 70)

        summary = self.storage.get_results_summary()
        print("\nResults Summary:")
        print(summary)

        elapsed = datetime.now() - start_time
        print(f"\nTotal time: {elapsed}")
        print(f"Results saved to: {self.storage.base_dir}")

        # Export to Parquet
        parquet_file = self.storage.export_to_parquet()
        print(f"Exported to: {parquet_file}")

    def _print_progress(self, completed: int, total: int, start_time: datetime):
        """Print progress summary (overwrites in place)."""
        elapsed = datetime.now() - start_time
        elapsed_sec = elapsed.total_seconds()
        configs_per_sec = completed / elapsed_sec if elapsed_sec > 0 else 0
        remaining_sec = (total - completed) / configs_per_sec if configs_per_sec > 0 else 0

        networks_done = len(self.storage.results)
        networks_total = total * self.config['networks_per_config']
        networks_per_sec = networks_done / elapsed_sec if elapsed_sec > 0 else 0

        # Move cursor up 6 lines if not first time
        if hasattr(self, '_progress_printed') and self._progress_printed:
            print("\033[6A", end="")  # Move up 6 lines
        self._progress_printed = True

        print(f"  ╔══════════════════════════════════════════════════════════════════╗")
        print(f"  ║ Progress: Config {completed}/{total} ({100 * completed / total:.1f}%) | "
              f"Networks {networks_done}/{networks_total} ({100 * networks_done / networks_total:.1f}%) ║")
        print(f"  ╠══════════════════════════════════════════════════════════════════╣")
        print(f"  ║ Elapsed: {str(elapsed).split('.')[0]:>20} | "
              f"Rate: {networks_per_sec:>6.2f} networks/sec ║")

        if remaining_sec > 0:
            remaining_min = int(remaining_sec / 60)
            eta_time = start_time + elapsed + timedelta(seconds=remaining_sec)
            print(f"  ║ Remaining: ~{remaining_min:>3} minutes | "
                  f"ETA: {eta_time.strftime('%H:%M:%S'):>19} ║")
        else:
            print(f"  ║ {'Almost done...':^64} ║")

        print(f"  ╚══════════════════════════════════════════════════════════════════╝")

    def process_pending(self):
        """Process any networks that were generated but not computed."""
        processor = BatchProcessor(self.storage, n_workers=self.config['n_workers'])
        result = processor.process_pending_networks()

        print(f"Processed {result['processed']} pending networks")
        print(f"  Success: {result['success']}")
        print(f"  Failed: {result['failed']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PhiBench - Systematic IIT Phi approximation validation"
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration JSON file'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory for storing results (default: results)'
    )
    parser.add_argument(
        '--pending-only',
        action='store_true',
        help='Only process pending networks (already generated)'
    )

    args = parser.parse_args()

    runner = BenchmarkRunner(args.config, args.results_dir)

    if args.pending_only:
        runner.process_pending()
    else:
        runner.run_full_benchmark()


if __name__ == "__main__":
    main()
