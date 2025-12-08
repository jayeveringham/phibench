# PhiBench

Systematic benchmark of Integrated Information Theory (IIT) Phi values across network topologies and TPM rules, plus a **high-performance C++ implementation** validated against PyPhi.

## C++/CUDA Implementation (NEW)

A C++ implementation of IIT Phi computation, designed for GPU acceleration:

```bash
cd cuda
g++ -std=c++20 -O2 -I include -o test_validation tests/test_validation.cpp
./test_validation  # 24/24 tests pass
```

**Status:** Phase 1 (CPU) complete - 3240/3240 PyPhi validation tests pass (~2e-6 precision)

See [PLAN.md](PLAN.md) for implementation details and roadmap.

---

## PyPhi Benchmark Results

This project uses [PyPhi](https://github.com/wmayner/pyphi) to compute exact integrated information. If you use this benchmark or PyPhi in your research, please cite:

> Mayner WGP, Marshall W, Albantakis L, Findlay G, Marchman R, Tononi G (2018) PyPhi: A toolbox for integrated information theory. PLOS Computational Biology 14(7): e1006343. https://doi.org/10.1371/journal.pcbi.1006343

## Executive Summary

**Benchmark scope:** 20,100 networks | 6 topologies | 4 TPM rules | 5 node sizes (n=4-8) | 298 compute hours

### Primary Findings

1. **TPM rule explains 2.4x more variance than topology** (ANOVA eta²: 0.075 vs 0.031). Majority produces mean phi of 1.57 vs 0.10 (AND) and 0.04 (OR).

2. **Both factors act as hard gates.** XOR produces 0% integration across all topologies. Barabasi_albert and modular produce 0% across all TPM rules. Ring_lattice guarantees 100% non-zero for AND/OR/majority.

3. **Strong interaction effects.** Combined, TPM rule and topology explain only ~11% of variance. The majority + watts_strogatz combination produces mean phi of 3.76, while majority + ring_lattice produces only 0.25.

4. **Optimal combination: majority + small-world.** Watts_strogatz with majority rule achieves the highest mean phi (3.76) and max phi (24.28).

### Results by TPM Rule

| TPM Rule | Networks | Non-zero Rate | Mean Phi | Max Phi |
|----------|----------|---------------|----------|---------|
| majority | 4,500 | 33.3% | 1.567 | 24.280 |
| AND | 5,200 | 27.9% | 0.105 | 0.375 |
| OR | 5,200 | 27.9% | 0.039 | 0.139 |
| XOR | 5,200 | 0.0% | 0.000 | 0.000 |

### Results by Topology

| Topology | Networks | Non-zero Rate | Mean Phi (non-zero) | Max Phi |
|----------|----------|---------------|---------------------|---------|
| ring_lattice | 1,700 | 73.5% | 0.255 | 0.375 |
| watts_strogatz | 5,850 | 40.2% | 2.323 | 24.280 |
| random_geometric | 2,850 | 21.1% | 3.168 | 24.280 |
| erdos_renyi | 4,000 | 5.0% | 0.586 | 1.592 |
| modular | 2,850 | 0.0% | - | 0.000 |
| barabasi_albert | 2,850 | 0.0% | - | 0.000 |

### Computation Time by Node Size

| Nodes | Mean Time | Max Time | Non-zero Rate |
|-------|-----------|----------|---------------|
| n=4 | 0.04s | 0.4s | 20.7% |
| n=5 | 0.7s | 7s | 33.3% |
| n=6 | 5.5s | 124s | 20.8% |
| n=7 | 103s | 58 min | 20.8% |
| n=8 | 192s | 89 min | 14.3% |

### Implications

- **For IIT research:** TPM rule explains more variance than topology (2.4x), but both act as hard constraints. Neither factor alone predicts phi well (~11% combined), suggesting interaction effects dominate.
- **For consciousness theory:** XOR dynamics produce zero integration regardless of topology, consistent with IIT's requirement for causal asymmetry. Parity-preserving dynamics cannot support integration.
- **For computational neuroscience:** The majority + watts_strogatz combination produces highest phi. Whether this reflects neural reality requires validation against biological network data.

## Quick Start

```bash
# Clone and setup
git clone <repo>
cd phibench
python3 -m venv venv
source venv/bin/activate
pip install pyphi numpy networkx pandas scipy

# Run benchmark
python run_benchmark.py --config config.json --results-dir results

# Analyze results
python analyze_results.py --export-json
```

## What This Does

Computes exact Phi (integrated information) across:
- **6 network topologies**: erdos_renyi, watts_strogatz, barabasi_albert, modular, ring_lattice, random_geometric
- **4 TPM rules**: OR, AND, XOR, majority
- **5 node sizes**: 4, 5, 6, 7, 8
- **50 networks per configuration**

Total: 20,100 networks (some configurations produce fewer due to topology constraints)

## Project Structure

```
phibench/
  README.md              # This file
  PLAN.md                # C++ implementation roadmap
  config.json            # Benchmark configuration
  run_benchmark.py       # Main benchmark runner
  analyze_results.py     # Analysis and visualization data export
  src/
    network_generator.py # Topology generators
    batch_processor.py   # Parallel phi computation
    storage.py           # Results storage
  results/
    phi_results/
      all_results.csv    # Combined results (20,100 networks)
    visualization.html   # Interactive results dashboard
  cuda/                  # C++/CUDA implementation
    include/phi/         # Header-only library
      core/types.hpp     # Core types (NodeSet, StateIndex, Real)
      data/              # TPM, Repertoire, Network, Subsystem
      partition/         # Bipartition enumeration
      compute/           # SmallPhi, BigPhi, EMD algorithms
    tests/               # Validation against PyPhi
```

## Configuration

Edit `config.json` to customize:

```json
{
  "node_sizes": [4, 5, 6, 7, 8],
  "networks_per_config": 50,
  "tpm_rules": ["or", "and", "xor", "majority"],
  "topologies": { ... }
}
```

## Analysis

Run the analysis script to generate statistics and visualization data:

```bash
python analyze_results.py                    # Print full report
python analyze_results.py --export-json      # Export JSON for visualization
python analyze_results.py --quiet            # JSON only, no console output
```

View the interactive dashboard:
```bash
cd results && python -m http.server 8888
# Open http://localhost:8888/visualization.html
```

## Known Limitations

1. **majority n=8** is computationally expensive (up to 89 min per network)
2. **XOR + odd nodes + all-ON state** is unreachable due to parity constraints (benchmark uses all-OFF state)
3. **modular topology** requires n_nodes divisible by n_modules

## Dependencies

### Python (PyPhi benchmark)
- Python 3.10+
- pyphi
- numpy
- networkx
- pandas
- scipy

### C++ Implementation
- C++20 compiler (g++ 10+ or clang++ 11+)
- CUDA Toolkit 11.0+ (for Phase 2 GPU acceleration)

## References

Mayner WGP, Marshall W, Albantakis L, Findlay G, Marchman R, Tononi G (2018) PyPhi: A toolbox for integrated information theory. PLOS Computational Biology 14(7): e1006343. https://doi.org/10.1371/journal.pcbi.1006343

Tononi G (2004) An information integration theory of consciousness. BMC Neuroscience 5:42. https://doi.org/10.1186/1471-2202-5-42

Oizumi M, Albantakis L, Tononi G (2014) From the Phenomenology to the Mechanisms of Consciousness: Integrated Information Theory 3.0. PLOS Computational Biology 10(5): e1003588. https://doi.org/10.1371/journal.pcbi.1003588

## License

MIT
