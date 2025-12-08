# PhiBench Implementation Plan

## Goal
Build a high-performance C/CUDA implementation of Integrated Information Theory (IIT) Φ calculation, validated against PyPhi.

---

## Phase 1: Understanding PyPhi (Week 1)

### Key Files to Study
```
pyphi/
├── compute/big_phi.py    # Main Φ computation entry point
├── compute/subsystem.py  # Subsystem analysis
├── models/subsystem.py   # Core data structures
├── partition.py          # Bipartition enumeration
├── direction.py          # Cause vs effect
├── distance.py           # EMD implementation
├── tpm.py               # Transition probability matrices
└── repertoire.py        # Cause/effect repertoires
```

### Tasks
- [ ] Read and document PyPhi's Φ algorithm flow
- [ ] Identify data structures (TPM, repertoires, partitions)
- [ ] Map computational bottlenecks
- [ ] Run PyPhi on test cases to understand I/O

---

## Phase 2: C Foundation (Week 2-3)

### Directory Structure
```
src/
├── tpm.h / tpm.c         # Transition probability matrix
├── partition.h / partition.c  # Partition enumeration
├── emd.h / emd.c         # Earth mover's distance
├── repertoire.h / repertoire.c  # Probability distributions
├── phi.h / phi.c         # Main Φ computation
└── main.c                # CLI interface

tests/
├── test_tpm.c
├── test_partition.c
├── test_emd.c
└── test_phi.c

python/
└── phicuda.py            # Python bindings for validation
```

### Tasks
- [ ] Implement TPM data structure in C
- [ ] Implement partition enumeration
- [ ] Implement EMD (Earth Mover's Distance)
- [ ] Implement cause/effect repertoire calculation
- [ ] Implement basic Φ computation (single-threaded)
- [ ] Validate against PyPhi outputs

---

## Phase 3: CUDA Parallelization (Week 4-6)

### Parallelization Strategy
1. **Partition-level parallelism**: Each CUDA thread evaluates one partition
2. **Batch-level parallelism**: Multiple networks computed simultaneously
3. **Matrix operations**: cuBLAS for probability calculations

### Files
```
cuda/
├── tpm.cu               # TPM operations on GPU
├── partition.cu         # Parallel partition evaluation
├── emd.cu              # GPU-accelerated EMD
├── phi.cu              # Main CUDA kernel
└── phi_cuda.h          # CUDA API
```

### Tasks
- [ ] Port TPM to CUDA
- [ ] Implement parallel partition evaluation kernel
- [ ] Optimize memory access patterns (coalescing)
- [ ] Implement reduction for finding MIP
- [ ] Benchmark vs CPU implementation

---

## Phase 4: Validation & Benchmarking (Week 7-8)

### Validation Suite
```python
# Generate test cases with PyPhi
for n in range(4, 16):
    for _ in range(1000):
        network = random_network(n)
        phi_pyphi = pyphi.compute.phi(network)
        save_test_case(network, phi_pyphi)

# Validate CUDA implementation
for test in load_test_cases():
    phi_cuda = phicuda.compute_phi(test.network)
    assert abs(phi_cuda - test.phi_pyphi) < 1e-10
```

### Benchmarks
- [ ] Time vs n (system size)
- [ ] Time vs network density
- [ ] Speedup vs PyPhi
- [ ] Memory usage
- [ ] Scaling across GPU architectures

---

## Phase 5: Paper & Release (Week 9-12)

### Paper Outline
1. Introduction: IIT computational bottleneck
2. Background: IIT formalism, PyPhi
3. Methods: C/CUDA implementation details
4. Results: Benchmarks, speedup factors
5. Discussion: Enabling new research directions
6. Availability: Open source release

### Release Checklist
- [ ] Clean API documentation
- [ ] Installation instructions (Linux, CUDA requirements)
- [ ] Python bindings (pip installable)
- [ ] Example scripts
- [ ] License (MIT or Apache 2.0)

---

## Key Algorithms to Implement

### 1. Transition Probability Matrix (TPM)
- 2^n × 2^n matrix for n-node system
- Entry [i,j] = P(state j | state i)

### 2. Bipartition Enumeration
- Generate all ways to split nodes into two non-empty sets
- For n nodes: Bell(n) - 1 partitions (excluding trivial)

### 3. Earth Mover's Distance (EMD)
- Optimal transport between probability distributions
- Used to measure information distance

### 4. Cause/Effect Repertoires
- Probability distribution over past/future states
- Given current state, what caused it? What will it cause?

### 5. Integrated Information (Φ)
- φ = min over partitions of: EMD(whole_repertoire, partitioned_repertoire)
- The "minimum information partition" (MIP) cuts the system at its weakest link

---

## Hardware Target

### Primary: RTX 3090
- 10,496 CUDA cores
- 24 GB GDDR6X
- 936 GB/s memory bandwidth

### Requirements
- CUDA Toolkit 11.0+
- GCC 9+ or Clang 10+
- Python 3.8+ (for validation)

---

## Success Criteria

1. **Correctness**: Match PyPhi to 1e-10 precision on all test cases
2. **Speed**: 100x+ faster than PyPhi for n ≥ 10
3. **Scalability**: Push tractable n from ~15 to ~20
4. **Usability**: Simple API, pip-installable Python bindings

---

## Risk Factors

| Risk | Mitigation |
|------|------------|
| EMD is complex to parallelize | Start with simpler distance metrics |
| Memory limits for large n | Streaming computation, memory-efficient partitioning |
| Numerical precision issues | Use double precision, careful reduction |
| PyPhi bugs (false ground truth) | Cross-reference with IIT papers |

---

## Next Steps

1. SSH into dev machine
2. Install PyPhi: `pip install pyphi`
3. Run first test: compute Φ for 4-node network
4. Begin reading PyPhi source code

