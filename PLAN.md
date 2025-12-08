# PhiBench: C++/CUDA Implementation Plan

## Project Evolution

PhiBench started as a PyPhi benchmarking project (20,100 networks, 298 compute hours). Now evolving into the **first high-performance C++/CUDA implementation of IIT Φ**.

---

## Why This Matters

- **No C++/CUDA implementation exists** - confirmed via extensive search
- PyPhi (Python) is the only real implementation, limited to n≈12-15 nodes
- IIT researchers are bottlenecked by compute
- A 100x speedup would enable research that's currently impossible

---

## Reference Materials (git ignored)

| Directory | Source | Purpose |
|-----------|--------|---------|
| `pyphi/` | github.com/wmayner/pyphi | Reference implementation, validation |
| `iit-pseudocode/` | github.com/CSC-UW/iit-pseudocode | Algorithm structure |

---

## Algorithm Overview (from pseudocode)

### Core Functions

```
BigPhi(Candidate Set C):
    for each unidirectional partition Z of C:
        distance = XEMD(ConceptualStructure(C), ConceptualStructure(C'))
    return min(distances)

ConceptualStructure(C):
    for each mechanism M in powerset(C):
        compute Concept(M)

Concept(M, C):
    core_cause = argmax over purviews P: SmallPhiCause(M, P)
    core_effect = argmax over purviews P: SmallPhiEffect(M, P)
    return [CauseRepertoire(M, core_cause), EffectRepertoire(M, core_effect)]

SmallPhiCause(Mechanism M, Purview P):
    for every partition Z of M/P:
        partitioned = CauseRepertoire(M1,P1) × CauseRepertoire(M2,P2)
        distance = EMD(partitioned, unpartitioned)
    return min(distances)
```

### Key Data Structures

1. **TPM (Transition Probability Matrix)**: 2^n × 2^n matrix
2. **Repertoire**: Probability distribution over states
3. **Partition**: Way to split mechanism/purview into parts
4. **Concept**: (cause_repertoire, effect_repertoire, phi)

---

## Parallelization Strategy

| Level | What | CUDA Approach |
|-------|------|---------------|
| System partitions | Independent BigPhi evals | 1 thread per partition |
| Mechanisms | Independent concepts | 1 thread per mechanism |
| M/P partitions | Independent SmallPhi evals | 1 thread per partition |
| EMD | Matrix operations | cuBLAS or custom kernel |
| Batch networks | Multiple networks at once | Stream parallelism |

**RTX 3090**: 10,496 CUDA cores, 24GB VRAM - can attack all levels simultaneously.

---

## Implementation Phases

### Phase 1: C++ Foundation (CPU) - COMPLETE
- [x] TPM data structure and operations
- [x] Partition enumeration (bipartitions)
- [x] Cause/effect repertoire calculation
- [x] EMD (Earth Mover's Distance) - Hamming-based
- [x] SmallPhi computation (MIC/MIE with proper partitioning)
- [x] BigPhi computation (XEMD between CES structures)
- [x] Validate against PyPhi on small networks (n=3-4)

**Validation Results:**
- 3240/3240 comprehensive tests passed
- BigPhi matches PyPhi to ~2e-6 precision
- All repertoire, concept, and SIA tests pass

### Phase 2: CUDA Kernels
- [ ] Port TPM to device memory
- [ ] Parallel partition evaluation kernel
- [ ] Parallel mechanism evaluation kernel  
- [ ] GPU-accelerated EMD
- [ ] Memory-efficient design for larger n
- [ ] Validate against CPU implementation

### Phase 3: Optimization
- [ ] Memory coalescing
- [ ] Shared memory usage
- [ ] Stream concurrency for batch processing
- [ ] Profile and eliminate bottlenecks

### Phase 4: Validation & Benchmarking
- [ ] Test against all 20,100 existing PyPhi results
- [ ] Measure speedup vs PyPhi
- [ ] Push to larger n (target: n=18-22)
- [ ] Document failure modes and precision limits

### Phase 5: Release
- [ ] Clean API
- [ ] Python bindings (pybind11)
- [ ] Documentation
- [ ] Paper draft

---

## Directory Structure

```
phibench/
├── README.md              # Benchmark results (existing)
├── PLAN.md               # This file
├── src/                  # Existing Python benchmarking code
├── results/              # 20,100 PyPhi results (validation data!)
├── cuda/                 # NEW: C++/CUDA implementation
│   ├── include/
│   │   ├── tpm.hpp
│   │   ├── partition.hpp
│   │   ├── repertoire.hpp
│   │   ├── emd.hpp
│   │   └── phi.hpp
│   ├── src/
│   │   ├── tpm.cpp
│   │   ├── partition.cpp
│   │   ├── repertoire.cpp
│   │   ├── emd.cpp
│   │   ├── phi.cpp
│   │   └── main.cpp
│   ├── kernels/
│   │   ├── partition.cu
│   │   ├── repertoire.cu
│   │   └── phi.cu
│   ├── tests/
│   │   └── test_against_pyphi.cpp
│   └── CMakeLists.txt
├── pyphi/                # Reference (git ignored)
└── iit-pseudocode/       # Algorithm reference (git ignored)
```

---

## Validation Strategy

```python
# We already have ground truth from existing benchmarks
for network in existing_results:
    phi_pyphi = network['phi']  # Already computed
    phi_cuda = phicuda.compute(network['tpm'])
    assert abs(phi_pyphi - phi_cuda) < 1e-10
```

The 20,100 networks with known Φ values = comprehensive test suite.

---

## Success Criteria

1. **Correctness**: Match PyPhi to ~1e-6 precision (achieved for Phase 1)
2. **Speed**: 100x+ faster than PyPhi for n >= 10
3. **Scale**: Push tractable n from ~15 to ~20
4. **Usability**: Simple C++ API + Python bindings

---

## Implementation Notes (Phase 1 Lessons)

### Critical PyPhi Compatibility Details

1. **Array Ordering**: PyPhi uses big-endian state ordering, C++ uses little-endian
   - Solution: Bit reversal conversion when loading PyPhi test data

2. **Repertoire Expansion**: When expanding repertoires to larger purviews:
   - CAUSE direction: Use max entropy (uniform) for new nodes
   - EFFECT direction: Use unconstrained effect repertoire (NOT uniform!)
   - This asymmetry is because effect is constrained by TPM and current state

3. **Null Concept Distance**: The null concept has:
   - Cause: max entropy over purview
   - Effect: unconstrained effect repertoire (empty mechanism) - NOT max entropy

4. **Single-Node Concepts**: Do NOT early-exit when unpartitioned repertoire is uniform
   - The partitioned repertoire may still be non-uniform, giving phi > 0

### C++ Header Structure

```
cuda/include/phi/
├── phi.hpp                    # Main include
├── core/
│   └── types.hpp              # NodeSet, StateIndex, Real, Direction
├── data/
│   ├── tpm.hpp                # Transition Probability Matrix
│   ├── connectivity.hpp       # Connectivity Matrix
│   ├── repertoire.hpp         # Probability distributions
│   ├── network.hpp            # Network (TPM + CM)
│   └── subsystem.hpp          # Subsystem with state
├── partition/
│   └── bipartition.hpp        # Partition enumeration
└── compute/
    ├── small_phi.hpp          # MIC/MIE computation
    ├── big_phi.hpp            # SIA/BigPhi computation
    └── emd.hpp                # EMD algorithms (Hamming, exact SSP)
```

---

## Hardware

**Primary development**: RTX 3090
- 10,496 CUDA cores
- 24 GB GDDR6X
- 936 GB/s bandwidth

**Requirements**:
- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 compiler

---

## Next Steps

1. ~~Read PyPhi source to understand implementation details~~ DONE
2. ~~Implement TPM and partition enumeration in C++~~ DONE
3. ~~Implement SmallPhi (CPU) and validate~~ DONE
4. ~~Implement BigPhi (CPU) and validate~~ DONE
5. **Begin Phase 2: CUDA port**
   - Start with parallel mechanism evaluation (embarrassingly parallel)
   - Port TPM and repertoire data structures to device memory
   - Implement parallel partition evaluation kernel
6. Benchmark CPU vs CUDA performance
7. Optimize memory access patterns for GPU

---

*Last updated: December 2024*
