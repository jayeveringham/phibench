#!/usr/bin/env python3
"""
Generate validation test cases for PhiCUDA.
Computes phi using PyPhi and exports TPM, CM, state, and expected phi.
"""

import os
import sys
import json
import warnings

os.environ['PYPHI_WELCOME_OFF'] = 'yes'
warnings.filterwarnings('ignore')

import pyphi
import numpy as np

# Configure PyPhi
pyphi.config.VALIDATE_SUBSYSTEM_STATES = False
pyphi.config.PICK_SMALLEST_PURVIEW = True


def and_gate_network():
    """
    Classic AND gate: node 2 = node 0 AND node 1.
    Nodes 0 and 1 have self-loops.
    """
    # State-by-node format: tpm[state, node] = P(node ON | state)
    tpm = np.array([
        [0, 0, 0],  # 000
        [1, 0, 0],  # 100
        [0, 1, 0],  # 010
        [1, 1, 1],  # 110
        [0, 0, 0],  # 001
        [1, 0, 0],  # 101
        [0, 1, 0],  # 011
        [1, 1, 1],  # 111
    ], dtype=float)

    cm = np.array([
        [1, 0, 1],  # node 0: self-loop, output to node 2
        [0, 1, 1],  # node 1: self-loop, output to node 2
        [0, 0, 0],  # node 2: no outputs
    ])

    return tpm, cm, "and_gate"


def or_gate_network():
    """
    OR gate: node 2 = node 0 OR node 1.
    """
    tpm = np.array([
        [0, 0, 0],  # 000
        [1, 0, 1],  # 100
        [0, 1, 1],  # 010
        [1, 1, 1],  # 110
        [0, 0, 0],  # 001
        [1, 0, 1],  # 101
        [0, 1, 1],  # 011
        [1, 1, 1],  # 111
    ], dtype=float)

    cm = np.array([
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 0],
    ])

    return tpm, cm, "or_gate"


def xor_gate_network():
    """
    XOR gate: node 2 = node 0 XOR node 1.
    """
    tpm = np.array([
        [0, 0, 0],  # 000 -> 0 XOR 0 = 0
        [1, 0, 1],  # 100 -> 1 XOR 0 = 1
        [0, 1, 1],  # 010 -> 0 XOR 1 = 1
        [1, 1, 0],  # 110 -> 1 XOR 1 = 0
        [0, 0, 0],  # 001
        [1, 0, 1],  # 101
        [0, 1, 1],  # 011
        [1, 1, 0],  # 111
    ], dtype=float)

    cm = np.array([
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 0],
    ])

    return tpm, cm, "xor_gate"


def copy_network():
    """
    Simple copy: A -> B -> C (chain).
    Each node copies its input.
    """
    tpm = np.array([
        [0, 0, 0],  # 000
        [1, 1, 0],  # 100: A=1 -> B=1, B=0 -> C=0
        [0, 0, 1],  # 010: B=1 -> C=1
        [1, 1, 1],  # 110
        [0, 0, 0],  # 001
        [1, 1, 0],  # 101
        [0, 0, 1],  # 011
        [1, 1, 1],  # 111
    ], dtype=float)

    cm = np.array([
        [0, 1, 0],  # A -> B
        [0, 0, 1],  # B -> C
        [0, 0, 0],  # C -> nothing
    ])

    return tpm, cm, "copy_chain"


def majority_network():
    """
    3-node majority gate: each node outputs majority of all inputs.
    """
    tpm = np.zeros((8, 3), dtype=float)
    for state in range(8):
        bits = [(state >> i) & 1 for i in range(3)]
        majority = 1 if sum(bits) >= 2 else 0
        for i in range(3):
            tpm[state, i] = majority

    cm = np.ones((3, 3), dtype=int)  # Fully connected

    return tpm, cm, "majority"


def iit_example_4node():
    """
    IIT 4.0 paper example network.
    Based on Albantakis et al. 2023.
    """
    # 4 nodes: A, B, C, D
    # Simplified version - majority rule
    n = 4
    tpm = np.zeros((2**n, n), dtype=float)

    for state in range(2**n):
        bits = [(state >> i) & 1 for i in range(n)]

        # Each node does XOR of its inputs
        tpm[state, 0] = (bits[1] ^ bits[3]) if (bits[1] + bits[3]) > 0 else 0
        tpm[state, 1] = (bits[0] ^ bits[2]) if (bits[0] + bits[2]) > 0 else 0
        tpm[state, 2] = (bits[1] ^ bits[3]) if (bits[1] + bits[3]) > 0 else 0
        tpm[state, 3] = (bits[0] ^ bits[2]) if (bits[0] + bits[2]) > 0 else 0

    cm = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
    ])

    return tpm, cm, "iit_4node"


def compute_phi(tpm, cm, state):
    """Compute BigPhi using PyPhi."""
    network = pyphi.Network(tpm, cm)
    n = tpm.shape[1]
    subsystem = pyphi.Subsystem(network, state, nodes=range(n))
    sia = pyphi.compute.sia(subsystem)
    return sia.phi, sia


def main():
    networks = [
        and_gate_network,
        or_gate_network,
        xor_gate_network,
        copy_network,
        majority_network,
        iit_example_4node,
    ]

    # Test states for each network size
    test_states = {
        3: [(0, 0, 0), (1, 1, 1), (1, 0, 1), (0, 1, 0)],
        4: [(0, 0, 0, 0), (1, 1, 1, 1), (1, 0, 1, 0), (0, 1, 0, 1)],
    }

    results = []

    for net_fn in networks:
        tpm, cm, name = net_fn()
        n = tpm.shape[1]
        states = test_states.get(n, [(1,) * n])

        for state in states:
            try:
                phi, sia = compute_phi(tpm, cm, state)

                result = {
                    'name': name,
                    'n_nodes': n,
                    'state': list(state),
                    'tpm': tpm.tolist(),
                    'cm': cm.tolist(),
                    'phi': phi,
                    'n_concepts': len(sia.ces) if hasattr(sia, 'ces') else 0,
                }
                results.append(result)

                state_str = ''.join(map(str, state))
                print(f"{name} (state={state_str}): phi={phi:.6f}")

            except Exception as e:
                print(f"Error computing {name} state {state}: {e}")

    # Save to JSON
    output_path = os.path.join(os.path.dirname(__file__), 'validation_data.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} test cases to {output_path}")

    # Also generate C++ header with test cases
    header_path = os.path.join(os.path.dirname(__file__), 'validation_data.hpp')
    generate_cpp_header(results, header_path)
    print(f"Generated C++ header: {header_path}")


def generate_cpp_header(results, path):
    """Generate C++ header with test case data."""

    lines = [
        '#pragma once',
        '',
        '#include "phi/phi.hpp"',
        '#include <vector>',
        '#include <string>',
        '',
        'namespace phi_test {',
        '',
        'struct ValidationCase {',
        '    std::string name;',
        '    size_t n_nodes;',
        '    phi::StateIndex state;',
        '    std::vector<phi::Real> tpm_data;  // Flattened state-by-node',
        '    std::vector<uint8_t> cm_data;     // Flattened row-major',
        '    phi::Real expected_phi;',
        '    size_t expected_n_concepts;',
        '};',
        '',
        'inline std::vector<ValidationCase> get_validation_cases() {',
        '    return {',
    ]

    for i, r in enumerate(results):
        tpm_flat = [v for row in r['tpm'] for v in row]
        cm_flat = [v for row in r['cm'] for v in row]
        state_idx = sum(s << i for i, s in enumerate(r['state']))

        tpm_str = ', '.join(f'{v:.1f}' for v in tpm_flat)
        cm_str = ', '.join(str(int(v)) for v in cm_flat)

        lines.append(f'        {{')
        lines.append(f'            "{r["name"]}_s{state_idx}",')
        lines.append(f'            {r["n_nodes"]},')
        lines.append(f'            {state_idx},')
        lines.append(f'            {{{tpm_str}}},')
        lines.append(f'            {{{cm_str}}},')
        lines.append(f'            {r["phi"]},')
        lines.append(f'            {r["n_concepts"]}')
        lines.append(f'        }}{","  if i < len(results)-1 else ""}')

    lines.extend([
        '    };',
        '}',
        '',
        '}  // namespace phi_test',
        '',
    ])

    with open(path, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
