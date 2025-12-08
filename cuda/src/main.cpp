#include "phi/phi.hpp"
#include <iostream>
#include <iomanip>

using namespace phi;

/**
 * Create a simple AND gate TPM for 3 nodes.
 *
 * Node 2 = Node 0 AND Node 1
 * This is a classic example from IIT literature.
 */
TPM create_and_gate_tpm() {
    // 3 nodes, 2^3 = 8 states
    TPM tpm(3);

    // For each state, set transition probabilities
    // State format: [node0, node1, node2] -> P(node_i = ON | state)

    // Node 0 and 1 are inputs (self-loops)
    // Node 2 = Node 0 AND Node 1

    for (StateIndex s = 0; s < 8; ++s) {
        uint8_t n0 = state::get_bit(s, 0);
        uint8_t n1 = state::get_bit(s, 1);
        uint8_t n2 = state::get_bit(s, 2);

        // Self-loops for nodes 0 and 1
        tpm(s, 0) = static_cast<Real>(n0);
        tpm(s, 1) = static_cast<Real>(n1);

        // Node 2 = AND of inputs
        tpm(s, 2) = static_cast<Real>(n0 & n1);
    }

    return tpm;
}

/**
 * Create connectivity matrix for AND gate.
 */
ConnectivityMatrix create_and_gate_cm() {
    ConnectivityMatrix cm(3);

    // Self-loops for input nodes
    cm(0, 0) = 1;  // Node 0 -> Node 0
    cm(1, 1) = 1;  // Node 1 -> Node 1

    // Inputs to AND gate
    cm(0, 2) = 1;  // Node 0 -> Node 2
    cm(1, 2) = 1;  // Node 1 -> Node 2

    return cm;
}

void test_types() {
    std::cout << "=== Testing Types ===" << std::endl;

    // Test NodeSet operations
    NodeSet set = bits::make_set({0, 2, 3});
    std::cout << "NodeSet {0,2,3}: " << set << " (popcount: " << bits::popcount(set) << ")" << std::endl;

    // Test state conversion
    std::vector<uint8_t> state_vec = {1, 0, 1};  // Node 0 ON, Node 1 OFF, Node 2 ON
    StateIndex idx = state::from_vector(state_vec);
    std::cout << "State [1,0,1] -> index: " << idx << std::endl;

    auto back = state::to_vector(idx, 3);
    std::cout << "Index " << idx << " -> state: [" << (int)back[0] << "," << (int)back[1] << "," << (int)back[2] << "]" << std::endl;

    std::cout << std::endl;
}

void test_tpm() {
    std::cout << "=== Testing TPM ===" << std::endl;

    TPM tpm = create_and_gate_tpm();
    std::cout << "Created AND gate TPM with " << tpm.num_nodes() << " nodes, "
              << tpm.num_states() << " states" << std::endl;

    // Print TPM
    std::cout << "TPM values:" << std::endl;
    for (StateIndex s = 0; s < tpm.num_states(); ++s) {
        auto sv = state::to_vector(s, 3);
        std::cout << "  [" << (int)sv[0] << "," << (int)sv[1] << "," << (int)sv[2] << "] -> [";
        for (NodeIndex n = 0; n < 3; ++n) {
            std::cout << tpm(s, n);
            if (n < 2) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    std::cout << std::endl;
}

void test_partitions() {
    std::cout << "=== Testing Partitions ===" << std::endl;

    NodeSet mechanism = bits::make_set({0, 1});
    NodeSet purview = bits::make_set({0, 1, 2});

    auto parts = mip_bipartitions(mechanism, purview);
    std::cout << "MIP bipartitions for mechanism {0,1} and purview {0,1,2}: "
              << parts.size() << " partitions" << std::endl;

    // Print first few partitions
    size_t to_print = std::min(parts.size(), size_t{5});
    for (size_t i = 0; i < to_print; ++i) {
        const auto& p = parts[i];
        std::cout << "  [" << i << "] M0=" << p.part0.mechanism
                  << " P0=" << p.part0.purview
                  << " | M1=" << p.part1.mechanism
                  << " P1=" << p.part1.purview << std::endl;
    }
    if (parts.size() > to_print) {
        std::cout << "  ... (" << (parts.size() - to_print) << " more)" << std::endl;
    }

    std::cout << std::endl;
}

void test_repertoire() {
    std::cout << "=== Testing Repertoire ===" << std::endl;

    TPM tpm = create_and_gate_tpm();
    ConnectivityMatrix cm = create_and_gate_cm();
    Network network(tpm, cm);

    // Create subsystem with all nodes, state = [1, 1, 1]
    StateIndex state = state::from_vector({1, 1, 1});
    Subsystem subsystem(network, bits::full_set(3), state);

    // Compute cause repertoire for mechanism {0} over purview {0, 1}
    NodeSet mechanism = bits::make_set({0});
    NodeSet purview = bits::make_set({0, 1});

    Repertoire cause_rep = subsystem.cause_repertoire(mechanism, purview);
    std::cout << "Cause repertoire for M={0}, P={0,1}, state=[1,1,1]:" << std::endl;
    for (StateIndex s = 0; s < cause_rep.num_states(); ++s) {
        auto sv = state::to_vector(s, 2);
        std::cout << "  [" << (int)sv[0] << "," << (int)sv[1] << "]: " << cause_rep[s] << std::endl;
    }

    std::cout << std::endl;
}

void test_small_phi() {
    std::cout << "=== Testing SmallPhi ===" << std::endl;

    TPM tpm = create_and_gate_tpm();
    ConnectivityMatrix cm = create_and_gate_cm();
    Network network(tpm, cm);

    StateIndex state = state::from_vector({1, 1, 1});
    Subsystem subsystem(network, bits::full_set(3), state);

    // Find MICE for mechanism {2} (the AND gate output)
    NodeSet mechanism = bits::make_set({2});

    MIC mic = find_mice(subsystem, Direction::CAUSE, mechanism);
    MIE mie = find_mice(subsystem, Direction::EFFECT, mechanism);

    std::cout << "Mechanism {2} (AND gate output):" << std::endl;
    std::cout << "  MIC: phi=" << mic.phi << ", purview=" << mic.purview << std::endl;
    std::cout << "  MIE: phi=" << mie.phi << ", purview=" << mie.purview << std::endl;

    std::cout << std::endl;
}

void test_big_phi() {
    std::cout << "=== Testing BigPhi ===" << std::endl;

    TPM tpm = create_and_gate_tpm();
    ConnectivityMatrix cm = create_and_gate_cm();
    Network network(tpm, cm);

    StateIndex state = state::from_vector({1, 1, 1});

    std::cout << "Computing BigPhi for AND gate in state [1,1,1]..." << std::endl;

    SIA sia = compute_sia(network, bits::full_set(3), state);

    std::cout << "BigPhi = " << sia.phi << std::endl;
    std::cout << "CES has " << sia.ces.size() << " concepts" << std::endl;

    for (size_t i = 0; i < sia.ces.size(); ++i) {
        const auto& c = sia.ces[i];
        std::cout << "  Concept " << i << ": mechanism=" << c.mechanism
                  << ", phi=" << c.phi()
                  << " (cause=" << c.cause.phi << ", effect=" << c.effect.phi << ")"
                  << std::endl;
    }

    std::cout << std::endl;
}

int main() {
    std::cout << "PhiCUDA " << phi::VERSION << " (" << phi::VERSION_NAME << ")" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::fixed << std::setprecision(6);

    test_types();
    test_tpm();
    test_partitions();
    test_repertoire();
    test_small_phi();
    test_big_phi();

    std::cout << "========================================" << std::endl;
    std::cout << "All tests completed." << std::endl;

    return 0;
}
