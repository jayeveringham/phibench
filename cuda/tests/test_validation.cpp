#include "phi/phi.hpp"
#include "validation_data.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace phi;

// Helper to construct TPM from flattened data
TPM build_tpm(size_t n_nodes, const std::vector<Real>& data) {
    size_t n_states = state::num_states(n_nodes);
    TPM tpm(n_nodes);

    for (StateIndex s = 0; s < n_states; ++s) {
        for (NodeIndex node = 0; node < n_nodes; ++node) {
            tpm(s, node) = data[s * n_nodes + node];
        }
    }

    return tpm;
}

// Helper to construct CM from flattened data
ConnectivityMatrix build_cm(size_t n_nodes, const std::vector<uint8_t>& data) {
    ConnectivityMatrix cm(n_nodes);

    for (NodeIndex from = 0; from < n_nodes; ++from) {
        for (NodeIndex to = 0; to < n_nodes; ++to) {
            cm(from, to) = data[from * n_nodes + to];
        }
    }

    return cm;
}

int main() {
    std::cout << "PhiCUDA Validation Tests" << std::endl;
    std::cout << "========================" << std::endl;
    std::cout << std::fixed << std::setprecision(6);

    auto cases = phi_test::get_validation_cases();

    int passed = 0;
    int failed = 0;
    int total = 0;

    // Tolerance for phi comparison
    const Real TOLERANCE = 1e-4;

    for (const auto& tc : cases) {
        total++;

        // Build network
        TPM tpm = build_tpm(tc.n_nodes, tc.tpm_data);
        ConnectivityMatrix cm = build_cm(tc.n_nodes, tc.cm_data);
        Network network(tpm, cm);

        // Compute phi
        NodeSet all_nodes = bits::full_set(tc.n_nodes);
        SIA sia = compute_sia(network, all_nodes, tc.state);

        Real computed_phi = sia.phi;
        Real expected_phi = tc.expected_phi;
        Real diff = std::abs(computed_phi - expected_phi);

        bool phi_ok = diff < TOLERANCE;

        if (phi_ok) {
            passed++;
            std::cout << "[PASS] " << tc.name
                      << ": phi=" << computed_phi
                      << " (expected " << expected_phi << ")" << std::endl;
        } else {
            failed++;
            std::cout << "[FAIL] " << tc.name
                      << ": phi=" << computed_phi
                      << " (expected " << expected_phi
                      << ", diff=" << diff << ")" << std::endl;

            // Debug output for failures
            std::cout << "       n_nodes=" << tc.n_nodes
                      << ", state=" << tc.state
                      << ", n_concepts=" << sia.ces.size()
                      << " (expected " << tc.expected_n_concepts << ")" << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "========================" << std::endl;
    std::cout << "Results: " << passed << "/" << total << " passed";
    if (failed > 0) {
        std::cout << " (" << failed << " failed)";
    }
    std::cout << std::endl;

    return failed > 0 ? 1 : 0;
}
