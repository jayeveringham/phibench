#include "phi/phi.hpp"
#include <iostream>
#include <iomanip>

using namespace phi;

int main() {
    std::cout << "Test: BigPhi with XEMD" << std::endl;
    std::cout << "=======================" << std::endl;
    std::cout << std::fixed << std::setprecision(6);

    // Majority network
    size_t n = 3;
    TPM tpm(n);
    for (StateIndex s = 0; s < 8; ++s) {
        int count = bits::popcount(static_cast<NodeSet>(s) & 0x7);
        Real majority = (count >= 2) ? 1.0 : 0.0;
        for (NodeIndex node = 0; node < n; ++node) {
            tpm(s, node) = majority;
        }
    }

    ConnectivityMatrix cm(n);
    for (NodeIndex from = 0; from < n; ++from) {
        for (NodeIndex to = 0; to < n; ++to) {
            cm(from, to) = 1;
        }
    }

    Network network(tpm, cm);
    StateIndex state = 0;

    // Compute SIA
    std::cout << "Computing SIA..." << std::endl;
    SIA sia = compute_sia(network, bits::full_set(n), state);

    std::cout << "\nResults:" << std::endl;
    std::cout << "  BigPhi = " << sia.phi << std::endl;
    std::cout << "  MIP: from=" << sia.cut.from_nodes << " to=" << sia.cut.to_nodes << std::endl;
    std::cout << "  Unpartitioned CES: " << sia.ces.size() << " concepts, total_phi="
              << sia.ces.total_phi() << std::endl;
    std::cout << "  Partitioned CES: " << sia.partitioned_ces.size() << " concepts, total_phi="
              << sia.partitioned_ces.total_phi() << std::endl;

    std::cout << "\n=== Expected ===" << std::endl;
    std::cout << "  PyPhi BigPhi: 0.674443" << std::endl;
    std::cout << "  PyPhi MIP: Cut [n0] -> [n1, n2] (from=1, to=6)" << std::endl;

    // Check if result is close
    Real expected = 0.674443;
    Real diff = std::abs(sia.phi - expected);
    std::cout << "\n=== Validation ===" << std::endl;
    std::cout << "  Difference from expected: " << diff << std::endl;
    if (diff < 0.01) {
        std::cout << "  PASS (within 1% tolerance)" << std::endl;
    } else {
        std::cout << "  FAIL (difference too large)" << std::endl;
    }

    return 0;
}
