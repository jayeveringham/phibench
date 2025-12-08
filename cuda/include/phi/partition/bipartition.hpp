#pragma once

#include "phi/core/types.hpp"
#include <vector>
#include <utility>

namespace phi {

/**
 * A part in a bipartition, containing mechanism and purview subsets.
 */
struct Part {
    NodeSet mechanism;
    NodeSet purview;

    Part() : mechanism(0), purview(0) {}
    Part(NodeSet m, NodeSet p) : mechanism(m), purview(p) {}

    bool is_empty() const {
        return mechanism == 0 && purview == 0;
    }

    bool operator==(const Part& other) const {
        return mechanism == other.mechanism && purview == other.purview;
    }
};

/**
 * Bipartition of mechanism and purview into two parts.
 *
 * Used for SmallPhi (MIP) computation.
 */
struct Bipartition {
    Part part0;
    Part part1;

    Bipartition() = default;
    Bipartition(Part p0, Part p1) : part0(p0), part1(p1) {}

    /**
     * Check if partition is valid (not trivially reducible).
     *
     * Invalid if either part has both mechanism and purview empty.
     */
    bool is_valid() const {
        return !part0.is_empty() && !part1.is_empty();
    }

    bool operator==(const Bipartition& other) const {
        return part0 == other.part0 && part1 == other.part1;
    }
};

/**
 * System-level cut for BigPhi computation.
 *
 * Represents severing connections from 'from_nodes' to 'to_nodes'.
 */
struct SystemCut {
    NodeSet from_nodes;
    NodeSet to_nodes;

    SystemCut() : from_nodes(0), to_nodes(0) {}
    SystemCut(NodeSet from, NodeSet to) : from_nodes(from), to_nodes(to) {}

    bool is_null() const {
        return from_nodes == 0 || to_nodes == 0;
    }

    /**
     * Check if a specific connection is cut.
     */
    bool is_cut(NodeIndex from, NodeIndex to) const {
        return bits::contains(from_nodes, from) && bits::contains(to_nodes, to);
    }

    bool operator==(const SystemCut& other) const {
        return from_nodes == other.from_nodes && to_nodes == other.to_nodes;
    }
};

/**
 * Generate all undirected bipartitions of a node set.
 *
 * Undirected means (A, B) and (B, A) are considered the same.
 * Returns 2^(n-1) partitions for n nodes.
 */
inline std::vector<std::pair<NodeSet, NodeSet>> undirected_bipartitions(NodeSet nodes) {
    std::vector<std::pair<NodeSet, NodeSet>> result;

    if (nodes == 0) {
        result.emplace_back(0, 0);
        return result;
    }

    size_t n = bits::popcount(nodes);
    std::vector<NodeIndex> node_vec = bits::to_vector(nodes);

    // Generate 2^(n-1) bipartitions (undirected, so only half)
    size_t num_partitions = 1ULL << (n - 1);
    result.reserve(num_partitions);

    for (size_t i = 0; i < num_partitions; ++i) {
        NodeSet part0 = 0, part1 = 0;
        for (size_t j = 0; j < n; ++j) {
            if ((i >> j) & 1) {
                part0 = bits::add(part0, node_vec[j]);
            } else {
                part1 = bits::add(part1, node_vec[j]);
            }
        }
        result.emplace_back(part0, part1);
    }

    return result;
}

/**
 * Generate all directed bipartitions of a node set.
 *
 * Directed means (A, B) and (B, A) are different.
 * Returns 2^n partitions for n nodes.
 */
inline std::vector<std::pair<NodeSet, NodeSet>> directed_bipartitions(NodeSet nodes) {
    std::vector<std::pair<NodeSet, NodeSet>> result;

    if (nodes == 0) {
        result.emplace_back(0, 0);
        return result;
    }

    size_t n = bits::popcount(nodes);
    std::vector<NodeIndex> node_vec = bits::to_vector(nodes);

    // All 2^n possible assignments
    size_t num_states = 1ULL << n;
    result.reserve(num_states);

    for (size_t i = 0; i < num_states; ++i) {
        NodeSet part0 = 0, part1 = 0;
        for (size_t j = 0; j < n; ++j) {
            if ((i >> j) & 1) {
                part0 = bits::add(part0, node_vec[j]);
            } else {
                part1 = bits::add(part1, node_vec[j]);
            }
        }
        result.emplace_back(part0, part1);
    }

    return result;
}

/**
 * Generate all MIP bipartitions for SmallPhi computation.
 *
 * These are all valid combinations of:
 * - Undirected bipartitions of mechanism
 * - Directed bipartitions of purview
 *
 * Valid means neither part is completely empty (both mechanism and purview empty).
 */
inline std::vector<Bipartition> mip_bipartitions(NodeSet mechanism, NodeSet purview) {
    std::vector<Bipartition> result;

    auto mech_parts = undirected_bipartitions(mechanism);
    auto purv_parts = directed_bipartitions(purview);

    result.reserve(mech_parts.size() * purv_parts.size());

    for (const auto& [m0, m1] : mech_parts) {
        for (const auto& [p0, p1] : purv_parts) {
            // Validity check: (m0 or p0) and (m1 or p1)
            // Neither part should be completely empty
            bool part0_nonempty = (m0 != 0) || (p0 != 0);
            bool part1_nonempty = (m1 != 0) || (p1 != 0);

            if (part0_nonempty && part1_nonempty) {
                result.emplace_back(Part(m0, p0), Part(m1, p1));
            }
        }
    }

    return result;
}

/**
 * Count MIP bipartitions without generating them.
 */
inline size_t count_mip_bipartitions(NodeSet mechanism, NodeSet purview) {
    size_t m = bits::popcount(mechanism);
    size_t p = bits::popcount(purview);

    if (m == 0 && p == 0) return 0;

    size_t num_mech = (m > 0) ? (1ULL << (m - 1)) : 1;
    size_t num_purv = (p > 0) ? (1ULL << p) : 1;

    // Total combinations minus invalid ones
    // Invalid: both parts empty, which happens when:
    // - m0=0, p0=0 (i.e., m_idx=0, p_idx has all bits in part1)
    // - m1=0, p1=0 (i.e., m_idx=max, p_idx has all bits in part0)

    size_t total = num_mech * num_purv;

    // Subtract invalid cases
    if (m > 0 && p > 0) {
        // Case: m0=empty (m_idx=0 for undirected) and p0=empty (p_idx=0)
        // and m1=empty (m_idx=max) and p1=empty (p_idx=max)
        // These are 2 invalid combinations
        total -= 2;
    } else if (m == 0) {
        // Only purview, invalid when p0=empty (p_idx=0) or p1=empty (p_idx=max)
        total -= 2;
    } else if (p == 0) {
        // Only mechanism, invalid when m0=empty (impossible for undirected with m>0)
        // or m1=empty (also impossible)
        // Actually for undirected, both parts are always non-empty if m >= 2
        // For m=1, one part is always empty
        if (m == 1) {
            total = 0;  // Cannot have valid partition of single mechanism with no purview
        }
    }

    return total;
}

/**
 * Generate all system cuts (directed bipartitions) for BigPhi.
 *
 * Nontrivial cuts exclude the null cut where from_nodes or to_nodes is empty.
 */
inline std::vector<SystemCut> system_cuts(NodeSet nodes, bool nontrivial = true) {
    std::vector<SystemCut> result;

    auto parts = directed_bipartitions(nodes);

    for (const auto& [from, to] : parts) {
        if (!nontrivial || (from != 0 && to != 0)) {
            result.emplace_back(from, to);
        }
    }

    return result;
}

/**
 * Count nontrivial system cuts.
 *
 * For n nodes: 2^n - 2 (all directed bipartitions except trivial ones)
 */
inline size_t count_system_cuts(size_t num_nodes) {
    if (num_nodes == 0) return 0;
    return (1ULL << num_nodes) - 2;
}

}  // namespace phi
