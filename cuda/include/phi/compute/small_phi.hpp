#pragma once

#include "phi/core/types.hpp"
#include "phi/data/tpm.hpp"
#include "phi/data/repertoire.hpp"
#include "phi/partition/bipartition.hpp"
#include "phi/metrics/emd.hpp"
#include <limits>
#include <optional>

namespace phi {

/**
 * Result of finding the Minimum Information Partition (MIP).
 */
struct MIPResult {
    Real phi;                    // Small phi value
    Bipartition partition;       // The MIP
    Repertoire unpartitioned;    // Unpartitioned repertoire
    Repertoire partitioned;      // Partitioned repertoire at MIP

    bool is_reducible() const {
        return phi <= EPSILON;
    }
};

/**
 * Result of finding Maximally Irreducible Cause/Effect (MICE).
 */
struct MICEResult {
    Real phi;                    // Small phi at core purview
    NodeSet mechanism;           // The mechanism
    NodeSet purview;             // Core cause/effect purview
    Direction direction;
    Repertoire repertoire;       // The repertoire at core purview
    Bipartition mip;             // The MIP for this purview

    bool is_null() const {
        return phi <= EPSILON;
    }
};

using MIC = MICEResult;  // Maximally Irreducible Cause
using MIE = MICEResult;  // Maximally Irreducible Effect

/**
 * Compute the partitioned repertoire for a bipartition.
 *
 * The partitioned repertoire is the product of the repertoires
 * of the two parts.
 */
inline Repertoire partitioned_repertoire(const Subsystem& subsystem,
                                          Direction direction,
                                          const Bipartition& partition) {
    // Compute repertoire for each part
    Repertoire rep0 = subsystem.repertoire(direction,
                                            partition.part0.mechanism,
                                            partition.part0.purview);
    Repertoire rep1 = subsystem.repertoire(direction,
                                            partition.part1.mechanism,
                                            partition.part1.purview);

    // Product of independent parts
    return Repertoire::product(rep0, rep1);
}

/**
 * Find the Minimum Information Partition (MIP) for a mechanism-purview pair.
 *
 * MIP is the partition that minimizes the EMD between unpartitioned
 * and partitioned repertoires.
 */
inline MIPResult find_mip(const Subsystem& subsystem,
                           Direction direction,
                           NodeSet mechanism,
                           NodeSet purview) {
    MIPResult result;
    result.phi = std::numeric_limits<Real>::max();

    // Compute unpartitioned repertoire
    result.unpartitioned = subsystem.repertoire(direction, mechanism, purview);

    // Handle degenerate cases
    if (mechanism == 0 || purview == 0) {
        result.phi = 0.0;
        result.partitioned = result.unpartitioned;
        return result;
    }

    // NOTE: We do NOT early-exit when unpartitioned is uniform!
    // The partitioned repertoire may still be non-uniform, giving phi > 0.
    // This is the case for single-node mechanisms in majority networks.

    // Generate and evaluate all bipartitions
    auto partitions = mip_bipartitions(mechanism, purview);

    for (const auto& partition : partitions) {
        // Compute partitioned repertoire
        Repertoire part_rep = partitioned_repertoire(subsystem, direction, partition);

        // Expand to match purview if needed
        if (part_rep.purview() != purview) {
            part_rep = part_rep.expand(purview);
        }

        // Compute EMD
        Real phi = repertoire_distance(result.unpartitioned, part_rep, direction);

        // Track minimum
        if (phi < result.phi) {
            result.phi = phi;
            result.partition = partition;
            result.partitioned = part_rep;
        }

        // Early exit if reducible
        if (result.phi <= EPSILON) {
            result.phi = 0.0;
            break;
        }
    }

    return result;
}

/**
 * Find potential purviews for a mechanism in given direction.
 *
 * Filters purviews based on connectivity - a purview is only potential
 * if it has some causal connection to the mechanism.
 */
inline std::vector<NodeSet> potential_purviews(const Subsystem& subsystem,
                                                Direction direction,
                                                NodeSet mechanism) {
    std::vector<NodeSet> purviews;
    NodeSet nodes = subsystem.nodes();
    size_t n = bits::popcount(nodes);
    size_t num_subsets = state::num_states(n);

    std::vector<NodeIndex> node_vec = bits::to_vector(nodes);

    // Enumerate all non-empty subsets of nodes
    for (StateIndex i = 1; i < num_subsets; ++i) {
        NodeSet purview = 0;
        for (size_t j = 0; j < n; ++j) {
            if ((i >> j) & 1) {
                purview = bits::add(purview, node_vec[j]);
            }
        }

        // Check connectivity
        if (subsystem.cm().is_connected(mechanism, purview, direction)) {
            purviews.push_back(purview);
        }
    }

    return purviews;
}

/**
 * Find MICE (Maximally Irreducible Cause/Effect) for a mechanism.
 *
 * MICE is the purview that maximizes SmallPhi for the mechanism.
 */
inline MICEResult find_mice(const Subsystem& subsystem,
                             Direction direction,
                             NodeSet mechanism) {
    MICEResult result;
    result.phi = -1.0;  // Will be set to 0 if no purview found
    result.mechanism = mechanism;
    result.direction = direction;
    result.purview = 0;

    // Handle empty mechanism
    if (mechanism == 0) {
        result.phi = 0.0;
        return result;
    }

    // Get potential purviews
    auto purviews = potential_purviews(subsystem, direction, mechanism);

    if (purviews.empty()) {
        result.phi = 0.0;
        return result;
    }

    // Find purview with maximum phi
    // On ties, prefer larger purviews (more nodes) - PyPhi behavior
    for (NodeSet purview : purviews) {
        MIPResult mip = find_mip(subsystem, direction, mechanism, purview);

        bool better = (mip.phi > result.phi + EPSILON);
        bool tie_larger_purview = (std::abs(mip.phi - result.phi) <= EPSILON) &&
                                   (bits::popcount(purview) > bits::popcount(result.purview));

        if (better || tie_larger_purview) {
            result.phi = mip.phi;
            result.purview = purview;
            result.repertoire = mip.unpartitioned;
            result.mip = mip.partition;
        }
    }

    // If no positive phi found
    if (result.phi < EPSILON) {
        result.phi = 0.0;
    }

    return result;
}

/**
 * Concept (distinction) formed by a mechanism.
 */
struct Concept {
    NodeSet mechanism;
    MIC cause;
    MIE effect;

    Real phi() const {
        return std::min(cause.phi, effect.phi);
    }

    bool is_null() const {
        return phi() <= EPSILON;
    }
};

/**
 * Compute the concept for a mechanism.
 *
 * A concept exists only if both cause and effect have positive phi.
 */
inline std::optional<Concept> compute_concept(const Subsystem& subsystem,
                                               NodeSet mechanism) {
    Concept cpt;
    cpt.mechanism = mechanism;

    // Find MIC (core cause)
    cpt.cause = find_mice(subsystem, Direction::CAUSE, mechanism);

    // Early exit if cause is null
    if (cpt.cause.is_null()) {
        return std::nullopt;
    }

    // Find MIE (core effect)
    cpt.effect = find_mice(subsystem, Direction::EFFECT, mechanism);

    // Concept exists only if both cause and effect have positive phi
    if (cpt.effect.is_null()) {
        return std::nullopt;
    }

    return cpt;
}

}  // namespace phi
