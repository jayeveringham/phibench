#pragma once

#include "phi/core/types.hpp"
#include "phi/data/tpm.hpp"
#include "phi/data/repertoire.hpp"
#include "phi/partition/bipartition.hpp"
#include "phi/compute/small_phi.hpp"
#include <vector>
#include <algorithm>
#include <limits>

namespace phi {

/**
 * Cause-Effect Structure (CES): Collection of all concepts.
 */
class CES {
public:
    CES() = default;

    void add(Concept&& cpt) {
        concepts_.push_back(std::move(cpt));
    }

    void add(const Concept& cpt) {
        concepts_.push_back(cpt);
    }

    const std::vector<Concept>& concepts() const { return concepts_; }
    size_t size() const { return concepts_.size(); }
    bool empty() const { return concepts_.empty(); }

    const Concept& operator[](size_t i) const { return concepts_[i]; }

    // Total phi (sum of concept phis)
    Real total_phi() const {
        Real total = 0.0;
        for (const auto& c : concepts_) {
            total += c.phi();
        }
        return total;
    }

private:
    std::vector<Concept> concepts_;
};

/**
 * System Irreducibility Analysis (SIA) result.
 */
struct SIA {
    Real phi;                    // BigPhi value
    CES ces;                     // Unpartitioned CES
    CES partitioned_ces;         // CES under MIP
    SystemCut cut;               // Minimum Information Partition
    NodeSet nodes;               // System nodes
    StateIndex state;            // System state

    bool is_null() const {
        return phi <= EPSILON;
    }
};

/**
 * Compute the Cause-Effect Structure for a subsystem.
 *
 * Enumerates all non-empty mechanisms and computes their concepts.
 */
inline CES compute_ces(const Subsystem& subsystem) {
    CES ces;
    NodeSet nodes = subsystem.nodes();
    size_t n = bits::popcount(nodes);

    if (n == 0) return ces;

    std::vector<NodeIndex> node_vec = bits::to_vector(nodes);

    // Enumerate all non-empty subsets (mechanisms)
    size_t num_subsets = state::num_states(n);

    for (StateIndex i = 1; i < num_subsets; ++i) {
        // Build mechanism from subset
        NodeSet mechanism = 0;
        for (size_t j = 0; j < n; ++j) {
            if ((i >> j) & 1) {
                mechanism = bits::add(mechanism, node_vec[j]);
            }
        }

        // Compute concept
        auto concept_opt = compute_concept(subsystem, mechanism);
        if (concept_opt) {
            ces.add(std::move(*concept_opt));
        }
    }

    return ces;
}

/**
 * Expand a repertoire to a new purview using proper unconstrained repertoire.
 *
 * For CAUSE: unconstrained = max entropy (uniform) for new nodes
 * For EFFECT: unconstrained = effect_repertoire(empty, new_nodes) - NOT uniform!
 *
 * This matches PyPhi's expand_repertoire logic.
 */
inline Repertoire expand_repertoire_proper(const Repertoire& rep,
                                            NodeSet new_purview,
                                            Direction dir,
                                            const Subsystem& subsystem) {
    if (new_purview == rep.purview()) return rep;

    // New purview must contain old purview
    if (!bits::is_subset(rep.purview(), new_purview)) {
        throw std::invalid_argument("New purview must contain old purview");
    }

    NodeSet new_nodes = bits::difference(new_purview, rep.purview());

    // Get unconstrained repertoire for new nodes
    Repertoire unconstrained;
    if (dir == Direction::CAUSE) {
        // Cause: max entropy (uniform)
        unconstrained = Repertoire::max_entropy(new_nodes);
    } else {
        // Effect: use actual unconstrained effect repertoire (NOT uniform!)
        unconstrained = subsystem.effect_repertoire(0, new_nodes);
    }

    // Multiply and normalize
    return Repertoire::product(rep, unconstrained).normalized();
}

/**
 * Compute distance between two concepts in concept space.
 *
 * The distance is the sum of:
 * - EMD between expanded cause repertoires
 * - EMD between expanded effect repertoires
 *
 * Uses proper expansion with unconstrained repertoires.
 */
inline Real concept_distance(const Concept& c1, const Concept& c2,
                              const Subsystem& subsystem1, const Subsystem& subsystem2) {
    // Combine purviews
    NodeSet combined_cause = bits::unite(c1.cause.purview, c2.cause.purview);
    NodeSet combined_effect = bits::unite(c1.effect.purview, c2.effect.purview);

    // Expand using proper unconstrained repertoires
    Repertoire cause1_exp = expand_repertoire_proper(c1.cause.repertoire, combined_cause,
                                                      Direction::CAUSE, subsystem1);
    Repertoire cause2_exp = expand_repertoire_proper(c2.cause.repertoire, combined_cause,
                                                      Direction::CAUSE, subsystem2);
    Real cause_dist = hamming_emd(cause1_exp, cause2_exp);

    Repertoire effect1_exp = expand_repertoire_proper(c1.effect.repertoire, combined_effect,
                                                       Direction::EFFECT, subsystem1);
    Repertoire effect2_exp = expand_repertoire_proper(c2.effect.repertoire, combined_effect,
                                                       Direction::EFFECT, subsystem2);
    Real effect_dist = hamming_emd(effect1_exp, effect2_exp);

    return cause_dist + effect_dist;
}

// Backwards compatibility version (uses uniform expansion)
inline Real concept_distance(const Concept& c1, const Concept& c2) {
    // Combine purviews
    NodeSet combined_cause = bits::unite(c1.cause.purview, c2.cause.purview);
    NodeSet combined_effect = bits::unite(c1.effect.purview, c2.effect.purview);

    // Expand with uniform for new nodes
    Repertoire cause1_exp = c1.cause.repertoire.expand(combined_cause);
    Repertoire cause2_exp = c2.cause.repertoire.expand(combined_cause);
    Real cause_dist = hamming_emd(cause1_exp, cause2_exp);

    Repertoire effect1_exp = c1.effect.repertoire.expand(combined_effect);
    Repertoire effect2_exp = c2.effect.repertoire.expand(combined_effect);
    Real effect_dist = hamming_emd(effect1_exp, effect2_exp);

    return cause_dist + effect_dist;
}

/**
 * Compute distance from concept to null concept.
 *
 * The null concept has:
 * - CAUSE: max entropy (uniform) when expanded to any purview
 * - EFFECT: unconstrained effect repertoire (empty mechanism) - NOT uniform!
 *
 * This asymmetry is because:
 * - Cause repertoire represents P(past | no mechanism) = uniform
 * - Effect repertoire represents P(future | no mechanism, current state) =
 *   constrained by TPM and current state, hence NOT uniform
 */
inline Real null_concept_distance(const Concept& c, const Subsystem& subsystem) {
    // Null cause is max entropy over the concept's cause purview
    Repertoire null_cause = Repertoire::max_entropy(c.cause.purview);

    // Null effect is the UNCONSTRAINED effect repertoire (empty mechanism)
    // This is NOT max entropy - it's constrained by TPM and current state
    Repertoire null_effect = subsystem.effect_repertoire(0, c.effect.purview);

    // Compare against concept's repertoire (same purview, no expansion needed)
    Real cause_dist = hamming_emd(c.cause.repertoire, null_cause);
    Real effect_dist = hamming_emd(c.effect.repertoire, null_effect);

    return cause_dist + effect_dist;
}

/**
 * Check if two concepts are "emd equal" (same mechanism and similar repertoires).
 *
 * Following PyPhi: expand repertoires to combined purviews and check EMD.
 * Concepts are equal if their repertoires have near-zero EMD when expanded.
 *
 * IMPORTANT: Each concept should be expanded using its own subsystem's
 * unconstrained repertoire. For simplicity, we use a single subsystem here.
 */
inline bool concepts_emd_eq(const Concept& c1, const Concept& c2,
                             const Subsystem& subsystem1, const Subsystem& subsystem2) {
    if (c1.mechanism != c2.mechanism) return false;

    // Expand repertoires to combined purviews
    NodeSet combined_cause = bits::unite(c1.cause.purview, c2.cause.purview);
    NodeSet combined_effect = bits::unite(c1.effect.purview, c2.effect.purview);

    // Expand using proper unconstrained repertoires
    Repertoire cause1_exp = expand_repertoire_proper(c1.cause.repertoire, combined_cause,
                                                      Direction::CAUSE, subsystem1);
    Repertoire cause2_exp = expand_repertoire_proper(c2.cause.repertoire, combined_cause,
                                                      Direction::CAUSE, subsystem2);
    Repertoire effect1_exp = expand_repertoire_proper(c1.effect.repertoire, combined_effect,
                                                       Direction::EFFECT, subsystem1);
    Repertoire effect2_exp = expand_repertoire_proper(c2.effect.repertoire, combined_effect,
                                                       Direction::EFFECT, subsystem2);

    Real cause_dist = hamming_emd(cause1_exp, cause2_exp);
    Real effect_dist = hamming_emd(effect1_exp, effect2_exp);

    return (cause_dist + effect_dist) < EPSILON;
}

// Backwards compatibility overload using single subsystem
inline bool concepts_emd_eq(const Concept& c1, const Concept& c2) {
    // Simple comparison without proper expansion - just use uniform expansion
    if (c1.mechanism != c2.mechanism) return false;

    NodeSet combined_cause = bits::unite(c1.cause.purview, c2.cause.purview);
    NodeSet combined_effect = bits::unite(c1.effect.purview, c2.effect.purview);

    Repertoire cause1_exp = c1.cause.repertoire.expand(combined_cause);
    Repertoire cause2_exp = c2.cause.repertoire.expand(combined_cause);
    Repertoire effect1_exp = c1.effect.repertoire.expand(combined_effect);
    Repertoire effect2_exp = c2.effect.repertoire.expand(combined_effect);

    Real cause_dist = hamming_emd(cause1_exp, cause2_exp);
    Real effect_dist = hamming_emd(effect1_exp, effect2_exp);

    return (cause_dist + effect_dist) < EPSILON;
}

/**
 * Compute Extended EMD between two CES structures.
 *
 * This computes the optimal transport cost to move concepts from
 * CES1 to CES2 in concept space.
 *
 * @param ces1 The unpartitioned CES
 * @param ces2 The partitioned CES
 * @param subsystem1 The unpartitioned subsystem (for ces1 concepts)
 * @param subsystem2 The cut subsystem (for ces2 concepts)
 */
inline Real ces_distance_xemd(const CES& ces1, const CES& ces2,
                               const Subsystem& subsystem1, const Subsystem& subsystem2) {
    // Find concepts unique to each CES
    // Use proper subsystems for expansion during comparison
    std::vector<const Concept*> unique_c1, unique_c2;

    for (const auto& c1 : ces1.concepts()) {
        bool found = false;
        for (const auto& c2 : ces2.concepts()) {
            if (concepts_emd_eq(c1, c2, subsystem1, subsystem2)) {
                found = true;
                break;
            }
        }
        if (!found) unique_c1.push_back(&c1);
    }

    for (const auto& c2 : ces2.concepts()) {
        bool found = false;
        for (const auto& c1 : ces1.concepts()) {
            if (concepts_emd_eq(c1, c2, subsystem1, subsystem2)) {
                found = true;
                break;
            }
        }
        if (!found) unique_c2.push_back(&c2);
    }

    // Simple case: only concepts disappeared (none created)
    if (unique_c1.empty() || unique_c2.empty()) {
        Real dist = 0.0;
        // Sum phi * distance_to_null for all destroyed/created concepts
        // Use each concept's own subsystem for null distance
        for (const auto* c : unique_c1) {
            dist += c->phi() * null_concept_distance(*c, subsystem1);
        }
        for (const auto* c : unique_c2) {
            dist += c->phi() * null_concept_distance(*c, subsystem2);
        }
        return dist;
    }

    // Full XEMD case: concepts both destroyed and created
    // Build the EMD problem following PyPhi's approach:
    // - d1 = [phi(C1_unique), 0..0 (M zeros), 0]
    // - d2 = [0..0 (N zeros), phi(C2_unique), phi_diff]
    // where phi_diff balances the distributions

    size_t N = unique_c1.size();
    size_t M = unique_c2.size();
    size_t total = N + M + 1;  // +1 for null concept

    // Build distance matrix
    std::vector<Real> dist_matrix(total * total, 0.0);
    Real max_dist = 0.0;

    // Pairwise distances between unique concepts from CES1 and CES2
    // Use proper subsystems for expansion
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            Real d = concept_distance(*unique_c1[i], *unique_c2[j], subsystem1, subsystem2);
            dist_matrix[i * total + (N + j)] = d;
            dist_matrix[(N + j) * total + i] = d;
            max_dist = std::max(max_dist, d);
        }
    }

    // Distances to null concept (last position)
    // Use each concept's own subsystem
    for (size_t i = 0; i < N; ++i) {
        Real d = null_concept_distance(*unique_c1[i], subsystem1);
        dist_matrix[i * total + (total - 1)] = d;
        dist_matrix[(total - 1) * total + i] = d;
        max_dist = std::max(max_dist, d);
    }
    for (size_t j = 0; j < M; ++j) {
        Real d = null_concept_distance(*unique_c2[j], subsystem2);
        dist_matrix[(N + j) * total + (total - 1)] = d;
        dist_matrix[(total - 1) * total + (N + j)] = d;
        max_dist = std::max(max_dist, d);
    }

    // Set diagonal blocks to large value (prevent moving within same CES)
    Real large = max_dist + 1.0;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            dist_matrix[i * total + j] = large;
        }
    }
    for (size_t i = N; i < N + M; ++i) {
        for (size_t j = N; j < N + M; ++j) {
            dist_matrix[i * total + j] = large;
        }
    }
    // Null to null is 0
    dist_matrix[(total - 1) * total + (total - 1)] = 0.0;

    // Build the two distributions (d1 and d2 in PyPhi)
    // d1 = [phi values from C1, zeros for C2 positions, 0 for null]
    // d2 = [zeros for C1 positions, phi values from C2, balance]
    std::vector<Real> d1(total, 0.0);
    std::vector<Real> d2(total, 0.0);

    for (size_t i = 0; i < N; ++i) {
        d1[i] = unique_c1[i]->phi();
    }
    for (size_t j = 0; j < M; ++j) {
        d2[N + j] = unique_c2[j]->phi();
    }

    // Balance: phi that "disappeared" goes to null in d2
    Real sum_d1 = 0.0, sum_d2 = 0.0;
    for (size_t i = 0; i < total; ++i) {
        sum_d1 += d1[i];
        sum_d2 += d2[i];
    }
    d2[total - 1] = sum_d1 - sum_d2;  // Can be negative if d2 > d1

    // If d2's null becomes negative, we need to adjust
    // This means partitioned CES has MORE phi, so we add to d1's null instead
    if (d2[total - 1] < 0) {
        d1[total - 1] = -d2[total - 1];
        d2[total - 1] = 0.0;
    }

    // Solve EMD using exact algorithm
    return exact_emd_ssp(d1, d2, dist_matrix, total);
}

/**
 * Create a subsystem with a cut applied.
 *
 * The cut severs connections from 'from_nodes' to 'to_nodes'.
 */
class CutSubsystem : public Subsystem {
public:
    CutSubsystem(const Network& network, NodeSet nodes, StateIndex state,
                 const SystemCut& cut)
        : Subsystem(network, nodes, state)
        , cut_(cut)
        , cut_cm_(create_cut_cm(network.cm(), cut))
    {}

    const ConnectivityMatrix& cm() const override { return cut_cm_; }

private:
    SystemCut cut_;
    ConnectivityMatrix cut_cm_;

    static ConnectivityMatrix create_cut_cm(const ConnectivityMatrix& original,
                                             const SystemCut& cut) {
        ConnectivityMatrix result = original;
        size_t n = original.num_nodes();

        for (NodeIndex from = 0; from < n; ++from) {
            for (NodeIndex to = 0; to < n; ++to) {
                if (cut.is_cut(from, to)) {
                    result(from, to) = 0;
                }
            }
        }

        return result;
    }
};

/**
 * Compute BigPhi (System Irreducibility Analysis).
 *
 * BigPhi measures how much a system is more than the sum of its parts.
 * It's the minimum "distance" between the unpartitioned CES and any
 * partitioned CES.
 */
inline SIA compute_sia(const Network& network, NodeSet nodes, StateIndex state) {
    SIA result;
    result.phi = 0.0;
    result.nodes = nodes;
    result.state = state;

    // Handle degenerate cases
    if (bits::popcount(nodes) <= 1) {
        return result;
    }

    // Create subsystem
    Subsystem subsystem(network, nodes, state);

    // Compute unpartitioned CES
    result.ces = compute_ces(subsystem);

    // If CES is empty, phi is 0
    if (result.ces.empty()) {
        return result;
    }

    // Generate all system cuts
    auto cuts = system_cuts(nodes);

    if (cuts.empty()) {
        // No valid cuts (shouldn't happen for n > 1)
        result.phi = result.ces.total_phi();
        return result;
    }

    // Find MIP (cut with minimum distance)
    // Using XEMD distance between CES structures
    result.phi = std::numeric_limits<Real>::max();

    for (const auto& cut : cuts) {
        // Create subsystem with cut
        CutSubsystem cut_subsystem(network, nodes, state, cut);

        // Compute partitioned CES
        CES partitioned = compute_ces(cut_subsystem);

        // XEMD distance between CES structures
        // Pass both subsystems for proper repertoire expansion
        Real distance = ces_distance_xemd(result.ces, partitioned, subsystem, cut_subsystem);

        // Track minimum distance cut
        if (distance < result.phi) {
            result.phi = distance;
            result.cut = cut;
            result.partitioned_ces = std::move(partitioned);
        }

        // Early exit if system is reducible (distance very close to 0)
        if (result.phi <= EPSILON) {
            result.phi = 0.0;
            break;
        }
    }

    // Ensure non-negative
    if (result.phi < 0.0) {
        result.phi = 0.0;
    }

    return result;
}

/**
 * Convenience function: compute BigPhi for a network in given state.
 */
inline Real compute_phi(const Network& network, StateIndex state) {
    NodeSet all_nodes = bits::full_set(network.num_nodes());
    SIA sia = compute_sia(network, all_nodes, state);
    return sia.phi;
}

}  // namespace phi
