#pragma once

#include "phi/core/types.hpp"
#include "phi/data/tpm.hpp"
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace phi {

/**
 * Probability distribution over purview states.
 *
 * A repertoire represents a probability distribution over the states
 * of a subset of nodes (the purview). It's stored as a flat array
 * indexed by purview state (little-endian).
 */
class Repertoire {
public:
    Repertoire() : purview_(0), num_states_(0) {}

    explicit Repertoire(NodeSet purview)
        : purview_(purview)
        , num_states_(state::num_states(bits::popcount(purview)))
        , data_(num_states_, 0.0)
    {}

    Repertoire(NodeSet purview, std::vector<Real>&& data)
        : purview_(purview)
        , num_states_(state::num_states(bits::popcount(purview)))
        , data_(std::move(data))
    {
        if (data_.size() != num_states_) {
            throw std::invalid_argument("Repertoire data size mismatch");
        }
    }

    Repertoire(NodeSet purview, const std::vector<Real>& data)
        : purview_(purview)
        , num_states_(state::num_states(bits::popcount(purview)))
        , data_(data)
    {
        if (data_.size() != num_states_) {
            throw std::invalid_argument("Repertoire data size mismatch");
        }
    }

    // Basic properties
    NodeSet purview() const { return purview_; }
    size_t purview_size() const { return bits::popcount(purview_); }
    size_t num_states() const { return num_states_; }
    bool empty() const { return num_states_ == 0; }

    // Access by purview state index
    Real& operator[](StateIndex idx) { return data_[idx]; }
    Real operator[](StateIndex idx) const { return data_[idx]; }

    // Raw data access
    const Real* data() const { return data_.data(); }
    Real* data() { return data_.data(); }
    const std::vector<Real>& vec() const { return data_; }

    // Sum of all probabilities
    Real sum() const {
        return std::accumulate(data_.begin(), data_.end(), Real{0});
    }

    // Normalize to sum to 1
    void normalize() {
        Real s = sum();
        if (s > EPSILON) {
            for (Real& p : data_) {
                p /= s;
            }
        }
    }

    // Create normalized copy
    Repertoire normalized() const {
        Repertoire result = *this;
        result.normalize();
        return result;
    }

    // Check if uniform distribution
    bool is_uniform() const {
        if (num_states_ == 0) return true;
        Real expected = 1.0 / static_cast<Real>(num_states_);
        for (Real p : data_) {
            if (!fp::equal(p, expected)) return false;
        }
        return true;
    }

    // Maximum entropy (uniform) distribution
    static Repertoire max_entropy(NodeSet purview) {
        size_t n = state::num_states(bits::popcount(purview));
        Real p = 1.0 / static_cast<Real>(n);
        return Repertoire(purview, std::vector<Real>(n, p));
    }

    /**
     * Expand repertoire to a larger purview.
     *
     * New nodes are filled with max-entropy distribution (uniform).
     * This is used when comparing repertoires with different purviews.
     */
    Repertoire expand(NodeSet new_purview) const {
        if (new_purview == purview_) return *this;

        // New purview must contain old purview
        if (!bits::is_subset(purview_, new_purview)) {
            throw std::invalid_argument("New purview must contain old purview");
        }

        NodeSet new_nodes = bits::difference(new_purview, purview_);
        size_t new_purview_size = bits::popcount(new_purview);
        size_t new_num_states = state::num_states(new_purview_size);
        size_t num_new_states = state::num_states(bits::popcount(new_nodes));

        // New distribution is uniform over new nodes
        Real factor = 1.0 / static_cast<Real>(num_new_states);

        Repertoire result(new_purview);

        for (StateIndex new_state = 0; new_state < new_num_states; ++new_state) {
            // Extract the old purview state from new state
            StateIndex old_state = extract_substate(new_state, new_purview, purview_);
            result[new_state] = data_[old_state] * factor;
        }

        return result;
    }

    /**
     * Compute product of two independent repertoires.
     *
     * The purviews must be disjoint.
     */
    static Repertoire product(const Repertoire& a, const Repertoire& b) {
        // Handle empty cases
        if (a.num_states_ == 0) return b;
        if (b.num_states_ == 0) return a;

        // Purviews must be disjoint
        if (bits::intersect(a.purview_, b.purview_) != 0) {
            throw std::invalid_argument("Repertoire product requires disjoint purviews");
        }

        NodeSet combined_purview = bits::unite(a.purview_, b.purview_);
        size_t combined_size = bits::popcount(combined_purview);
        size_t combined_states = state::num_states(combined_size);

        Repertoire result(combined_purview);

        for (StateIndex s = 0; s < combined_states; ++s) {
            StateIndex a_state = extract_substate(s, combined_purview, a.purview_);
            StateIndex b_state = extract_substate(s, combined_purview, b.purview_);
            result[s] = a[a_state] * b[b_state];
        }

        return result;
    }

    /**
     * Compute marginal probability of a node being OFF (state 0).
     *
     * Used for effect_emd closed-form computation.
     */
    Real marginal_off(size_t purview_node_idx) const {
        Real sum_off = 0.0;
        for (StateIndex s = 0; s < num_states_; ++s) {
            if (!state::get_bit(s, static_cast<NodeIndex>(purview_node_idx))) {
                sum_off += data_[s];
            }
        }
        return sum_off;
    }

private:
    NodeSet purview_;
    StateIndex num_states_;
    std::vector<Real> data_;

    // Extract bits corresponding to sub_purview from a state over full_purview
    static StateIndex extract_substate(StateIndex full_state, NodeSet full_purview,
                                       NodeSet sub_purview) {
        StateIndex result = 0;
        size_t full_pos = 0;
        size_t sub_pos = 0;

        bits::for_each(full_purview, [&](NodeIndex node) {
            if (bits::contains(sub_purview, node)) {
                if ((full_state >> full_pos) & 1) {
                    result |= (StateIndex{1} << sub_pos);
                }
                ++sub_pos;
            }
            ++full_pos;
        });

        return result;
    }
};

/**
 * Subsystem representing a subset of nodes in a specific state.
 *
 * This is the main computation context for repertoires and phi values.
 */
class Subsystem {
public:
    Subsystem(const Network& network, NodeSet nodes, StateIndex system_state)
        : network_(&network)
        , nodes_(nodes)
        , system_state_(system_state)
    {}

    virtual ~Subsystem() = default;

    const Network& network() const { return *network_; }
    const TPM& tpm() const { return network_->tpm(); }
    virtual const ConnectivityMatrix& cm() const { return network_->cm(); }
    NodeSet nodes() const { return nodes_; }
    StateIndex state() const { return system_state_; }
    size_t size() const { return bits::popcount(nodes_); }

    /**
     * Compute cause repertoire: P(purview | mechanism in current state)
     */
    Repertoire cause_repertoire(NodeSet mechanism, NodeSet purview) const {
        if (purview == 0) {
            // Empty purview -> multiplicative identity
            return Repertoire(0, {1.0});
        }
        if (mechanism == 0) {
            // No mechanism -> max entropy over purview
            return Repertoire::max_entropy(purview);
        }

        // Product of single-node cause repertoires
        Repertoire result = single_node_cause_repertoire(
            bits::lowest_bit(mechanism), purview);

        NodeSet remaining = bits::clear_lowest(mechanism);
        bits::for_each(remaining, [&](NodeIndex m) {
            Repertoire single = single_node_cause_repertoire(m, purview);
            result = multiply_repertoires(result, single);
        });

        result.normalize();
        return result;
    }

    /**
     * Compute effect repertoire: P(purview | mechanism in current state)
     */
    Repertoire effect_repertoire(NodeSet mechanism, NodeSet purview) const {
        if (purview == 0) {
            return Repertoire(0, {1.0});
        }

        // Get mechanism state
        StateIndex mech_state = state::extract_bits(system_state_, mechanism);

        // Product of single-node effect repertoires (disjoint purviews)
        NodeIndex first_purview = bits::lowest_bit(purview);
        Repertoire result = single_node_effect_repertoire(first_purview, mechanism, mech_state);

        NodeSet remaining = bits::clear_lowest(purview);
        bits::for_each(remaining, [&](NodeIndex p) {
            Repertoire single = single_node_effect_repertoire(p, mechanism, mech_state);
            result = Repertoire::product(result, single);  // Product for disjoint purviews
        });

        return result;
    }

    /**
     * Compute repertoire based on direction.
     */
    Repertoire repertoire(Direction dir, NodeSet mechanism, NodeSet purview) const {
        if (dir == Direction::CAUSE) {
            return cause_repertoire(mechanism, purview);
        } else {
            return effect_repertoire(mechanism, purview);
        }
    }

private:
    const Network* network_;
    NodeSet nodes_;
    StateIndex system_state_;

    /**
     * Single-node cause repertoire contribution using Bayesian inversion.
     *
     * The cause repertoire uses the BACKWARD probability:
     *   P(past_purview_state | mechanism_node_in_current_state)
     *
     * Using Bayes' theorem with uniform prior:
     *   P(past | current) proportional to P(current | past)
     *
     * For a mechanism node in state mech_state:
     *   P(purview_past | mech_node = mech_state) proportional to
     *   P(mech_node = mech_state | purview_past) * uniform_prior
     *
     * We marginalize over nodes not in the EFFECTIVE purview (i.e., nodes
     * not actually connected to this mechanism via the current CM).
     * Cut connections result in uniform marginals for those purview nodes.
     */
    Repertoire single_node_cause_repertoire(NodeIndex mech_node, NodeSet purview) const {
        uint8_t mech_state = state::get_bit(system_state_, mech_node);

        // Get inputs to mechanism node from the CURRENT CM (may be cut)
        NodeSet cm_inputs = cm().inputs_to(mech_node);

        // Effective purview: only nodes that ACTUALLY affect this mechanism
        // Nodes in purview but not in cm_inputs are CUT - they become uniform
        NodeSet effective_purview = bits::intersect(cm_inputs, purview);

        // If mechanism has no inputs from purview, distribution is uniform
        if (effective_purview == 0) {
            return Repertoire::max_entropy(purview);
        }

        // Nodes to marginalize over: everything not in effective_purview
        // This includes: non-purview nodes AND cut purview nodes
        NodeSet marginalize_nodes = bits::difference(nodes_, effective_purview);

        size_t eff_purview_size = bits::popcount(effective_purview);
        size_t num_eff_purview_states = state::num_states(eff_purview_size);
        size_t num_marginalize_states = state::num_states(bits::popcount(marginalize_nodes));

        // First compute repertoire over EFFECTIVE purview only
        Repertoire eff_result(effective_purview);

        // For each effective purview state, compute P(mech_state | eff_purview_state)
        // marginalized over all other nodes (with uniform prior)
        for (StateIndex eps = 0; eps < num_eff_purview_states; ++eps) {
            Real sum_prob = 0.0;

            // Marginalize over all non-effective-purview nodes
            for (StateIndex ms = 0; ms < num_marginalize_states; ++ms) {
                // Build full network state
                StateIndex full_state = system_state_;

                // Set effective purview nodes
                size_t eff_pos = 0;
                bits::for_each(effective_purview, [&](NodeIndex node) {
                    uint8_t val = (eps >> eff_pos) & 1;
                    full_state = state::set_bit(full_state, node, val);
                    ++eff_pos;
                });

                // Set marginalized nodes
                size_t m_pos = 0;
                bits::for_each(marginalize_nodes, [&](NodeIndex node) {
                    uint8_t val = (ms >> m_pos) & 1;
                    full_state = state::set_bit(full_state, node, val);
                    ++m_pos;
                });

                // P(mech_node = mech_state | full_state)
                sum_prob += tpm().prob(full_state, mech_node, mech_state);
            }

            // Average over marginalized nodes (uniform prior)
            eff_result[eps] = sum_prob / static_cast<Real>(num_marginalize_states);
        }

        // Expand to full purview (cut nodes become uniform)
        if (effective_purview == purview) {
            return eff_result;
        }
        return eff_result.expand(purview);
    }

    // Single-node effect repertoire contribution
    //
    // PyPhi's approach:
    // 1. Each node has a TPM pre-marginalized over non-inputs
    // 2. Condition on mechanism_inputs (intersection of mechanism and actual inputs)
    // 3. Marginalize over non-mechanism inputs
    //
    // We simulate this by marginalizing over ALL nodes not in mech_inputs.
    // This includes both non-inputs (cut connections) and non-mechanism inputs.
    Repertoire single_node_effect_repertoire(NodeIndex purview_node,
                                              NodeSet mechanism,
                                              StateIndex mech_state) const {
        // Effective inputs: only mechanism nodes that are actually connected
        NodeSet mech_inputs = bits::intersect(cm().inputs_to(purview_node), mechanism);

        // Build the conditioning state from mechanism inputs
        StateIndex condition_state = 0;
        size_t mech_pos = 0;
        bits::for_each(mechanism, [&](NodeIndex m) {
            if (bits::contains(mech_inputs, m)) {
                // Set this mechanism node to its state
                if ((mech_state >> mech_pos) & 1) {
                    condition_state = state::set_bit(condition_state, m, 1);
                }
            }
            ++mech_pos;
        });

        // Marginalize over ALL nodes not in mech_inputs
        // This includes: (1) non-inputs due to cut, (2) inputs not in mechanism
        NodeSet marginalize_nodes = bits::difference(nodes_, mech_inputs);
        size_t num_margin = bits::popcount(marginalize_nodes);
        size_t num_margin_states = state::num_states(num_margin);

        // Compute P(purview_node = ON) averaged over marginalized nodes
        Real sum_prob = 0.0;
        for (StateIndex ms = 0; ms < num_margin_states; ++ms) {
            // Build full state: condition_state + marginalized nodes' states
            StateIndex full_state = condition_state;

            size_t pos = 0;
            bits::for_each(marginalize_nodes, [&](NodeIndex m) {
                if ((ms >> pos) & 1) {
                    full_state = state::set_bit(full_state, m, 1);
                }
                ++pos;
            });

            sum_prob += tpm()(full_state, purview_node);
        }

        Real avg_prob = sum_prob / static_cast<Real>(num_margin_states);

        // Return repertoire: [P(node=0), P(node=1)]
        NodeSet single_purview = bits::add(0, purview_node);
        return Repertoire(single_purview, {1.0 - avg_prob, avg_prob});
    }

    // Expand purview state to full network state
    StateIndex expand_purview_state(StateIndex purview_state, NodeSet purview) const {
        StateIndex result = system_state_;
        size_t pos = 0;
        bits::for_each(purview, [&](NodeIndex node) {
            uint8_t val = (purview_state >> pos) & 1;
            result = state::set_bit(result, node, val);
            ++pos;
        });
        return result;
    }

    // Multiply two repertoires with same purview (element-wise)
    static Repertoire multiply_repertoires(const Repertoire& a, const Repertoire& b) {
        if (a.purview() != b.purview()) {
            throw std::invalid_argument("Repertoires must have same purview for multiplication");
        }
        Repertoire result(a.purview());
        for (size_t i = 0; i < a.num_states(); ++i) {
            result[i] = a[i] * b[i];
        }
        return result;
    }
};

}  // namespace phi
