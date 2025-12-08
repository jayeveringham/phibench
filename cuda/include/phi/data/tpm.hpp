#pragma once

#include "phi/core/types.hpp"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace phi {

/**
 * Transition Probability Matrix in state-by-node format.
 *
 * Storage: Flat array of size 2^n * n
 * Layout: tpm[state * n + node] = P(node = ON | state)
 *
 * State indexing is little-endian: state = s0 + 2*s1 + 4*s2 + ...
 * where s0 is the state of node 0, etc.
 */
class TPM {
public:
    TPM() : num_nodes_(0), num_states_(0) {}

    explicit TPM(size_t num_nodes)
        : num_nodes_(num_nodes)
        , num_states_(state::num_states(num_nodes))
        , data_(num_states_ * num_nodes, 0.0)
    {}

    TPM(size_t num_nodes, const Real* data)
        : num_nodes_(num_nodes)
        , num_states_(state::num_states(num_nodes))
        , data_(data, data + num_states_ * num_nodes)
    {}

    TPM(size_t num_nodes, std::vector<Real>&& data)
        : num_nodes_(num_nodes)
        , num_states_(state::num_states(num_nodes))
        , data_(std::move(data))
    {
        if (data_.size() != num_states_ * num_nodes_) {
            throw std::invalid_argument("TPM data size mismatch");
        }
    }

    // Basic properties
    size_t num_nodes() const { return num_nodes_; }
    size_t num_states() const { return num_states_; }
    size_t size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }

    // Access probability: P(node = ON | state)
    Real& operator()(StateIndex state, NodeIndex node) {
        return data_[state * num_nodes_ + node];
    }

    Real operator()(StateIndex state, NodeIndex node) const {
        return data_[state * num_nodes_ + node];
    }

    // Get probability that node takes value 'value' given state
    Real prob(StateIndex state, NodeIndex node, uint8_t value) const {
        Real p_on = (*this)(state, node);
        return value ? p_on : (1.0 - p_on);
    }

    // Raw data access
    const Real* data() const { return data_.data(); }
    Real* data() { return data_.data(); }
    const std::vector<Real>& vec() const { return data_; }

    /**
     * Condition TPM on fixed node states.
     *
     * Returns a new TPM where the specified nodes are fixed to given states.
     * The returned TPM has the same number of nodes, but the fixed node
     * dimensions are effectively collapsed (they only have one valid state).
     *
     * @param fixed_nodes Bitmask of nodes to fix
     * @param fixed_state State values for fixed nodes (packed into StateIndex)
     * @return Conditioned TPM
     */
    TPM condition_on(NodeSet fixed_nodes, StateIndex fixed_state) const {
        TPM result(num_nodes_);

        // For each possible state of non-fixed nodes
        NodeSet free_nodes = bits::difference(bits::full_set(num_nodes_), fixed_nodes);
        size_t num_free = bits::popcount(free_nodes);
        size_t num_free_states = state::num_states(num_free);

        for (StateIndex free_idx = 0; free_idx < num_free_states; ++free_idx) {
            // Expand free state to full state, filling in fixed values
            StateIndex full_state = expand_state(free_idx, free_nodes, fixed_state, fixed_nodes);

            // Copy probabilities for this state
            for (NodeIndex node = 0; node < num_nodes_; ++node) {
                result(full_state, node) = (*this)(full_state, node);
            }
        }

        return result;
    }

    /**
     * Marginalize out nodes from TPM.
     *
     * Averages over all states of the specified nodes.
     *
     * @param nodes_to_remove Bitmask of nodes to marginalize out
     * @return Marginalized TPM
     */
    TPM marginalize_out(NodeSet nodes_to_remove) const {
        if (nodes_to_remove == 0) {
            return *this;
        }

        TPM result(num_nodes_);
        std::fill(result.data_.begin(), result.data_.end(), 0.0);

        NodeSet kept_nodes = bits::difference(bits::full_set(num_nodes_), nodes_to_remove);
        size_t num_removed = bits::popcount(nodes_to_remove);
        size_t num_removed_states = state::num_states(num_removed);
        Real normalizer = 1.0 / static_cast<Real>(num_removed_states);

        // For each state
        for (StateIndex s = 0; s < num_states_; ++s) {
            // Get the "representative" state (marginalized state)
            StateIndex kept_state = state::extract_bits(s, kept_nodes);
            StateIndex rep_state = state::expand_bits(kept_state, kept_nodes);

            // Accumulate probabilities
            for (NodeIndex node = 0; node < num_nodes_; ++node) {
                result(rep_state, node) += (*this)(s, node) * normalizer;
            }
        }

        return result;
    }

    /**
     * Validate TPM probabilities are in [0, 1].
     */
    bool is_valid() const {
        for (Real p : data_) {
            if (p < 0.0 || p > 1.0) return false;
        }
        return true;
    }

    /**
     * Check if TPM is deterministic (all probabilities are 0 or 1).
     */
    bool is_deterministic() const {
        for (Real p : data_) {
            if (!fp::is_zero(p) && !fp::equal(p, 1.0)) return false;
        }
        return true;
    }

private:
    size_t num_nodes_;
    StateIndex num_states_;
    std::vector<Real> data_;

    // Helper: expand a packed state (only free node bits) to full state
    static StateIndex expand_state(StateIndex free_state, NodeSet free_nodes,
                                   StateIndex fixed_state, NodeSet fixed_nodes) {
        StateIndex result = 0;
        size_t free_pos = 0;
        size_t fixed_pos = 0;

        for (NodeIndex i = 0; i < 64; ++i) {
            if (bits::contains(free_nodes, i)) {
                if ((free_state >> free_pos) & 1) {
                    result |= (StateIndex{1} << i);
                }
                ++free_pos;
            } else if (bits::contains(fixed_nodes, i)) {
                if ((fixed_state >> fixed_pos) & 1) {
                    result |= (StateIndex{1} << i);
                }
                ++fixed_pos;
            }
        }
        return result;
    }
};

/**
 * Connectivity Matrix representing network structure.
 *
 * cm(i, j) = 1 if there is an edge from node i to node j.
 */
class ConnectivityMatrix {
public:
    ConnectivityMatrix() : num_nodes_(0) {}

    explicit ConnectivityMatrix(size_t num_nodes)
        : num_nodes_(num_nodes)
        , data_(num_nodes * num_nodes, 0)
    {}

    ConnectivityMatrix(size_t num_nodes, const std::vector<uint8_t>& data)
        : num_nodes_(num_nodes)
        , data_(data)
    {
        if (data_.size() != num_nodes * num_nodes) {
            throw std::invalid_argument("CM data size mismatch");
        }
    }

    size_t num_nodes() const { return num_nodes_; }

    // Access: cm(from, to) = 1 if edge from -> to exists
    uint8_t& operator()(NodeIndex from, NodeIndex to) {
        return data_[from * num_nodes_ + to];
    }

    uint8_t operator()(NodeIndex from, NodeIndex to) const {
        return data_[from * num_nodes_ + to];
    }

    // Check if there's an edge from any node in 'from_set' to node 'to'
    bool has_input_from(NodeSet from_set, NodeIndex to) const {
        bool result = false;
        bits::for_each(from_set, [&](NodeIndex from) {
            if ((*this)(from, to)) result = true;
        });
        return result;
    }

    // Get all inputs to a node as a NodeSet
    NodeSet inputs_to(NodeIndex node) const {
        NodeSet result = 0;
        for (NodeIndex i = 0; i < num_nodes_; ++i) {
            if ((*this)(i, node)) {
                result = bits::add(result, i);
            }
        }
        return result;
    }

    // Get all outputs from a node as a NodeSet
    NodeSet outputs_from(NodeIndex node) const {
        NodeSet result = 0;
        for (NodeIndex j = 0; j < num_nodes_; ++j) {
            if ((*this)(node, j)) {
                result = bits::add(result, j);
            }
        }
        return result;
    }

    // Check if mechanism has any connection to purview
    bool is_connected(NodeSet mechanism, NodeSet purview, Direction dir) const {
        if (dir == Direction::CAUSE) {
            // For cause: check if any purview node has output to mechanism
            bool connected = false;
            bits::for_each(purview, [&](NodeIndex p) {
                bits::for_each(mechanism, [&](NodeIndex m) {
                    if ((*this)(p, m)) connected = true;
                });
            });
            return connected;
        } else {
            // For effect: check if any mechanism node has output to purview
            bool connected = false;
            bits::for_each(mechanism, [&](NodeIndex m) {
                bits::for_each(purview, [&](NodeIndex p) {
                    if ((*this)(m, p)) connected = true;
                });
            });
            return connected;
        }
    }

    const std::vector<uint8_t>& vec() const { return data_; }

private:
    size_t num_nodes_;
    std::vector<uint8_t> data_;
};

/**
 * Network combining TPM and connectivity matrix.
 */
class Network {
public:
    Network() = default;

    Network(TPM tpm, ConnectivityMatrix cm)
        : tpm_(std::move(tpm))
        , cm_(std::move(cm))
    {
        if (tpm_.num_nodes() != cm_.num_nodes()) {
            throw std::invalid_argument("TPM and CM node count mismatch");
        }
    }

    const TPM& tpm() const { return tpm_; }
    const ConnectivityMatrix& cm() const { return cm_; }
    size_t num_nodes() const { return tpm_.num_nodes(); }

private:
    TPM tpm_;
    ConnectivityMatrix cm_;
};

}  // namespace phi
