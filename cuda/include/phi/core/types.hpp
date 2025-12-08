#pragma once

#include <cstdint>
#include <cstddef>
#include <bit>
#include <vector>
#include <array>
#include <initializer_list>
#include <functional>

namespace phi {

// Configurable precision via compile-time option
#ifdef PHI_REAL_TYPE
using Real = PHI_REAL_TYPE;
#else
using Real = double;
#endif

// Precision constants
constexpr int PRECISION_DIGITS = 10;
constexpr Real EPSILON = 1e-10;

// Node set represented as bitmask (supports up to 64 nodes)
using NodeSet = uint64_t;

// State index for 2^n state space
using StateIndex = uint32_t;

// Node index (0-63)
using NodeIndex = uint8_t;

// Direction for cause/effect computation
enum class Direction { CAUSE, EFFECT };

// Bit manipulation utilities for NodeSet
namespace bits {

// Count set bits
constexpr size_t popcount(NodeSet set) {
    return static_cast<size_t>(std::popcount(set));
}

// Check if node is in set
constexpr bool contains(NodeSet set, NodeIndex node) {
    return (set >> node) & 1;
}

// Add node to set
constexpr NodeSet add(NodeSet set, NodeIndex node) {
    return set | (NodeSet{1} << node);
}

// Remove node from set
constexpr NodeSet remove(NodeSet set, NodeIndex node) {
    return set & ~(NodeSet{1} << node);
}

// Create set from initializer list
constexpr NodeSet make_set(std::initializer_list<NodeIndex> nodes) {
    NodeSet result = 0;
    for (auto n : nodes) {
        result |= (NodeSet{1} << n);
    }
    return result;
}

// Get lowest set bit index (undefined if set == 0)
constexpr NodeIndex lowest_bit(NodeSet set) {
    return static_cast<NodeIndex>(std::countr_zero(set));
}

// Clear lowest set bit
constexpr NodeSet clear_lowest(NodeSet set) {
    return set & (set - 1);
}

// Iterate over all set bits, calling f(node_index) for each
template<typename Func>
void for_each(NodeSet set, Func&& f) {
    while (set) {
        NodeIndex node = lowest_bit(set);
        f(node);
        set = clear_lowest(set);
    }
}

// Convert node set to vector of indices
inline std::vector<NodeIndex> to_vector(NodeSet set) {
    std::vector<NodeIndex> result;
    result.reserve(popcount(set));
    for_each(set, [&](NodeIndex n) { result.push_back(n); });
    return result;
}

// Create full set of n nodes: {0, 1, ..., n-1}
constexpr NodeSet full_set(size_t n) {
    if (n >= 64) return ~NodeSet{0};
    return (NodeSet{1} << n) - 1;
}

// Intersection
constexpr NodeSet intersect(NodeSet a, NodeSet b) {
    return a & b;
}

// Union
constexpr NodeSet unite(NodeSet a, NodeSet b) {
    return a | b;
}

// Difference (a - b)
constexpr NodeSet difference(NodeSet a, NodeSet b) {
    return a & ~b;
}

// Check if a is subset of b
constexpr bool is_subset(NodeSet a, NodeSet b) {
    return (a & b) == a;
}

}  // namespace bits

// State conversion utilities
namespace state {

// Convert state index to bit vector representation
inline std::vector<uint8_t> to_vector(StateIndex idx, size_t n) {
    std::vector<uint8_t> result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = (idx >> i) & 1;
    }
    return result;
}

// Convert bit vector to state index (little-endian)
inline StateIndex from_vector(const std::vector<uint8_t>& state) {
    StateIndex idx = 0;
    for (size_t i = 0; i < state.size(); ++i) {
        idx |= (static_cast<StateIndex>(state[i] & 1) << i);
    }
    return idx;
}

// Get bit at position
constexpr uint8_t get_bit(StateIndex state, NodeIndex pos) {
    return (state >> pos) & 1;
}

// Set bit at position
constexpr StateIndex set_bit(StateIndex state, NodeIndex pos, uint8_t value) {
    if (value) {
        return state | (StateIndex{1} << pos);
    } else {
        return state & ~(StateIndex{1} << pos);
    }
}

// Extract bits at positions specified by mask, pack into lower bits
// Example: extract_bits(0b11010, 0b10110) = 0b110 (bits at positions 1,2,4)
inline StateIndex extract_bits(StateIndex state, NodeSet mask) {
    StateIndex result = 0;
    size_t out_pos = 0;
    bits::for_each(mask, [&](NodeIndex pos) {
        if ((state >> pos) & 1) {
            result |= (StateIndex{1} << out_pos);
        }
        ++out_pos;
    });
    return result;
}

// Expand packed bits into positions specified by mask
// Inverse of extract_bits
inline StateIndex expand_bits(StateIndex packed, NodeSet mask) {
    StateIndex result = 0;
    size_t in_pos = 0;
    bits::for_each(mask, [&](NodeIndex pos) {
        if ((packed >> in_pos) & 1) {
            result |= (StateIndex{1} << pos);
        }
        ++in_pos;
    });
    return result;
}

// Number of states for n nodes
constexpr StateIndex num_states(size_t n) {
    return StateIndex{1} << n;
}

}  // namespace state

// Floating point comparison utilities
namespace fp {

constexpr bool is_zero(Real x) {
    return x >= -EPSILON && x <= EPSILON;
}

constexpr bool equal(Real a, Real b) {
    Real diff = a - b;
    return diff >= -EPSILON && diff <= EPSILON;
}

constexpr bool less_than(Real a, Real b) {
    return a < b - EPSILON;
}

constexpr bool greater_than(Real a, Real b) {
    return a > b + EPSILON;
}

}  // namespace fp

}  // namespace phi
