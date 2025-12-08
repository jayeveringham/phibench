#pragma once

#include "phi/core/types.hpp"
#include "phi/data/repertoire.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>

namespace phi {

/**
 * Compute Hamming distance between two states.
 *
 * The Hamming distance is the number of bits that differ.
 */
inline size_t hamming_distance(StateIndex a, StateIndex b) {
    return bits::popcount(static_cast<NodeSet>(a ^ b));
}

/**
 * Generate a Hamming distance matrix for n-bit states.
 *
 * Returns a 2^n x 2^n matrix where entry (i,j) is hamming_distance(i,j).
 */
inline std::vector<Real> hamming_matrix(size_t num_nodes) {
    size_t num_states = state::num_states(num_nodes);
    std::vector<Real> matrix(num_states * num_states);

    for (StateIndex i = 0; i < num_states; ++i) {
        for (StateIndex j = 0; j < num_states; ++j) {
            matrix[i * num_states + j] = static_cast<Real>(hamming_distance(i, j));
        }
    }

    return matrix;
}

/**
 * Compute effect EMD using closed-form solution (O(n)).
 *
 * For effect repertoires, nodes are independent, so EMD equals
 * the sum of absolute differences in marginal OFF probabilities.
 *
 * @param p First repertoire
 * @param q Second repertoire
 * @return EMD between p and q
 */
inline Real effect_emd(const Repertoire& p, const Repertoire& q) {
    if (p.purview() != q.purview()) {
        throw std::invalid_argument("Repertoires must have same purview for EMD");
    }

    size_t purview_size = p.purview_size();
    Real total = 0.0;

    for (size_t node_idx = 0; node_idx < purview_size; ++node_idx) {
        Real p_marginal_off = p.marginal_off(node_idx);
        Real q_marginal_off = q.marginal_off(node_idx);
        total += std::abs(p_marginal_off - q_marginal_off);
    }

    return total;
}

/**
 * Compute exact EMD using successive shortest paths algorithm.
 *
 * This is a simple O(n^3) implementation for small transport problems.
 * Uses Bellman-Ford for shortest paths to handle the augmenting paths correctly.
 */
inline Real exact_emd_ssp(const std::vector<Real>& p, const std::vector<Real>& q,
                           const std::vector<Real>& cost, size_t n) {
    // Successive shortest path algorithm for min-cost flow
    // Build network: source S, nodes 0..n-1 (supply), nodes n..2n-1 (demand), sink T

    const Real INF = 1e9;
    const size_t S = 2 * n;      // Source
    const size_t T = 2 * n + 1;  // Sink
    const size_t V = 2 * n + 2;  // Total vertices

    // Adjacency list with (to, capacity, cost, reverse_edge_index)
    struct Edge {
        size_t to;
        Real cap, cost;
        size_t rev;
    };
    std::vector<std::vector<Edge>> adj(V);

    auto add_edge = [&](size_t u, size_t v, Real cap, Real c) {
        adj[u].push_back({v, cap, c, adj[v].size()});
        adj[v].push_back({u, 0, -c, adj[u].size() - 1});
    };

    // Source to supply nodes
    for (size_t i = 0; i < n; ++i) {
        if (p[i] > EPSILON) {
            add_edge(S, i, p[i], 0);
        }
    }

    // Demand nodes to sink
    for (size_t j = 0; j < n; ++j) {
        if (q[j] > EPSILON) {
            add_edge(n + j, T, q[j], 0);
        }
    }

    // Supply to demand edges with cost
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            add_edge(i, n + j, INF, cost[i * n + j]);
        }
    }

    // Min-cost max-flow using SPFA (Bellman-Ford variant)
    Real total_cost = 0.0;
    Real total_flow = 0.0;
    Real required_flow = 0.0;
    for (size_t i = 0; i < n; ++i) required_flow += p[i];

    while (total_flow < required_flow - EPSILON) {
        // Find shortest path from S to T using Bellman-Ford
        std::vector<Real> dist(V, INF);
        std::vector<size_t> parent(V, V);
        std::vector<size_t> parent_edge(V, 0);
        dist[S] = 0;

        // SPFA
        std::vector<bool> in_queue(V, false);
        std::vector<size_t> queue_vec;
        queue_vec.push_back(S);
        in_queue[S] = true;

        size_t head = 0;
        while (head < queue_vec.size()) {
            size_t u = queue_vec[head++];
            in_queue[u] = false;

            for (size_t e = 0; e < adj[u].size(); ++e) {
                const auto& edge = adj[u][e];
                if (edge.cap > EPSILON && dist[u] + edge.cost < dist[edge.to] - EPSILON) {
                    dist[edge.to] = dist[u] + edge.cost;
                    parent[edge.to] = u;
                    parent_edge[edge.to] = e;
                    if (!in_queue[edge.to]) {
                        queue_vec.push_back(edge.to);
                        in_queue[edge.to] = true;
                    }
                }
            }
        }

        if (dist[T] > INF - 1) break;  // No augmenting path

        // Find min capacity along path
        Real path_flow = INF;
        for (size_t v = T; v != S; v = parent[v]) {
            size_t u = parent[v];
            path_flow = std::min(path_flow, adj[u][parent_edge[v]].cap);
        }

        // Limit flow to what's needed
        path_flow = std::min(path_flow, required_flow - total_flow);

        // Augment flow along path
        for (size_t v = T; v != S; v = parent[v]) {
            size_t u = parent[v];
            size_t e = parent_edge[v];
            adj[u][e].cap -= path_flow;
            adj[v][adj[u][e].rev].cap += path_flow;
        }

        total_flow += path_flow;
        total_cost += path_flow * dist[T];
    }

    return total_cost;
}

/**
 * Compute exact EMD using network simplex algorithm (legacy - for reference).
 *
 * This solves the optimal transport problem exactly for small purviews.
 * For larger cases, consider using Sinkhorn approximation.
 */
inline Real exact_emd(const std::vector<Real>& p, const std::vector<Real>& q,
                       const std::vector<Real>& cost, size_t n) {
    // Network simplex implementation for small transport problems
    // Based on minimum cost flow formulation

    // For very small cases, use simpler methods
    if (n <= 1) return 0.0;
    if (n == 2) return std::abs(p[0] - q[0]) * cost[1];  // Only one possible transport

    // Build the bipartite transport network
    // Sources: nodes 0..n-1 with supply p[i]
    // Sinks: nodes n..2n-1 with demand q[j]
    // Edges: from each source i to each sink j with cost cost[i*n+j]

    const size_t INF_ITER = 1000;
    const Real INF = 1e9;

    // Use the stepping stone / MODI method for transportation problem
    // Initialize using northwest corner rule, then optimize

    // Transport plan T[i][j] represents flow from source i to sink j
    std::vector<std::vector<Real>> T(n, std::vector<Real>(n, 0.0));

    // Copy supply and demand
    std::vector<Real> supply(p), demand(q);

    // Northwest corner rule initialization
    size_t i = 0, j = 0;
    while (i < n && j < n) {
        Real amount = std::min(supply[i], demand[j]);
        T[i][j] = amount;
        supply[i] -= amount;
        demand[j] -= amount;

        if (supply[i] < EPSILON) i++;
        if (demand[j] < EPSILON) j++;
    }

    // MODI method (Modified Distribution Method) for optimization
    // Iterate until no improvement possible

    for (size_t iter = 0; iter < INF_ITER; ++iter) {
        // Calculate dual variables u[i], v[j] such that u[i] + v[j] = cost[i,j] for basic cells
        std::vector<Real> u(n, INF), v(n, INF);

        // Find basic cells (T[i][j] > 0)
        std::vector<std::pair<size_t, size_t>> basic;
        for (size_t ii = 0; ii < n; ++ii) {
            for (size_t jj = 0; jj < n; ++jj) {
                if (T[ii][jj] > EPSILON) {
                    basic.emplace_back(ii, jj);
                }
            }
        }

        // Solve for u and v: u[i] + v[j] = cost[i*n+j] for basic cells
        // Start with u[0] = 0
        u[0] = 0.0;
        bool changed = true;
        size_t solve_iter = 0;
        while (changed && solve_iter++ < n * n) {
            changed = false;
            for (const auto& [ii, jj] : basic) {
                if (u[ii] < INF - 1 && v[jj] > INF - 1) {
                    v[jj] = cost[ii * n + jj] - u[ii];
                    changed = true;
                } else if (v[jj] < INF - 1 && u[ii] > INF - 1) {
                    u[ii] = cost[ii * n + jj] - v[jj];
                    changed = true;
                }
            }
        }

        // Fill in any unset dual variables
        for (size_t ii = 0; ii < n; ++ii) {
            if (u[ii] > INF - 1) u[ii] = 0.0;
        }
        for (size_t jj = 0; jj < n; ++jj) {
            if (v[jj] > INF - 1) v[jj] = 0.0;
        }

        // Find entering variable (most negative reduced cost)
        Real min_rc = -EPSILON;
        size_t enter_i = n, enter_j = n;
        for (size_t ii = 0; ii < n; ++ii) {
            for (size_t jj = 0; jj < n; ++jj) {
                if (T[ii][jj] < EPSILON) {  // Non-basic cell
                    Real rc = cost[ii * n + jj] - u[ii] - v[jj];
                    if (rc < min_rc) {
                        min_rc = rc;
                        enter_i = ii;
                        enter_j = jj;
                    }
                }
            }
        }

        // If no negative reduced cost, we're optimal
        if (enter_i >= n) break;

        // Find a cycle (loop) containing the entering cell
        // Use BFS to find augmenting path in bipartite graph

        // Simple approach: try to find a cycle by alternating rows/cols
        std::vector<std::pair<size_t, size_t>> cycle;
        cycle.emplace_back(enter_i, enter_j);

        // Build adjacency for basic cells
        std::vector<std::vector<size_t>> row_to_cols(n), col_to_rows(n);
        for (const auto& [ii, jj] : basic) {
            row_to_cols[ii].push_back(jj);
            col_to_rows[jj].push_back(ii);
        }

        // DFS to find cycle
        std::function<bool(size_t, size_t, bool, std::vector<std::pair<size_t,size_t>>&)> find_cycle;
        find_cycle = [&](size_t ci, size_t cj, bool move_in_row,
                         std::vector<std::pair<size_t,size_t>>& path) -> bool {
            if (path.size() > 1 && ci == enter_i && cj == enter_j) {
                return true;
            }
            if (path.size() > 2 * n) return false;

            if (move_in_row) {
                // Move along row ci to another column
                for (size_t nj : row_to_cols[ci]) {
                    if (nj != cj) {
                        path.emplace_back(ci, nj);
                        if (find_cycle(ci, nj, false, path)) return true;
                        path.pop_back();
                    }
                }
            } else {
                // Move along column cj to another row
                for (size_t ni : col_to_rows[cj]) {
                    if (ni != ci) {
                        path.emplace_back(ni, cj);
                        if (find_cycle(ni, cj, true, path)) return true;
                        path.pop_back();
                    }
                }
            }
            return false;
        };

        std::vector<std::pair<size_t,size_t>> path;
        path.emplace_back(enter_i, enter_j);
        bool found = false;
        for (size_t nj : row_to_cols[enter_i]) {
            path.emplace_back(enter_i, nj);
            if (find_cycle(enter_i, nj, false, path)) {
                found = true;
                break;
            }
            path.pop_back();
        }

        if (!found) {
            // Try starting from column
            path.clear();
            path.emplace_back(enter_i, enter_j);
            for (size_t ni : col_to_rows[enter_j]) {
                path.emplace_back(ni, enter_j);
                if (find_cycle(ni, enter_j, true, path)) {
                    found = true;
                    break;
                }
                path.pop_back();
            }
        }

        if (!found || path.size() < 4) {
            // Degenerate case - add a small amount to entering cell
            T[enter_i][enter_j] = EPSILON;
            continue;
        }

        // Find minimum flow on negative positions in cycle (odd indices)
        Real min_flow = INF;
        for (size_t k = 1; k < path.size(); k += 2) {
            min_flow = std::min(min_flow, T[path[k].first][path[k].second]);
        }

        // Adjust flow along cycle
        for (size_t k = 0; k < path.size(); ++k) {
            if (k % 2 == 0) {
                T[path[k].first][path[k].second] += min_flow;
            } else {
                T[path[k].first][path[k].second] -= min_flow;
            }
        }
    }

    // Calculate total cost
    Real total = 0.0;
    for (size_t ii = 0; ii < n; ++ii) {
        for (size_t jj = 0; jj < n; ++jj) {
            total += T[ii][jj] * cost[ii * n + jj];
        }
    }

    return total;
}

/**
 * Compute Hamming EMD (Earth Mover's Distance with Hamming cost).
 *
 * This is used for cause repertoires where the closed-form doesn't apply.
 * Uses exact network simplex algorithm for small purviews.
 *
 * @param p First repertoire
 * @param q Second repertoire
 * @return EMD between p and q
 */
inline Real hamming_emd(const Repertoire& p, const Repertoire& q) {
    if (p.purview() != q.purview()) {
        throw std::invalid_argument("Repertoires must have same purview for EMD");
    }

    size_t num_states = p.num_states();

    // For very small cases, use simpler methods
    if (num_states <= 1) return 0.0;
    if (num_states == 2) {
        // For 2 states: EMD is |p[0] - q[0]| * distance(0,1)
        // Hamming distance between 0 and 1 is 1
        return std::abs(p[0] - q[0]);
    }

    // Generate Hamming matrix
    auto cost = hamming_matrix(p.purview_size());

    // Convert to vectors
    std::vector<Real> pv(num_states), qv(num_states);
    for (size_t i = 0; i < num_states; ++i) {
        pv[i] = p[i];
        qv[i] = q[i];
    }

    return exact_emd_ssp(pv, qv, cost, num_states);
}

/**
 * Compute EMD based on direction.
 *
 * Uses closed-form effect_emd for EFFECT direction,
 * hamming_emd for CAUSE direction.
 */
inline Real emd(const Repertoire& p, const Repertoire& q, Direction direction) {
    if (direction == Direction::EFFECT) {
        return effect_emd(p, q);
    } else {
        return hamming_emd(p, q);
    }
}

/**
 * Compute repertoire distance (SmallPhi between unpartitioned and partitioned).
 */
inline Real repertoire_distance(const Repertoire& unpartitioned,
                                 const Repertoire& partitioned,
                                 Direction direction) {
    return emd(unpartitioned, partitioned, direction);
}

}  // namespace phi
