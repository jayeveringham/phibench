#pragma once

/**
 * PhiCUDA: High-performance C++/CUDA implementation of IIT Phi
 *
 * This is the main header file that includes all components.
 */

// Core types and utilities
#include "phi/core/types.hpp"

// Data structures
#include "phi/data/tpm.hpp"
#include "phi/data/repertoire.hpp"

// Partition generation
#include "phi/partition/bipartition.hpp"

// Metrics
#include "phi/metrics/emd.hpp"

// Computation algorithms
#include "phi/compute/small_phi.hpp"
#include "phi/compute/big_phi.hpp"

namespace phi {

/**
 * Version information
 */
constexpr const char* VERSION = "0.1.0";
constexpr const char* VERSION_NAME = "Alpha";

}  // namespace phi
