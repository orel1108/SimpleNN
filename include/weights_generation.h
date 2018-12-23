#pragma once

#include "types.h"

namespace nn
{
  /**
  * @brief Generates matrix of given size with randomly distributed coefficients.
  * @param[in] num_rows       Number of rows in resulting matrix.
  * @param[in] num_cols       Number of columns in resulting matrix.
  * @param[in] distribution   Distribution to be used during generation.
  * @return Matrix with randomly distributed coefficients.
  */
  template<class TDistribution>
  matrix_t<double> generate
  (
    std::size_t num_rows,
    std::size_t num_cols,
    TDistribution& distribution
  )
  noexcept;
}

#include "weights_generation_impl.h"
