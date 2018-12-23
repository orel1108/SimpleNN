#pragma once

#include <random>

namespace nn
{
  template<class TDistribution>
  matrix_t<double> generate
  (
    std::size_t num_rows,
    std::size_t num_cols,
    TDistribution& distribution
  )
  noexcept
  {
    std::default_random_engine generator;

    matrix_t<double> matrix(num_rows, row_t<double>(num_cols, 0.0));
    for (std::size_t r = 0; r < num_rows; ++r)
      for (std::size_t c = 0; c < num_cols; ++c)
        matrix[r][c] = distribution(generator);

    return std::move(matrix);
  }
}
