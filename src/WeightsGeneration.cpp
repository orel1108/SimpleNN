#include "../include/WeightsGeneration.h"

#include <random>

///////////////////////////////////////////////////////////////////////////////

matrix_type GenerateNormalWeights(size_type i_num_rows,
                                  size_type i_num_cols) noexcept
{
  std::default_random_engine generator;
  std::normal_distribution<real_type> distribution(0.0, std::pow(i_num_rows, -0.5));

  matrix_type matrix(i_num_rows, i_num_cols);
  for (size_type r = 0; r < i_num_rows; ++r)
    for (size_type c = 0; c < i_num_cols; ++c)
      matrix(r, c) = distribution(generator);

  return matrix;
}

///////////////////////////////////////////////////////////////////////////////

matrix_type GenerateUniformWeights(size_type i_num_rows,
                                   size_type i_num_cols) noexcept
{
  std::default_random_engine generator;
  std::uniform_real_distribution<real_type> distribution(-0.5, 0.5);

  matrix_type matrix(i_num_rows, i_num_cols);
  for (size_type r = 0; r < i_num_rows; ++r)
    for (size_type c = 0; c < i_num_cols; ++c)
      matrix(r, c) = distribution(generator);

  return matrix;
}
