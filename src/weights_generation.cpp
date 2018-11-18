#include "../include/weights_generation.h"

///////////////////////////////////////////////////////////////////////////////

matrix_type GenerateNormalWeights
(
  std::size_t num_rows,
  std::size_t num_cols
)
noexcept
{
  std::default_random_engine generator;
  std::normal_distribution<real_type> distribution(0.0, std::pow(num_rows, -0.5));

  matrix_type matrix(num_rows, num_cols);
  for (std::size_t r = 0; r < num_rows; ++r)
    for (std::size_t c = 0; c < num_cols; ++c)
      matrix(r, c) = distribution(generator);

  return matrix;
}

///////////////////////////////////////////////////////////////////////////////

matrix_type GenerateUniformWeights
(
  std::size_t num_rows,
  std::size_t num_cols
)
noexcept
{
  std::default_random_engine generator;
  std::uniform_real_distribution<real_type> distribution(-0.5, 0.5);

  matrix_type matrix(num_rows, num_cols);
  for (std::size_t r = 0; r < num_rows; ++r)
    for (std::size_t c = 0; c < num_cols; ++c)
      matrix(r, c) = distribution(generator);

  return matrix;
}
