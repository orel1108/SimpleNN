#pragma once

#include "types.h"

#include <random>

/**
 * @brief Generates random matrix with normally distributed coefficients.
 * @param[in] num_rows Number of rows in resulting matrix.
 * @param[in] num_cols Number of columns in resulting matrix.
 * @return Generated matrix.
 */
matrix_type GenerateNormalWeights
(
  std::size_t num_rows,
  std::size_t num_cols
)
noexcept;

/**
 * @brief Generates random matrix with uniformly distributed coefficients.
 * @param[in] num_rows Number of rows in resulting matrix.
 * @param[in] num_cols Number of columns in resulting matrix.
 * @return Generated matrix.
 */
matrix_type GenerateUniformWeights
(
  std::size_t num_rows,
  std::size_t num_cols
)
noexcept;

/**
 * @brief Generates random matrix with normally distributed coefficients.
 * @tparam[in] TRow Number of rows in resulting matrix.
 * @tparam[in] TCol Number of columns in resulting matrix.
 * @return Generated matrix.
 */
template<std::size_t TRow, std::size_t TCol>
matrix_t<TRow, TCol> generate_normal_weights
(
)
noexcept
{
  matrix_t<TRow, TCol> res;

  std::default_random_engine generator;
  std::normal_distribution<real_type> distribution(0.0, std::pow(TRow, -0.5));

  for (std::size_t r = 0; r < TRow; ++r)
    for (std::size_t c = 0; c < TCol; ++c)
      res[r][c] = distribution(generator);

  return std::move(res);
}

/**
 * @brief Generates random matrix with uniformly distributed coefficients.
 * @tparam[in] TRow Number of rows in resulting matrix.
 * @tparam[in] TCol Number of columns in resulting matrix.
 * @return Generated matrix.
 */
template<std::size_t TRow, std::size_t TCol>
matrix_t<TRow, TCol> generate_uniform_weights
(
)
noexcept
{
  matrix_t<TRow, TCol> res;

  std::default_random_engine generator;
  std::uniform_real_distribution<real_type> distribution(-0.5, 0.5);

  for (std::size_t r = 0; r < TRow; ++r)
    for (std::size_t c = 0; c < TCol; ++c)
      res[r][c] = distribution(generator);

  return std::move(res);
}
