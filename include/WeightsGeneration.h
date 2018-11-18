#pragma once

#include "Types.h"

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
