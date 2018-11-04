#pragma once

#include "Types.h"

/**
 * @brief Generates random matrix with normally distributed coefficients.
 * @param[in] i_num_rows Number of rows in resulting matrix.
 * @param[in] i_num_cols Number of columns in resulting matrix.
 * @return Generated matrix.
 */
matrix_type GenerateNormalWeights(size_type i_num_rows,
                                  size_type i_num_cols) noexcept;

/**
 * @brief Generates random matrix with uniformly distributed coefficients.
 * @param[in] i_num_rows Number of rows in resulting matrix.
 * @param[in] i_num_cols Number of columns in resulting matrix.
 * @return Generated matrix.
 */
matrix_type GenerateUniformWeights(size_type i_num_rows,
                                   size_type i_num_cols) noexcept;
