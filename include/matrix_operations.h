#pragma once

#include "Types.h"

/**
 * @brief Multiplies two matrices.
 * @tparam K Number of rows in the first matrix.
 * @tparam L Number of columns in the first matrix and number of rows in the second matrix.
 * @tparam M Number of columns in the second matrix.
 */
template<std::size_t K, std::size_t L, std::size_t M>
matrix_t<K, M> prod
(
  const matrix_t<K, L>& i_left,
  const matrix_t<L, M>& i_right
)
noexcept
{
  matrix_t<K, M> res;
  for (std::size_t k = 0; k < K; ++k)
    for (std::size_t m = 0; m < M; ++m)
      for (std::size_t l = 0; l < L; ++l)
        res[k][m] += i_left[k][l] * i_right[l][m];

  return std::move(res);
}

/**
 * @brief Transposes the given matrix.
 * @tparam TRow Number of rows in the input matrix.
 * @tparam TCol Number of columns in the input matrix.
 */
template<std::size_t TRow, std::size_t TCol>
matrix_t<TCol, TRow> trans
(
  const matrix_t<TRow, TCol>& i_matrix
)
noexcept
{
  matrix_t<TCol, TRow> res;
  for (std::size_t r = 0; r < TRow; ++r)
    for (std::size_t c = 0; c < TCol; ++c)
      res[c][r] = i_matrix[r][c];

  return std::move(res);
}

/**
 * @brief Performs per element multiplication of the given matrices.
 * @tparam TRow Number of rows in the input matrices.
 * @tparam TCol Number of columns in the input matrices.
 */
template<std::size_t TRow, std::size_t TCol>
matrix_t<TRow, TCol> elem_prod
(
  const matrix_t<TRow, TCol>& i_left,
  const matrix_t<TRow, TCol>& i_right
)
noexcept
{
  matrix_t<TRow, TCol> res;
  for (std::size_t r = 0; r < TRow; ++r)
    for (std::size_t c = 0; c < TCol; ++c)
      res[r][c] = i_left[r][c] * i_right[r][c];

  return std::move(res);
}
