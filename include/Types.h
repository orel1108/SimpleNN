#pragma once

#include <boost/serialization/array_wrapper.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

#include <array>

/**
 * @brief Alias for values type.
 */
using real_type                 = double;

/**
 * @brief Alias for list of values.
 */
using list_type                 = boost::numeric::ublas::vector<real_type, std::vector<real_type>>;

/**
 * @brief Alias for weights matrix type.
 */
using matrix_type               = boost::numeric::ublas::matrix<real_type>;

/**
 * @brief Alias for weights matrix type.
 * @tparam TRow Number of rows in the matrix.
 * @tparam TCol Number of columns in the matrix.
 */
template<std::size_t TRow, std::size_t TCol>
using matrix_t = std::array<std::array<real_type, TCol>, TRow>;

/**
 * @brief Type of input.
 * @tparam TNum Number of nodes in input layer.
 */
template<std::size_t TNum>
using input_t = matrix_t<TNum, 1>;

/**
 * @brief Type of output.
 * @tparam TNum Number of nodes in output layer.
 */
template<std::size_t TNum>
using output_t = matrix_t<1, TNum>;
