#pragma once

#include <boost/serialization/array_wrapper.hpp>
#include <boost/numeric/ublas/matrix.hpp>

/**
 * @brief Alias for size type.
 */
using size_type                 = std::size_t;

/**
 * @brief Alias for values type.
 */
using real_type                 = double;

/**
 * @brief Alias for weights matrix type.
 */
using matrix_type               = boost::numeric::ublas::matrix<real_type>;
