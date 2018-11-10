#pragma once

#include <boost/serialization/array_wrapper.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

/**
 * @brief Alias for size type.
 */
using size_type                 = std::size_t;

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
