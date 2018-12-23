#pragma once

#include <vector>

namespace nn
{
  /**
   * @brief Matrix row type definition.
   * @param TElement Type of matrix row element.
   */
  template<class TElement>
  using row_t = std::vector<TElement>;

  /**
   * @brief Matrix type definition.
   * @param TElement Type of matrix element.
   */
  template<class TElement>
  using matrix_t = std::vector<row_t<TElement>>;
}
