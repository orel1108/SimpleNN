#include "../include/NeuralNetwork.h"

#include <algorithm>
#include <random>

void NeuralNetwork::Train()
{

}

std::vector<real_type> NeuralNetwork::Query(const std::vector<real_type>& i_inputs) const
{
  if (i_inputs.size() != m_input_nodes)
    return {};

  return _Apply(m_weights_hidden_output,
           _Apply(m_weights_input_hidden, i_inputs));
}

///////////////////////////////////////////////////////////////////////////////

std::vector<real_type> NeuralNetwork::_Apply(const matrix_type& i_weights,
                                             const std::vector<real_type>& i_input) const noexcept
{
  std::vector<real_type> result(i_weights.size1(), 0);
  for (size_type r = 0; r < i_weights.size1(); ++r)
    for (size_type c = 0; c < i_weights.size2(); ++c)
      result[r] += i_weights(r, c) * i_input[c];

  std::for_each(result.begin(), result.end(), m_activation_function);
  return result;
}
