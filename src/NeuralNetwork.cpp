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

  std::vector<real_type> hidden_inputs(m_hidden_nodes, 0);
  for (size_type r = 0; r < m_hidden_nodes; ++r)
    for (size_type c = 0; c < m_input_nodes; ++c)
      hidden_inputs[r] += m_weights_input_hidden(r, c) * i_inputs[c];

  std::for_each(hidden_inputs.begin(), hidden_inputs.end(), m_activation_function);

  std::vector<real_type> final_inputs(m_output_nodes, 0);
  for (size_type r = 0; r < m_output_nodes; ++r)
    for (size_type c = 0; c < m_hidden_nodes; ++c)
      final_inputs[r] += m_weights_hidden_output(r, c) * hidden_inputs[c];

  std::for_each(final_inputs.begin(), final_inputs.end(), m_activation_function);

  return final_inputs;
}
