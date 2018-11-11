#include "../include/NeuralNetwork.h"

#include <algorithm>
#include <random>

using namespace boost::numeric::ublas;

///////////////////////////////////////////////////////////////////////////////

void NeuralNetwork::Train
(
  const list_type& inputs,
  const list_type& expected
)
noexcept
{
  if (inputs.size() != m_layer_nodes_number[0] || expected.size() != m_layer_nodes_number[2])
    return;

  // calculate outputs from hidden layer
  const auto hidden_outputs = _Apply(0, inputs);
  // calculate outputs from output layer
  const auto final_outputs  = _Apply(1, hidden_outputs);

  // calculate error on output layer
  const auto output_error = expected - final_outputs;
  // calculate error on hidden layer
  const auto hidden_error = prod(trans(m_weights[1]), output_error);

  // update weights between hidden and output layers
  m_weights[1] += m_learning_rate * outer_prod(
                               element_prod(
                                 element_prod(output_error, final_outputs),
                                 list_type(final_outputs.size(), 1.0) - final_outputs
                               ),
                               trans(hidden_outputs)
                             );

  // update weights between input and hidden and layers
  m_weights[0] += m_learning_rate * outer_prod(
                              element_prod(
                                element_prod(hidden_error, hidden_outputs),
                                list_type(hidden_outputs.size(), 1.0) - hidden_outputs
                              ),
                              trans(inputs)
                            );
}

///////////////////////////////////////////////////////////////////////////////

list_type NeuralNetwork::Query
(
  const list_type& inputs
)
const noexcept
{
  if (inputs.size() != m_layer_nodes_number[0])
    return {};

  return _Apply(1,
                _Apply(0, inputs));
}

///////////////////////////////////////////////////////////////////////////////

list_type NeuralNetwork::_Apply
(
  size_type index,
  const list_type& input
)
const noexcept
{
  list_type result = prod(m_weights[index], input);
  std::for_each(result.begin(), result.end(), m_activation_function);
  return result;
}
