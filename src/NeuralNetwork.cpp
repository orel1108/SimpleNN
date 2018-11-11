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
  if (inputs.size() != m_input_nodes || expected.size() != m_output_nodes)
    return;

  // calculate outputs from hidden layer
  const auto hidden_outputs = _Apply(m_weights_input_hidden, inputs);
  // calculate outputs from output layer
  const auto final_outputs  = _Apply(m_weights_hidden_output, hidden_outputs);

  // calculate error on output layer
  const auto output_error = expected - final_outputs;
  // calculate error on hidden layer
  const auto hidden_error = prod(trans(m_weights_hidden_output), output_error);

  // update weights between hidden and output layers
  m_weights_hidden_output += m_learning_rate * outer_prod(
                               element_prod(
                                 element_prod(output_error, final_outputs),
                                 list_type(final_outputs.size(), 1.0) - final_outputs
                               ),
                               trans(hidden_outputs)
                             );

  // update weights between input and hidden and layers
  m_weights_input_hidden += m_learning_rate * outer_prod(
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
  if (inputs.size() != m_input_nodes)
    return {};

  return _Apply(m_weights_hidden_output,
                _Apply(m_weights_input_hidden, inputs));
}

///////////////////////////////////////////////////////////////////////////////

list_type NeuralNetwork::_Apply
(
  const matrix_type& weights,
  const list_type& input
)
const noexcept
{
  list_type result = prod(weights, input);
  std::for_each(result.begin(), result.end(), m_activation_function);
  return result;
}
