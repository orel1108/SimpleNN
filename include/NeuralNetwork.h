#pragma once

#include "WeightsGeneration.h"

#include <algorithm>
#include <random>

/**
 * @brief Implementation of simple Neural Network.
 * @brief TNodesPerLayer Number of nodes at each layer.
 * @note Only 3 layers are currently supported.
 */
template<size_type... TNodesPerLayer>
class NeuralNetwork final
{
  static constexpr size_type NUMBER_OF_LAYERS = sizeof...(TNodesPerLayer);
  static_assert(NUMBER_OF_LAYERS == 3);

  public:
    /**
     * @brief Creates Neural Network with given learning rate.
     * @param[in] learning_rate Learning rate of the network.
     */
    NeuralNetwork
    (
      real_type learning_rate
    )
    noexcept
      : m_layer_nodes_number{ std::forward<size_type>(TNodesPerLayer)... }
      , m_learning_rate(learning_rate)
      , m_activation_function([](real_type& value) { value = 1.0 / (1.0 + std::exp(-1.0 * value)); })
    {
      for (size_type weights_idx = 0; weights_idx < NUMBER_OF_LAYERS - 1; ++weights_idx)
        m_weights[weights_idx] = GenerateNormalWeights(
                                 m_layer_nodes_number[weights_idx + 1],
                                 m_layer_nodes_number[weights_idx + 0]
                               );
    }

    /**
     * @brief Performs one train iteration (updates weights) based on given and expected input data.
     * @param[in] inputs Input data.
     * @param[in] expected Expected result.
     */
    void Train
    (
      const list_type& inputs,
      const list_type& expected
    )
    noexcept;

    /**
     * @brief Queries Neural Network.
     * @param[in] input Input data.
     * @return Return result of query.
     */
    list_type Query
    (
      const list_type& input
    )
    const noexcept;

  private:
    /// Number of nodes at each layer.
    const std::array<size_type, NUMBER_OF_LAYERS> m_layer_nodes_number;
    /// Learning rate of the network.
    const real_type m_learning_rate;
    /// Activation function.
    const std::function<void(real_type&)> m_activation_function;

    /// Transition weights matrices.
    /// 0 - from input to hidden layer,
    /// 1 - from first hidden layer to second layer, ...,
    /// n - 1 from last hidden layer to output layer.
    std::array<matrix_type, NUMBER_OF_LAYERS - 1> m_weights;

    /**
     * @brief Applies weights and activation function to input data.
     * @param[in] index Index of transition weights to be applied.
     * @param[in] input Input data.
     * @return Result of operation.
     */
    list_type _Apply
    (
      size_type index,
      const list_type& input
    )
    const noexcept;
};

///////////////////////////////////////////////////////////////////////////////
//// IMPL
///////////////////////////////////////////////////////////////////////////////

template<size_type... TNodesPerLayer>
void NeuralNetwork<TNodesPerLayer...>::Train
(
  const list_type& inputs,
  const list_type& expected
)
noexcept
{
  if (inputs.size()   != m_layer_nodes_number[0]                  ||
      expected.size() != m_layer_nodes_number[NUMBER_OF_LAYERS - 1])
    return;

  // calculate outputs from hidden layer
  const auto hidden_outputs = _Apply(0, inputs);
  // calculate outputs from output layer
  const auto final_outputs  = _Apply(1, hidden_outputs);

  // calculate error on output layer
  const auto output_error = expected - final_outputs;
  // calculate error on hidden layer
  const auto hidden_error = boost::numeric::ublas::prod(
                              boost::numeric::ublas::trans(m_weights[1]),
                              output_error
                            );

  // update weights between hidden and output layers
  m_weights[1] += m_learning_rate * boost::numeric::ublas::outer_prod(
                    boost::numeric::ublas::element_prod(
                      boost::numeric::ublas::element_prod(output_error, final_outputs),
                      list_type(final_outputs.size(), 1.0) - final_outputs
                    ),
                    boost::numeric::ublas::trans(hidden_outputs)
                  );

  // update weights between input and hidden and layers
  m_weights[0] += m_learning_rate * boost::numeric::ublas::outer_prod(
                    boost::numeric::ublas::element_prod(
                      boost::numeric::ublas::element_prod(hidden_error, hidden_outputs),
                      list_type(hidden_outputs.size(), 1.0) - hidden_outputs
                    ),
                    boost::numeric::ublas::trans(inputs)
                  );
}

///////////////////////////////////////////////////////////////////////////////

template<size_type... TNodesPerLayer>
list_type NeuralNetwork<TNodesPerLayer...>::Query
(
  const list_type& input
)
const noexcept
{
  if (input.size() != m_layer_nodes_number[0])
    return {};

  auto res = input;
  for (size_type index = 0; index < NUMBER_OF_LAYERS - 1; ++index)
    res = _Apply(index, res);

  return res;
}

///////////////////////////////////////////////////////////////////////////////

template<size_type... TNodesPerLayer>
list_type NeuralNetwork<TNodesPerLayer...>::_Apply
(
  size_type index,
  const list_type& input
)
const noexcept
{
  list_type result = boost::numeric::ublas::prod(m_weights[index], input);
  std::for_each(result.begin(), result.end(), m_activation_function);
  return result;
}
