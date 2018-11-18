#pragma once

#include "WeightsGeneration.h"

#include <algorithm>
#include <random>

/**
 * @brief Implementation of simple Neural Network.
 * @brief TNodesPerLayer Number of nodes at each layer.
 * @note Currently only 3 layers are supported.
 */
template<std::size_t... TNodesPerLayer>
class NeuralNetwork final
{
    static constexpr std::size_t NUMBER_OF_LAYERS = sizeof...(TNodesPerLayer);
    static_assert(NUMBER_OF_LAYERS >= 3);

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
      : m_nodes_per_layer { std::forward<std::size_t>(TNodesPerLayer)... }
      , m_learning_rate(learning_rate)
      , m_activation_function([](real_type& value) { value = 1.0 / (1.0 + std::exp(-1.0 * value)); })
    {
      for (std::size_t weights_idx = 1; weights_idx < NUMBER_OF_LAYERS; ++weights_idx)
        m_weights[weights_idx - 1] = GenerateNormalWeights(
                                       m_nodes_per_layer[weights_idx + 0],
                                       m_nodes_per_layer[weights_idx - 1]
                                     );
    }

    /**
     * @brief Performs one train iteration (updates weights) based on given and expected input data.
     * @param[in] input     Input data.
     * @param[in] expected  Expected result.
     */
    void Train
    (
      const list_type& input,
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
    const std::array<std::size_t, NUMBER_OF_LAYERS> m_nodes_per_layer;
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
      std::size_t index,
      const list_type& input
    )
    const noexcept;

    /**
     * @brief Calculates output at each layer.
     * @param[in] input Input data.
     * @return Output from each layer (The first element is original input, the last is output at last layer).
     */
    std::vector<list_type> _GetPerLayerOutput
    (
      const list_type& input
    )
    const noexcept;

    /**
     * @brief Calculates error at each layer.
     * @param[in] total_error Total error calculated at last layer (expected output - actual output).
     * @return Error at each layer.
     */
    std::vector<list_type> _GetPerLayerError
    (
      const list_type& total_error
    )
    const noexcept;

    /**
     * @brief Updates weights.
     * @param[in] outputs   Per layer outputs.
     * @param[in] errors    Per layer errors.
     */
    void _UpdateWeights
    (
      const std::vector<list_type>& outputs,
      const std::vector<list_type>& errors
    )
    noexcept;
};

///////////////////////////////////////////////////////////////////////////////
//// IMPL
///////////////////////////////////////////////////////////////////////////////

template<std::size_t... TNodesPerLayer>
void NeuralNetwork<TNodesPerLayer...>::Train
(
  const list_type& input,
  const list_type& expected
)
noexcept
{
  if (input.size()    != m_nodes_per_layer[0]                  ||
      expected.size() != m_nodes_per_layer[NUMBER_OF_LAYERS - 1])
    return;

  const auto outputs = _GetPerLayerOutput(input);

  const auto errors = _GetPerLayerError(expected - outputs.back());

  _UpdateWeights(outputs, errors);
}

///////////////////////////////////////////////////////////////////////////////

template<std::size_t... TNodesPerLayer>
list_type NeuralNetwork<TNodesPerLayer...>::Query
(
  const list_type& input
)
const noexcept
{
  if (input.size() != m_nodes_per_layer[0])
    return {};

  auto res = input;
  for (std::size_t index = 1; index < NUMBER_OF_LAYERS; ++index)
    res = _Apply(index - 1, res);

  return res;
}

///////////////////////////////////////////////////////////////////////////////

template<std::size_t... TNodesPerLayer>
list_type NeuralNetwork<TNodesPerLayer...>::_Apply
(
  std::size_t index,
  const list_type& input
)
const noexcept
{
  list_type result = boost::numeric::ublas::prod(m_weights[index], input);
  std::for_each(result.begin(), result.end(), m_activation_function);
  return result;
}

///////////////////////////////////////////////////////////////////////////////

template<std::size_t... TNodesPerLayer>
std::vector<list_type> NeuralNetwork<TNodesPerLayer...>::_GetPerLayerOutput
(
  const list_type& input
)
const noexcept
{
  std::vector<list_type> outputs;
  outputs.reserve(NUMBER_OF_LAYERS);
  outputs.push_back(input);
  for (std::size_t index = 1; index < NUMBER_OF_LAYERS; ++index)
    outputs.push_back(_Apply(index - 1, outputs.back()));

  return std::move(outputs);
}

///////////////////////////////////////////////////////////////////////////////

template<std::size_t... TNodesPerLayer>
std::vector<list_type> NeuralNetwork<TNodesPerLayer...>::_GetPerLayerError
(
  const list_type& total_error
)
const noexcept
{
  std::vector<list_type> errors;
  errors.reserve(NUMBER_OF_LAYERS - 1);
  errors.push_back(total_error);
  for (std::size_t index = NUMBER_OF_LAYERS - 2; index > 0; --index)
    errors.push_back(boost::numeric::ublas::prod(
                       boost::numeric::ublas::trans(m_weights[index]),
                       errors.back()
                     ));
  std::reverse(errors.begin(), errors.end());

  return std::move(errors);
}

///////////////////////////////////////////////////////////////////////////////

template<std::size_t... TNodesPerLayer>
void NeuralNetwork<TNodesPerLayer...>::_UpdateWeights
(
  const std::vector<list_type>& outputs,
  const std::vector<list_type>& errors
)
noexcept
{
  for (std::size_t index = NUMBER_OF_LAYERS - 2; index != static_cast<std::size_t>(-1); --index)
    m_weights[index] += m_learning_rate * boost::numeric::ublas::outer_prod(
                          boost::numeric::ublas::element_prod(
                            boost::numeric::ublas::element_prod(errors[index], outputs[index + 1]),
                            list_type(outputs[index + 1].size(), 1.0) - outputs[index + 1]
                          ),
                          boost::numeric::ublas::trans(outputs[index])
                        );
}
