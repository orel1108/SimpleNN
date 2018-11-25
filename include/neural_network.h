#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <random>

/**
 * @brief Generates matrix of given size with randomly distributed coefficients.
 * @param[in] num_rows Number of rows in resulting matrix.
 * @param[in] num_cols Number of columns in resulting matrix.
 * @param[in] distribution Distribution to be used during generation.
 * @return Matrix with randomly distributed coefficients.
 */
template<class TDistribution>
std::vector<std::vector<double>> GenerateRandomWeights
(
  std::size_t num_rows,
  std::size_t num_cols,
  TDistribution& distribution
)
noexcept
{
  std::default_random_engine generator;

  std::vector<std::vector<double>> matrix(num_rows, std::vector<double>(num_cols, 0.0));
  for (std::size_t r = 0; r < num_rows; ++r)
    for (std::size_t c = 0; c < num_cols; ++c)
      matrix[r][c] = distribution(generator);

  return std::move(matrix);
}

/**
 * @brief Implementation of a simple Neural Network.
 * @param TNodesPerLayer Number of nodes at each layer.
 * @note Number of layers should be at least 3.
 */
template<std::size_t... TNodesPerLayer>
class NeuralNetwork final
{
    static constexpr std::size_t NUMBER_OF_LAYERS = sizeof...(TNodesPerLayer);
    static_assert(NUMBER_OF_LAYERS >= 3, "Number of layers in the Neural Network should be at least 3.");

  public:
    /**
     * @brief Creates a Neural Network with given learning rate.
     * @param[in] learning_rate Learning rate of the network.
     */
    NeuralNetwork
    (
      double learning_rate
    )
    noexcept
      : m_nodes_per_layer { std::forward<std::size_t>(TNodesPerLayer)... }
      , m_learning_rate(learning_rate)
      , m_activation_function([](double& value) { value = 1.0 / (1.0 + std::exp(-1.0 * value)); })
    {
      // std::uniform_real_distribution<double> distribution(-0.5, 0.5);
      for (std::size_t weights_idx = 1; weights_idx < NUMBER_OF_LAYERS; ++weights_idx)
      {
        std::normal_distribution<double> distribution(0.0, std::pow(m_nodes_per_layer[weights_idx - 0], -0.5));
        m_weights[weights_idx - 1] = GenerateRandomWeights
                                      (
                                        m_nodes_per_layer[weights_idx - 0],
                                        m_nodes_per_layer[weights_idx - 1],
                                        distribution
                                      );
      }
    }

    /**
     * @brief Performs one train iteration (updates weights) based on given and expected input data.
     * @param[in] input     Input data.
     * @param[in] expected  Expected result.
     * @return True if operation one train iteration succeeded and false otherwise.
     */
    bool Train
    (
      const std::vector<double>& input,
      const std::vector<double>& expected
    )
    noexcept;

    /**
     * @brief Queries Neural Network.
     * @param[in] input Input data.
     * @return Returns a pair of result of query if operation succeeded and true value
     *         and empty result with false value otherwise.
     */
    std::pair<std::vector<double>, bool> Query
    (
      const std::vector<double>& input
    )
    const noexcept;

  private:
    /// Number of nodes at each layer.
    const std::array<std::size_t, NUMBER_OF_LAYERS> m_nodes_per_layer;
    /// Learning rate of the network.
    const double m_learning_rate;
    /// Activation function.
    const std::function<void(double&)> m_activation_function;

    /// Transition weights matrices.
    /// 0 - from input to hidden layer,
    /// 1 - from first hidden layer to second layer, ...,
    /// n - 1 from last hidden layer to output layer.
    std::array<std::vector<std::vector<double>>, NUMBER_OF_LAYERS - 1> m_weights;

    /**
     * @brief Applies weights and activation function to the input data.
     * @param[in] index Index of transition weights to be applied.
     * @param[in] input Input data.
     * @return Result of operation.
     */
    std::vector<double> _Apply
    (
      std::size_t index,
      const std::vector<double>& input
    )
    const noexcept;

    /**
     * @brief Calculates output at each layer.
     * @param[in] input Input data.
     * @return Output from each layer (The first element is original input, the last is output at last layer).
     */
    std::vector<std::vector<double>> _GetPerLayerOutput
    (
      const std::vector<double>& input
    )
    const noexcept;

    /**
     * @brief Calculates error at each layer.
     * @param[in] total_error Total error calculated at last layer (expected output - actual output).
     * @return Error at each layer.
     */
    std::vector<std::vector<double>> _GetPerLayerError
    (
      const std::vector<double>& total_error
    )
    const noexcept;

    /**
     * @brief Updates weights.
     * @param[in] per_layer_outputs   Per layer outputs.
     * @param[in] per_layer_errors    Per layer errors.
     */
    void _UpdateWeights
    (
      const std::vector<std::vector<double>>& per_layer_outputs,
      const std::vector<std::vector<double>>& per_layer_errors
    )
    noexcept;
};

///////////////////////////////////////////////////////////////////////////////
//// IMPL
///////////////////////////////////////////////////////////////////////////////

template<std::size_t... TNodesPerLayer>
bool NeuralNetwork<TNodesPerLayer...>::Train
(
  const std::vector<double>& input,
  const std::vector<double>& expected
)
noexcept
{
  if (input.size()    != m_nodes_per_layer[0]                  ||
      expected.size() != m_nodes_per_layer[NUMBER_OF_LAYERS - 1])
    return false;

  // calculate output at each layer
  const auto per_layer_outputs = std::move(_GetPerLayerOutput(input));
  // calculate error at each layer
  std::vector<double> total_error(m_nodes_per_layer[NUMBER_OF_LAYERS - 1]);
  std::transform(expected.cbegin(), expected.cend(), per_layer_outputs.back().cbegin(), total_error.begin(),
                 [](double l, double r)
                 {
                   return l - r;
                 });
  const auto per_layer_errors  = std::move(_GetPerLayerError(total_error));
  // update weights
  _UpdateWeights(per_layer_outputs, per_layer_errors);
  return true;
}

///////////////////////////////////////////////////////////////////////////////

template<std::size_t... TNodesPerLayer>
std::pair<std::vector<double>, bool> NeuralNetwork<TNodesPerLayer...>::Query
(
  const std::vector<double>& input
)
const noexcept
{
  if (input.size() != m_nodes_per_layer[0])
    return { {}, false };

  auto res = input;
  for (std::size_t index = 1; index < NUMBER_OF_LAYERS; ++index)
    res = _Apply(index - 1, res);

  return { res, true };
}

///////////////////////////////////////////////////////////////////////////////

template<std::size_t... TNodesPerLayer>
std::vector<double> NeuralNetwork<TNodesPerLayer...>::_Apply
(
  std::size_t index,
  const std::vector<double>& input
)
const noexcept
{
  std::vector<double> result(m_weights[index].size());
  std::transform(m_weights[index].cbegin(), m_weights[index].cend(), result.begin(),
                 [&](const std::vector<double>& row) -> double
                 {
                   return std::inner_product(row.cbegin(), row.cend(), input.cbegin(), 0.0);
                 });
  std::for_each(result.begin(), result.end(), m_activation_function);
  return std::move(result);
}

///////////////////////////////////////////////////////////////////////////////

template<std::size_t... TNodesPerLayer>
std::vector<std::vector<double>> NeuralNetwork<TNodesPerLayer...>::_GetPerLayerOutput
(
  const std::vector<double>& input
)
const noexcept
{
  std::vector<std::vector<double>> per_layer_outputs;
  per_layer_outputs.reserve(NUMBER_OF_LAYERS);
  per_layer_outputs.push_back(input);
  for (std::size_t index = 1; index < NUMBER_OF_LAYERS; ++index)
    per_layer_outputs.push_back(_Apply(index - 1, per_layer_outputs.back()));

  return std::move(per_layer_outputs);
}

///////////////////////////////////////////////////////////////////////////////

template<std::size_t... TNodesPerLayer>
std::vector<std::vector<double>> NeuralNetwork<TNodesPerLayer...>::_GetPerLayerError
(
  const std::vector<double>& total_error
)
const noexcept
{
  std::vector<std::vector<double>> per_layer_errors;
  per_layer_errors.reserve(NUMBER_OF_LAYERS - 1);
  per_layer_errors.push_back(total_error);
  for (std::size_t index = NUMBER_OF_LAYERS - 2; index > 0; --index)
  {
    std::vector<double> prod(m_nodes_per_layer[index], 0.0);
    for (std::size_t c = 0; c < m_nodes_per_layer[index]; ++c)
      for (std::size_t r = 0; r < m_nodes_per_layer[index + 1]; ++r)
        prod[c] += m_weights[index][r][c] * per_layer_errors.back()[r];
    per_layer_errors.emplace_back(std::move(prod));
  }

  std::reverse(per_layer_errors.begin(), per_layer_errors.end());

  return std::move(per_layer_errors);
}

///////////////////////////////////////////////////////////////////////////////

template<std::size_t... TNodesPerLayer>
void NeuralNetwork<TNodesPerLayer...>::_UpdateWeights
(
  const std::vector<std::vector<double>>& per_layer_outputs,
  const std::vector<std::vector<double>>& per_layer_errors
)
noexcept
{
  for (std::size_t index = NUMBER_OF_LAYERS - 2; index != static_cast<std::size_t>(-1); --index)
  {
    std::vector<double> prod(m_nodes_per_layer[index + 1], 0.0);
    std::transform(per_layer_errors[index].cbegin(), per_layer_errors[index].cend(),
                   per_layer_outputs[index + 1].cbegin(),
                   prod.begin(),
                   [](double error, double output)
                   {
                     return error * output * (1 - output);
                   });
    for (std::size_t r = 0; r < m_nodes_per_layer[index + 1]; ++r)
      for (std::size_t c = 0; c < m_nodes_per_layer[index + 0]; ++c)
        m_weights[index][r][c] += m_learning_rate * prod[r] * per_layer_outputs[index][c];
  }
}
