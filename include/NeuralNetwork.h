#pragma once

#include "WeightsGeneration.h"

/**
 * @brief Implementation of simple Neural Network.
 */
class NeuralNetwork final
{
public:
  /**
   * @brief Creates Neural Network with given parameters.
   * @param[in] i_input_nodes   Number of input nodes in the network.
   * @param[in] i_hidden_nodes  Number of hidden nodes in the network.
   * @param[in] i_output_nodes  Number of output nodes in the network.
   * @param[in] i_learning_rate Learning rate of the network.
   */
  NeuralNetwork(size_type i_input_nodes,
                size_type i_hidden_nodes,
                size_type i_output_nodes,
                real_type i_learning_rate) noexcept
    : m_input_nodes(i_input_nodes)
    , m_hidden_nodes(i_hidden_nodes)
    , m_output_nodes(i_output_nodes)
    , m_learning_rate(i_learning_rate)
    , m_activation_function([](real_type& io_value) { io_value = 1.0 / (1.0 + std::exp(-1.0 * io_value)); })
    {
      // Initialize transition weights between input and hidden layers.
      m_weights_input_hidden  = GenerateNormalWeights(m_hidden_nodes,
                                                      m_input_nodes);
      // Initialize transition weights between hidden and output layers.
      m_weights_hidden_output = GenerateNormalWeights(m_output_nodes,
                                                      m_hidden_nodes);
    }

    void Train();

    /**
     * @brief Queries Neural Network.
     * @param[in] i_inputs Input data.
     * @return Return result of query.
     */
    std::vector<real_type> Query(const std::vector<real_type>& i_inputs) const;

private:
  /// Number of input nodes in the network.
  const size_type m_input_nodes;
  /// Number of hidden nodes in the network.
  const size_type m_hidden_nodes;
  /// Number of output nodes in the network.
  const size_type m_output_nodes;
  /// Learning rate of the network.
  const real_type m_learning_rate;
  /// Activation function.
  const std::function<void(real_type&)> m_activation_function;

  /// Matrix of transition weights between input and hidden layers.
  matrix_type m_weights_input_hidden;
  /// Matrix of transition weights between hidden and output layers.
  matrix_type m_weights_hidden_output;

  /**
   * @brief Applies weights and activation function to input data.
   * @param[in] i_weights Weights to be applied.
   * @param[in] i_input Input data.
   * @return Result of operation.
   */
  std::vector<real_type> _Apply(const matrix_type& i_weights,
                                const std::vector<real_type>& i_input) const noexcept;
};
