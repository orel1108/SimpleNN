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
     * @param[in] input_nodes   Number of input nodes in the network.
     * @param[in] hidden_nodes  Number of hidden nodes in the network.
     * @param[in] output_nodes  Number of output nodes in the network.
     * @param[in] learning_rate Learning rate of the network.
     */
    NeuralNetwork
    (
      size_type input_nodes,
      size_type hidden_nodes,
      size_type output_nodes,
      real_type learning_rate
    )
    noexcept
      : m_input_nodes(input_nodes)
      , m_hidden_nodes(hidden_nodes)
      , m_output_nodes(output_nodes)
      , m_learning_rate(learning_rate)
      , m_activation_function([](real_type& value)
    {
      value = 1.0 / (1.0 + std::exp(-1.0 * value));
    })
    {
      // Initialize transition weights between input and hidden layers.
      m_weights_input_hidden  = GenerateNormalWeights(m_hidden_nodes, m_input_nodes);
      // Initialize transition weights between hidden and output layers.
      m_weights_hidden_output = GenerateNormalWeights(m_output_nodes, m_hidden_nodes);
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
     * @param[in] inputs Input data.
     * @return Return result of query.
     */
    list_type Query
    (
      const list_type& inputs
    )
    const noexcept;

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
     * @param[in] weights Weights to be applied.
     * @param[in] input Input data.
     * @return Result of operation.
     */
    list_type _Apply
    (
      const matrix_type& weights,
      const list_type& input
    )
    const noexcept;
};
