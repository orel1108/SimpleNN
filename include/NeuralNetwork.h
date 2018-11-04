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
    {
      m_weights_input_hidden  = GenerateNormalWeights(m_hidden_nodes,
                                                      m_input_nodes);

      m_weights_hidden_output = GenerateNormalWeights(m_output_nodes,
                                                      m_hidden_nodes);
    }

    void Train();

    void Query() const;

private:
  /// Number of input nodes in the network.
  const size_type m_input_nodes;
  /// Number of hidden nodes in the network.
  const size_type m_hidden_nodes;
  /// Number of output nodes in the network.
  const size_type m_output_nodes;
  /// Learning rate of the network.
  const real_type m_learning_rate;

  /// Matrix of transition weights between input and hidden layers.
  matrix_type m_weights_input_hidden;
  /// Matrix of transition weights between hidden and output layers.
  matrix_type m_weights_hidden_output;
};
