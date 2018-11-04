#pragma once

#include <cstddef>

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
  constexpr NeuralNetwork(std::size_t i_input_nodes,
                          std::size_t i_hidden_nodes,
                          std::size_t i_output_nodes,
                          double      i_learning_rate) noexcept
    : m_input_nodes(i_input_nodes)
    , m_hidden_nodes(i_hidden_nodes)
    , m_output_nodes(i_output_nodes)
    , m_learning_rate(i_learning_rate)
    {
    }

    void Train();

    void Query() const;

private:
  /// Number of input nodes in the network.
  const std::size_t m_input_nodes;
  /// Number of hidden nodes in the network.
  const std::size_t m_hidden_nodes;
  /// Number of output nodes in the network.
  const std::size_t m_output_nodes;
  /// Learning rate of the network.
  const double      m_learning_rate;
};
