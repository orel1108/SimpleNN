#pragma once

#include <array>
#include <functional>
#include <random>

#include "types.h"
#include "weights_generation.h"

namespace nn
{
  /**
  * @brief Implementation of a simple Neural Network.
  * @param TNodesPerLayer Number of nodes at each layer.
  * @note Number of layers should be at least 3.
  */
  template<std::size_t... TNodesPerLayer>
  class neural_network final
  {
      static constexpr std::size_t NUMBER_OF_LAYERS = sizeof...(TNodesPerLayer);
      static_assert(NUMBER_OF_LAYERS >= 3, "Number of layers in the Neural Network should be at least 3.");

    public:
      /**
       * @brief Creates a Neural Network with given learning rate.
       * @param[in] learning_rate Learning rate of the network.
       */
      neural_network
      (
        double learning_rate
      )
      noexcept
      : m_nodes_per_layer{ std::forward<std::size_t>(TNodesPerLayer)... }
      , m_learning_rate(learning_rate)
      , m_activation_function([](double& value){ value = 1.0 / (1.0 + std::exp(-1.0 * value)); })
      {
        // std::uniform_real_distribution<double> distribution(-0.5, 0.5);
        for (std::size_t weights_idx = 1; weights_idx < NUMBER_OF_LAYERS; ++weights_idx)
          {
            std::normal_distribution<double> distribution(0.0, std::pow(m_nodes_per_layer[weights_idx - 0], -0.5));
            m_weights[weights_idx - 1] = generate
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
}

#include "neural_network_impl.h"
