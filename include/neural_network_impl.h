#pragma once

#include <algorithm>

namespace nn
{
  template<std::size_t... TNodesPerLayer>
  bool neural_network<TNodesPerLayer...>::Train
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
  std::pair<std::vector<double>, bool> neural_network<TNodesPerLayer...>::Query
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
  std::vector<double> neural_network<TNodesPerLayer...>::_Apply
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
  std::vector<std::vector<double>> neural_network<TNodesPerLayer...>::_GetPerLayerOutput
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
  std::vector<std::vector<double>> neural_network<TNodesPerLayer...>::_GetPerLayerError
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
  void neural_network<TNodesPerLayer...>::_UpdateWeights
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

}
