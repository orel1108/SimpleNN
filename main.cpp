#include "include/neural_network.h"

#include <fstream>
#include <iostream>
#include <sstream>

int main()
{
  constexpr std::size_t NUMBER_OF_INPUT_LAYERS  = 784;
  constexpr std::size_t NUMBER_OF_HIDDEN_LAYERS = 100;
  constexpr std::size_t NUMBER_OF_OUTPUT_LAYERS = 10;

  constexpr double learning_rate = 0.3;

  nn::neural_network<NUMBER_OF_INPUT_LAYERS, NUMBER_OF_HIDDEN_LAYERS, NUMBER_OF_OUTPUT_LAYERS> nn(learning_rate);

  if (true)
    {
      std::ifstream train_data("../test/mnist_train_100.csv");

      std::vector<std::vector<double>> input_list_for_train;
      std::vector<std::vector<double>> expected_list_for_train;

      std::string line;
      while (std::getline(train_data, line))
        {
          std::stringstream ss(line);

          char c;

          std::size_t expected = -1;
          ss >> expected >> c;

          expected_list_for_train.push_back(std::vector<double>(10, 0.01));
          expected_list_for_train.back()[expected] = 0.99;

          std::vector<double> input(784, 0.0);
          for (std::size_t idx = 0; idx < 784; ++idx)
            ss >> input[idx] >> c;

          std::for_each(input.begin(), input.end(),
                        [](double& value)
          {
            value = (value / 255.0 * 0.99) + 0.01;
          });

          input_list_for_train.push_back(input);
        }

      for (std::size_t idx = 0; idx < input_list_for_train.size(); ++idx)
        nn.Train(input_list_for_train[idx], expected_list_for_train[idx]);
    }

  if (true)
    {
      std::ifstream test_data("../test/mnist_test_10.csv");

      std::vector<std::vector<double>> input_list_for_test;
      std::vector<std::size_t> expected_list_for_test;

      std::string line;
      while (std::getline(test_data, line))
        {
          std::stringstream ss(line);

          char c;

          std::size_t expected = -1;
          ss >> expected >> c;

          expected_list_for_test.push_back(expected);

          std::vector<double> input(784, 0.0);
          for (std::size_t idx = 0; idx < 784; ++idx)
            ss >> input[idx] >> c;

          std::for_each(input.begin(), input.end(),
                        [](double& value)
          {
            value = (value / 255.0 * 0.99) + 0.01;
          });

          input_list_for_test.push_back(input);
        }
      for (std::size_t idx = 0; idx < input_list_for_test.size(); ++idx)
        {
          const auto res = nn.Query(input_list_for_test[idx]);
          const auto actual = std::distance(res.first.begin(), std::max_element(res.first.begin(), res.first.end()));

          std::cout << "Actual: " << actual << "; Expected: " << expected_list_for_test[idx] << std::endl;
        }
    }

  return 0;
}
