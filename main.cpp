#include "include/NeuralNetwork.h"

#include "include/matrix_operations.h"

#include <fstream>
#include <iostream>

int main()
{
  constexpr std::size_t NUMBER_OF_INPUT_LAYERS  = 784;
  constexpr std::size_t NUMBER_OF_HIDDEN_LAYERS = 100;
  constexpr std::size_t NUMBER_OF_OUTPUT_LAYERS = 10;

  constexpr real_type learning_rate = 0.3;

  NeuralNetwork<NUMBER_OF_INPUT_LAYERS, NUMBER_OF_HIDDEN_LAYERS, NUMBER_OF_OUTPUT_LAYERS> nn(learning_rate);

  if (true)
    {
      std::ifstream train_data("../test/mnist_train_100.csv");

      std::vector<list_type> input_list_for_train;
      std::vector<list_type> expected_list_for_train;

      std::string line;
      while (std::getline(train_data, line))
        {
          std::stringstream ss(line);

          char c;

          std::size_t expected = -1;
          ss >> expected >> c;

          expected_list_for_train.push_back(list_type(10, 0.01));
          expected_list_for_train.back()[expected] = 0.99;

          list_type input(784, 0.0);
          for (std::size_t idx = 0; idx < 784; ++idx)
            ss >> input[idx] >> c;

          std::for_each(input.begin(), input.end(),
                        [](real_type& value)
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

      std::vector<list_type> input_list_for_test;
      std::vector<std::size_t> expected_list_for_test;

      std::string line;
      while (std::getline(test_data, line))
        {
          std::stringstream ss(line);

          char c;

          std::size_t expected = -1;
          ss >> expected >> c;

          expected_list_for_test.push_back(expected);

          list_type input(784, 0.0);
          for (std::size_t idx = 0; idx < 784; ++idx)
            ss >> input[idx] >> c;

          std::for_each(input.begin(), input.end(),
                        [](real_type& value)
          {
            value = (value / 255.0 * 0.99) + 0.01;
          });

          input_list_for_test.push_back(input);
        }
      for (std::size_t idx = 0; idx < input_list_for_test.size(); ++idx)
        {
          const auto res = nn.Query(input_list_for_test[idx]);
          const auto actual = std::distance(res.begin(), std::max_element(res.begin(), res.end()));

          std::cout << "Actual: " << actual << "; Expected: " << expected_list_for_test[idx] << std::endl;
        }
    }

  matrix_t<2, 3> l;
  matrix_t<3, 4> r;

  const auto res = prod(l, r);
  const auto t = trans(l);
  const auto e = elem_prod(l, l);
  return 0;
}
