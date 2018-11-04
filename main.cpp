#include "include/NeuralNetwork.h"

#include <iostream>

int main()
{
    NeuralNetwork nn(3, 3, 3, 0.5);
    nn.Train();
    const auto res = nn.Query({0.1, 0.5, 0.9});
    std::cout << "It works" << std::endl;
    for (const auto& val : res)
      std::cout << val << " ";
    return 0;
}
