#include "include/NeuralNetwork.h"

#include <iostream>

int main()
{
    NeuralNetwork<5, 4, 3, 3> nn(0.5);

    const list_type inputs({ 0.1, 0.3, 0.5, 0.7, 0.9 });
    const list_type expected({0.01, 0.98, 0.01});

    nn.Train( inputs, expected);
    const auto res = nn.Query(inputs);
    std::cout << "It works" << std::endl;
    for (const auto& val : res)
      std::cout << val << " ";
    return 0;
}
