#include "include/NeuralNetwork.h"

#include <iostream>

int main()
{
    NeuralNetwork nn(3, 3, 3, 0.5);
    nn.Train();
    nn.Query();
    std::cout << "It works" << std::endl;
    return 0;
}
