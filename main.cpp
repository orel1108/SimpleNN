#include "include/NeuralNetwork.h"

#include <iostream>

int main()
{
    NeuralNetwork nn;
    nn.Train();
    nn.Query();
    std::cout << "It works" << std::endl;
    return 0;
}
