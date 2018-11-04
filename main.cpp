#include "include/NeuralNetwork.h"

#include <iostream>

#include <boost/serialization/array_wrapper.hpp>
#include <boost/numeric/ublas/matrix.hpp>

int main()
{
    NeuralNetwork nn(3, 3, 3, 0.5);
    nn.Train();
    nn.Query();
    std::cout << "It works" << std::endl;
    return 0;
}
