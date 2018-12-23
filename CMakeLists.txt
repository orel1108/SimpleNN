cmake_minimum_required (VERSION 2.8)

project (NeuralNetwork)

set(CMAKE_CXX_STANDARD 17)

set(INCLUDE_DIR
    include/neural_network.h
    include/neural_network_impl.h
    include/types.h
    include/weights_generation.h
    include/weights_generation_impl.h)

add_executable (NeuralNetwork ${INCLUDE_DIR} main.cpp)
