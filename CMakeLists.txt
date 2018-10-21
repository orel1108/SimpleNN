cmake_minimum_required (VERSION 2.8)

project (NeuralNetwork)

set(INCLUDE_DIR
    include/NeuralNetwork.h)

set(SRC_DIR
    src/NeuralNetwork.cpp)

add_executable (NeuralNetwork ${INCLUDE_DIR} ${SRC_DIR} main.cpp)
