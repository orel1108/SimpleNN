cmake_minimum_required (VERSION 2.8)

project (NeuralNetwork)

set(CMAKE_CXX_STANDARD 17)

find_package(Boost 1.64.0)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()

set(INCLUDE_DIR
    include/matrix_operations.h
    include/neural_network.h
    include/types.h
    include/weights_generation.h)

set(SRC_DIR
    src/weights_generation.cpp)

add_executable (NeuralNetwork ${INCLUDE_DIR} ${SRC_DIR} main.cpp)
