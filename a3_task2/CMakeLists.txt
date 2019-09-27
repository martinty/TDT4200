cmake_minimum_required(VERSION 3.10)
project(Assignment3)
set(TARGET edgeDetection.out)

find_package(MPI REQUIRED)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu11 -Wall -O3")

include_directories(inc ${MPI_INCLUDE_PATH})
file(GLOB_RECURSE SRC_FILES src/*.c)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/program)
add_executable(${TARGET} ${SRC_FILES})
target_link_libraries(${TARGET} ${MPI_LIBRARIES})
