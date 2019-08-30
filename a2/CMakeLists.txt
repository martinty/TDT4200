cmake_minimum_required(VERSION 3.10)
project(assignment1)

find_package(MPI REQUIRED)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu11 -Wall -g")

include_directories(inc ${MPI_INCLUDE_PATH})
file(GLOB_RECURSE SRC_FILES src/*.c)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/program)
add_executable(bitmap ${SRC_FILES})
#target_link_libraries(bitmap ${MPI_LIBRARIES})
target_link_libraries(bitmap MPI::MPI_C)