cmake_minimum_required(VERSION 3.10)
project(assignment1)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu11 -Wall -g")

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/program)

include_directories(inc)
file(GLOB_RECURSE SRC_FILES src/*.c)

add_executable(bitmap ${SRC_FILES})