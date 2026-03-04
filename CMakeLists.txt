cmake_minimum_required(VERSION 3.16)
project(didactic_lamp_universe_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(universe_sim src/main.cpp)

target_compile_options(universe_sim PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/O2 /W4>
)
