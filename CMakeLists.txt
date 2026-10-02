cmake_minimum_required(VERSION 3.16)
project(didactic_lamp_universe_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(universe_sim src/main.cpp)

target_compile_options(universe_sim PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

include(CTest)
if(BUILD_TESTING)
  foreach(test_name core metrics)
    add_executable(test_${test_name} tests/test_${test_name}.cpp)
    target_compile_options(test_${test_name} PRIVATE
      $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
      $<$<CXX_COMPILER_ID:MSVC>:/W4>)
    add_test(NAME ${test_name} COMMAND test_${test_name})
  endforeach()
  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(NAME python_contracts
      COMMAND ${Python3_EXECUTABLE} -m unittest discover -s tests -p test_*.py -v)
    set_tests_properties(python_contracts PROPERTIES
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      ENVIRONMENT "SIM_BINARY=$<TARGET_FILE:universe_sim>;MPLBACKEND=Agg")
  endif()
endif()
