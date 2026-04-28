cmake_minimum_required(VERSION 3.15)
project(fastipc CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wno-unused-parameter)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -march=native)
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# pybind11: prefer pip-installed (via `pip install pybind11`)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11, sys; sys.stdout.write(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    RESULT_VARIABLE _pb_rc
)
if(NOT _pb_rc EQUAL 0)
    message(FATAL_ERROR "pybind11 not found. Install with: pip install pybind11")
endif()
find_package(pybind11 REQUIRED CONFIG)

# ----- Core static library -----
add_library(fastipc_core STATIC
    src/shm_pool.cpp
    src/ring_shm.cpp
    src/server.cpp
    src/client.cpp
)
target_include_directories(fastipc_core PUBLIC include)
target_link_libraries(fastipc_core PUBLIC pthread rt)

# ----- Python extension -----
pybind11_add_module(_fastipc src/pybind_module.cpp)
target_link_libraries(_fastipc PRIVATE fastipc_core)
set_target_properties(_fastipc PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python/fastipc
)

# Copy Python wrapper package into build dir too (optional convenience).
add_custom_command(TARGET _fastipc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/python/fastipc
        ${CMAKE_BINARY_DIR}/python/fastipc
)

# ----- C++ standalone test binary (self-test without python) -----
add_executable(fastipc_selftest tests/selftest.cpp)
target_link_libraries(fastipc_selftest PRIVATE fastipc_core)

install(TARGETS _fastipc LIBRARY DESTINATION fastipc)
