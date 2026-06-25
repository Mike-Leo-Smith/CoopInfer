cmake_minimum_required(VERSION 3.18)

project(coopinfer_core LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

pybind11_add_module(_core src/coopinfer/cpp/core.cpp)
target_compile_features(_core PRIVATE cxx_std_17)
target_link_libraries(_core PRIVATE Threads::Threads)

install(
    TARGETS _core
    LIBRARY DESTINATION coopinfer
    RUNTIME DESTINATION coopinfer
)
