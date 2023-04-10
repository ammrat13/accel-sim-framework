cmake_minimum_required(VERSION 3.26.2)
set(CMAKE_CXX_STANDARD 20)
project(gasan-alloc-sim
  DESCRIPTION "Simulator for the allocator used by GASan"
  HOMEPAGE_URL "https://github.com/ammrat13/accel-sim-framework/tree/gasan-sim"
  LANGUAGES CXX)

add_subdirectory("./third-party/argparse/" EXCLUDE_FROM_ALL)

add_executable(gasan-alloc-sim)
target_link_libraries(gasan-alloc-sim
  argparse)
target_include_directories(gasan-alloc-sim
  PRIVATE
    "./include/")
target_sources(gasan-alloc-sim
  PRIVATE
    "./src/main.cpp")
