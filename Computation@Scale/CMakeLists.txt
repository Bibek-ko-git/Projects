# CMake script for Navier-Stokes solver:
cmake_minimum_required(VERSION 3.13.4)

project(IncompressibleNavierStokesSolver
  VERSION 0.1
  LANGUAGES CXX)
  
# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the name of the project and target:
set(TARGET "navier_stokes_pipe")

# Common source files for both targets
set(COMMON_SRC
  NavierStokesSolver.cpp
  ChannelFlowSolver.cpp 
  ManufacturedSolutionSolver.cpp
)

# Header files
set(TARGET_INC
  ChannelFlowSolver.hpp
)

# Find the deal.II library
find_package(deal.II 9.7.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)
if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
endif()

# Initialize the cached variables
deal_ii_initialize_cached_variables()

# Setup main target with main.cpp and common sources
add_executable(${TARGET} main.cpp ${COMMON_SRC} ${TARGET_INC})
deal_ii_setup_target(${TARGET})

# Setup test target if Catch2 is available
find_package(Catch2 QUIET)
if(Catch2_FOUND)
  message(STATUS "Catch2 found, building test executable")
  add_executable(manufactured_test manufactured_test.cpp ${COMMON_SRC} ${TARGET_INC})
  deal_ii_setup_target(manufactured_test)
  target_link_libraries(manufactured_test Catch2::Catch2)
else()
  message(STATUS "Catch2 not found, test executable will not be built")
endif()

# Output the configuration
message(STATUS "Configuration summary:")
message(STATUS "  C++ compiler:       ${CMAKE_CXX_COMPILER}")
message(STATUS "  C++ compiler flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "  Common source files: ${COMMON_SRC}")
message(STATUS "  Header files:       ${TARGET_INC}")