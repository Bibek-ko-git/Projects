cmake_minimum_required(VERSION 3.10)
project(GeneralizedAlpha1D)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Find deal.II package
find_package(deal.II 9.2.0 REQUIRED
  HINTS ${DEAL_II_DIR} ${DEAL_II_DIR}/lib/cmake/deal.II
  $ENV{DEAL_II_DIR} $ENV{DEAL_II_DIR}/lib/cmake/deal.II
)
deal_ii_initialize_cached_variables()

# Create library with implementation
add_library(generalized_alpha_lib 
  src/generalized_alpha_1d.cpp
)
deal_ii_setup_target(generalized_alpha_lib)

# Prepare the executable target
add_executable(generalized_alpha_1d main.cpp)
target_link_libraries(generalized_alpha_1d PRIVATE generalized_alpha_lib)
deal_ii_setup_target(generalized_alpha_1d)

# Catch2 tests
include(FetchContent)

# Download Catch2 automatically
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v2.13.7 # Specify a version that's compatible with C++17
)
FetchContent_MakeAvailable(Catch2)

# Create test executable
add_executable(run_tests tests/generalized_alpha_tests.cpp)
target_link_libraries(run_tests PRIVATE generalized_alpha_lib Catch2::Catch2)
deal_ii_setup_target(run_tests)

# Enable testing
enable_testing()
add_test(NAME generalized_alpha_tests COMMAND run_tests)

# Install targets
install(TARGETS generalized_alpha_1d DESTINATION bin)
install(TARGETS generalized_alpha_lib DESTINATION lib)
install(FILES include/generalized_alpha_1d.h DESTINATION include)
