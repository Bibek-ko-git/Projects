#include "ChannelFlowSolver.hpp"
#include <catch2/catch_test_macros.hpp>

// Define the main function required by Catch2
#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

TEST_CASE("Manufactured solution test", "[navier-stokes]") {
  // Initialize MPI environment
  int argc = 1;
  char name[] = "test";
  char* argv_local[] = {name};
  char** argv = argv_local;
  
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  
  // Create the solver
  ManufacturedSolutionSolver<2> solver(MPI_COMM_SELF);
  
  // Run with steady state solver
  solver.run();
  
  // Check error against exact solution
  double error = solver.compute_error();
  
  std::cout << "L2 error in velocity: " << error << std::endl;
  
  // For a sufficiently refined mesh, we expect the error to be small
  REQUIRE(error < 0.05);
}
// Standalone main function for running without test framework
int main(int argc, char *argv[])
{
  try {
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
    ManufacturedSolutionSolver<2> solver(MPI_COMM_SELF);
    solver.run();
    
    double error = solver.compute_error();
    std::cout << "L2 error in velocity: " << error << std::endl;
  }
  catch (std::exception &exc) {
    std::cerr << std::endl
              << "----------------------------------------------------" 
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" 
              << std::endl;
    return 1;
  }
  catch (...) {
    std::cerr << std::endl
              << "----------------------------------------------------" 
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" 
              << std::endl;
    return 1;
  }
  
  return 0;
}