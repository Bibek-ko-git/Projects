#include "ChannelFlowSolver.hpp"

int main(int argc, char *argv[])
{
  try {
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
    
    // Create and run the channel flow solver
    ChannelFlowSolver<2> solver(MPI_COMM_SELF);
    solver.run();
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