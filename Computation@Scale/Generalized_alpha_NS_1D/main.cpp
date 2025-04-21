#include "generalized_alpha_1d.h"

int main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    
    try {
        dealii::deallog.depth_console(0);
        
        GeneralizedAlpha1D<1> solver;
        solver.run();
    }
    catch (std::exception &exc) {
        std::cerr << std::endl
                 << std::endl
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
                 << std::endl
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
