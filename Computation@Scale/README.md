# NavierStokesFEM

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

A finite element implementation of the incompressible Navier-Stokes equations using the deal.II library.

## Overview

The software package provides a parallel implementation of the incompressible Navier-Stokes equations using the finite element method. It features a stabilized P2-P1 Taylor-Hood element formulation for spatial discretization and a generalized-alpha method for time integration. The package includes a test case of a steady method of manufactured solutions (MMS), for verification of the solver's accuracy and convergence properties. More rigorous test cases will be added in the future.
Based on previous implementation in FENICSx using the similar method. That can be reviewed here: [git@github.com:Bibek-ko-git/FEniCSx-for-NS.git](https://github.com/Bibek-ko-git/FEniCSx-for-NS.git)

## Features

- **P2-P1 Taylor-Hood Elements**: Quadratic velocity, linear pressure elements that satisfy the LBB condition
- **Stabilized Formulation**: SUPG, PSPG, and LSIC stabilization for robust handling of convection-dominated flows
- **Generalized-Alpha Time Integration**: Second-order accurate time integration with controllable numerical dissipation
- **Method of Manufactured Solutions**: Built-in verification using analytical solutions
- **MPI Parallelization**: Efficient parallel execution for large-scale simulations
- **Automated Testing**: Comprehensive test suite using Catch2 and CI/CD using github actions.

## Dependencies

- [deal.II](https://www.dealii.org/) (version 9.0.0 or newer)
- MPI implementation (OpenMPI, MPICH, etc.)
- [Trilinos](https://trilinos.github.io/)
- [Catch2](https://github.com/catchorg/Catch2) (for testing)
- CMake (3.13 or newer)

Note : One can easily configure and run this software package by just installing a docker (https://www.docker.com/) on  their device and adding a deal.II image with everything build on it. One such example is dealii:master-focal. To install the image run this command on the device terminal after installing and running the docker.
```
docker pull dealii/dealii:master-focal
docker run -i -t dealii/dealii:master-focal
```
Note: Catch2 needs to installed separately even in the container image.

## Building and running

```bash
# Clone the repository
git clone git@github.com:Bibek-ko-git/Projects.git
cd Computation@Scale

# Create build directory
cmake -B build .
cd build

# Configure and build
make -j$(nproc)

# Run testing 
./manufactured_test

# Run solver
./navier_stokes_pipe

# For MPI sun
mpirun -np number of processors ./navier_stokes_pipe
```
## Code Structure
- **NavierStokesSolver**: Base class implementing the core solver functionality
- **ChannelFlowSolver**: Derived class for 2D pipe flow simulations more 2D cases can be built upon and added eg. Cavity, flow over cylinder.
- **ManufacturedSolutionSolver**: Specialized class for verification using manufactured solution

## Post processing 
- [Paraview](https://www.paraview.org/) can be used for post processing the results.

## License 
This project is licensed under the BSD 3-Clause License - see LICENSE file for details.

## Contributing
Contributions are welcomed! Especially bug and fixes for the code base. Please feel free to submit a Pull Request. 

## Acknowledgements
- The [deal.II](https://www.dealii.org/) developers for their finite element library
- [Trilinos](https://trilinos.github.io/) for parallel linear algebra capabilities
- [Catch2](https://github.com/catchorg/Catch2) for testing framework
- [Github](https://github.com/) for automated testing and hosting the code base

