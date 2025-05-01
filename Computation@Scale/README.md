# NavierStokesFEM

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

A finite element implementation of the incompressible Navier-Stokes equations using the deal.II library.

## Overview

The software package provides a parallel implementation of the incompressible Navier-Stokes equations using the finite element method. It features a stabilized P2-P1 Taylor-Hood element formulation for spatial discretization and a generalized-alpha method for time integration. The package includes a test case of a steady method of manufactured solutions (MMS), for verification of the solver's accuracy and convergence properties. More rigorous test cases will be added in the future.

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
- CMake (3.10 or newer)

Note : One can easily configure and run this software package by just installing a docker (https://www.docker.com/) on  their device and adding a deal.II image with everything build on it. 
     : One such example is Master:focal 
     : To install the image run this command on the device terminal after installing and running the docker.
     : > docker pull dealii/dealii:master-focal
     : > docker run -i -t dealii/dealii:master-focal
       
Note: Catch2 needs to installed separately even in the container image.

## Building

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

