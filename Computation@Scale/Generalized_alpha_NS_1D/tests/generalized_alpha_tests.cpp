#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/function.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

// Include our implementation
#include "generalized_alpha_1d.h"

using namespace dealii;

// Test fixture for the generalized-alpha solver
class GeneralizedAlphaTest : public testing::Test {
protected:
    void SetUp() override {
        // Initialize MPI if not already initialized
        if (!Utilities::MPI::job_supports_mpi()) {
            mpi_init = std::make_unique<Utilities::MPI::MPI_InitFinalize>();
        }
    }
    
    std::unique_ptr<Utilities::MPI::MPI_InitFinalize> mpi_init;
};

// Helper function to compute L2 error between numerical and exact solution
template <int dim>
double compute_l2_error(const GeneralizedAlpha1D<dim>& solver, const double time) {
    // Get the solution vector
    const LA::MPI::Vector& numerical_solution = solver.get_solution();
    
    // Create exact solution
    ManufacturedSolution<dim> exact_solution(time);
    
    // Compute L2 error
    Vector<float> error_per_cell(solver.get_triangulation().n_active_cells());
    
    VectorTools::integrate_difference(
        solver.get_dof_handler(),
        numerical_solution,
        exact_solution,
        error_per_cell,
        QGauss<dim>(solver.get_fe().degree + 2),
        VectorTools::L2_norm);
    
    return std::sqrt(Utilities::MPI::sum(
        error_per_cell.norm_sqr(), MPI_COMM_WORLD));
}

// Test case 1: Test convergence for a single time step
TEST_CASE_METHOD(GeneralizedAlphaTest, "Convergence for a single time step", "[convergence]") {
    // Create a solver with different time step sizes
    std::vector<double> time_steps = {0.1, 0.05, 0.025, 0.0125};
    std::vector<double> errors;
    
    for (const double dt : time_steps) {
        GeneralizedAlpha1D<1> solver(dt);
        solver.setup_system();
        solver.initialize_solution();
        
        // Take a single time step
        solver.set_time(0.0);
        solver.advance_time_step();
        
        // Compute error at t = dt
        double error = compute_l2_error(solver, dt);
        errors.push_back(error);
    }
    
    // Check that errors are decreasing with expected order
    for (size_t i = 1; i < errors.size(); ++i) {
        // Expected convergence rate for generalized-alpha with current parameters should be ~2
        double rate = std::log(errors[i-1]/errors[i]) / std::log(time_steps[i-1]/time_steps[i]);
        REQUIRE(rate > 1.8); // Allow for some numerical error
    }
}

// Test case 2: Verify conservation properties
TEST_CASE_METHOD(GeneralizedAlphaTest, "Conservation of properties", "[conservation]") {
    GeneralizedAlpha1D<1> solver(0.01); // Small time step
    solver.setup_system();
    solver.initialize_solution();
    
    // Integrate the solution initially
    double initial_integral = solver.compute_integral();
    
    // Run for 10 time steps
    for (int i = 0; i < 10; ++i) {
        solver.advance_time_step();
    }
    
    // Check that the integral is conserved (up to tolerance)
    double final_integral = solver.compute_integral();
    REQUIRE(std::abs(final_integral - initial_integral) < 1e-6);
}

// Test case 3: Test stability for long time integration
TEST_CASE_METHOD(GeneralizedAlphaTest, "Stability for long-time integration", "[stability]") {
    GeneralizedAlpha1D<1> solver(0.05);
    solver.setup_system();
    solver.initialize_solution();
    
    // Initial norm
    double initial_norm = solver.get_solution().l2_norm();
    
    // Run for a long time (100 time steps)
    for (int i = 0; i < 100; ++i) {
        solver.advance_time_step();
    }
    
    // Check that solution hasn't blown up
    double final_norm = solver.get_solution().l2_norm();
    REQUIRE(final_norm < 10.0 * initial_norm); // Solution should remain bounded
}

// Test case 4: Verify manufactured solution
TEST_CASE_METHOD(GeneralizedAlphaTest, "Manufactured solution", "[manufactured]") {
    // Create solver with very small time step to minimize time discretization error
    GeneralizedAlpha1D<1> solver(0.001);
    solver.setup_system();
    solver.initialize_solution();
    
    // Run for 10 steps
    for (int i = 0; i < 10; ++i) {
        solver.advance_time_step();
    }
    
    // Compute error at final time
    double error = compute_l2_error(solver, solver.get_time());
    
    // Error should be small with the manufactured solution
    REQUIRE(error < 1e-3);
}

// Test case 5: SUPG stabilization effectiveness
TEST_CASE_METHOD(GeneralizedAlphaTest, "SUPG stabilization effectiveness", "[stabilization]") {
    // Test with high advection (where stabilization matters)
    GeneralizedAlpha1D<1> solver_with_supg(0.01);
    solver_with_supg.set_convection_coefficient(10.0); // Strong advection
    solver_with_supg.setup_system();
    solver_with_supg.initialize_solution();
    
    // Same solver but with SUPG turned off
    GeneralizedAlpha1D<1> solver_without_supg(0.01);
    solver_without_supg.set_convection_coefficient(10.0);
    solver_without_supg.set_use_supg(false);
    solver_without_supg.setup_system();
    solver_without_supg.initialize_solution();
    
    // Run both for 10 steps
    for (int i = 0; i < 10; ++i) {
        solver_with_supg.advance_time_step();
        solver_without_supg.advance_time_step();
    }
    
    // Compute errors
    double error_with_supg = compute_l2_error(solver_with_supg, solver_with_supg.get_time());
    double error_without_supg = compute_l2_error(solver_without_supg, solver_without_supg.get_time());
    
    // SUPG should give better results in advection-dominated case
    REQUIRE(error_with_supg < error_without_supg);
}
