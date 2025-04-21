#ifndef GENERALIZED_ALPHA_1D_HPP
#define GENERALIZED_ALPHA_1D_HPP

#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/la_parallel_sparse_matrix.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

using namespace dealii;

// Manufactured solution for testing
template <int dim>
class ManufacturedSolution : public Function<dim>
{
public:
    ManufacturedSolution(const double time = 0)
        : Function<dim>(dim + 1, time) // dim velocity components + 1 pressure component
    {}

    virtual void vector_value(const Point<dim> &p,
                              Vector<double>   &values) const override
    {
        const double t = this->get_time();
        const double x = p[0];
        
        // Velocity (simple wave)
        values[0] = std::sin(2 * numbers::PI * (x - t));
        
        // Pressure (derivative of velocity)
        values[1] = -2 * numbers::PI * std::cos(2 * numbers::PI * (x - t));
    }
};

template <int dim>
class GeneralizedAlpha1D
{
public:
    GeneralizedAlpha1D(double time_step_size = 1e-2);
    
    // Methods required by the test suite
    void setup_system();
    void initialize_solution();
    void advance_time_step();
    double compute_integral() const;
    void set_time(double new_time);
    double get_time() const { return time; }
    void set_convection_coefficient(double coeff) { convection_coefficient = coeff; }
    void set_use_supg(bool use) { use_supg = use; }
    
    // Accessors for testing
    const DoFHandler<dim>& get_dof_handler() const { return dof_handler; }
    const FESystem<dim>& get_fe() const { return fe; }
    const LA::MPI::Vector& get_solution() const { return solution; }
    const parallel::distributed::Triangulation<dim>& get_triangulation() const { return triangulation; }
    
    void run();
    
private:
    void assemble_system();
    void solve_time_step();
    void output_results(unsigned int timestep) const;
    
    // Helper functions
    double get_velocity(const unsigned int component, 
                        const std::vector<double> &U) const;
    double get_pressure(const std::vector<double> &U) const;
    double get_acceleration(const std::vector<double> &U,
                            const std::vector<double> &U_old,
                            const std::vector<double> &U_old2) const;
    
    MPI_Comm                                  mpi_communicator;
    parallel::distributed::Triangulation<dim> triangulation;
    FESystem<dim>                             fe;
    DoFHandler<dim>                           dof_handler;
    IndexSet                                  locally_owned_dofs;
    IndexSet                                  locally_relevant_dofs;
    LA::MPI::SparseMatrix                     system_matrix;
    LA::MPI::Vector                           solution, solution_old, solution_old2;
    LA::MPI::Vector                           system_rhs;
    
    double                                    time, time_step;
    const double                              alpha_m, alpha_f, gamma;
    double                                    viscosity = 0.01; // Physical parameter
    double                                    convection_coefficient = 1.0; // Scaling for convection term
    bool                                      use_supg = true; // Whether to use SUPG stabilization
    
    AffineConstraints<double>                 constraints;
    ManufacturedSolution<dim>                 exact_solution;
};

template <int dim>
GeneralizedAlpha1D<dim>::GeneralizedAlpha1D(double time_step_size)
    : mpi_communicator(MPI_COMM_WORLD)
    , triangulation(mpi_communicator)
    , fe(FE_Q<dim>(1), dim, FE_Q<dim>(1), 1)  // {u, p}
    , dof_handler(triangulation)
    , time(0), time_step(time_step_size)
    , alpha_m(0.5), alpha_f(0.5), gamma(0.5)
{}

// Implementation details are in the .cpp file
// This header only declares the interface needed for testing

#endif // GENERALIZED_ALPHA_1D_HPP
