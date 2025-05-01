#ifndef CHANNEL_FLOW_SOLVER_HPP
#define CHANNEL_FLOW_SOLVER_HPP

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_time.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/distributed/tria.h>

#include <fstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <chrono>
#include <algorithm>

using namespace dealii;

template <int dim>
class NavierStokesSolver
{
public:
  NavierStokesSolver(const MPI_Comm &mpi_comm = MPI_COMM_SELF);
  virtual ~NavierStokesSolver() = default;
  
  // Main driver function
  virtual void run();
  
  // Get access to solution for testing
  const TrilinosWrappers::MPI::Vector& get_solution() const { return solution; }
  
protected:
  // Core methods that must be implemented by derived classes
  virtual void make_grid() = 0;
  virtual void set_boundary_conditions() = 0;
  
  // Common methods for all solvers
  virtual void setup_dofs();
  virtual void assemble_system();
  virtual void solve_newton_system();
  virtual void solve_stokes();
  bool check_solution_validity();
  virtual void output_results(unsigned int step);
  virtual void post_process_solution();
  double compute_time_step();
  double get_max_velocity();
  double get_minimum_cell_size() const;
  
  // Physical parameters
  double mu, rho, Re;
  double dt, dt_min, dt_max, cfl_number;
  double T;
  double time;
  unsigned int timestep;

  // Generalized-alpha parameters
  double specRad, alpf, alpm, gamm;

  // Stabilization parameters
  double tau_SUPG, tau_PSPG, tau_LSIC;

  MPI_Comm mpi_communicator;
  const unsigned int n_mpi_processes, this_mpi_process;
  ConditionalOStream pcout;

  parallel::distributed::Triangulation<dim> triangulation{mpi_communicator};  
  FESystem<dim>      fe;
  DoFHandler<dim>    dof_handler;
  MappingQ<dim>      mapping;
  AffineConstraints<double> constraints;

  TrilinosWrappers::SparsityPattern  sparsity_pattern;
  TrilinosWrappers::SparseMatrix     system_matrix;
  TrilinosWrappers::MPI::Vector      system_rhs;
  TrilinosWrappers::MPI::Vector      newton_update;

  TrilinosWrappers::MPI::Vector solution, old_solution, old_solution2;
  TrilinosWrappers::MPI::Vector acceleration, old_acceleration;
  TrilinosWrappers::SparseMatrix     constant_block;
  TrilinosWrappers::PreconditionILU  preconditioner;

  std::ofstream output_file;
  std::vector<unsigned int> output_steps;
  std::vector<double>       output_times;

  bool stokes_solved;
  TimerOutput timer;

  std::string output_dir;
  std::string checkpoint_dir;
  unsigned int checkpoint_interval;
  
  unsigned int max_newton_iterations;
  double newton_tolerance;
};

// Derived class for the channel flow case
template <int dim>
class ChannelFlowSolver : public NavierStokesSolver<dim>
{
public:
  ChannelFlowSolver(const MPI_Comm &mpi_comm = MPI_COMM_SELF);
  
protected:
  void make_grid() override;
  void set_boundary_conditions() override;
  void save_checkpoint();
  void load_checkpoint();
};

// Derived class for manufactured solution testing
template <int dim>
class ManufacturedSolutionSolver : public NavierStokesSolver<dim>
{
public:
  ManufacturedSolutionSolver(const MPI_Comm &mpi_comm = MPI_COMM_SELF);
  
  // Compute error against exact solution
  double compute_error() const;
  
  void solve_newton_system() override;
  // Solve the system using Newton's method

  // Override run method to handle time-dependent solution
  void run() override;
  
protected:
  // Exact solution for testing that depends on time
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution() : Function<dim>(dim+1) {}
    
    void set_time(double t) override { time = t; }
    double get_time() const { return time; }

    void vector_value(const Point<dim>& p, Vector<double>& values) const override;
    void vector_gradient(const Point<dim>& p, std::vector<Tensor<1,dim>>& gradients) const override;
    
  private:
    double time = 0.0;
  };
  
  // Source term (forcing function) for the manufactured solution
  class ForcingFunction : public Function<dim>
  {
  public:
    ForcingFunction() : Function<dim>(dim) {}
    
    void set_time(double t) override { time = t; }
    double get_time() const { return time; }
    
    void vector_value(const Point<dim>& p, Vector<double>& values) const override;
    
  private:
    double time = 0.0;
  };
  
  void make_grid() override;
  void set_boundary_conditions() override;
  void setup_dofs() override;
  void assemble_system() override;
  void output_results(unsigned int step) override;
  void add_forcing_terms();
  
  mutable ExactSolution exact_solution;
  ForcingFunction forcing_function;
};

// Helper function for directory operations
bool directory_exists(const std::string &path);
bool create_directory(const std::string &path);

#endif // CHANNEL_FLOW_SOLVER_HPP