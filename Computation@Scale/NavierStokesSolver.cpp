#include "ChannelFlowSolver.hpp"

// Directory helper functions
bool directory_exists(const std::string &path)
{
  struct stat info;
  if (stat(path.c_str(), &info) != 0) return false;
  return (info.st_mode & S_IFDIR);
}

bool create_directory(const std::string &path)
{
  if (directory_exists(path)) return true;
#ifndef _WIN32
  int result = mkdir(path.c_str(), 0777);
#else
  int result = mkdir(path.c_str());
#endif
  return (result == 0 || errno == EEXIST);
}

// Base class implementation
template <int dim>
NavierStokesSolver<dim>::NavierStokesSolver(const MPI_Comm &mpi_comm)
  : mu(0.001), rho(1.0), Re(100.0)
  , dt(1.0/3200.0), dt_min(1.0/6400.0), dt_max(1.0/1600.0), cfl_number(0.5)
  , T(1.0)
  , time(0.0), timestep(0)
  , specRad(0.0)
  , alpf(1.0/(1.0+specRad))
  , alpm(0.5*(3.0-specRad)/(1.0+specRad))
  , gamm(0.5+alpm-alpf)
  , mpi_communicator(mpi_comm)
  , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_comm))
  , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_comm))
  , pcout(std::cout, this_mpi_process==0)
  , fe(FE_Q<dim>(2), dim, FE_Q<dim>(1), 1)  // P2-P1 elements
  , dof_handler(triangulation)
  , mapping(1)
  , stokes_solved(false)
  , timer(pcout, TimerOutput::summary, TimerOutput::wall_times)
  , output_dir("results")
  , checkpoint_dir("checkpoints")
  , checkpoint_interval(20)
  , max_newton_iterations(20)
  , newton_tolerance(1e-6)
{
  if (this_mpi_process == 0)
  {
    if (!directory_exists(output_dir))
      create_directory(output_dir);
    
    if (!directory_exists(checkpoint_dir))
      create_directory(checkpoint_dir);
      
    output_file.open(output_dir + "/flow_data.dat");
  }
}

template <int dim>
void NavierStokesSolver<dim>::setup_dofs()
{
  TimerOutput::Scope t(timer, "Setup DoFs");

  dof_handler.distribute_dofs(fe);

  // Owned / relevant
  IndexSet locally_owned = dof_handler.locally_owned_dofs();
  IndexSet locally_relevant;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant);

  // Set up constraints
  DoFRenumbering::Cuthill_McKee(dof_handler);
  constraints.clear();
  set_boundary_conditions();
  constraints.close();

  // Build the DSP with the correct dimensions
  const unsigned int n_dofs = dof_handler.n_dofs(); // Total number of DoFs
  (void) n_dofs; // Avoid unused variable warning
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);

  // Trilinos pattern & matrices
  sparsity_pattern.reinit(locally_owned, locally_owned, dsp, mpi_communicator);
  system_matrix.reinit(sparsity_pattern);
  constant_block.reinit(sparsity_pattern);

  // Vector initialization
  system_rhs.reinit(locally_owned, mpi_communicator);
  solution.reinit(locally_owned, locally_relevant, mpi_communicator);
  old_solution.reinit(locally_owned, locally_relevant, mpi_communicator);
  old_solution2.reinit(locally_owned, locally_relevant, mpi_communicator);
  acceleration.reinit(locally_owned, locally_relevant, mpi_communicator);
  old_acceleration.reinit(locally_owned, locally_relevant, mpi_communicator);
  newton_update.reinit(locally_owned, mpi_communicator);

  pcout << "DoFs: " << dof_handler.n_dofs() << std::endl;

  // Build constant block for preconditioning
  {
    QGauss<dim> quad(fe.degree+2);
    FEValues<dim> fev(mapping, fe, quad,
                     update_values|update_gradients|
                     update_quadrature_points|update_JxW_values);
    FullMatrix<double> local_mat(fe.n_dofs_per_cell(), fe.n_dofs_per_cell());
    Vector<double> dummy_rhs(fe.n_dofs_per_cell());
    std::vector<types::global_dof_index> local_dofs(fe.n_dofs_per_cell());

    const auto U = FEValuesExtractors::Vector(0);
    const auto P = FEValuesExtractors::Scalar(dim);

    // Only iterate over locally owned cells (not artificial ones)
    for (const auto &cell : dof_handler.active_cell_iterators_on_level(0))
    {
      if (cell->is_locally_owned()) 
      {
        fev.reinit(cell);
        local_mat = 0;
        for (unsigned q=0; q<quad.size(); ++q)
        {
          const double JxW = fev.JxW(q);
          for (unsigned i=0; i<fe.n_dofs_per_cell(); ++i)
          {
            const auto phi_i_u = fev[U].value(i,q);
            const auto grad_i_u = fev[U].gradient(i,q);
            const double div_i_u = trace(grad_i_u);
            
            for (unsigned j=0; j<fe.n_dofs_per_cell(); ++j)
            {
              const auto phi_j_u = fev[U].value(j,q);
              const auto grad_j_u = fev[U].gradient(j,q);
              const double div_j_u = trace(grad_j_u);
              
              local_mat(i,j) += rho*scalar_product(phi_j_u, phi_i_u)*JxW
                              + mu*scalar_product(grad_j_u, grad_i_u)*JxW
                              - fev[P].value(j,q)*div_i_u*JxW
                              - div_j_u*fev[P].value(i,q)*JxW;
            }
          }
        }
        cell->get_dof_indices(local_dofs);
        constraints.distribute_local_to_global(
        local_mat, dummy_rhs, local_dofs, constant_block, system_rhs);
      }
    
      // Add small values to diagonal for stability
      for (unsigned int i=0; i<constant_block.m(); ++i)
        if (std::abs(constant_block.diag_element(i)) < 1e-12)
          constant_block.set(i, i, constant_block.diag_element(i) + 1e-12);
    }
  
    // Initialize preconditioner
    TrilinosWrappers::PreconditionILU::AdditionalData ilu_data;
    ilu_data.overlap = 0;
    ilu_data.ilu_fill = 10.0;
    ilu_data.ilu_atol = 1e-8;
    ilu_data.ilu_rtol = 1.0;
    preconditioner.initialize(constant_block, ilu_data);

    pcout << "DoFs: " << dof_handler.n_dofs() << std::endl;
  }
}

template <int dim>
void NavierStokesSolver<dim>::assemble_system()
{
    // updating ghost values
    this->solution.update_ghost_values();
    this->old_solution.update_ghost_values();
    this->old_solution2.update_ghost_values();
    this->old_acceleration.update_ghost_values();

  system_matrix = 0;
  system_rhs = 0;

  QGauss<dim> quad(fe.degree+3);  // Use higher quadrature for better integration
  FEValues<dim> fev(mapping, fe, quad,
                    update_values|update_gradients|
                    update_quadrature_points|update_JxW_values);
  
  std::vector<types::global_dof_index> local_dofs(fe.n_dofs_per_cell());
  std::vector<Tensor<1,dim>> u_present(quad.size()), u_old(quad.size()), a_old(quad.size());
  std::vector<Tensor<2,dim>> grad_u_present(quad.size()), grad_u_old(quad.size());
  std::vector<double> p_present(quad.size()), p_old(quad.size());
  std::vector<Tensor<1,dim>> grad_p_present(quad.size()), grad_p_old(quad.size());
  
  const auto U = FEValuesExtractors::Vector(0);
  const auto P = FEValuesExtractors::Scalar(dim);
  
  for (const auto cell : dof_handler.active_cell_iterators())
  {
    if (cell->is_locally_owned())
    {
        fev.reinit(cell);
        FullMatrix<double> local_m(fe.n_dofs_per_cell(), fe.n_dofs_per_cell());
        Vector<double> local_rhs(fe.n_dofs_per_cell());
        
        local_m = 0;
        local_rhs = 0;
        
        fev[U].get_function_values(solution, u_present);
        fev[U].get_function_gradients(solution, grad_u_present);
        fev[P].get_function_values(solution, p_present);
        fev[P].get_function_gradients(solution, grad_p_present);
        
        fev[U].get_function_values(old_solution, u_old);
        fev[U].get_function_gradients(old_solution, grad_u_old);
        fev[P].get_function_values(old_solution, p_old);
        fev[P].get_function_gradients(old_solution, grad_p_old);
        
        fev[U].get_function_values(old_acceleration, a_old);
        
        double h = std::pow(cell->measure(), 1.0/dim);
        
        for (unsigned q=0; q<quad.size(); ++q)
        {
          const double JxW = fev.JxW(q);
          
          // Compute averaged quantities for generalized-alpha - use consistent alpha parameters for all
          Tensor<1,dim> u_avg = alpf*u_present[q] + (1.0-alpf)*u_old[q];
          Tensor<2,dim> grad_u_avg = alpf*grad_u_present[q] + (1.0-alpf)*grad_u_old[q];
          double p_avg = alpf*p_present[q] + (1.0-alpf)*p_old[q];
          (void)p_avg;  // Avoid unused variable warning
          Tensor<1,dim> grad_p_avg = alpf*grad_p_present[q] + (1.0-alpf)*grad_p_old[q];
          
          // Compute stabilization parameters 
          double nu = mu/rho;
          double u_mag = std::max(u_avg.norm(), 1e-6);
          
          // Modified stabilization parameters based on element Reynolds number
          double Re_elem = u_mag * h / (2.0 * nu);
          double alpha_supg = (Re_elem <= 1.0) ? Re_elem/3.0 : 1.0/3.0;
          
          tau_SUPG = alpha_supg * h / (2.0 * u_mag);
          tau_PSPG = h * h / (4.0 * nu);
          tau_LSIC = h * u_mag / 2.0;
          
          for (unsigned i=0; i<fe.n_dofs_per_cell(); ++i)
          {
            const auto phi_i_u = fev[U].value(i,q);
            const auto grad_phi_i_u = fev[U].gradient(i,q);
            const double div_phi_i_u = trace(grad_phi_i_u);
            const double phi_i_p = fev[P].value(i,q);
            (void) phi_i_p;  // Avoid unused variable warning
            const auto grad_phi_i_p = fev[P].gradient(i,q);
            
            // Residual computation for RHS - use more consistent time derivative
            Tensor<1,dim> residual;
            residual = rho * (u_present[q] - u_old[q])/(gamm*dt) 
                    + rho * (1.0-gamm)/gamm * a_old[q];
            
            // Convection term using alpha-averaged velocity
            for (unsigned d=0; d<dim; ++d)
              for (unsigned e=0; e<dim; ++e)
                residual[d] += rho * u_avg[e] * grad_u_avg[d][e];
            
            // Pressure gradient term using alpha-averaged pressure
            for (unsigned d=0; d<dim; ++d)
              residual[d] += grad_p_avg[d];
            
            // Viscous term using alpha-averaged velocity gradient
            for (unsigned d=0; d<dim; ++d)
              for (unsigned e=0; e<dim; ++e)
                residual[d] -= mu * (grad_u_avg[d][e] + grad_u_avg[e][d]);
            
            // Compute RHS
            local_rhs(i) -= scalar_product(residual, phi_i_u) * JxW;
            
            // Continuity residual with alpha-averaged velocity
            double div_u = trace(grad_u_avg);
            local_rhs(i) -= div_u * phi_i_p * JxW;
            
            // Assembly of Jacobian matrix
            for (unsigned j=0; j<fe.n_dofs_per_cell(); ++j)
            {
              const auto phi_j_u = fev[U].value(j,q);
              const auto grad_phi_j_u = fev[U].gradient(j,q);
              const double div_phi_j_u = trace(grad_phi_j_u);
              const double phi_j_p = fev[P].value(j,q);
              const auto grad_phi_j_p = fev[P].gradient(j,q);
              
              // Mass term
              local_m(i,j) += rho/(gamm*dt) * scalar_product(phi_j_u, phi_i_u) * JxW;
              
              // Convection term (linearized)
              for (unsigned d=0; d<dim; ++d)
                for (unsigned e=0; e<dim; ++e)
                  local_m(i,j) += alpf * rho * u_avg[e] * grad_phi_j_u[d][e] * phi_i_u[d] * JxW;
              
              // Additional convection term (Newton linearization)
              for (unsigned d=0; d<dim; ++d)
                for (unsigned e=0; e<dim; ++e)
                  local_m(i,j) += alpf * rho * phi_j_u[e] * grad_u_avg[d][e] * phi_i_u[d] * JxW;
              
              // Viscous term
              for (unsigned d=0; d<dim; ++d)
                for (unsigned e=0; e<dim; ++e)
                  local_m(i,j) += alpf * mu * (grad_phi_j_u[d][e] + grad_phi_j_u[e][d]) * 
                                  0.5 * (grad_phi_i_u[d][e] + grad_phi_i_u[e][d]) * JxW;
              
              // Pressure terms - use consistent alpha parameter
              for (unsigned d=0; d<dim; ++d)
                local_m(i,j) += alpf * grad_phi_j_p[d] * phi_i_u[d] * JxW;
              local_m(i,j) -= alpf * div_phi_j_u * phi_i_p * JxW;
              
              // SUPG stabilization
              if (u_mag > 1e-6) 
              {
                Tensor<1,dim> supg_test;
                for (unsigned d=0; d<dim; ++d)
                  for (unsigned e=0; e<dim; ++e)
                    supg_test[d] += u_avg[e] * grad_phi_i_u[d][e];
                
                Tensor<1,dim> linearized_residual;
                for (unsigned d=0; d<dim; ++d) 
                {
                  linearized_residual[d] = rho * phi_j_u[d]/(gamm*dt);
                  for (unsigned e=0; e<dim; ++e)
                    linearized_residual[d] += alpf * rho * (u_avg[e] * grad_phi_j_u[d][e] + phi_j_u[e] * grad_u_avg[d][e]);
                  
                  linearized_residual[d] += alpf * grad_phi_j_p[d];
                  
                  for (unsigned e=0; e<dim; ++e)
                    linearized_residual[d] -= alpf * mu * (grad_phi_j_u[d][e] + grad_phi_j_u[e][d]);
                }
                
                local_m(i,j) += tau_SUPG * scalar_product(linearized_residual, supg_test) * JxW;
              }
              
              // PSPG stabilization
              for (unsigned d=0; d<dim; ++d) 
              {
                local_m(i,j) += tau_PSPG * rho * phi_j_u[d]/(gamm*dt) * grad_phi_i_p[d] * JxW;
                for (unsigned e=0; e<dim; ++e)
                  local_m(i,j) += tau_PSPG * alpf * rho * (u_avg[e] * grad_phi_j_u[d][e] + phi_j_u[e] * grad_u_avg[d][e]) * grad_phi_i_p[d] * JxW;
                local_m(i,j) += tau_PSPG * alpf * grad_phi_j_p[d] * grad_phi_i_p[d] * JxW;
              }
              
              // LSIC stabilization
              local_m(i,j) += tau_LSIC * rho * div_phi_j_u * div_phi_i_u * JxW;
            }
          }
        }
    
        
      cell->get_dof_indices(local_dofs);
      constraints.distribute_local_to_global(
        local_m, local_rhs, local_dofs, system_matrix, system_rhs);
    }
  }
}

template <int dim>
void NavierStokesSolver<dim>::solve_newton_system()
{
  TimerOutput::Scope t(timer, "Solve Newton system");
  
  unsigned int newton_iteration = 0;
  double residual_norm;
  double initial_residual_norm = 0.0;
  
  // Use a reasonable damping factor
  const double damping = 0.3;
  
  for (newton_iteration = 0; newton_iteration < max_newton_iterations; ++newton_iteration)
  {
    system_rhs = 0;
    system_matrix = 0;
    assemble_system();
    
    // Compute residual norm
    TrilinosWrappers::MPI::Vector residual(system_rhs);
    system_matrix.vmult(residual, solution);
    residual -= system_rhs;
    residual_norm = residual.l2_norm();
    
    if (newton_iteration == 0)
      initial_residual_norm = residual_norm;
    
    pcout << "   Newton iteration " << newton_iteration
          << ", residual = " << residual_norm 
          << ", relative = " << residual_norm/initial_residual_norm << std::endl;
    
    // Check convergence
    if (residual_norm < newton_tolerance * initial_residual_norm || residual_norm < 1e-10)
      break;
    
    // Solve linear system
    SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
    SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);
    
    newton_update = 0;
    solver.solve(system_matrix, newton_update, system_rhs, preconditioner);
    constraints.distribute(newton_update);
    
    pcout << "   Linear solve: " 
          << (solver_control.last_step() == solver_control.max_steps() ? "failed" : "converged")
          << " in " << solver_control.last_step() << " iterations." << std::endl;
    
    // Apply damping
    solution.add(damping, newton_update);
    
    // Check if solution is changing significantly
    double update_norm = newton_update.l2_norm();
    double solution_norm = solution.l2_norm();
    double relative_change = damping * update_norm / solution_norm;
    
    pcout << "   Relative change: " << relative_change << std::endl;
    
    // If solution is not changing much, stop
    if (relative_change < 1e-6)
      break;
  }
  
  if (newton_iteration == max_newton_iterations)
    pcout << "   WARNING: Newton solver did not converge after " 
          << max_newton_iterations << " iterations!" << std::endl;
  
  // Update acceleration
  TrilinosWrappers::MPI::Vector temp(solution);
  temp -= old_solution;
  temp *= (1.0/(gamm*dt));
  
  acceleration = temp;
  acceleration.add((gamm-1.0)/gamm, old_acceleration);
}

template <int dim>
void NavierStokesSolver<dim>::solve_stokes()
{
  pcout << "Solving initial Stokes problem..." << std::endl;
  
  // Create matrix for Stokes problem (no convection terms)
  QGauss<dim> quad(fe.degree+2);
  FEValues<dim> fev(mapping, fe, quad,
                   update_values|update_gradients|
                   update_quadrature_points|update_JxW_values);
  
  TrilinosWrappers::SparseMatrix stokes_matrix;
  stokes_matrix.reinit(sparsity_pattern);
  TrilinosWrappers::MPI::Vector stokes_rhs;
  stokes_rhs.reinit(solution.locally_owned_elements(), mpi_communicator);
  
  std::vector<types::global_dof_index> local_dofs(fe.n_dofs_per_cell());
  const auto U = FEValuesExtractors::Vector(0);
  const auto P = FEValuesExtractors::Scalar(dim);
  
  // Assemble Stokes system
  for (auto cell : dof_handler.active_cell_iterators())
  {
    if (!cell->is_artificial())  // Add this check
    {
      fev.reinit(cell);
      FullMatrix<double> local_mat(fe.n_dofs_per_cell(),fe.n_dofs_per_cell());
      Vector<double> local_rhs(fe.n_dofs_per_cell());
      local_mat = 0;
      local_rhs = 0;
      
      for (unsigned q=0; q<quad.size(); ++q)
      {
        const double JxW = fev.JxW(q);
        
        for (unsigned i=0;i<fe.n_dofs_per_cell();++i)
        {
          const auto phi_i_u = fev[U].value(i,q);
          (void)phi_i_u; // Avoid unused variable warning
          const auto grad_phi_i_u = fev[U].gradient(i,q);
          const double div_phi_i_u = trace(grad_phi_i_u);
          
          for (unsigned j=0;j<fe.n_dofs_per_cell();++j)
          {
            const auto phi_j_u = fev[U].value(j,q);
            (void)phi_j_u; // Avoid unused variable warning
            const auto grad_phi_j_u = fev[U].gradient(j,q);
            const double div_phi_j_u = trace(grad_phi_j_u);
            
            // Viscous term
            local_mat(i,j) += mu*scalar_product(grad_phi_j_u,grad_phi_i_u)*JxW;
            
            // Pressure terms
            local_mat(i,j) -= fev[P].value(j,q)*div_phi_i_u*JxW;
            local_mat(i,j) -= div_phi_j_u*fev[P].value(i,q)*JxW;
            
            // Mass term for stability
            local_mat(i,j) += 1e-6*fev[P].value(j,q)*fev[P].value(i,q)*JxW;
          }
        }
      }
      
      cell->get_dof_indices(local_dofs);
      constraints.distribute_local_to_global(
        local_mat,local_rhs,local_dofs,stokes_matrix,stokes_rhs);
    }
  }
  
  // Solve Stokes problem with stronger preconditioner
  TrilinosWrappers::PreconditionILU stokes_prec;
  TrilinosWrappers::PreconditionILU::AdditionalData ilu_data;
  ilu_data.overlap = 0;
  ilu_data.ilu_fill = 8.0;
  ilu_data.ilu_atol = 1e-10;
  ilu_data.ilu_rtol = 1.0;
  stokes_prec.initialize(stokes_matrix, ilu_data);
  
  // Use GMRES with high iteration count
  SolverControl sc(50000, 1e-8);
  SolverGMRES<TrilinosWrappers::MPI::Vector> solver(sc);
  
  // Solve
  solution = 0;
  try {
    solver.solve(stokes_matrix, solution, stokes_rhs, stokes_prec);
    constraints.distribute(solution);
    
    pcout << "  Stokes solver converged in " << sc.last_step() << " iterations." << std::endl;
    stokes_solved = true;
  }
  catch (std::exception &e) {
    pcout << "  Stokes solver failed: " << e.what() << std::endl;
    pcout << "  Proceeding with zero initial condition." << std::endl;
    solution = 0;
  }
}

template <int dim>
double NavierStokesSolver<dim>::get_minimum_cell_size() const
{
  // Get the minimum cell size across all locally owned cells
  // and ghost cells
  double h_min_local = std::numeric_limits<double>::max();
  for (const auto &cell : this->triangulation.active_cell_iterators()) 
  {
    // Check if the cell is locally owned or a ghost cell
    if (cell->is_locally_owned() || cell->is_ghost()) 
    { 
      // Get the minimum cell size
      h_min_local = std::min(h_min_local, cell->diameter());
    }
  }
  return Utilities::MPI::min(h_min_local, this->mpi_communicator);
}

template <int dim>
double NavierStokesSolver<dim>::compute_time_step()
{
  if (this->timestep == 0)
    return this->dt_min;  // Just use the minimum time step to start
  
  // After the first step, try to adapt based on the solution
  const double h_min = get_minimum_cell_size();
  
  double u_max = 1.0;  // Use a reasonable initial guess
  
  // Calculate CFL time step
  const double dt_cfl = cfl_number * (h_min / std::max(u_max, 1e-8));
  return std::min(dt_max, std::max(dt_min, dt_cfl));
}

template <int dim>
double NavierStokesSolver<dim>::get_max_velocity()
{
  // Use VectorTools to compute Linfty norm for velocity components
  const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim), dim+1);
  Vector<float> norm_per_cell(triangulation.n_active_cells());
  
  VectorTools::integrate_difference(
    dof_handler,
    solution,
    Functions::ZeroFunction<dim>(dim+1),
    norm_per_cell,
    QGauss<dim>(fe.degree+1),
    VectorTools::Linfty_norm,
    &velocity_mask);
  
  // Get the maximum norm across all cells and all processors
  float max_local = norm_per_cell.linfty_norm();
  return Utilities::MPI::max(max_local, mpi_communicator);
}

template <int dim>
bool NavierStokesSolver<dim>::check_solution_validity()
{
  TimerOutput::Scope t(timer, "Check solution");
  bool valid = true;

  // Check local entries for non-finite values
  for (auto idx = solution.local_range().first;
       idx < solution.local_range().second;
       ++idx) {
    if (!std::isfinite(solution[idx])) {
      valid = false;
      break;
    }
  }

  // Ensure all processes agree
  return Utilities::MPI::min(static_cast<int>(valid), mpi_communicator) > 0;
}

template <int dim>
void NavierStokesSolver<dim>::post_process_solution()
{
  // Placeholder for post-processing steps
  TimerOutput::Scope t(timer, "Post-process solution");
  
  // No additional post-processing for now
}

template <int dim>
void NavierStokesSolver<dim>::output_results(unsigned int step)
{
  TimerOutput::Scope t(timer, "Output results");
  
  // Create results directory if it doesn't exist
  if (this_mpi_process == 0) {
    bool dir_created = create_directory(output_dir);
    pcout << "Directory creation " << (dir_created ? "successful" : "failed") << std::endl;
  }

  // Set up data output
  pcout << "Setting up output for timestep " << step << "..." << std::endl;
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  
  std::vector<std::string> names(dim, "velocity");
  names.push_back("pressure");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    comp_interp(dim, DataComponentInterpretation::component_is_part_of_vector);
  comp_interp.push_back(DataComponentInterpretation::component_is_scalar);
  
  // Make sure ghost values are updated
  this->solution.update_ghost_values();
  this->old_solution.update_ghost_values();
  this->old_solution2.update_ghost_values();
  this->old_acceleration.update_ghost_values();

  data_out.add_data_vector(solution, names,
    DataOut<dim>::type_dof_data, comp_interp);
  
  // Add partition ID to visualize the domain decomposition - MODIFIED PART
  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");
  
  data_out.build_patches(mapping, 2);

  // Use the deal.II parallel VTU output functionality
  const std::string filename = output_dir + "/solution-" + 
                               Utilities::int_to_string(step, 6);
  
  data_out.write_vtu_with_pvtu_record("./", filename, step, mpi_communicator, 4);

  // Only process 0 creates the PVD file that combines all timesteps
  if (this_mpi_process == 0) {
    // Create a new PVD file each time (more robust)
    std::string pvd_filename = output_dir + "/solution.pvd";
    std::ofstream pvd(pvd_filename);
    if (!pvd) {
      pcout << "ERROR: Could not open PVD file for writing: " << pvd_filename << std::endl;
      return;
    }
    
    // Always write a complete PVD file with all current outputs
    pcout << "Writing complete PVD file with " << output_steps.size()+1 << " timesteps..." << std::endl;
    pvd << "<?xml version=\"1.0\"?>\n"
        << "<VTKFile type=\"Collection\">\n"
        << "  <Collection>\n";
    
    // Add all previous timesteps
    for (unsigned int i = 0; i < output_steps.size(); ++i) {
      pvd << "    <DataSet timestep=\"" << output_times[i] << "\" file=\"solution-"
          << Utilities::int_to_string(output_steps[i], 6) << ".pvtu\"/>\n";
    }
    
    // Add current timestep
    pcout << "Adding timestep " << time << " to PVD file..." << std::endl;
    pvd << "    <DataSet timestep=\"" << time << "\" file=\"solution-"
        << Utilities::int_to_string(step, 6) << ".pvtu\"/>\n";
    
    // Store this step for future updates
    output_steps.push_back(step);
    output_times.push_back(time);
    
    // Close the file
    pvd << "  </Collection>\n"
        << "</VTKFile>\n";
    pvd.close();
  }
}
  
template <int dim>
void NavierStokesSolver<dim>::run()
{
  make_grid();
  setup_dofs();

  // Initial Stokes solve to get a good starting point for the NS solver
  if (timestep == 0) 
  {
    solve_stokes();
    old_solution = solution;
    old_solution2 = solution;
    acceleration = 0;
    old_acceleration = 0;
    output_results(0);
  }

  while (time < T) 
  {
    dt = compute_time_step();
    time += dt;
    ++timestep;

    pcout << "Timestep " << timestep << ": t = " << time << ", dt = " << dt << std::endl;

    old_solution2 = old_solution;
    old_solution  = solution;
    old_acceleration = acceleration;

    solve_newton_system();
    
    if (!check_solution_validity()) 
    {
      pcout << "ERROR: Non-finite values detected in solution!" << std::endl;
      break;
    }
    
    post_process_solution();

    if (timestep % 5 == 0 || timestep == 1) 
    {
      output_results(timestep);
    }
    
    // Monitor maximum velocity
    double max_vel = get_max_velocity();
    pcout << "  Maximum velocity: " << max_vel << " m/s" << std::endl;
  }

  output_results(timestep);
}
  
// Explicit instantiation
template class NavierStokesSolver<2>;
template class NavierStokesSolver<3>;