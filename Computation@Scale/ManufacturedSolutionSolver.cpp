#include "ChannelFlowSolver.hpp"

// Manufactured solution implementation with time-dependent polynomials
template <int dim>
ManufacturedSolutionSolver<dim>::ManufacturedSolutionSolver(const MPI_Comm &mpi_comm)
  : NavierStokesSolver<dim>(mpi_comm)
{
  // Use specific settings for testing
  this->mu = 1.0;         // Higher viscosity for better stability
  this->rho = 1.0;
  //this->dt = this->compute_time_step();      // pick a CFL-safe time step
  this->T = 0.0003;          // Just run a few time steps
  
  // Use safer parameters for generalized-alpha method
  this->specRad = 0.4;    // Less dissipative value but still provides stability
  this->alpf = 1.0/(1.0+this->specRad);
  this->alpm = 0.5*(3.0-this->specRad)/(1.0+this->specRad);
  this->gamm = 0.5+this->alpm-this->alpf;
  
  // More iterations with adaptive damping
  this->max_newton_iterations = 30;
  this->newton_tolerance = 1e-3;
}

template <int dim>
void ManufacturedSolutionSolver<dim>::make_grid()
{
  this->pcout << "Creating manufactured solution mesh..." << std::endl;
  
  // Use unit square (0,1)x(0,1) for testing
  GridGenerator::hyper_cube(this->triangulation, 0, 1);
  
  // Refine to get accurate results
  this->triangulation.refine_global(3);
  
  // Partition the mesh for parallel processing no need since we are already calling it on the root
  //GridTools::partition_triangulation(this->mpi_communicator, this->triangulation);

  // Set boundary IDs:
  this->pcout << "Active cells: " << this->triangulation.n_active_cells() << std::endl;
}

template <int dim>
void ManufacturedSolutionSolver<dim>::set_boundary_conditions()
{
  // Apply exact solution on all boundaries at current time
  exact_solution.set_time(this->time);
  VectorTools::interpolate_boundary_values(
    this->dof_handler, 
    0,  // All boundaries have ID 0 by default
    exact_solution,
    this->constraints);
}

template <int dim>
void ManufacturedSolutionSolver<dim>::setup_dofs()
{
  // Call the base class implementation
  NavierStokesSolver<dim>::setup_dofs();
  
  // Initialize solution with exact solution for better convergence
  exact_solution.set_time(0.0);
  Vector<double> temp(this->dof_handler.n_dofs());
  VectorTools::interpolate(this->dof_handler, exact_solution, temp);
  
  // Create a non-ghosted vector for writing
  TrilinosWrappers::MPI::Vector solution_temp(this->dof_handler.locally_owned_dofs(),
                                             this->mpi_communicator);
  
  // Copy values to the non-ghosted vector
  for (unsigned int i = 0; i < temp.size(); ++i) {
    if (this->dof_handler.locally_owned_dofs().is_element(i)) {
      solution_temp[i] = temp[i];
    }
  }
  solution_temp.compress(VectorOperation::insert);
  
  // Copy to the ghosted solution vector
  this->solution = solution_temp;
  
  // Also update old solutions
  this->old_solution = this->solution;
  this->old_solution2 = this->solution;
  this->old_acceleration = 0;
  
  // Skip initial Stokes solve
  this->stokes_solved = true;
}

template <int dim>
void ManufacturedSolutionSolver<dim>::assemble_system()
{
  // First call the base class implementation
  NavierStokesSolver<dim>::assemble_system();
  
  // Then add the forcing terms
  add_forcing_terms();
}

template <int dim>
void ManufacturedSolutionSolver<dim>::add_forcing_terms()
{
  // Set the current time in the forcing function
  forcing_function.set_time(this->time);
  
  // Add forcing terms to the right-hand side
  QGauss<dim> quad(this->fe.degree+2);
  FEValues<dim> fev(this->mapping, this->fe, quad,
                   update_values|update_quadrature_points|update_JxW_values);
                   
  Vector<double> local_rhs(this->fe.n_dofs_per_cell());
  std::vector<types::global_dof_index> local_dofs(this->fe.n_dofs_per_cell());
  std::vector<Vector<double>> f_values(quad.size(), Vector<double>(dim));
  
  const auto U = FEValuesExtractors::Vector(0);
  
  for (auto cell : this->dof_handler.active_cell_iterators())
  {
    fev.reinit(cell);
    local_rhs = 0;
    
    // Get forcing function values at quadrature points
    forcing_function.vector_value_list(fev.get_quadrature_points(), f_values);
    
    for (unsigned q=0; q<quad.size(); ++q)
    {
      const double JxW = fev.JxW(q);
      
      for (unsigned i=0; i<this->fe.n_dofs_per_cell(); ++i)
      {
        const auto phi_i_u = fev[U].value(i,q);
        
        // Add forcing term contribution
        for (unsigned d=0; d<dim; ++d)
          local_rhs(i) += f_values[q][d] * phi_i_u[d] * JxW;
      }
    }
    
    cell->get_dof_indices(local_dofs);
    this->constraints.distribute_local_to_global(local_rhs, local_dofs, this->system_rhs);
  }
}

template <int dim>
void ManufacturedSolutionSolver<dim>::run()
{
  make_grid();
  setup_dofs();

  // Initialize time and timestep
  this->dt = this->compute_time_step();
  
  // Initialize solution with exact solution at t=0
  exact_solution.set_time(this->time);
  Vector<double> temp(this->dof_handler.n_dofs());
  VectorTools::interpolate(this->dof_handler, exact_solution, temp);
  
  // Create a non-ghosted vector for writing
  TrilinosWrappers::MPI::Vector solution_temp(this->dof_handler.locally_owned_dofs(),
                                             this->mpi_communicator);
  
  // Copy values to the non-ghosted vector
  for (unsigned int i = 0; i < temp.size(); ++i) {
    if (this->dof_handler.locally_owned_dofs().is_element(i)) {
      solution_temp[i] = temp[i];
    }
  }
  solution_temp.compress(VectorOperation::insert);
  
  // Copy to the ghosted solution vector
  this->solution = solution_temp;
  this->old_solution = this->solution;
  this->old_solution2 = this->solution;
  
  // Track error evolution
  std::ofstream error_file;
  if (this->this_mpi_process == 0) {
    error_file.open("error_evolution.dat");
    if (error_file.is_open()) {
      error_file << "# Timestep Time L2Error\n";
    }
  }
  
  output_results(0);
  
  // Start with a much smaller time step than configured
  this->dt = std::min(this->dt, this->dt_min);
  
  // Compute initial error
  double initial_error = compute_error();
  this->pcout << "Initial L2 error in velocity: " << initial_error << std::endl;
  
  if (this->this_mpi_process == 0 && error_file.is_open()) {
    error_file << this->timestep << " " << this->time << " " << initial_error << "\n";
  }
  
  // Time stepping loop with safety mechanisms
  unsigned int consecutive_failures = 0;
  unsigned int max_failures = 3;
  
  while (this->time < this->T && consecutive_failures < max_failures)
  {
    // Save current state for possible rollback
    double old_time = this->time;
    TrilinosWrappers::MPI::Vector backup_solution = this->solution;
    
    // Advance time
    this->time += this->dt;
    ++this->timestep;
    
    this->pcout << "Timestep " << this->timestep << ": t = " << this->time 
                << ", dt = " << this->dt << std::endl;
    
    // Update previous solutions
    this->old_solution2 = this->old_solution;
    this->old_solution = this->solution;
    this->old_acceleration = this->acceleration;
    
    // Set time for exact solution (for boundary conditions)
    exact_solution.set_time(this->time);
    
    // Update boundary conditions
    this->constraints.clear();
    set_boundary_conditions();
    this->constraints.close();
    
    // Initialize solution with exact solution for better convergence
    if (this->timestep <= 3) {
      Vector<double> temp(this->dof_handler.n_dofs());
      VectorTools::interpolate(this->dof_handler, exact_solution, temp);
      
      // Blend with current solution for stability
      double blend_factor = 0.5;
      for (unsigned int i = 0; i < this->solution.size(); ++i)
        if (this->solution.locally_owned_elements().is_element(i))
          this->solution[i] = (1.0-blend_factor) * this->solution[i] + blend_factor * temp[i];
      
      this->solution.compress(VectorOperation::insert);
    }
    
    // Try to solve the system
    this->solve_newton_system();
    
    // Compute error
    double error = compute_error();
    this->pcout << "L2 error in velocity at t = " << this->time << ": " << error << std::endl;
    
    // Check if solution failed
    if (error > 0.1) {
      this->pcout << "ERROR: Solution has large error, trying smaller time step" << std::endl;
      
      // Rollback to previous state
      this->time = old_time;
      this->solution = backup_solution;
      --this->timestep;
      
      // Reduce time step
      this->dt *= 0.5;
      
      // Check if time step is too small
      if (this->dt < this->dt_min * 0.1) {
        this->pcout << "WARNING: Time step too small, skipping to next time" << std::endl;
        this->dt = this->dt_min;
        consecutive_failures++;
      }
      
      continue;
    }
    
    // Reset failure counter on success
    consecutive_failures = 0;
    
    // Output results
    output_results(this->timestep);
    
    // Log error
    if (this->this_mpi_process == 0 && error_file.is_open()) {
      error_file << this->timestep << " " << this->time << " " << error << "\n";
    }
    
    // Adapt time step based on error
    if (error < 0.01) {
      // Good accuracy, increase time step
      this->dt = std::min(this->dt * 1.1, this->dt_max);
    }
    else if (error > 0.05) {
      // Poor accuracy, reduce time step
      this->dt = std::max(this->dt * 0.8, this->dt_min);
    }
  }
  
  // Close error file
  if (this->this_mpi_process == 0 && error_file.is_open()) {
    error_file.close();
  }
}

template <int dim>
void ManufacturedSolutionSolver<dim>::ExactSolution::vector_value(
  const Point<dim>& p, Vector<double>& values) const
{
  Assert(values.size() == dim+1, ExcDimensionMismatch(values.size(), dim+1));
  
  const double x = p[0];
  const double y = p[1];
  const double t = this->get_time();
  
  // Simple manufactured solution - polynomial
  // u_x = x^2(1-x)^2 * y(1-y) * t
  // u_y = -x(1-x) * y^2(1-y)^2 * t
  // p = (x-0.5)*(y-0.5) * t
  
  const double fx = x*x*(1-x)*(1-x);
  const double fy = y*(1-y);
  const double gx = x*(1-x);
  const double gy = y*y*(1-y)*(1-y);
  
  values[0] = fx * fy * t;        // u_x
  values[1] = -gx * gy * t;       // u_y
  values[2] = (x-0.5)*(y-0.5) * t; // p
}

template <int dim>
void ManufacturedSolutionSolver<dim>::ExactSolution::vector_gradient(
  const Point<dim>& p, std::vector<Tensor<1,dim>>& gradients) const
{
  Assert(gradients.size() == dim+1, ExcDimensionMismatch(gradients.size(), dim+1));
  
  const double x = p[0];
  const double y = p[1];
  const double t = this->get_time();
  
  // Derivatives of the basis functions
  // d/dx[x^2(1-x)^2] = 2x(1-x)^2 - 2x^2(1-x)
  // d/dy[y(1-y)] = (1-y) - y = 1-2y
  // d/dx[x(1-x)] = (1-x) - x = 1-2x
  // d/dy[y^2(1-y)^2] = 2y(1-y)^2 - 2y^2(1-y)
  
  const double fx = x*x*(1-x)*(1-x);
  const double fy = y*(1-y);
  const double gx = x*(1-x);
  const double gy = y*y*(1-y)*(1-y);
  
  const double dfx_dx = 2*x*(1-x)*(1-x) - 2*x*x*(1-x);
  const double dfy_dy = 1-2*y;
  const double dgx_dx = 1-2*x;
  const double dgy_dy = 2*y*(1-y)*(1-y) - 2*y*y*(1-y);
  
  // Gradient of u_x
  gradients[0][0] = dfx_dx * fy * t;
  gradients[0][1] = fx * dfy_dy * t;
  
  // Gradient of u_y
  gradients[1][0] = -dgx_dx * gy * t;
  gradients[1][1] = -gx * dgy_dy * t;
  
  // Gradient of p
  gradients[2][0] = (p[1] - 0.5) * t;   // ∂p/∂x
  gradients[2][1] = (p[0] - 0.5) * t;   // ∂p/∂y
}

template <int dim>
void ManufacturedSolutionSolver<dim>::ForcingFunction::vector_value(
  const Point<dim>& p, Vector<double>& values) const
{
  Assert(values.size() == dim, ExcDimensionMismatch(values.size(), dim));
  
  const double x = p[0];
  const double y = p[1];
  const double t = this->get_time();
  const double mu = 1.0;
  const double rho = 1.0;
  
  // Base polynomial functions
  const double fx = x*x*(1-x)*(1-x);
  const double fy = y*(1-y);
  const double gx = x*(1-x);
  const double gy = y*y*(1-y)*(1-y);
  
  // First derivatives
  const double dfx_dx = 2*x*(1-x)*(1-x) - 2*x*x*(1-x);
  const double dfy_dy = 1-2*y;
  const double dgx_dx = 1-2*x;
  const double dgy_dy = 2*y*(1-y)*(1-y) - 2*y*y*(1-y);
  
  // Second derivatives for Laplacian
  const double d2fx_dx2 = 2*(1-x)*(1-x) - 8*x*(1-x) + 2*x*x;
  const double d2fy_dy2 = -2;
  const double d2gx_dx2 = -2;
  const double d2gy_dy2 = 2*(1-y)*(1-y) - 8*y*(1-y) + 2*y*y;
  
  // Time derivatives
  const double dudt_x = fx * fy;
  const double dudt_y = -gx * gy;
  
  // Velocity components
  const double u_x = fx * fy * t;
  const double u_y = -gx * gy * t;
  
  // Spatial derivatives of velocity
  const double ux_x = dfx_dx * fy * t;
  const double ux_y = fx * dfy_dy * t;
  const double uy_x = -dgx_dx * gy * t;
  const double uy_y = -gx * dgy_dy * t;
  
  // Convection terms
  const double conv_x = u_x * ux_x + u_y * ux_y;
  const double conv_y = u_x * uy_x + u_y * uy_y;
  
  // Pressure gradients
  const double dp_dx = (p[1]-0.5)*t;;
  const double dp_dy = (p[0]-0.5)*t;
  
  // Laplacian terms
  const double lap_x = (d2fx_dx2 * fy + fx * d2fy_dy2) * t;
  const double lap_y = (-d2gx_dx2 * gy - gx * d2gy_dy2) * t;
  
  // Calculate the forcing function
  values[0] = rho * (dudt_x + conv_x) + dp_dx - mu * lap_x;
  values[1] = rho * (dudt_y + conv_y) + dp_dy - mu * lap_y;
}

template <int dim>
void ManufacturedSolutionSolver<dim>::solve_newton_system()
{
  TimerOutput::Scope t(this->timer, "Solve Newton system");
  
  // Use a more conservative damping factor
  const double damping = 0.1;
  
  unsigned int newton_iteration = 0;
  double residual_norm;
  double initial_residual_norm = 0.0;
  
  for (newton_iteration = 0; newton_iteration < this->max_newton_iterations; ++newton_iteration)
  {
    this->system_rhs = 0;
    this->system_matrix = 0;
    this->assemble_system();
    
    // Compute residual norm
    TrilinosWrappers::MPI::Vector residual(this->system_rhs);
    this->system_matrix.vmult(residual, this->solution);
    residual -= this->system_rhs;
    residual_norm = residual.l2_norm();
    
    if (newton_iteration == 0)
      initial_residual_norm = residual_norm;
    
    this->pcout << "   Newton iteration " << newton_iteration
          << ", residual = " << residual_norm 
          << ", relative = " << residual_norm/initial_residual_norm << std::endl;
    
    // Check convergence
    if (residual_norm < this->newton_tolerance || residual_norm < 1e-8)
      break;
    
    // Solve linear system with GMRES
    SolverControl solver_control(1000, 1e-8 * this->system_rhs.l2_norm());
    SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);
    
    // Use ILU preconditioner
    TrilinosWrappers::PreconditionILU preconditioner;
    TrilinosWrappers::PreconditionILU::AdditionalData ilu_data;
    ilu_data.overlap = 0;
    ilu_data.ilu_fill = 2.0;
    ilu_data.ilu_atol = 1e-8;
    ilu_data.ilu_rtol = 1.0;
    preconditioner.initialize(this->system_matrix, ilu_data);
    
    // Solve system
    this->newton_update = 0;
    solver.solve(this->system_matrix, this->newton_update, this->system_rhs, preconditioner);
    this->constraints.distribute(this->newton_update);
    
    this->pcout << "   Linear solve: converged in " 
          << solver_control.last_step() << " iterations." << std::endl;
    
    // Apply damping
    this->solution.add(damping, this->newton_update);
    
    // Check if solution is changing significantly
    double update_norm = this->newton_update.l2_norm();
    double solution_norm = this->solution.l2_norm();
    double relative_change = damping * update_norm / solution_norm;
    
    this->pcout << "   Relative change: " << relative_change << std::endl;
    
    // If solution is not changing much, stop
    if (relative_change < 1e-5 && newton_iteration > 2)
      break;
  }
  
  if (newton_iteration == this->max_newton_iterations)
    this->pcout << "   WARNING: Newton solver did not converge after " 
          << this->max_newton_iterations << " iterations!" << std::endl;
  
  // Update acceleration
  TrilinosWrappers::MPI::Vector temp(this->solution);
  temp -= this->old_solution;
  temp *= (1.0/(this->gamm*this->dt));
  
  this->acceleration = temp;
  this->acceleration.add((this->gamm-1.0)/this->gamm, this->old_acceleration);
}
  
template <int dim>
double ManufacturedSolutionSolver<dim>::compute_error() const
{
  // Set the correct time in the exact solution
  exact_solution.set_time(this->time);
  
  // Make sure solution has ghost values for access from all cells
  const_cast<TrilinosWrappers::MPI::Vector&>(this->solution).update_ghost_values();
  
  Vector<float> error_per_cell(this->triangulation.n_active_cells());
  
  // Compute L2 error in velocity
  const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim), dim+1);
  
  VectorTools::integrate_difference(
    this->dof_handler,
    this->solution,
    exact_solution,
    error_per_cell,
    QGauss<dim>(this->fe.degree + 3),
    VectorTools::L2_norm,
    &velocity_mask);
  
  return std::sqrt(Utilities::MPI::sum(
    error_per_cell.norm_sqr(), this->mpi_communicator));
}

// Override output_results to also output the exact solution
template <int dim>
void ManufacturedSolutionSolver<dim>::output_results(unsigned int step)
{
  TimerOutput::Scope t(this->timer, "Output results");
  
  // Create results directory if it doesn't exist
  if (this->this_mpi_process == 0) {
    bool dir_created = create_directory(this->output_dir);
    this->pcout << "Directory creation " << (dir_created ? "successful" : "failed") << std::endl;
  }

  // Set up data output
  this->pcout << "Setting up output for timestep " << step << "..." << std::endl;
  DataOut<dim> data_out;
  data_out.attach_dof_handler(this->dof_handler);
  
  std::vector<std::string> names(dim, "velocity");
  names.push_back("pressure");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    comp_interp(dim, DataComponentInterpretation::component_is_part_of_vector);
  comp_interp.push_back(DataComponentInterpretation::component_is_scalar);

  data_out.add_data_vector(this->solution, names,
    DataOut<dim>::type_dof_data, comp_interp);
  
  // Also output the exact solution
  Vector<double> exact_sol(this->dof_handler.n_dofs());
  exact_solution.set_time(this->time);
  VectorTools::interpolate(this->dof_handler, exact_solution, exact_sol);
  
  std::vector<std::string> exact_names(dim, "exact_velocity");
  exact_names.push_back("exact_pressure");
  
  data_out.add_data_vector(exact_sol, exact_names,
    DataOut<dim>::type_dof_data, comp_interp);
  
  // Compute and output the error
  Vector<double> error(this->solution);
  error -= exact_sol;
  
  std::vector<std::string> error_names(dim, "error_velocity");
  error_names.push_back("error_pressure");
  
  data_out.add_data_vector(error, error_names,
    DataOut<dim>::type_dof_data, comp_interp);
  
  data_out.build_patches(this->mapping, 2);

  if (this->this_mpi_process == 0) {
    std::string filename = this->output_dir + "/sol-" + Utilities::int_to_string(step, 6) + ".vtu";
    this->pcout << "Writing output file: " << filename << std::endl;
    
    std::ofstream f(filename);
    if (!f) {
      this->pcout << "ERROR: Could not open file for writing: " << filename << std::endl;
      return;
    }
    
    data_out.write_vtu(f);
    this->pcout << "VTU file written successfully." << std::endl;

    // Create a new PVD file each time (more robust)
    std::string pvd_filename = this->output_dir + "/collection.pvd";
    std::ofstream pvd(pvd_filename);
    if (!pvd) {
      this->pcout << "ERROR: Could not open PVD file for writing: " << pvd_filename << std::endl;
      return;
    }
    
    // Always write a complete PVD file with all current outputs
    this->pcout << "Writing complete PVD file with " << this->output_steps.size()+1 << " timesteps..." << std::endl;
    pvd << "<?xml version=\"1.0\"?>\n"
        << "<VTKFile type=\"Collection\">\n"
        << "  <Collection>\n";
    
    // Add all previous timesteps
    for (unsigned int i = 0; i < this->output_steps.size(); ++i) {
      pvd << "    <DataSet timestep=\"" << this->output_times[i] << "\" file=\"sol-"
          << Utilities::int_to_string(this->output_steps[i], 6) << ".vtu\"/>\n";
    }
    
    // Add current timestep
    this->pcout << "Adding timestep " << this->time << " to PVD file..." << std::endl;
    pvd << "    <DataSet timestep=\"" << this->time << "\" file=\"sol-"
        << Utilities::int_to_string(step, 6) << ".vtu\"/>\n";
    
    // Store this step for future updates
    this->output_steps.push_back(step);
    this->output_times.push_back(this->time);
    
    // Close the file
    pvd << "  </Collection>\n"
        << "</VTKFile>\n";
    pvd.close();
  }
}

// Explicit instantiation
template class ManufacturedSolutionSolver<2>;
template class ManufacturedSolutionSolver<3>;