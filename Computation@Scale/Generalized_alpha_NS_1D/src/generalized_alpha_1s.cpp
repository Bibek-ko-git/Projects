#include "generalized_alpha_1d.h"

template <int dim>
void GeneralizedAlpha1D<dim>::setup_system()
{
    GridGenerator::hyper_cube(triangulation, 0, 1);
    triangulation.refine_global(5);
    dof_handler.distribute_dofs(fe);
    
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    
    // Set up constraints
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    
    // Apply Dirichlet boundary conditions using the manufactured solution
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    boundary_functions[0] = &exact_solution; // Left boundary
    boundary_functions[1] = &exact_solution; // Right boundary
    
    // Apply to velocity components only
    ComponentMask vel_mask(dim + 1, false);
    for (unsigned int i = 0; i < dim; ++i)
        vel_mask.set(i, true);
    
    VectorTools::interpolate_boundary_values(dof_handler,
                                          boundary_functions,
                                          constraints,
                                          vel_mask);
    
    constraints.close();
    
    // Initialize matrix with sparsity pattern
    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                              locally_owned_dofs,
                                              mpi_communicator,
                                              locally_relevant_dofs);
    
    system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
    
    // Initialize vectors
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    solution_old.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    solution_old2.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);
}

template <int dim>
void GeneralizedAlpha1D<dim>::initialize_solution()
{
    // Initialize with manufactured solution at t=0
    exact_solution.set_time(time);
    VectorTools::interpolate(dof_handler, exact_solution, solution);
    solution_old = solution;
    solution_old2 = solution;
}

template <int dim>
void GeneralizedAlpha1D<dim>::set_time(double new_time)
{
    time = new_time;
    exact_solution.set_time(time);
}

template <int dim>
double GeneralizedAlpha1D<dim>::get_velocity(const unsigned int component,
                                           const std::vector<double> &U) const
{
    Assert(component < dim, ExcIndexRange(component, 0, dim));
    return U[component];
}

template <int dim>
double GeneralizedAlpha1D<dim>::get_pressure(const std::vector<double> &U) const
{
    return U[dim];
}

template <int dim>
double GeneralizedAlpha1D<dim>::get_acceleration(const std::vector<double> &U,
                                               const std::vector<double> &U_old,
                                               const std::vector<double> &U_old2) const
{
    // Compute acceleration using BDF-2 formula
    const double a = (U[0] - 2.0 * U_old[0] + U_old2[0]) / (time_step * time_step);
    return a;
}

template <int dim>
double GeneralizedAlpha1D<dim>::compute_integral() const
{
    const QGauss<dim> quadrature_formula(fe.degree + 2);
    
    FEValues<dim> fe_values(fe,
                           quadrature_formula,
                           update_values | update_JxW_values);
    
    const unsigned int n_q_points = quadrature_formula.size();
    
    std::vector<Vector<double>> solution_values(n_q_points, Vector<double>(dim + 1));
    
    double integral = 0.0;
    
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;
            
        fe_values.reinit(cell);
        fe_values.get_function_values(solution, solution_values);
        
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            // Integrate just the velocity component
            integral += solution_values[q][0] * fe_values.JxW(q);
        }
    }
    
    return Utilities::MPI::sum(integral, mpi_communicator);
}

template <int dim>
void GeneralizedAlpha1D<dim>::assemble_system()
{
    system_matrix = 0;
    system_rhs = 0;
    
    const QGauss<dim> quadrature_formula(fe.degree + 2);
    
    FEValues<dim> fe_values(fe,
                           quadrature_formula,
                           update_values | update_gradients |
                           update_quadrature_points | update_JxW_values);
    
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();
    
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    
    // Allocate arrays for values at quadrature points
    std::vector<double> u_n(n_q_points);
    std::vector<double> u_n_old(n_q_points);
    std::vector<double> u_n_old2(n_q_points);
    std::vector<double> p_n(n_q_points);
    std::vector<Tensor<1, dim>> grad_u_n(n_q_points);
    std::vector<Tensor<1, dim>> grad_p_n(n_q_points);
    
    std::vector<std::vector<double>> phi_u(dofs_per_cell, std::vector<double>(n_q_points));
    std::vector<std::vector<Tensor<1, dim>>> grad_phi_u(dofs_per_cell, std::vector<Tensor<1, dim>>(n_q_points));
    std::vector<std::vector<double>> phi_p(dofs_per_cell, std::vector<double>(n_q_points));
    std::vector<std::vector<Tensor<1, dim>>> grad_phi_p(dofs_per_cell, std::vector<Tensor<1, dim>>(n_q_points));
    
    // Time integration parameters for generalized-alpha
    const double alpha_f_prime = 1.0 - alpha_f;
    const double alpha_m_prime = 1.0 - alpha_m;
    const double dt = time_step;
    const double beta = 0.25 * std::pow(gamma + 0.5, 2); // Recommended value
    
    // Get current solution vectors
    const auto local_solution = solution.locally_owned_elements();
    const auto local_solution_old = solution_old.locally_owned_elements();
    const auto local_solution_old2 = solution_old2.locally_owned_elements();
    
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;
            
        cell_matrix = 0;
        cell_rhs = 0;
        
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        
        // Extract current solution values at quadrature points
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            u_n[q] = 0.0;
            u_n_old[q] = 0.0;
            u_n_old2[q] = 0.0;
            p_n[q] = 0.0;
            grad_u_n[q] = 0;
            grad_p_n[q] = 0;
            
            // Precompute basis functions at quadrature points
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                // Get velocity component (first component, index 0)
                if (fe.system_to_component_index(i).first == 0)
                {
                    phi_u[i][q] = fe_values.shape_value(i, q);
                    grad_phi_u[i][q] = fe_values.shape_grad(i, q);
                    
                    // Extract solution values
                    const auto idx = local_dof_indices[i];
                    if (local_solution.is_element(idx))
                    {
                        u_n[q] += solution(idx) * phi_u[i][q];
                        u_n_old[q] += solution_old(idx) * phi_u[i][q];
                        u_n_old2[q] += solution_old2(idx) * phi_u[i][q];
                        grad_u_n[q] += solution(idx) * grad_phi_u[i][q];
                    }
                }
                // Get pressure component (second component, index 1)
                else if (fe.system_to_component_index(i).first == 1)
                {
                    phi_p[i][q] = fe_values.shape_value(i, q);
                    grad_phi_p[i][q] = fe_values.shape_grad(i, q);
                    
                    // Extract pressure values
                    const auto idx = local_dof_indices[i];
                    if (local_solution.is_element(idx))
                    {
                        p_n[q] += solution(idx) * phi_p[i][q];
                        grad_p_n[q] += solution(idx) * grad_phi_p[i][q];
                    }
                }
            }
        }
        
        // Element length for stabilization
        const double h_e = std::sqrt(cell->measure());
        
        // Iterate over quadrature points
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            // Calculate acceleration at quadrature point (BDF-2 formula)
            const double acc_n = (u_n[q] - 2.0 * u_n_old[q] + u_n_old2[q]) / (dt * dt);
            
            // Velocity at intermediate time step t_{n+alpha_f}
            const double u_alpha_f = alpha_f * u_n[q] + alpha_f_prime * u_n_old[q];
            
            // Acceleration at intermediate time step t_{n+alpha_m}
            const double acc_alpha_m = alpha_m * acc_n + alpha_m_prime * 
                                     (u_n_old[q] - 2.0 * u_n_old2[q] + u_n_old2[q]) / (dt * dt);
            
            // Convective term at intermediate time step
            const double convective_term = convection_coefficient * u_alpha_f * grad_u_n[q][0];
            
            // SUPG stabilization parameter
            const double tau_supg = use_supg ? h_e / (2.0 * std::abs(u_alpha_f) + 1e-10) : 0.0;
            
            // Galerkin formulation with time integration
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const unsigned int component_i = fe.system_to_component_index(i).first;
                
                // Add contribution to RHS vector
                if (component_i == 0) // velocity component
                {
                    // Standard Galerkin terms for momentum equation
                    cell_rhs(i) += (-acc_alpha_m * phi_u[i][q]                // Acceleration term
                                  - convective_term * phi_u[i][q]            // Convection term
                                  + viscosity * grad_u_n[q][0] * grad_phi_u[i][q][0] // Diffusion term
                                  - p_n[q] * grad_phi_u[i][q][0])            // Pressure gradient term
                                  * fe_values.JxW(q);
                                  
                    // SUPG stabilization - add to RHS
                    if (use_supg) {
                        cell_rhs(i) += tau_supg * 
                                     (acc_alpha_m + convective_term + grad_p_n[q][0]) * 
                                     (u_alpha_f * grad_phi_u[i][q][0]) * 
                                     fe_values.JxW(q);
                    }
                }
                else if (component_i == 1) // pressure component
                {
                    // Continuity equation (incompressibility)
                    cell_rhs(i) += grad_u_n[q][0] * phi_p[i][q] * fe_values.JxW(q);
                }
                
                // Add contributions to system matrix
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const unsigned int component_j = fe.system_to_component_index(j).first;
                    
                    if (component_i == 0 && component_j == 0) // u-u coupling
                    {
                        // Time derivative term with alpha_m factor
                        cell_matrix(i, j) += alpha_m * gamma / (beta * dt) * phi_u[i][q] * phi_u[j][q] * fe_values.JxW(q);
                        
                        // Convection term with alpha_f factor
                        cell_matrix(i, j) += alpha_f * convection_coefficient * u_alpha_f * grad_phi_u[j][q][0] * phi_u[i][q] * fe_values.JxW(q);
                        
                        // Diffusion term with alpha_f factor
                        cell_matrix(i, j) += alpha_f * viscosity * grad_phi_u[i][q][0] * grad_phi_u[j][q][0] * fe_values.JxW(q);
                        
                        // SUPG stabilization for u-u
                        if (use_supg) {
                            const double supg_test = tau_supg * u_alpha_f * grad_phi_u[i][q][0];
                            cell_matrix(i, j) += alpha_m * gamma / (beta * dt) * supg_test * phi_u[j][q] * fe_values.JxW(q);
                            cell_matrix(i, j) += alpha_f * convection_coefficient * supg_test * u_alpha_f * grad_phi_u[j][q][0] * fe_values.JxW(q);
                            cell_matrix(i, j) += alpha_f * viscosity * supg_test * grad_phi_u[j][q][0] * fe_values.JxW(q);
                        }
                    }
                    else if (component_i == 0 && component_j == 1) // u-p coupling
                    {
                        // Pressure gradient term
                        cell_matrix(i, j) += alpha_f * grad_phi_u[i][q][0] * phi_p[j][q] * fe_values.JxW(q);
                        
                        // SUPG stabilization for u-p
                        if (use_supg) {
                            cell_matrix(i, j) += alpha_f * tau_supg * u_alpha_f * grad_phi_u[i][q][0] * grad_phi_p[j][q][0] * fe_values.JxW(q);
                        }
                    }
                    else if (component_i == 1 && component_j == 0) // p-u coupling (continuity)
                    {
                        cell_matrix(i, j) += grad_phi_u[j][q][0] * phi_p[i][q] * fe_values.JxW(q);
                    }
                    // No p-p coupling in standard formulation
                }
            }
        }
        
        // Add local contributions to global system
        constraints.distribute_local_to_global(cell_matrix, cell_rhs,
                                            local_dof_indices,
                                            system_matrix, system_rhs);
    }
    
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
}

template <int dim>
void GeneralizedAlpha1D<dim>::solve_time_step()
{
    SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
    LA::SolverGMRES solver(solver_control, mpi_communicator);
    
    LA::MPI::PreconditionAMG preconditioner;
    LA::MPI::PreconditionAMG::AdditionalData data;
    
    // AMG parameters tailored for systems with velocity/pressure coupling
    data.elliptic = false;
    data.higher_order_elements = true;
    data.smoother_sweeps = 2;
    data.aggregation_threshold = 0.02;
    
    preconditioner.initialize(system_matrix, data);
    
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
    
    // Apply constraints
    constraints.distribute(solution);
}

template <int dim>
void GeneralizedAlpha1D<dim>::advance_time_step()
{
    time += time_step;
    exact_solution.set_time(time);
    
    assemble_system();
    solve_time_step();
    
    // Update old solutions for next time step
    solution_old2 = solution_old;
    solution_old = solution;
}

template <int dim>
void GeneralizedAlpha1D<dim>::output_results(unsigned int timestep) const
{
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    
    std::vector<std::string> solution_names(dim, "u");
    solution_names.push_back("p");
    
    std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    
    data_out.add_data_vector(solution, solution_names, DataOut<dim>::type_dof_data, interpretation);
    
    // Add exact solution for comparison
    Vector<double> exact(solution.size());
    VectorTools::interpolate(dof_handler, exact_solution, exact);
    
    std::vector<std::string> exact_names(dim, "exact_u");
    exact_names.push_back("exact_p");
    
    data_out.add_data_vector(exact, exact_names, DataOut<dim>::type_dof_data, interpretation);
    
    // Compute and output error
    Vector<double> error(solution.size());
    error = solution;
    error -= exact;
    
    std::vector<std::string> error_names(dim, "error_u");
    error_names.push_back("error_p");
    
    data_out.add_data_vector(error, error_names, DataOut<dim>::type_dof_data, interpretation);
    
    data_out.build_patches();
    
    std::ofstream output("output-" + std::to_string(timestep) + ".vtu");
    data_out.write_vtu(output);
}

template <int dim>
void GeneralizedAlpha1D<dim>::run()
{
    setup_system();
    initialize_solution();
    
    unsigned int timestep = 0;
    output_results(timestep);
    
    while (time < 1.0 - 1e-12)
    {
        advance_time_step();
        ++timestep;
        
        std::cout << "Time step " << timestep 
                 << " at t = " << time
                 << std::endl;
        
        output_results(timestep);
    }
}

// Explicit instantiation
template class GeneralizedAlpha1D<1>;
