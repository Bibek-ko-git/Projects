#include "ChannelFlowSolver.hpp"
// Solves a 2D Channel (pipe) flow problem using the Navier-Stokes equations
// with a specific velocity profile at the inlet and no-slip conditions at the walls.


// Channel flow implementation
template <int dim>
ChannelFlowSolver<dim>::ChannelFlowSolver(const MPI_Comm &mpi_comm)
  : NavierStokesSolver<dim>(mpi_comm)
{
  // Use specific settings for channel flow
  this->mu = 0.001;
  this->rho = 1.0;
  this->Re = 100.0;
  this->T = 1.0;
}

template <int dim>
void ChannelFlowSolver<dim>::make_grid()
{
  this->pcout << "Creating channel flow mesh..." << std::endl;
  
  const double L = 2.2, H = 0.41;  // Channel dimensions
  
  GridGenerator::hyper_rectangle(this->triangulation,
                               Point<dim>(0,0), Point<dim>(L,H), true);
  
  // Refine the mesh for better resolution
  this->triangulation.refine_global(3);
  
  // Set boundary IDs:
  // 1: Inlet (left)
  // 2: Outlet (right)
  // 3: Wall (top)
  // 4: Wall (bottom)
  for (auto cell : this->triangulation.active_cell_iterators())
    for (unsigned f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
      if (cell->face(f)->at_boundary())
      {
        const auto c = cell->face(f)->center();
        if (std::abs(c[0])<1e-10)        cell->face(f)->set_boundary_id(1);
        else if (std::abs(c[0]-L)<1e-10) cell->face(f)->set_boundary_id(2);
        else if (std::abs(c[1]-H)<1e-10) cell->face(f)->set_boundary_id(3);
        else if (std::abs(c[1])<1e-10)   cell->face(f)->set_boundary_id(4);
      }

  this->pcout << "Active cells: " << this->triangulation.n_active_cells() << std::endl;
}

template <int dim>
void ChannelFlowSolver<dim>::set_boundary_conditions()
{
  std::vector<bool> vel_mask(dim+1, false), pres_mask(dim+1, false);
  for (unsigned d=0; d<dim; ++d) vel_mask[d] = true;
  pres_mask[dim] = true;
  ComponentMask vel_comp(vel_mask), pres_comp(pres_mask);

  // No-slip condition on walls (top and bottom)
  for (auto id : {3u, 4u})
    VectorTools::interpolate_boundary_values(
      this->dof_handler, id,
      Functions::ZeroFunction<dim>(dim+1),
      this->constraints, vel_comp);

  // Inlet velocity profile
  class InletVel : public Function<dim>
  {
  public:
    InletVel():Function<dim>(dim+1){}
    void vector_value(const Point<dim>&p, Vector<double>&v) const override
    {
      static const double H=0.41, UIN=1.0;
      
      // Use a simple profile: zero at walls, constant in middle
      const double y=p[1];
      double prof = 0.0;
      
      if (y > 0.02 && y < H-0.02) {
        prof = UIN;  // Constant velocity away from walls
      } else {
        // Linearly go to zero at walls
        prof = UIN * std::min(y/0.02, (H-y)/0.02);
      }
      
      for (unsigned i=0; i<dim+1; ++i) v[i]=0;
      v[0]=prof;
    }
  } inlet_vel;
  
  VectorTools::interpolate_boundary_values(
    this->dof_handler, 1, inlet_vel, this->constraints, vel_comp);

  // Zero pressure at outlet
  VectorTools::interpolate_boundary_values(
    this->dof_handler, 2,
    Functions::ZeroFunction<dim>(dim+1),
    this->constraints, pres_comp);
}

template <int dim>
void ChannelFlowSolver<dim>::save_checkpoint()
{
  if (this->this_mpi_process != 0)
    return;

  const std::string fn = this->checkpoint_dir + "/chkpt_" +
                        Utilities::int_to_string(this->timestep, 6) + ".bin";
  std::ofstream out(fn, std::ios::binary);
  out.write(reinterpret_cast<const char*>(&this->timestep), sizeof(this->timestep));
  out.write(reinterpret_cast<const char*>(&this->time), sizeof(this->time));

  // Save local vector portion
  std::pair<unsigned int, unsigned int> range = this->solution.local_range();
  const unsigned int n_local = range.second - range.first;
  out.write(reinterpret_cast<const char*>(&n_local), sizeof(n_local));
  std::vector<double> tmp(n_local);
  for (unsigned int i = 0; i < n_local; ++i)
    tmp[i] = this->solution[range.first + i];
  out.write(reinterpret_cast<const char*>(tmp.data()), n_local * sizeof(double));
}

template <int dim>
void ChannelFlowSolver<dim>::load_checkpoint()
{
  // Gather checkpoint files on root
  std::vector<std::string> files;
  if (this->this_mpi_process == 0) {
    DIR* dir = opendir(this->checkpoint_dir.c_str());
    if (dir) {
      struct dirent* e;
      while ((e = readdir(dir)) != nullptr) {
        std::string name = e->d_name;
        if (name.rfind("chkpt_", 0) == 0)
          files.push_back(name);
      }
      closedir(dir);
      std::sort(files.begin(), files.end());
    }
  }

  // Broadcast number of files
  unsigned int count = files.size();
  Utilities::MPI::broadcast(this->mpi_communicator, count, 0);
  if (count == 0)
    return;

  // Root selects latest
  std::string latest;
  if (this->this_mpi_process == 0)
    latest = files.back();
  Utilities::MPI::broadcast(this->mpi_communicator, latest, 0);

  // Read on root
  unsigned int n_local = 0;
  std::vector<double> tmp;
  unsigned int local_offset = 0;
  if (this->this_mpi_process == 0) {
    std::ifstream in(this->checkpoint_dir + "/" + latest, std::ios::binary);
    in.read(reinterpret_cast<char*>(&this->timestep), sizeof(this->timestep));
    in.read(reinterpret_cast<char*>(&this->time), sizeof(this->time));
    in.read(reinterpret_cast<char*>(&n_local), sizeof(n_local));
    tmp.resize(n_local);
    in.read(reinterpret_cast<char*>(tmp.data()), n_local * sizeof(double));
  }

  // Broadcast local size and offset
  Utilities::MPI::broadcast(this->mpi_communicator, n_local, 0);
  auto range = this->solution.local_range();
  local_offset = range.first;
  Utilities::MPI::broadcast(this->mpi_communicator, local_offset, 0);

  // Broadcast data
  if (this->this_mpi_process != 0)
    tmp.resize(n_local);
  Utilities::MPI::broadcast(this->mpi_communicator, tmp, 0);

  // Build a list of the global indices we own:
  std::vector<types::global_dof_index> inds(n_local);
  for (unsigned int i = 0; i < n_local; ++i)
    inds[i] = range.first + i;

  // And set them all at once:
  this->solution.set(n_local,
               inds.data(),     // pointer to global indices
               tmp.data());     // pointer to values
  this->solution.compress(VectorOperation::insert);

  this->stokes_solved = true;
}

// Explicit instantiation
template class ChannelFlowSolver<2>;
template class ChannelFlowSolver<3>;