// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

// Module contributed by Richard Booth, then at University of Leeds

#ifndef RADIATION_MULTIGRID_HPP_
#define RADIATION_MULTIGRID_HPP_

#include <memory>
#include <vector>

#include "grid.hpp"
#include "idefix.hpp"
#include "input.hpp"
#include "stencil_matrix.hpp"

#ifdef WITH_MPI
#include "mpi.hpp"
#endif

template <class T> class IterativeSolver;
enum class RadiationBoundaryType;

class MG_Grid {
 public:
  MG_Grid() = default;
  MG_Grid(const MG_Grid& grid) = delete;
  MG_Grid(MG_Grid&& grid) = default;
  MG_Grid(DataBlock* data, std::array<bool,3> coarsen_dir);
  MG_Grid(DataBlock* data, std::array<bool,3> coarsen_dir,
          std::array<BoundaryType,3> lbound, std::array<BoundaryType,3> rbound);

  std::array<bool,3> coarsened; // Whether the grid is coarsened in each direction

  std::array<int,3> np_tot;  // Total number of grid points (including ghosts)
  std::array<int,3> nghost;  // Number of ghost cells
  std::array<int,3> beg;     // Beginning of grid (including ghosts)
  std::array<int,3> end;     // End of grid (including ghosts)


  // Grid coordinates
  std::unique_ptr<DataBlock> data;  // DataBlock for this g1rid
  std::vector<IdefixArray1D<real>> x;    ///< geometrical central points

  std::vector<IdefixArray1D<real>> interpolation_weights;
  IdefixArray3D<real> rhs, solution;
  StencilMatrix matrix;
  IdefixArray3D<real> workspace, workspace2;

  std::array<BoundaryType, 3> lbound;  // Boundary condition to the left
  std::array<BoundaryType, 3> rbound;  // Boundary condition to the right

 private:
  void Init();
  std::unique_ptr<Grid> _grid;  // Grid for this level

#ifdef WITH_MPI
  Mpi mpi;        ///< MPI communicatior
#endif
};

class MultiGrid {
 public:
  enum class ProlongationType { ADD, SUBTRACT, REPLACE};
  enum class ScalingType {NONE, LEFT, RIGHT, SYMMETRIC};
  enum class CycleType {V, F};

  MultiGrid() = default;
  MultiGrid(Input& input, DataBlock* datain, int verbose=0);

  void SetMatrix(IdefixArray4D<real> M, int level) {
    grids[level].matrix.Reset();
    Kokkos::deep_copy(grids[level].matrix.M, M);
  }

  void SetScaling(IdefixArray3D<real> scale, ScalingType type=ScalingType::SYMMETRIC) {
    this->scale = scale;
    scale_system = type;
  }

  void FinishSetup(); // Call once all matrices have been set to prepare the multigrid solver

  void Solve(IdefixArray3D<real> x, IdefixArray3D<real> b);

  void Restrict(const IdefixArray3D<real> fine, IdefixArray3D<real> coarse, int level);
  void Prolongate(const IdefixArray3D<real> coarse, IdefixArray3D<real> fine,
                  int level, ProlongationType type);
  void Smooth(IdefixArray3D<real> x, IdefixArray3D<real> b, int level);


  void Cycle(IdefixArray3D<real> x, IdefixArray3D<real> b, int level=0, bool isFcycle=false);

  void Zero(IdefixArray3D<real> x, int level);
  void InitLevel(IdefixArray3D<real> x, IdefixArray3D<real> b, int level);
  void ComputeResidual(IdefixArray3D<real> x, IdefixArray3D<real> b,
                       IdefixArray3D<real> r, int level);
  void CreateInterpolationWeights(int level);

  int NumLevels() const { return num_levels; }
  MG_Grid& GetGrid(int level) { return grids[level]; }

  void ShowConfig();

  void PrintL2Norm(IdefixArray3D<real> x, IdefixArray3D<real> rhs, int level);

 private:
  DataBlock* data;
  int num_levels, num_smooth, num_smooth_coarse, num_repeat;
  std::vector<MG_Grid> grids;
  CycleType cycle_type{CycleType::V};

  std::unique_ptr<IterativeSolver<StencilMatrix>> coarse_solver;

  std::array<RadiationBoundaryType, 3> lbound;  // Boundary condition to the left
  std::array<RadiationBoundaryType, 3> rbound;  // Boundary condition to the right

  ScalingType scale_system{ScalingType::NONE};  // Whether to scale the system for preconditioning
  IdefixArray3D<real> scale;  // Scaling factor for the system, used for preconditioning
  int verbose{0};  // Whether to print information about the multigrid solver
};

#endif  // RADIATION_MULTIGRID_HPP_
