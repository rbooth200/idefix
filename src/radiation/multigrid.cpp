// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

// Module contributed by Richard Booth, then at University of Leeds
#include <algorithm>
#include <cctype>
#include <memory>
#include <string>
#include <utility>

#include "physics.hpp"
#include "boundary_parser.hpp"
#include "fluid.hpp"
#include "dataBlock.hpp"
#include "multigrid_mode.hpp"
#include "multigrid.hpp"

#include "bicgstab.hpp"
#include "cg.hpp"
#include "minres.hpp"

namespace {
BoundaryType MapBoundaryType(RadiationBoundaryType boundary) {
  if (boundary == RadiationBoundaryType::periodic) {
    return BoundaryType::periodic;
  }
  if (boundary == RadiationBoundaryType::axis) {
    return BoundaryType::axis;
  }

  // Dirichlet/neumann/userdef are physical constraints handled by the matrix terms,
  // not by Grid topology. Use undefined so coarse-grid geometry coarsening treats
  // them as non-periodic/non-axis boundaries.
  return BoundaryType::undefined;
}
}  // namespace

MG_Grid::MG_Grid(DataBlock* data, std::array<bool,3> coarsen_dir)
  : MG_Grid(data, coarsen_dir, data->mygrid->lbound, data->mygrid->rbound) {}

MG_Grid::MG_Grid(DataBlock* data, std::array<bool,3> coarsen_dir,
                 std::array<BoundaryType,3> lbound, std::array<BoundaryType,3> rbound)
  : coarsened(coarsen_dir) {
  idfx::RegionWrapper region("MG_Grid::MG_Grid(DataBlock)");

  // Make the coarsened grid
  CoarseGrid cg(data->mygrid, coarsened, lbound, rbound);
  this->data = std::make_unique<DataBlock>(&cg);
  std::swap(this->_grid, cg.grid); // Transfer ownership of the grid to this MG_Grid instance

  beg = this->data->beg;
  end = this->data->end;
  np_tot = this->data->np_tot;
  nghost = this->data->mygrid->nghost;

  Init();
}

void MG_Grid::Init() {
  idfx::RegionWrapper region("MG_Grid::Init");

  x.resize(3);
  for (int d=0; d < 3; d++) {
    x[d] = data->x[d];
  }

  rhs = IdefixArray3D<real>("rhs", np_tot[KDIR], np_tot[JDIR], np_tot[IDIR]);
  solution = IdefixArray3D<real>("solution", np_tot[KDIR], np_tot[JDIR], np_tot[IDIR]);
  workspace = IdefixArray3D<real>("workspace", np_tot[KDIR], np_tot[JDIR], np_tot[IDIR]);
  workspace2 = IdefixArray3D<real>("workspace2", np_tot[KDIR], np_tot[JDIR], np_tot[IDIR]);

  lbound = data->lbound;
  rbound = data->rbound;

  std::array<bool,3> isPeriodic;
  for (int d=0; d < 3; d++) {
    isPeriodic[d] = (lbound[d] == BoundaryType::periodic) && (rbound[d] == BoundaryType::periodic);
  }
  matrix = StencilMatrix(data.get(), isPeriodic);

#ifdef WITH_MPI
  mpi.Init(data.get(), {0}, nghost.data(), data->np_int.data());
#endif
}


MultiGrid::MultiGrid(Input& input, DataBlock* datain, int verbosein) {
  idfx::RegionWrapper region("MultiGrid::MultiGrid");

#if DEBUG_MULTIGRID
  verbosein=2;
#endif
  verbose = verbosein;
  data = datain;

  if (input.CheckEntry("Radiation", "multigrid") <= 0) {
    IDEFIX_ERROR("MultiGrid:: No multigrid levels specified in input file");
  }

  MultigridMode mode = ParseMultigridModeToken(
      input.GetOrSet<std::string>("Radiation", "multigrid", 0, "V"), false, "MultiGrid");
  if (mode == MultigridMode::V) {
    cycle_type = CycleType::V;
  } else {
    cycle_type = CycleType::F;
  }

  this->num_levels = input.GetOrSet<int>("Radiation", "multigrid", 1, 1);
  this->num_smooth = input.GetOrSet<int>("Radiation", "multigrid", 2, 2);
  this->num_repeat = input.GetOrSet<int>("Radiation", "multigrid", 3, 2);
  real inner_tol = input.GetOrSet<real>("Radiation", "multigrid", 4, 1e-6);
  int max_iter = input.GetOrSet<int>("Radiation", "multigrid", 5, 100);


  // Sort out the boundary conditions for the multigrid solver
  // We can ignore internal boundaries, as these are handled by MPI

  for (int dir = 0; dir < DIMENSIONS; dir++) {
    std::string label = std::string("boundary-X") + std::to_string(dir + 1) + std::string("-beg");
    lbound[dir] = ParseRadiationPhysicalBoundaryTypeToken(
        input.Get<std::string>("Radiation", label, 0), false, "MultiGrid");

    label = std::string("boundary-X") + std::to_string(dir + 1) + std::string("-end");
    rbound[dir] = ParseRadiationPhysicalBoundaryTypeToken(
        input.Get<std::string>("Radiation", label, 0), false, "MultiGrid");
  }

  ValidateRadiationAxisBoundaries(lbound, rbound, data->mygrid->nproc[KDIR], "MultiGrid");

  std::array<BoundaryType,3> grid_lbound, grid_rbound;
  for (int dir = 0; dir < DIMENSIONS; dir++) {
    grid_lbound[dir] = MapBoundaryType(lbound[dir]);
    grid_rbound[dir] = MapBoundaryType(rbound[dir]);
  }


  // Create the base grid.
  grids.reserve(num_levels);
  grids.emplace_back(datain, std::array<bool,3>{false, false, false}, grid_lbound, grid_rbound);

  // Create the coarse grid hierarchy
  for (int level = 1; level < num_levels; ++level) {
    // Coarsen direction if possible
    std::array<bool,3> coarsen_dir;
    bool coarsen_possible = false;
    for (int d=0; d < 3; d++) {
      int np_int = grids[level-1].data->np_int[d];
      coarsen_dir[d] = (np_int >= 4) && (np_int % 2 == 0);
      coarsen_possible = coarsen_possible || coarsen_dir[d];
    }
    if (!coarsen_possible) {
      num_levels = level;
      break;
    }

    if (verbose > 1) {
      idfx::cout << "MultiGrid:: Creating level " << level << " with coarsening: "
                 << coarsen_dir[0] << " " << coarsen_dir[1] << " " << coarsen_dir[2] << "\n";
    }

    grids.emplace_back(grids[level - 1].data.get(), coarsen_dir);

    CreateInterpolationWeights(level-1);
  }

  // Setup the coarse grid solver:
  auto& gc = grids[num_levels-1];
  gc.matrix.enablePreconditioner();

  coarse_solver = std::make_unique<Bicgstab<StencilMatrix>>(
    gc.matrix, inner_tol, max_iter, gc.data->np_tot, gc.data->beg, gc.data->end
  );
}

void MultiGrid::FinishSetup() {
  idfx::RegionWrapper region("MultiGrid::FinishSetup");

  auto& gc = grids[num_levels-1];

  gc.matrix.Reset();
  gc.matrix.ScaleMatrix();
}


void MultiGrid::Solve(IdefixArray3D<real> x, IdefixArray3D<real> b) {
  idfx::RegionWrapper region("MultiGrid::Solve");

  // Scale the input array if needed
  IdefixArray3D<real> b_scaled = grids[0].rhs;
  if (scale_system==ScalingType::LEFT || scale_system==ScalingType::SYMMETRIC) {
    idefix_for("ScaleRHS", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, data->np_tot[IDIR],
      KOKKOS_LAMBDA(int k, int j, int i) {
        b_scaled(k,j,i) = b(k,j,i) * scale(k,j,i);
      });
  } else {
    b_scaled = b;
  }

  // Start with the unpreconditioned value as the solution
  InitLevel(x, b_scaled, 0);

  if (verbose > 0) {
    ComputeResidual(x, b_scaled, grids[0].workspace, 0);
    PrintL2Norm(grids[0].workspace, b_scaled, 0);
  }

  for(int repeat = 0; repeat < num_repeat; repeat++) {
    Cycle(x, b_scaled, 0, cycle_type == CycleType::F);

    if (verbose > 0 && repeat < num_repeat - 1)
      idfx::cout << "\n";
  }

  // Scale the output if needed.
  if (scale_system==ScalingType::RIGHT || scale_system==ScalingType::SYMMETRIC) {
    idefix_for("UnscaleSolution",
      0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, data->np_tot[IDIR],
      KOKKOS_LAMBDA(int k, int j, int i) {
        x(k,j,i) = x(k,j,i) * scale(k,j,i);
      });
  }


  if (verbose > 0) {
    if (verbose > 1) {
      idfx::cout << "MultiGrid::Solve: Solution after V-cycle\n";
      for (int i=grids[0].beg[IDIR]; i < grids[0].beg[IDIR]+3; i++) {
        idfx::cout << "  " << i << " "
            << x(grids[0].beg[KDIR],grids[0].beg[JDIR],i) << " "
            << b(grids[0].beg[KDIR],grids[0].beg[JDIR],i) << "\n";
      }
    }
    idfx::cout <<"\n\n\n";
  }
}


void MultiGrid::Zero(IdefixArray3D<real> x, int level) {
  idfx::RegionWrapper region("MultiGrid::Zero");

  int nx = grids[level].np_tot[IDIR];
  int ny = grids[level].np_tot[JDIR];
  int nz = grids[level].np_tot[KDIR];

  idefix_for("zero", 0, nz, 0, ny, 0, nx,
    KOKKOS_LAMBDA(int k, int j, int i) {
        x(k,j,i) = 0.0;
  });
}

void MultiGrid::InitLevel(IdefixArray3D<real> x, IdefixArray3D<real> b, int level) {
    idfx::RegionWrapper region("MultiGrid::InitLevel");

  int ibeg = grids[level].beg[IDIR];
  int iend = grids[level].end[IDIR];
  int jbeg = grids[level].beg[JDIR];
  int jend = grids[level].end[JDIR];
  int kbeg = grids[level].beg[KDIR];
  int kend = grids[level].end[KDIR];

  IdefixArray4D<real> M = grids[level].matrix.M;

  idefix_for("Init", kbeg, kend, jbeg, jend, ibeg, iend,
      KOKKOS_LAMBDA(int k, int j, int i) {
            x(k,j,i) = b(k,j,i) / M(0, k, j, i);
      });

  grids[level].matrix.SetBoundaries(x);
}

void MultiGrid::Cycle(IdefixArray3D<real> x, IdefixArray3D<real> b, int level, bool isFcycle) {
  idfx::RegionWrapper region("MultiGrid::FCycle");

  if (level < num_levels - 1) {
    Smooth(x, b, level);

    // Compute the residual
    IdefixArray3D<real> r = grids[level].workspace;
    ComputeResidual(x, b, r, level);

    if (verbose > 0)
      PrintL2Norm(r, b, level);

    // Coarsen the problem
    IdefixArray3D<real> b_coarse = grids[level + 1].rhs;
    Restrict(r, b_coarse, level);

    // Solve the coarse problem
    IdefixArray3D<real> x_coarse = grids[level + 1].solution;
    InitLevel(x_coarse, b_coarse, level + 1);

    // Start the cycle on the next level
    Cycle(x_coarse, b_coarse, level + 1, isFcycle);

    // Prolongate the solution
    Prolongate(x_coarse, x, level, ProlongationType::ADD);

    Smooth(x, b, level);

    if (isFcycle) {
      // If we are doing an F-cycle, we need to go back down to the next level and
      // do a V-cycle.
      Cycle(x, b, level, false);
    }

    if (verbose > 0) {
      auto res = grids[level].workspace;
      ComputeResidual(x, b, res, level);
      PrintL2Norm(res, b, level);
    }

  } else {
    grids[level].matrix.ScaleSolAndRhs(x, b, false);
    int nsteps = coarse_solver->Solve(x, b);
    grids[level].matrix.ScaleSolAndRhs(x, b, true);

    grids[level].matrix.SetBoundaries(x);

    if (verbose > 0) {
      idfx::cout << "MultiGrid::Cycle: Coarse grid solver took " << nsteps << " iterations\n"
                 << "  Error Norm: " << coarse_solver->GetError() << "\n";
    }
  }
}



void MultiGrid::ComputeResidual(const IdefixArray3D<real> x, const IdefixArray3D<real> b,
                                IdefixArray3D<real> r, int level) {
  idfx::RegionWrapper region("MultiGrid::ComputeResidual");

  grids[level].matrix.ComputeResidual(x, b, r);

  if (verbose > 1) {
    int ibeg = grids[level].beg[IDIR];

    idfx::cout << "Residual " << level << " "
               << r(grids[level].beg[KDIR],grids[level].beg[JDIR],ibeg-1) << " "
               << r(grids[level].beg[KDIR],grids[level].beg[JDIR],ibeg)   << " "
               << r(grids[level].beg[KDIR],grids[level].beg[JDIR],ibeg+1) << "\n";
  }
}


void MultiGrid::Smooth(IdefixArray3D<real> x, const IdefixArray3D<real> b, int level) {
  idfx::RegionWrapper region("MultiGrid::Smooth");

  int ibeg = grids[level].beg[IDIR];
  int iend = grids[level].end[IDIR];
  int jbeg = grids[level].beg[JDIR];
  int jend = grids[level].end[JDIR];
  int kbeg = grids[level].beg[KDIR];
  int kend = grids[level].end[KDIR];

  IdefixArray4D<real> M = grids[level].matrix.M;
  double weight = 2.0/3.0;

  IdefixArray3D<real> x_ptr = x;
  IdefixArray3D<real> work = grids[level].workspace;

  // Make sure we have an even number of smoothing iterations so
  // that we end up with the smoothed solution in the correct array
  for(int iter = 0; iter < this->num_smooth; ++iter) {
    IdefixArray3D<real> x_old = x_ptr;
    IdefixArray3D<real> x_new = work;
    idefix_for("smooth", kbeg, kend, jbeg, jend, ibeg, iend,
        KOKKOS_LAMBDA(int k, int j, int i) {
            // Use a weighted Jacobi smoothing
            real x = b(k,j,i)
                D_EXPAND(
                    - M(1, k, j, i)* x_old(k, j, i - 1) - M(2, k, j, i) * x_old(k, j, i + 1),
                    - M(3, k, j, i)* x_old(k, j - 1, i) - M(4, k, j, i) * x_old(k, j + 1, i),
                    - M(5, k, j, i)* x_old(k - 1, j, i) - M(6, k, j, i) * x_old(k + 1, j, i));

            x_new(k,j,i) = (1 - weight) * x_old(k,j,i) + weight * x / M(0, k, j, i);
        });
    // Fill the boundaries
    grids[level].matrix.SetBoundaries(x_new);

    std::swap(x_ptr, work);
  }
  // Copy final result back to x if we have an odd number of smoothing iterations
  if (num_smooth % 2 == 1)
    Kokkos::deep_copy(x, x_ptr);

  if (verbose > 1) {
    idfx::cout << "Smooth " << level << " "
               << x_ptr(grids[level].beg[KDIR],grids[level].beg[JDIR],ibeg-1) << " "
               << x_ptr(grids[level].beg[KDIR],grids[level].beg[JDIR],ibeg)   << " "
               << x_ptr(grids[level].beg[KDIR],grids[level].beg[JDIR],ibeg+1) << "\n";
  }
}


void MultiGrid::CreateInterpolationWeights(int level) {
  idfx::RegionWrapper region("MultiGrid::CreateInterpolationWeights");

  grids[level].interpolation_weights.resize(DIMENSIONS);
  for (int d = 0; d < DIMENSIONS; d++) {
    if (!grids[level+1].coarsened[d]) {
      continue;
    }

    grids[level].interpolation_weights[d] = IdefixArray1D<real>(
      "interpolation_weights", grids[level].np_tot[d]
    );

    IdefixArray1D<real> weights = grids[level].interpolation_weights[d];

    IdefixArray1D<real> x_fine = grids[level].x[d];
    IdefixArray1D<real> x_coarse = grids[level + 1].x[d];

    // Need to fill the boundary cells to make restriction and prolongation work correctly.
    // For periodic boundaries, we need to fill the ghost cells with the correct weights.
    // For non-periodic boundaries, we need to fill the ghost cells with 0 or 1 depending
    // on whether they are on the left or right boundary to make sure they are not used in the sum.
    int nx = grids[level].np_tot[d];
    int ngh = grids[level].nghost[d];
    int need_left = (grids[level].lbound[d] == BoundaryType::periodic ||
                     grids[level].lbound[d] == BoundaryType::internal);
    int need_right = (grids[level].rbound[d] == BoundaryType::periodic ||
                      grids[level].rbound[d] == BoundaryType::internal);
    idefix_for("interpolation_weights", 0, nx,
      KOKKOS_LAMBDA(int i) {
        if (i <= ngh && !need_left) {
          weights(i) = (i == ngh);
        } else if (i >= nx - ngh - 1 && !need_right) {
          weights(i) = (i != (nx - ngh - 1));
        } else {
          int i_c = (i+ngh-1) / 2;
          real w = (x_fine(i) - x_coarse(i_c)) / (x_coarse(i_c + 1) - x_coarse(i_c));

          weights(i) = Kokkos::min(Kokkos::max(w, 0.0), 1.0);
        }
      });


    // Handle dirichlet boundaries
    if (lbound[d] == RadiationBoundaryType::dirichlet &&
        grids[level+1].lbound[d] != BoundaryType::internal) {
      IdefixArray1D<real> xl = grids[level].data->xl[d];

      idefix_for("interpolation_weights_dirichlet_left", ngh, ngh+1,
        KOKKOS_LAMBDA(int i) {
          int i_c = (i+ngh-1) / 2;
          weights(i) = (x_fine(i) - xl(i)) / (x_coarse(i_c + 1) - xl(i));
        });
    }

    if (rbound[d] == RadiationBoundaryType::dirichlet &&
        grids[level+1].rbound[d] != BoundaryType::internal) {
      IdefixArray1D<real> xr = grids[level].data->xr[d];

      idefix_for("interpolation_weights_dirichlet_right", nx-ngh-1, nx-ngh,
        KOKKOS_LAMBDA(int i) {
          int i_c = (i+ngh-1) / 2;
          weights(i) = (x_fine(i) - x_coarse(i_c)) / (xr(i) - x_coarse(i_c));
        });
    }
  }


  if (verbose > 2) {
    for (int d = 0; d < DIMENSIONS; d++) {
      if (!grids[level+1].coarsened[d]) {
        continue;
      }
      idfx::cout << "Interpolation weights for level " << level << " direction " << d << "\n";
      for (int i = 0; i < grids[level].np_tot[d]; i++) {
        idfx::cout << "  " << i << " " << grids[level].interpolation_weights[d](i) << "\n";
      }
    }
  }
}

void MultiGrid::Prolongate(const IdefixArray3D<real> q_c, IdefixArray3D<real> q_f,
                           int level, ProlongationType type) {
  idfx::RegionWrapper region("MultiGrid::Prolongate");

  int ibeg = 0;
  int iend = grids[level+1].np_tot[IDIR];
  int jbeg = 0;
  int jend = grids[level+1].np_tot[JDIR];
  int kbeg = 0;
  int kend = grids[level+1].np_tot[KDIR];

  IdefixArray3D<real> q_tmp = grids[level].workspace;
  IdefixArray3D<real> q_tmp2 = grids[level].workspace2;

  if (grids[level+1].coarsened[IDIR]) {
    IdefixArray1D<real> w = grids[level].interpolation_weights[0];

    ibeg = grids[level].beg[IDIR];
    iend = grids[level].end[IDIR];

    int ngh = grids[level].nghost[IDIR];
    idefix_for("prolongate_x", kbeg, kend, jbeg, jend, ibeg, iend,
      KOKKOS_LAMBDA(int k, int j, int i) {
        int i_c = (i+ngh-1) / 2;
        q_tmp(k, j, i) = (1 - w(i)) * q_c(k,j,i_c) + w(i) * q_c(k,j,i_c+1);
      });

    ibeg = 0;
    iend = grids[level].np_tot[IDIR];
  } else {
    idefix_for("prolongate_copy", kbeg, kend, jbeg, jend, ibeg, iend,
      KOKKOS_LAMBDA(int k, int j, int i) {
        q_tmp(k,j,i) = q_c(k,j,i);
      });
  }

  if (DIMENSIONS > 1) {
    if (grids[level+1].coarsened[JDIR]) {
      IdefixArray1D<real> w = grids[level].interpolation_weights[1];

      jbeg = grids[level].beg[JDIR];
      jend = grids[level].end[JDIR];

      int ngh = grids[level].nghost[JDIR];
      idefix_for("prolongate_y", kbeg, kend, jbeg, jend, ibeg, iend,
        KOKKOS_LAMBDA(int k, int j, int i) {
          int j_c = (j+ngh-1) / 2;
          q_tmp2(k, j, i) = (1 - w(j)) * q_tmp(k,j_c,i) + w(j) * q_tmp(k,j_c+1,i);
        });

      std::swap(q_tmp, q_tmp2);
      jbeg = 0;
      jend = grids[level].np_tot[JDIR];
    }
  }

  if (DIMENSIONS > 2) {
    if (grids[level+1].coarsened[KDIR]) {
      IdefixArray1D<real> w = grids[level].interpolation_weights[2];

      kbeg = grids[level].beg[KDIR];
      kend = grids[level].end[KDIR];

      int ngh = grids[level].nghost[KDIR];
      idefix_for("prolongate_z", kbeg, kend, jbeg, jend, ibeg, iend,
        KOKKOS_LAMBDA(int k, int j, int i) {
          int k_c = (k+ngh-1) / 2;
          q_tmp2(k, j, i) = (1 - w(k)) * q_tmp(k_c,j,i) + w(k) * q_tmp(k_c+1,j,i);
        });

      std::swap(q_tmp, q_tmp2);
    }
  }

  // Combine the result into the final solution

  int sign = (type == ProlongationType::SUBTRACT) ? -1 : 1;
  int keep = (type == ProlongationType::REPLACE) ? 0 : 1;

  ibeg = grids[level].beg[IDIR];
  iend = grids[level].end[IDIR];
  jbeg = grids[level].beg[JDIR];
  jend = grids[level].end[JDIR];
  kbeg = grids[level].beg[KDIR];
  kend = grids[level].end[KDIR];

  idefix_for("prolongate_combine", kbeg, kend, jbeg, jend, ibeg, iend,
    KOKKOS_LAMBDA(int k, int j, int i) {
      q_f(k,j,i) = keep * q_f(k,j,i) + sign * q_tmp(k,j,i);
    });

  grids[level].matrix.SetBoundaries(q_f);

  if (verbose > 1) {
    idfx::cout << "Prolongate: " << level << " "
               << q_f(kbeg,jbeg,ibeg-1) << " "
               << q_f(kbeg,jbeg,ibeg) << " "
               << q_f(kbeg,jbeg,ibeg+1) << "\n";
  }
}

void MultiGrid::Restrict(IdefixArray3D<real> q_fine, IdefixArray3D<real> q_crs, int level) {
  idfx::RegionWrapper region("MultiGrid::Restrict");

  int ibeg = 0;
  int iend = grids[level].np_tot[IDIR];
  int jbeg = 0;
  int jend = grids[level].np_tot[JDIR];
  int kbeg = grids[level+1].beg[KDIR];
  int kend = grids[level+1].end[KDIR];

  IdefixArray3D<real> q_tmp = grids[level].workspace;
  IdefixArray3D<real> q_tmp2 = grids[level].workspace2;

  if (DIMENSIONS > 2) {
      if (grids[level+1].coarsened[KDIR]) {
      IdefixArray1D<real> w = grids[level].interpolation_weights[2];

      // If the grid is not coarsened in the x & y directions, we can fill the coarse grid directly
      if ((grids[level+1].coarsened[JDIR] == false) && (grids[level+1].coarsened[IDIR] == false))
        q_tmp = q_crs;

      int ngh = grids[level].nghost[KDIR];
      idefix_for("restrict_z", kbeg, kend, jbeg, jend, ibeg, iend,
        KOKKOS_LAMBDA(int k, int j, int i) {
          int k_f = 2*(k-ngh) + ngh;

          q_tmp(k,j,i) = w(k_f-1) * q_fine(k_f-1,j,i) + w(k_f) * q_fine(k_f,j,i)
           + (1- w(k_f+1)) * q_fine(k_f+1,j,i) + (1 - w(k_f+2)) * q_fine(k_f+2,j,i);
        });
    }
  } else {
    Kokkos::deep_copy(q_tmp, q_fine);
  }

  jbeg = grids[level+1].beg[JDIR];
  jend = grids[level+1].end[JDIR];

  if (DIMENSIONS > 1) {
    if (grids[level+1].coarsened[JDIR]) {
      IdefixArray1D<real> w = grids[level].interpolation_weights[1];

      // If the grid is not coarsened in the x direction, we can fill the coarse grid directly
      if (grids[level+1].coarsened[IDIR] == false)
        q_tmp2 = q_crs;

      int ngh = grids[level].nghost[JDIR];
      idefix_for("restrict_y", kbeg, kend, jbeg, jend, ibeg, iend,
        KOKKOS_LAMBDA(int k, int j, int i) {
          int j_f = 2*(j-ngh) + ngh;

          q_tmp2(k,j,i) = w(j_f-1) * q_tmp(k,j_f-1,i) + w(j_f) * q_tmp(k,j_f,i)
           + (1- w(j_f+1)) * q_tmp(k,j_f+1,i) + (1 - w(j_f+2)) * q_tmp(k,j_f+2,i);
        });

      if (grids[level+1].coarsened[IDIR])
        std::swap(q_tmp, q_tmp2);
    }
  }

  ibeg = grids[level+1].beg[IDIR];
  iend = grids[level+1].end[IDIR];

  if (grids[level+1].coarsened[IDIR]) {
    IdefixArray1D<real> w = grids[level].interpolation_weights[0];

      int ngh = grids[level].nghost[IDIR];
      idefix_for("restrict_x", kbeg, kend, jbeg, jend, ibeg, iend,
        KOKKOS_LAMBDA(int k, int j, int i) {
          int i_f = 2*(i-ngh) + ngh;

          q_crs(k,j,i) = w(i_f-1) * q_tmp(k,j,i_f-1) + w(i_f) * q_tmp(k,j,i_f)
           + (1- w(i_f+1)) * q_tmp(k,j,i_f+1) + (1 - w(i_f+2)) * q_tmp(k,j,i_f+2);
        });
    }

  if (verbose > 1) {
    idfx::cout << "Restrict: " << level << " "
               << q_crs(kbeg,jbeg,ibeg-1) << " "
               << q_crs(kbeg,jbeg,ibeg) << " "
               << q_crs(kbeg,jbeg,ibeg+1) << "\n";
  }
}

void MultiGrid::PrintL2Norm(IdefixArray3D<real> res, IdefixArray3D<real> rhs, int level) {
  idfx::RegionWrapper region("MultiGrid::PrintL2Norm");

  // Loading needed attributes
  int ibeg = this->grids[level].beg[IDIR];
  int iend = this->grids[level].end[IDIR];
  int jbeg = this->grids[level].beg[JDIR];
  int jend = this->grids[level].end[JDIR];
  int kbeg = this->grids[level].beg[KDIR];
  int kend = this->grids[level].end[KDIR];

  // Do the reduction on a vector
  MyVector normL2Vector;

  // Sum of squared residuals over the grid and sum of squared rhs
  // both stored in a 2D reduction vector
  idefix_reduce("SumRes2", kbeg, kend, jbeg, jend, ibeg, iend,
                KOKKOS_LAMBDA (int k, int j, int i, MyVector &localVector) {
                  localVector.v[0] += res(k,j,i) * res(k,j,i);
                  localVector.v[1] += rhs(k,j,i) * rhs(k,j,i);
                },
                Kokkos::Sum<MyVector>(normL2Vector));

  // Reduction on the whole grid
  #ifdef WITH_MPI
  MPI_Allreduce(MPI_IN_PLACE, &normL2Vector.v, 2, realMPI, MPI_SUM, MPI_COMM_WORLD);
  #endif

  std::stringstream s_res, s_rhs, s_err;
  (normL2Vector.v[0] == 0) ? (s_res << "ZERO") : s_res << sqrt(normL2Vector.v[0]);
  (normL2Vector.v[1] == 0) ? (s_rhs << "ZERO") : s_rhs << sqrt(normL2Vector.v[1]);
  s_err << sqrt(normL2Vector.v[0]) / sqrt(normL2Vector.v[1]);

  idfx::cout << "MultiGrid::level=" << level << ", norm(res), norm(rhs), L2 = "
          << s_res.str() << " " << s_rhs.str()
          << " " << s_err.str()
          << std::endl;
}


void MultiGrid::ShowConfig() {
  idfx::cout << "MultiGrid: " << num_levels << " levels, " << num_smooth
            << " smoothing iterations, " << num_repeat << " "
            << ((cycle_type == CycleType::F) ? "F-cycles" : "V-cycles") << std::endl;
}
