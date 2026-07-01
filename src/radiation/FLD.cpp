// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

// Module contributed by Alex Ziampras, then at Queen Mary University of London
// Modified by Richard Booth, then at University of Leeds

#include "FLD.hpp"

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>
#include <vector>

#include "bicgstab.hpp"
#include "boundary_utils.hpp"
#include "cg.hpp"
#include "dataBlock.hpp"
#include "fluid.hpp"
#include "idefix.hpp"
#include "minres.hpp"
#include "multigrid_mode.hpp"
#include "stencil_matrix.hpp"
#include "units.hpp"
#include "vector.hpp"

#define SECOND_ORDER_FLUXES

KOKKOS_INLINE_FUNCTION real RadiationFluxLimiterKley1989(real R) {
  if (R <= 2.0)
    return 2.0 / (3.0 + sqrt(9.0 + 10.0 * R * R));
  else
    return 10.0 / (10.0 * R + 9.0 + sqrt(81.0 + 180.0 * R));
}

KOKKOS_INLINE_FUNCTION real RadiationFluxLimiterMinerbo1978(real R) {
  if (R <= 1.5)
    return 2.0 / (3.0 + sqrt(9.0 + 12.0 * R * R));
  else
    return 1.0 / (1.0 + R + sqrt(1.0 + 2.0 * R));
}

KOKKOS_INLINE_FUNCTION real RadiationFluxLimiterLevermorePomraning1981(real R) {
  real one_over_R = 1.0 / (R + 1e-15);
  return one_over_R * (1.0 / tanh(R) - one_over_R);
}

KOKKOS_INLINE_FUNCTION real lageval1(real x1, real x2, real y1, real y2, real x) {
  real dx = x1 - x2;
  real c1 = (x - x2) / dx;
  real c2 = (x1 - x) / dx;
  return c1 * y1 + c2 * y2;
}

KOKKOS_INLINE_FUNCTION real lagdiff1(real x1, real x2, real y1, real y2, real x) {
  // evaluates the first derivative of the lagrange polynomial defined at (x1,y1), (x2,y2)
  // at x to interpolate dy/dx
  real c1 = 1.0 / (x1 - x2);
  real c2 = -c1;
  return c1 * y1 + c2 * y2;
}

FluxLimitedDiffusion::FluxLimitedDiffusion(Input& input, DataBlock* datain) {
  idfx::RegionWrapper region("FLD::FLD");
  this->data = datain;

  Init(input, this->data);
  haveInitialisedRadiation = true;
}

// done
void FluxLimitedDiffusion::Init(Input& input, DataBlock* datain) {
  idfx::RegionWrapper region("FluxLimitedDiffusion::Init");

  // Save the parents data objects
  this->data = datain;

  // Initialize (default) solver parameters
  this->dt = 0.;
  this->mu = input.GetOrSet<real>("Radiation", "mu", 0, 2.353);  // mean molecular weight
  this->gamma = data->hydro->eos->GetGamma();

  // set units
  this->unit_opacity = 1.0 / (idfx::units.GetLength() * idfx::units.GetDensity());
  this->code_c = idfx::units.c / idfx::units.GetVelocity();  // speed of light
  real unit_aR = idfx::units.GetEnergy() / pow(idfx::units.GetKelvin(), 4);
  this->code_aR = idfx::units.ar / unit_aR;           // radiation constant
  this->cV = 1.0 / (this->mu * (this->gamma - 1.0));  // Rgas = 1 in code units

  // Update targetError when provided
  real targetError = input.GetOrSet<real>("Radiation", "targetError", 0, 1e-6);

  // Get maxiter when provided
  real maxiter = input.GetOrSet<int>("Radiation", "maxIter", 0, 1000);

  // configure flux limiter
  if (input.CheckEntry("Radiation", "fluxlimiter") >= 0) {
    std::string fluxlim = input.Get<std::string>("Radiation", "fluxlimiter", 0);
    if (fluxlim.compare("one_third") == 0) {
      this->fluxLimiterType = one_third;
    } else if (fluxlim.compare("Kley1989") == 0) {
      this->fluxLimiterType = Kley1989;
    } else if (fluxlim.compare("Minerbo1978") == 0) {
      this->fluxLimiterType = Minerbo1978;
    } else if (fluxlim.compare("LevermorePomraning1981") == 0) {
      this->fluxLimiterType = LevermorePomraning1981;
    } else {
      std::stringstream msg;
      msg << "FLD:: Unknown flux limiter " << fluxlim;
      IDEFIX_ERROR(msg);
    }
  } else {
    this->fluxLimiterType = Kley1989;
  }

  if (input.CheckBlock("Dust")) {
    num_species = 1 + input.Get<int>("Dust", "nSpecies", 0);
  }

  // configure Rosseland opacity
  this->haveUserDefOpacity = false;
  if (input.CheckEntry("Radiation", "kappaR") >= 0) {
    std::string rosseland = input.Get<std::string>("Radiation", "kappaR", 0);
    if (rosseland.compare("userdef") == 0) {
      this->haveUserDefOpacity = true;
      this->opacityType = userdefkappa;
    } else if (rosseland.compare("constant") == 0) {
      this->constkappaR = IdefixArray1D<real>("kappaR", num_species);

      IdefixHostArray1D<real> kappa_host = Kokkos::create_mirror_view(this->constkappaR);
      for (int s = 0; s < num_species; s++)
        kappa_host(s) = input.GetOrSet<real>("Radiation", "kappaR", 1 + s, 1.0);
      Kokkos::deep_copy(this->constkappaR, kappa_host);

      this->opacityType = constantkappa;
    } else {
      std::stringstream msg;
      msg << "FLD:: Unknown Rosseland opacity " << rosseland;
      IDEFIX_ERROR(msg);
    }
  } else {
    std::stringstream msg;
    msg << "FLD:: Rosseland opacity (kappaR) not specified";
    IDEFIX_ERROR(msg);
  }

  // configure Planck opacity
  bool userdefPlanck = false;
  if (input.CheckEntry("Radiation", "kappaP") >= 0) {
    std::string planck = input.Get<std::string>("Radiation", "kappaP", 0);
    if (planck.compare("userdef") == 0) {
      userdefPlanck = true;
    } else if (planck.compare("constant") == 0) {
      this->constkappaP = IdefixArray1D<real>("kappaP", num_species);

      IdefixHostArray1D<real> kappa_host = Kokkos::create_mirror_view(this->constkappaP);
      for (int s = 0; s < num_species; s++)
        kappa_host(s) = input.GetOrSet<real>("Radiation", "kappaP", 1 + s, 1.0);
      Kokkos::deep_copy(this->constkappaP, kappa_host);

      userdefPlanck = false;
    } else {
      std::stringstream msg;
      msg << "FLD:: Unknown Planck opacity " << planck;
      IDEFIX_ERROR(msg);
    }
  } else {
    std::stringstream msg;
    msg << "FLD:: Planck opacity (kappaP) not specified";
    IDEFIX_ERROR(msg);
  }

  if ((this->haveUserDefOpacity == false && userdefPlanck == true) ||
      (this->haveUserDefOpacity == true && userdefPlanck == false)) {
    std::stringstream msg;
    msg << "FLD:: Error only one of the Planck and Rosseland opacities"
        << "are userdef. Both must be the same";
    IDEFIX_ERROR(msg);
  }

  // Get the radiation-related boundary conditions
  for (int dir = 0; dir < 3; dir++) {
    std::string label = std::string("boundary-X") + std::to_string(dir + 1) + std::string("-beg");
    ParsedRadiationBoundary leftBoundary =
        ParseRadiationPhysicalBoundary(input, label, true, "FLD");
    rad_boundary.lbound[dir] = leftBoundary.type;
    rad_boundary.lvalue[dir] = leftBoundary.value;

    label = std::string("boundary-X") + std::to_string(dir + 1) + std::string("-end");
    ParsedRadiationBoundary rightBoundary =
        ParseRadiationPhysicalBoundary(input, label, true, "FLD");
    rad_boundary.rbound[dir] = rightBoundary.type;
    rad_boundary.rvalue[dir] = rightBoundary.value;
  }

  ValidateRadiationAxisBoundaries(rad_boundary.lbound, rad_boundary.rbound,
                                  data->mygrid->nproc[KDIR], "FLD");

  // Update internal boundaries in case of domain decomposition
#ifdef WITH_MPI
  for (int dir = 0; dir < DIMENSIONS; dir++) {
    if (data->mygrid->nproc[dir] > 1) {
      if (this->data->lbound[dir] == internal) {
        rad_boundary.lbound[dir] = RadiationBoundaryType::internal;
      }
      if (this->data->rbound[dir] == internal) {
        rad_boundary.rbound[dir] = RadiationBoundaryType::internal;
      }
    }
  }
#endif

#if GEOMETRY == SPHERICAL
  if ((rad_boundary.rbound[JDIR] == RadiationBoundaryType::axis) ||
      (rad_boundary.lbound[JDIR] == RadiationBoundaryType::axis)) {
    // Check wether the x3 spherical axis is full two pi
    rad_boundary.isTwoPi = IsFullTwoPiDomain(data);
  }
#endif

// Init MPI stack when needed
#ifdef WITH_MPI
  int ntarget = 0;
  std::vector<int> mapVars;
  mapVars.push_back(ntarget);

  this->mpi.Init(data->mygrid, {0}, this->nghost.data(), data->np_int.data());
#endif

  // Update solver when provided
  if (input.CheckEntry("Radiation", "solver") >= 0) {
    std::string strSolver = input.Get<std::string>("Radiation", "solver", 0);
    if (strSolver.compare("BICGSTAB") == 0) {
      solver = BICGSTAB;
    } else if (strSolver.compare("PBICGSTAB") == 0) {
      solver = PBICGSTAB;
    } else if (strSolver.compare("CG") == 0) {
      solver = CG;
    } else if (strSolver.compare("PCG") == 0) {
      solver = PCG;
    } else if (strSolver.compare("MINRES") == 0) {
      solver = MINRES;
    } else if (strSolver.compare("PMINRES") == 0) {
      solver = PMINRES;
    } else {
      std::stringstream msg;
      msg << "FLD: Unknown solver \"" << strSolver << "\"."
          << "Use \"BICGSTAB\" or \"PBICGSTAB\"." << std::endl;
      IDEFIX_ERROR(msg);
    }
  } else {
    this->solver = BICGSTAB;
  }


  FLD_matrix = std::make_unique<StencilMatrix>(data, rad_boundary.isPeriodic());

  // Enable the diagonal preconditioner
  if (this->solver == PBICGSTAB || this->solver == PCG || this->solver == PMINRES) {
    FLD_matrix->enablePreconditioner();
  }


  // Instantiate the bicgstab solver
  if (solver == BICGSTAB || solver == PBICGSTAB) {
    // Check whether we need a multigrid solver or not
    if (input.CheckEntry("Radiation", "multigrid") >= 0) {
      MultigridMode mg_mode = ParseMultigridModeToken(
          input.Get<std::string>("Radiation", "multigrid", 0), true, "FLD");

      if (mg_mode == MultigridMode::NONE) {
        idfx::cout << "FLD: Multigrid solver disabled by input (N/None).\n";
      } else {
        if (input.Get<int>("Radiation", "multigrid", 1) > 1) {
          haveMultiGrid = true;
          multigrid = std::make_unique<MultiGrid>(input, data);
        } else {
          idfx::cout << "FLD: Multigrid solver disabled because number of levels is <= 1\n";
        }
      }
    }

    if (haveMultiGrid) {
      mgPrecond = MultiGridPreconditioner(multigrid.get());
      iterativeSolver = std::make_unique<Bicgstab<StencilMatrix, MultiGridPreconditioner>>(
          *FLD_matrix.get(), targetError, maxiter, data->np_tot, data->beg, data->end, &mgPrecond
        );
    } else {
      iterativeSolver = std::make_unique<Bicgstab<StencilMatrix>>(
        *FLD_matrix.get(), targetError, maxiter, data->np_tot, data->beg, data->end
      );
    }

  } else if (solver == CG || solver == PCG) {
    iterativeSolver = std::make_unique<Cg<StencilMatrix>>(
      *FLD_matrix.get(), targetError, maxiter, data->np_tot, data->beg, data->end
    );
  } else if (solver == MINRES || solver == PMINRES) {
    iterativeSolver = std::make_unique<Minres<StencilMatrix>>(
      *FLD_matrix.get(), targetError, maxiter, data->np_tot, data->beg, data->end
    );
  } else {
    IDEFIX_ERROR("FLD iterative solver not defined: should never reach here.");
  }

  // Arrays initialization
  this->Erad =
      IdefixArray3D<real>("Erad", data->np_tot[KDIR], data->np_tot[JDIR], data->np_tot[IDIR]);

  this->nu = IdefixArray3D<real>("nu",
                                  data->np_tot[KDIR], data->np_tot[JDIR], data->np_tot[IDIR]);
  this->diag = IdefixArray3D<real>("diag",
                                    data->np_tot[KDIR], data->np_tot[JDIR], data->np_tot[IDIR]);
  this->rhs = IdefixArray3D<real>("rhs",
                                    data->np_tot[KDIR], data->np_tot[JDIR], data->np_tot[IDIR]);


  idefix_for(
      "InitRadiationArrays", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, data->np_tot[IDIR],
      KOKKOS_LAMBDA(int k, int j, int i) {
        Erad(k, j, i) = ZERO_F;
      });

  // Output radiation energy density
  data->dump->RegisterVariable(this->Erad, "ERAD");
  data->vtk->RegisterVariable(this->Erad, "ERAD");
}

// should be done
void FluxLimitedDiffusion::InitSolver() {
  idfx::RegionWrapper region("FluxLimitedDiffusion::InitSolver");

  // Loading needed attributes
  IdefixArray3D<real> Erad = this->Erad;
  this->dt = data->dt;

  // Look for Nans in the input field
  int nanErad = 0;
  idefix_reduce(
      "checkNanErad", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, data->np_tot[IDIR],
      KOKKOS_LAMBDA(int k, int j, int i, int& nnan) {
        if (std::isnan(Erad(k, j, i))) nnan++;
      },
      Kokkos::Sum<int>(nanErad)  // reduction variable
  );
#ifdef WITH_MPI
  MPI_Allreduce(MPI_IN_PLACE, &nanErad, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

  if (nanErad > 0) {
    std::stringstream msg;
    msg << "Input Erad in radiation contains " << nanErad << " NaNs" << std::endl;
    throw std::runtime_error(msg.str());
  }

  // Handling boundaries before Matrix calculation
  this->SetBoundaries(Erad);

  FLD_matrix->Reset();
  FillUtils();
  FillMatrixCouplingTerms();
  FillMatrix();


  // if we're "preconditioning" (i.e., rescaling), do it before solving
  if (FLD_matrix->havePreconditioner) {
    FLD_matrix->ScaleMatrix();
  }
}


void _fill_FLD_matrix(DataBlock* data, IdefixArray4D<real> M, IdefixArray3D<real> rhs,
                      IdefixArray3D<real> diag, IdefixArray3D<real> nu, IdefixArray3D<real> Erad,
                      bool havePreconditioner, RadiationBoundary rad_boundary,
                      real dt) {
  idfx::RegionWrapper region("_fill_FLD_matrix");

  int ibeg, iend, jbeg, jend, kbeg, kend;
  ibeg = data->beg[IDIR];
  iend = data->end[IDIR];
  jbeg = data->beg[JDIR];
  jend = data->end[JDIR];
  kbeg = data->beg[KDIR];
  kend = data->end[KDIR];

  // If preconditioning, we need the diagonal matrix elements in the ghost cells.
  // See FillFluxLimiter for explanation.
  const int pad = havePreconditioner;
  D_EXPAND(ibeg -= pad; iend += pad; , jbeg -= pad; jend += pad; , kbeg -= pad; kend += pad;)

  IdefixArray3D<real> dV = data->dV;

  D_EXPAND(  // dimensions = 0, 1, 2
      IdefixArray1D<real> x1 = data->x[IDIR]; IdefixArray1D<real> x1l = data->xl[IDIR];
      IdefixArray1D<real> x1r = data->xr[IDIR]; IdefixArray1D<real> dx1 = data->dx[IDIR];
      IdefixArray3D<real> Ax1 = data->A[IDIR]; ,
      IdefixArray1D<real> x2 = data->x[JDIR]; IdefixArray1D<real> x2l = data->xl[JDIR];
      IdefixArray1D<real> x2r = data->xr[JDIR]; IdefixArray1D<real> dx2 = data->dx[JDIR];
      IdefixArray3D<real> Ax2 = data->A[JDIR]; ,
      IdefixArray1D<real> x3 = data->x[KDIR]; IdefixArray1D<real> x3l = data->xl[KDIR];
      IdefixArray1D<real> x3r = data->xr[KDIR]; IdefixArray1D<real> dx3 = data->dx[KDIR];
      IdefixArray3D<real> Ax3 = data->A[KDIR];
#if GEOMETRY == SPHERICAL
      IdefixArray1D<real> sinth = data->sinx2;
#endif
  )

  std::array<RadiationBoundaryType, 3> lbound = rad_boundary.lbound;
  std::array<RadiationBoundaryType, 3> rbound = rad_boundary.rbound;
  std::array<real, 3> lvalue = rad_boundary.lvalue;
  std::array<real, 3> rvalue = rad_boundary.rvalue;

  /*
  idefix_for("Zero Matrix", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, data->np_tot[IDIR],
    KOKKOS_LAMBDA(int k, int j, int i) {
      for (int d = 0; d < 1 + 2 * DIMENSIONS; d++) {
        M(d, k, j, i) = ZERO_F;
      }
  });
  */

  idefix_for(
      "FiniteDifference", kbeg, kend, jbeg, jend, ibeg, iend, KOKKOS_LAMBDA(int k, int j, int i) {
        D_EXPAND(real h1; , real h2; , real h3;)
#if GEOMETRY == CARTESIAN
        D_EXPAND(h1 = 1.; , h2 = 1.; , h3 = 1.;)
#elif GEOMETRY == POLAR
      D_EXPAND(h1 = 1.; , h2 = x1(i); , h3 = 1.;)
#else
      D_EXPAND(h1 = 1.; , h2 = x1(i); , h3 = x1(i) * sinth(j);)
#endif

        M(0, k, j, i) = diag(k,j,i) * dV(k, j, i);
        rhs(k, j, i) *= dV(k, j, i);

#ifndef SECOND_ORDER_FLUXES
        D_EXPAND(
            const real nu1m = lageval1(x1(i - 1), x1(i), nu(k, j, i - 1), nu(k, j, i), x1l(i));
            const real nu1p = lageval1(x1(i), x1(i + 1), nu(k, j, i), nu(k, j, i + 1), x1r(i));
            M(1, k, j, i) =
                -dt * 2. / h1 * nu1m * Ax1(k, j, i) / (dx1(i) + dx1(i - 1));
            M(2, k, j, i) =
                -dt * 2. / h1 * nu1p * Ax1(k, j, i + 1) / (dx1(i) + dx1(i + 1));
            M(0, k, j, i) -= M(1, k, j, i) + M(2, k, j, i);
            ,
            const real nu2m = lageval1(x2(j - 1), x2(j), nu(k, j - 1, i), nu(k, j, i), x2l(j));
            const real nu2p = lageval1(x2(j), x2(j + 1), nu(k, j, i), nu(k, j + 1, i), x2r(j));
            M(3, k, j, i) =
                -dt * 2. / h2 * nu2m * Ax2(k, j, i) / (dx2(j) + dx2(j - 1));
            M(4, k, j, i) =
                -dt * 2. / h2 * nu2p * Ax2(k, j + 1, i) / (dx2(j) + dx2(j + 1));
            M(0, k, j, i) -= M(3, k, j, i) + M(4, k, j, i);
            ,
            const real nu3m = lageval1(x3(k - 1), x3(k), nu(k - 1, j, i), nu(k, j, i), x3l(k));
            const real nu3p = lageval1(x3(k), x3(k + 1), nu(k, j, i), nu(k + 1, j, i), x3r(k));
            M(5, k, j, i) =
                -dt * 2. / h3 * nu3m * Ax3(k, j, i) / (dx3(k) + dx3(k - 1));
            M(6, k, j, i) =
                -dt * 2. / h3 * nu3p * Ax3(k + 1, j, i) / (dx3(k) + dx3(k + 1));
            M(0, k, j, i) -= M(5, k, j, i) + M(6, k, j, i);
          )
#else
D_EXPAND(
        real nudx1m = (2./h1) / (dx1(i)/nu(k,j,i) + dx1(i-1)/nu(k,j,i-1));
        real nudx1p = (2./h1) / (dx1(i)/nu(k,j,i) + dx1(i+1)/nu(k,j,i+1));
        M(1,k,j,i) = -dt * nudx1m * Ax1(k,j,i);
        M(2,k,j,i) = -dt * nudx1p * Ax1(k,j,i+1);
        M(0,k,j,i) -= M(1,k,j,i) + M(2,k,j,i);
      ,
        real nudx2m = (2./h2) / (dx2(j)/nu(k,j,i) + dx2(j-1)/nu(k,j-1,i));
        real nudx2p = (2./h2) / (dx2(j)/nu(k,j,i) + dx2(j+1)/nu(k,j+1,i));
        M(3,k,j,i) = -dt * nudx2m * Ax2(k,j,i);
        M(4,k,j,i) = -dt * nudx2p * Ax2(k,j+1,i);
        M(0,k,j,i) -= M(3,k,j,i) + M(4,k,j,i);
      ,
        real nudx3m = (2./h3) / (dx3(k)/nu(k,j,i) + dx3(k-1)/nu(k-1,j,i));
        real nudx3p = (2./h3) / (dx3(k)/nu(k,j,i) + dx3(k+1)/nu(k+1,j,i));
        M(5,k,j,i) = -dt * nudx3m * Ax3(k,j,i);
        M(6,k,j,i) = -dt * nudx3p * Ax3(k+1,j,i);
        M(0,k,j,i) -= M(5,k,j,i) + M(6,k,j,i);
      )
#endif
        // now handle boundary conditions
        real delta;
        D_EXPAND(
            if (i == ibeg + pad) {
              switch (lbound[0]) {
                case RadiationBoundaryType::dirichlet:
                  delta = dx1(i - 1) * nu(k, j, i) / (dx1(i) * nu(k, j, i - 1));
                  M(0, k, j, i) -= M(1, k, j, i) * delta;
                  rhs(k, j, i) -= Erad(k, j, i - 1) * M(1, k, j, i) * (1 + delta);
                  M(1, k, j, i) = 0;
                  break;
                case RadiationBoundaryType::neumann:
                  M(0, k, j, i) += M(1, k, j, i);
                  real dx_nu = 0.5 * dx1(i - 1) / nu(k, j, i - 1) + 0.5 * dx1(i) / nu(k, j, i);
                  rhs(k, j, i) += lvalue[0] * M(1, k, j, i) * dx_nu;
                  M(1, k, j, i) = 0;
                  break;
              }
            } else if (i == iend - 1 - pad) {
              switch (rbound[0]) {
                case RadiationBoundaryType::dirichlet:
                  delta = dx1(i + 1) * nu(k, j, i) / (dx1(i) * nu(k, j, i + 1));
                  M(0, k, j, i) -= M(2, k, j, i) * delta;
                  rhs(k, j, i) -= Erad(k, j, i + 1) * M(2, k, j, i) * (1 + delta);
                  M(2, k, j, i) = 0;
                  break;
                case RadiationBoundaryType::neumann:
                  M(0, k, j, i) += M(2, k, j, i);
                  real dx_nu = 0.5 * dx1(i + 1) / nu(k, j, i + 1) + 0.5 * dx1(i) / nu(k, j, i);
                  rhs(k, j, i) += rvalue[0] * M(2, k, j, i) * dx_nu;
                  M(2, k, j, i) = 0;
                  break;
              }
            },
            if (j == jbeg + pad) {
              switch (lbound[1]) {
                case RadiationBoundaryType::dirichlet:
                  delta = dx2(j - 1) * nu(k, j, i) / (dx2(j) * nu(k, j - 1, i));
                  M(0, k, j, i) -= M(3, k, j, i) * delta;
                  rhs(k, j, i) -= Erad(k, j - 1, i) * M(3, k, j, i) * (1 + delta);
                  M(3, k, j, i) = 0;
                  break;
                case RadiationBoundaryType::neumann:
                  M(0, k, j, i) += M(3, k, j, i);
                  const real dx_nu =
                      0.5 * dx2(j - 1) / nu(k, j - 1, i) + 0.5 * dx2(j) / nu(k, j, i);
                  rhs(k, j, i) += lvalue[1] * M(3, k, j, i) * dx_nu;
                  M(3, k, j, i) = 0;
                  break;
              }
            } else if (j == jend - 1 - pad) {
              switch (rbound[1]) {
                case RadiationBoundaryType::dirichlet:
                  delta = dx2(j + 1) * nu(k, j, i) / (dx2(j) * nu(k, j + 1, i));
                  M(0, k, j, i) -= M(4, k, j, i) * delta;
                  rhs(k, j, i) -= Erad(k, j + 1, i) * M(4, k, j, i) * (1 + delta);
                  M(4, k, j, i) = 0;
                  break;
                case RadiationBoundaryType::neumann:
                  M(0, k, j, i) += M(4, k, j, i);
                  const real dx_nu =
                      0.5 * dx2(j + 1) / nu(k, j + 1, i) + 0.5 * dx2(j) / nu(k, j, i);
                  rhs(k, j, i) += rvalue[1] * M(4, k, j, i) * dx_nu;
                  M(4, k, j, i) = 0;
                  break;
              }
            },
            if (k == kbeg + pad) {
              switch (lbound[2]) {
                case RadiationBoundaryType::dirichlet:
                  delta = dx3(k - 1) * nu(k, j, i) / (dx3(k) * nu(k - 1, j, i));
                  M(0, k, j, i) -= M(5, k, j, i) * delta;
                  rhs(k, j, i) -= Erad(k - 1, j, i) * M(5, k, j, i) * (1 + delta);
                  M(5, k, j, i) = 0;
                  break;
                case RadiationBoundaryType::neumann:
                  const real dx_nu =
                      0.5 * dx3(k - 1) / nu(k - 1, j, i) + 0.5 * dx3(k) / nu(k, j, i);
                  rhs(k, j, i) += lvalue[2] * M(5, k, j, i) * dx_nu;
                  M(0, k, j, i) += M(5, k, j, i);
                  M(5, k, j, i) = 0;
                  break;
              }
            } else if (k == kend - 1 - pad) {
              switch (rbound[2]) {
                case RadiationBoundaryType::dirichlet:
                  delta = dx3(k + 1) * nu(k, j, i) / (dx3(k) * nu(k + 1, j, i));
                  M(0, k, j, i) -= M(6, k, j, i) * delta;
                  rhs(k, j, i) -= Erad(k + 1, j, i) * M(6, k, j, i) * (1 + delta);
                  M(6, k, j, i) = 0;
                  break;
                case RadiationBoundaryType::neumann:
                  M(0, k, j, i) += M(6, k, j, i);
                  const real dx_nu =
                      0.5 * dx3(k + 1) / nu(k + 1, j, i) + 0.5 * dx3(k) / nu(k, j, i);
                  rhs(k, j, i) += rvalue[2] * M(6, k, j, i) * dx_nu;
                  M(6, k, j, i) = 0;
                  break;
              }
            })
      });
}


void FluxLimitedDiffusion::FillMatrix() {
  idfx::RegionWrapper region("FLD::FillMatrix");

  _fill_FLD_matrix(
    data, FLD_matrix->M, rhs, diag, nu, Erad,
    FLD_matrix->havePreconditioner, rad_boundary, dt
  );

  setup_multigrid_precond();
}


void FluxLimitedDiffusion::setup_multigrid_precond() {
  idfx::RegionWrapper region("FLD::setup_multigrid_precond");

  if (!haveMultiGrid) {
    return;
  }

  // Copy the solver to the finest grid
  multigrid->SetMatrix(this->FLD_matrix->M, 0);

  if (FLD_matrix->havePreconditioner)
    multigrid->SetScaling(FLD_matrix->precond);

  IdefixArray3D<real> Erad = this->Erad;
  IdefixArray3D<real> nu = this->nu;
  IdefixArray3D<real> diag = this->diag;

  for (int level = 1; level < multigrid->NumLevels(); level++) {
    auto& gf = multigrid->GetGrid(level - 1);
    auto& gc = multigrid->GetGrid(level);

    // Volume average Erad, nu, and the diagonal elements.
    // We don't use the Restrict method because we need the ghost cells.
    IdefixArray3D<real> Erad_coarse = gc.solution;
    IdefixArray3D<real> nu_coarse = gc.workspace;
    IdefixArray3D<real> diag_coarse = gc.workspace2;

    IdefixArray3D<real> dV_f = gf.data->dV;
    IdefixArray3D<real> dV_c = gc.data->dV;

    std::array<bool, 3> coarsend = gc.coarsened;

    int ibeg = gc.data->beg[IDIR], iend = gc.data->end[IDIR];
    int jbeg = gc.data->beg[JDIR], jend = gc.data->end[JDIR];
    int kbeg = gc.data->beg[KDIR], kend = gc.data->end[KDIR];
    int igh = gc.data->nghost[IDIR];
    int jgh = gc.data->nghost[JDIR];
    int kgh = gc.data->nghost[KDIR];

    idefix_for("RestrictEradNuDiag", 0, gc.np_tot[KDIR], 0, gc.np_tot[JDIR], 0, gc.np_tot[IDIR],
      KOKKOS_LAMBDA(int k, int j, int i) {
        // Get the start and end indices for the fine grid, accounting for coarsening and
        // boundaries. Periodic boundaries will be handled after.
        auto idx_start = [](int l, int beg, int end, bool coarsened) {
          if (l < beg) return l;
          if (l >= end) return l + (end-beg) * coarsened;
          return (l - beg) * (1 + coarsened) + beg;
        };
        auto idx_end = [](int l, int beg, int end, bool coarsened) {
          if (l < beg) return l+1;
          if (l >= end) return l+1 + (end-beg) * coarsened;
          return (l+1 - beg) * (1 + coarsened) + beg;
        };
        int ks_ = idx_start(k, kbeg, kend, coarsend[2]), ke = idx_end(k, kbeg, kend, coarsend[2]);
        int js_ = idx_start(j, jbeg, jend, coarsend[1]), je = idx_end(j, jbeg, jend, coarsend[1]);
        int is_ = idx_start(i, ibeg, iend, coarsend[0]), ie = idx_end(i, ibeg, iend, coarsend[0]);

        real Etot = 0, nutot = 0, diagtot = 0;
        for (int kk = ks_; kk < ke; kk++)
          for (int jj = js_; jj < je; jj++)
            for (int ii = is_; ii < ie; ii++) {
              Etot += Erad(kk, jj, ii) * dV_f(kk, jj, ii);
              nutot += dV_f(kk, jj, ii) / nu(kk, jj, ii);  // Harmonic average of nu
              diagtot += diag(kk, jj, ii) * dV_f(kk, jj, ii);
        }
        Erad_coarse(k, j, i) = Etot / dV_c(k, j, i);
        nu_coarse(k, j, i) = dV_c(k, j, i) / nutot;  // Harmonic average of nu
        diag_coarse(k, j, i) = diagtot / dV_c(k, j, i);
      });

      // Handle periodic and MPI boundaries
      //  (ignore other ghost cells)
      gc.matrix.ApplyPeriodicAxisBoundariesOnly(Erad_coarse);
      gc.matrix.ApplyPeriodicAxisBoundariesOnly(nu_coarse);
      gc.matrix.ApplyPeriodicAxisBoundariesOnly(diag_coarse);

    // Create the matrix on the coarse grid
    //   Note: the lowest level needs the preconditioner since we use an iterative solver on it.
    bool precond = (level == multigrid->NumLevels()-1);
    _fill_FLD_matrix(
      gc.data.get(), gc.matrix.M, gc.rhs, diag_coarse, nu_coarse, Erad_coarse, precond,
      rad_boundary, dt
    );

    gc.matrix.havePreconditioner = precond;

    Erad = Erad_coarse;
    nu = nu_coarse;
    diag = diag_coarse;
  }

  multigrid->FinishSetup();
}


// done
void FluxLimitedDiffusion::SetBoundaries(IdefixArray3D<real>& arr) {
  idfx::RegionWrapper region("FLD::SetBoundaries");

#ifdef WITH_MPI
  IdefixArray4D<real> arr4D = IdefixArray4D<real>(
    arr.data(), 1, data->np_tot[KDIR], data->np_tot[JDIR], data->np_tot[IDIR]
  );
#endif

  ApplyBoundaryByDirection(
      data,
      [&](int dir) {
// MPI Exchange data when needed
#ifdef WITH_MPI
  MPIBoundaryExchange(data, mpi, arr4D, dir);
#endif
      },
      [&](int dir, BoundarySide side) { EnforceBoundary(dir, side, arr); });
}

// done
void FluxLimitedDiffusion::EnforceBoundary(int dir, BoundarySide side, IdefixArray3D<real>& arr) {
  idfx::RegionWrapper region("FLD::EnforceBoundary");

  IdefixArray3D<real> localVar = arr;

  // Number of active cells
  const int nxi = data->np_int[IDIR];
  const int nxj = data->np_int[JDIR];
  const int nxk = data->np_int[KDIR];

  // Number of ghost cells
  const int ighost = data->nghost[IDIR];
  const int jghost = data->nghost[JDIR];
  const int kghost = data->nghost[KDIR];

  // Boundaries of the loop
  BoundaryLoopRange range = ComputeBoundaryLoopRange(data, dir, side);
  const int ibeg = range.ibeg;
  const int iend = range.iend;
  const int jbeg = range.jbeg;
  const int jend = range.jend;
  const int kbeg = range.kbeg;
  const int kend = range.kend;

  RadiationBoundaryType type = (side == left) ? rad_boundary.lbound[dir] : rad_boundary.rbound[dir];
  std::array<real, 3> lvalue = rad_boundary.lvalue;
  std::array<real, 3> rvalue = rad_boundary.rvalue;

  switch (type) {
    case RadiationBoundaryType::internal:
      // internal is used for MPI-enforced boundary conditions. Nothing to be done here.
      break;

    case RadiationBoundaryType::periodic: {
      if (data->mygrid->nproc[dir] > 1) break;  // Periodicity already enforced by MPI calls
      ApplyPeriodicBoundary(data, dir, side, arr);
      break;
    }

    case RadiationBoundaryType::userdef: {
      if (haveUserDefBoundary) {
        // Warning: unlike hydro userdef boundary functions, the Radiation
        // userdef boundary functions take an additional argument arr which
        // specifies the array for which boundaries are to be handled
        userDefBoundaryFunc(*data, dir, side, data->t, arr);
      } else {
        IDEFIX_ERROR("FLD:: No function enrolled to define your own boundary conditions");
      }
      break;
    }

    case RadiationBoundaryType::dirichlet: {
      real val = side == left ? lvalue[dir] : rvalue[dir];
      idefix_for(
          "BoundaryDirichlet", kbeg, kend, jbeg, jend, ibeg, iend,
          KOKKOS_LAMBDA(int k, int j, int i) { localVar(k, j, i) = val; });
      break;
    }

    case RadiationBoundaryType::neumann: {
      idefix_for(
          "BoundaryNeumann", kbeg, kend, jbeg, jend, ibeg, iend,
          KOKKOS_LAMBDA(int k, int j, int i) {
            const int iref = (dir == IDIR) ? ighost + side * (nxi - 1) : i;
            const int jref = (dir == JDIR) ? jghost + side * (nxj - 1) : j;
            const int kref = (dir == KDIR) ? kghost + side * (nxk - 1) : k;

            localVar(k, j, i) = localVar(kref, jref, iref);
          });
      break;
    }

    case RadiationBoundaryType::axis: {
      BoundaryLoopRange range{ibeg, iend, jbeg, jend, kbeg, kend};
      ApplyAxisBoundary(data, side, range, arr, this->isTwoPi);
      break;
    }

    default: {
      std::stringstream msg("FLD:: Boundary condition type is not yet implemented");
      IDEFIX_ERROR(msg);
    }
  }
}

void FluxLimitedDiffusion::EnrollUserDefBoundary(UserDefBoundaryFunc myFunc) {
  IDEFIX_ERROR("FLD::EnrollUserDefBoundary: Not currently implemented");

  /*
  userDefBoundaryFunc = myFunc;
  haveUserDefBoundary = true;
  idfx::cout << "FLD:: User-defined boundary condition has been enrolled" << std::endl;
  */
}

void FluxLimitedDiffusion::EnrollUserDefOpacityCGS(PrototypeOpacityFuncCGS myFunc) {
  if (!haveUserDefOpacity) {
    IDEFIX_ERROR(
        "FLD:: Attempt to enroll opacity function but a constant value has already been set");
  }
  UserOpacityFuncCGS = myFunc;
  idfx::cout << "FLD:: User-defined opacity function has been enrolled" << std::endl;
}

// Compute the flux limiter for each cell
void FluxLimitedDiffusion::FillFluxLimiter(IdefixArray4D<real> rho_kappaR,
                                           IdefixArray3D<real> lambda) {
  idfx::RegionWrapper region("FluxLimitedDiffusion::FillFluxLimiter");

  // configure flux limiter

  // First, compute the flux limiter parameter, R
  int kbeg = data->beg[KDIR];
  int kend = data->end[KDIR];
  int jbeg = data->beg[JDIR];
  int jend = data->end[JDIR];
  int ibeg = data->beg[IDIR];
  int iend = data->end[IDIR];

  // The flux limiter is needed for nu, which is needed in one layer of ghost cells
  // in order to compute the fluxes in the active domain.
  // For this reason we need to add a padding to the loop limits.
  // If preconditioning, the diagonal matrix elements are also needed in the ghosts,
  // so we need to add an additional layer then.
  // In both cases, the flux limiter also needs the gradient of Erad
  // -> total of 2+(havePreconditioner) layers of ghost cells are needed for Erad!
  const int pad = FLD_matrix->havePreconditioner + 1;
  D_EXPAND(ibeg -= pad; iend += pad; , jbeg -= pad; jend += pad; , kbeg -= pad; kend += pad;)

  IdefixArray1D<real> x1 = data->x[IDIR];
  IdefixArray1D<real> x2 = data->x[JDIR];
  IdefixArray1D<real> x3 = data->x[KDIR];

  IdefixArray3D<real> Erad = this->Erad;

  const FluxLimiterType fluxLimiterType = this->fluxLimiterType;

  idefix_for(
      "SetFluxLimiter", kbeg, kend, jbeg, jend, ibeg, iend, KOKKOS_LAMBDA(int k, int j, int i) {
        real gradE;
        real normE2 = 0.0;

        D_EXPAND(  // dimensions = 0, 1, 2
            gradE = lagdiff1(x1(i - 1), x1(i + 1), Erad(k, j, i - 1), Erad(k, j, i + 1), x1(i));
            normE2 += gradE * gradE;
            , gradE = lagdiff1(x2(j - 1), x2(j + 1), Erad(k, j - 1, i), Erad(k, j + 1, i), x2(j));
#if (GEOMETRY == CYLINDRICAL) || (GEOMETRY == POLAR) || (GEOMETRY == SPHERICAL)
            gradE /= x1(i);
#endif
            normE2 += gradE * gradE;
            , gradE = lagdiff1(x3(k - 1), x3(k + 1), Erad(k - 1, j, i), Erad(k + 1, j, i), x3(k));
#if GEOMETRY == SPHERICAL
            gradE /= x1(i) * sin(x2(j));
#endif
            normE2 += gradE * gradE;)

        // Compute R and store in lambda
        const real rho_kapp = rho_kappaR(0, k, j, i);
        real R = (1.0 / rho_kapp) * sqrt(normE2) / Erad(k, j, i);

        switch (fluxLimiterType) {
          case one_third:
            lambda(k, j, i) = 1.0 / 3;
            break;
          case Kley1989:
            lambda(k, j, i) = RadiationFluxLimiterKley1989(R);
            break;
          case Minerbo1978:
            lambda(k, j, i) = RadiationFluxLimiterMinerbo1978(R);
            break;
          case LevermorePomraning1981:
            lambda(k, j, i) = RadiationFluxLimiterLevermorePomraning1981(R);
            break;
        }
      });
}

// done
void FluxLimitedDiffusion::SolveSystem() {
  idfx::RegionWrapper region("FLD::SolveSystem");

  IdefixArray3D<real> Erad = this->Erad;
  IdefixArray3D<real> rhs = this->rhs;
  Kokkos::Timer timer;

  elapsedTime -= timer.seconds();

  InitSolver();  // (Re)initialise the solver

  if (FLD_matrix->havePreconditioner) {
    FLD_matrix->ScaleSolAndRhs(Erad, rhs, false);  // scale the solution and rhs for preconditioning
  }

  this->nsteps = iterativeSolver->Solve(Erad, rhs);

  if (this->nsteps < 0) {
    idfx::cout << "FLD:: BICGSTAB failed, resetting radiation field" << std::endl;

    // Look for Nans to explain the repetitive failing
    if (data->CheckNan() > 0) {
      std::stringstream msg;
      msg << "Nan found after BICGSTAB failed at time " << data->t << std::endl;
      throw std::runtime_error(msg.str());
    }

    // Re-initialise Erad to gas temperature.
    IdefixArray4D<real> Vc = data->hydro->Vc;

    const real mu = this->mu;
    const real code_aR = this->code_aR;

    idefix_for(
        "ResetErad", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, data->np_tot[IDIR],
        KOKKOS_LAMBDA(int k, int j, int i) {
          real tmp = mu * Vc(PRS, k, j, i) / Vc(RHO, k, j, i);
          Erad(k, j, i) = code_aR * pow(tmp, 4);
        });

    // if we're "preconditioning", we need to redo it
    if (FLD_matrix->havePreconditioner) FLD_matrix->ScaleSolution(Erad, false);

    // Try again !
    this->nsteps = iterativeSolver->Solve(this->Erad, rhs);
    if (this->nsteps < 0) {
      IDEFIX_ERROR("FLD:: BICGSTAB failed despite restart");
    }
  }

  currentError = iterativeSolver->GetError();

  // if we were "preconditioning", rescale back to normal
  if (FLD_matrix->havePreconditioner) {
    FLD_matrix->ScaleSolution(Erad, true);
  }

  this->SetBoundaries(Erad);
  UpdatePressure();
  elapsedTime += timer.seconds();
}

void FluxLimitedDiffusion::_ComputeRadiationPressureSourceTerm(real dt, int species,
                                                               bool update_Vc) {
  idfx::RegionWrapper region("FLD::ComputeRadiationPressureSourceTerm");

  // Step 1: Get the Rosseland opacity:
  IdefixArray3D<real> kappaR = this->rhs;

  int kbeg = data->beg[KDIR];
  int kend = data->end[KDIR];
  int jbeg = data->beg[JDIR];
  int jend = data->end[JDIR];
  int ibeg = data->beg[IDIR];
  int iend = data->end[IDIR];

  if (haveUserDefOpacity) {
    data->radiation->UserOpacityFuncCGS(data, species, FLD_matrix->precond, kappaR);
  } else {
    auto _kappaR = data->radiation->constkappaR;
    auto _kappaP = data->radiation->constkappaP;

    idefix_for(
        "FillKappa", kbeg, kend, jbeg, jend, ibeg, iend,
        KOKKOS_LAMBDA(int k, int j, int i) { kappaR(k, j, i) = _kappaR[species]; });
  }

  // Step 2: Compute radiation pressure force and add it to the momentum
  real code_c = this->code_c;
  IdefixArray3D<real> nu = this->nu;  // diffusivity
  IdefixArray4D<real> Uc = data->hydro->Uc;
  IdefixArray4D<real> Vc = data->hydro->Uc;
  IdefixArray3D<real> Erad = this->Erad;

  if (species > 0) {
    Uc = data->dust[species - 1]->Uc;
    Vc = data->dust[species - 1]->Vc;
  }

  D_EXPAND(  // dimensions = 0, 1, 2
      IdefixArray1D<real> x1 = data->x[IDIR]; IdefixArray1D<real> dx1 = data->dx[IDIR];
      , IdefixArray1D<real> x2 = data->x[JDIR]; IdefixArray1D<real> dx2 = data->dx[JDIR];
      , IdefixArray1D<real> x3 = data->x[KDIR]; IdefixArray1D<real> dx3 = data->dx[KDIR];
#if GEOMETRY == SPHERICAL
      IdefixArray1D<real> sinth = data->sinx2;
#endif
  )

  real dt_over_c = 0.5 * dt / (this->unit_opacity * code_c);

  idefix_for("FLDRadPressure", kbeg, kend, jbeg, jend, ibeg, iend,
    KOKKOS_LAMBDA(int k, int j, int i) {
      D_EXPAND(real h1; , real h2; , real h3;)
#if GEOMETRY == CARTESIAN
      D_EXPAND(h1 = 1.; , h2 = 1.; , h3 = 1.;)
#elif GEOMETRY == POLAR
      D_EXPAND(h1 = 1.; , h2 = x1(i); , h3 = 1.;)
#else
      D_EXPAND(h1 = 1.; , h2 = x1(i); , h3 = x1(i) * sinth(j);)
#endif

      D_EXPAND(real flux1m; , real flux2m; , real flux3m;)
      D_EXPAND(real flux1p; , real flux2p; , real flux3p;)

#ifndef SECOND_ORDER_FLUXES
      D_EXPAND(
            const real nu1m = lageval1(x1(i - 1), x1(i), nu(k, j, i - 1), nu(k, j, i), x1l(i));
            const real nu1p = lageval1(x1(i), x1(i + 1), nu(k, j, i), nu(k, j, i + 1), x1r(i));
            flux1m =
                -nu1m * (Erad(k, j, i) - Erad(k, j, i - 1)) * (2. / h1) / (dx1(i) + dx1(i - 1));
            flux1p =
                -nu1p * (Erad(k, j, i + 1) - Erad(k, j, i)) * (2. / h1) / (dx1(i) + dx1(i + 1));
            ,
            const real nu2m = lageval1(x2(j - 1), x2(j), nu(k, j - 1, i), nu(k, j, i), x2l(j));
            const real nu2p = lageval1(x2(j), x2(j + 1), nu(k, j, i), nu(k, j + 1, i), x2r(j));
            flux2m =
                -nu2m * (Erad(k, j, i) - Erad(k, j - 1, i)) * (2. / h2) / (dx2(j) + dx2(j - 1));
            flux2p =
                -nu2p * (Erad(k, j + 1, i) - Erad(k, j, i)) * (2. / h2) / (dx2(j) + dx2(j + 1));
            ,
            const real nu3m = lageval1(x3(k - 1), x3(k), nu(k - 1, j, i), nu(k, j, i), x3l(k));
            const real nu3p = lageval1(x3(k), x3(k + 1), nu(k, j, i), nu(k + 1, j, i), x3r(k));
            flux3m =
                -nu3m * (Erad(k, j, i) - Erad(k - 1, j, i)) * (2. / h3) / (dx3(k) + dx3(k - 1));
            flux3p =
                -nu3p * (Erad(k + 1, j, i) - Erad(k, j, i)) * (2. / h3) / (dx3(k) + dx3(k + 1));)
#else
      D_EXPAND(
        real nud1xm = (2./h1) / (dx1(i)/nu(k,j,i) + dx1(i-1)/nu(k,j,i-1));
        real nud1xp = (2./h1) / (dx1(i)/nu(k,j,i) + dx1(i+1)/nu(k,j,i+1));
        flux1m = - nud1xm * (Erad(k,j,i) - Erad(k,j,i-1));
        flux1p = - nud1xp * (Erad(k,j,i+1) - Erad(k,j,i));
        ,
        real nud2xm = (2./h2) / (dx2(j)/nu(k,j,i) + dx2(j-1)/nu(k,j-1,i));
        real nud2xp = (2./h2) / (dx2(j)/nu(k,j,i) + dx2(j+1)/nu(k,j+1,i));
        flux2m = - nud2xm * (Erad(k,j,i) - Erad(k,j-1,i));
        flux2p = - nud2xp * (Erad(k,j+1,i) - Erad(k,j,i));
        ,
        real nud3xm = (2./h3) / (dx3(k)/nu(k,j,i) + dx3(k-1)/nu(k-1,j,i));
        real nud3xp = (2./h3) / (dx3(k)/nu(k,j,i) + dx3(k+1)/nu(k+1,j,i));
        flux3m = - nud3xm * (Erad(k,j,i) - Erad(k-1,j,i));
        flux3p = - nud3xp * (Erad(k+1,j,i) - Erad(k,j,i));
      )
#endif

      if (update_Vc) {
        D_EXPAND(
          Vc(VX1, k, j, i) += dt_over_c * (flux1m + flux1p) * kappaR(k, j, i); ,
          Vc(VX2, k, j, i) += dt_over_c * (flux2m + flux2p) * kappaR(k, j, i); ,
          Vc(VX3, k, j, i) += dt_over_c * (flux3m + flux3p) * kappaR(k, j, i);
        )
      }
      else {
        D_EXPAND(
          Uc(VX1, k, j, i) += dt_over_c * (flux1m + flux1p) * Vc(RHO, k, j, i) * kappaR(k, j, i); ,
          Uc(VX2, k, j, i) += dt_over_c * (flux2m + flux2p) * Vc(RHO, k, j, i) * kappaR(k, j, i); ,
          Uc(VX3, k, j, i) += dt_over_c * (flux3m + flux3p) * Vc(RHO, k, j, i) * kappaR(k, j, i);
        )
    }
  });
}

TwoTemperatureFLD::TwoTemperatureFLD(Input& input, DataBlock* datain)
    : FluxLimitedDiffusion(input, datain) {
  Init(input, datain);
}

void TwoTemperatureFLD::Init(Input& input, DataBlock* datain) {
  idfx::RegionWrapper region("TwoTempFLD::Init");
  auto data = this->data;

  // Initialize our workspace arrays
  this->radX = IdefixArray4D<real>("RadiationX", num_species, data->np_tot[KDIR],
                                   data->np_tot[JDIR], data->np_tot[IDIR]);

  this->radY = IdefixArray4D<real>("RadiationY", num_species, data->np_tot[KDIR],
                                   data->np_tot[JDIR], data->np_tot[IDIR]);

  // Fill arrays with 0
  auto radX = this->radX;
  auto radY = this->radY;
  int num_species = this->num_species;

  idefix_for(
      "InitRadiationArrays", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, data->np_tot[IDIR],
      KOKKOS_LAMBDA(int k, int j, int i) {
        for (int s = 0; s < num_species; s++) {
          radX(s, k, j, i) = ZERO_F;
          radY(s, k, j, i) = ZERO_F;
        }
      });
}

// done
void TwoTemperatureFLD::UpdatePressure() {
  idfx::RegionWrapper region("FLD::UpdatePressure");

  IdefixArray3D<real> Erad = this->Erad;  // Radiation energy
  IdefixArray4D<real> radY = this->radY;  // κP ρ c dt -> dimensionless
  IdefixArray4D<real> radX = this->radX;  // aR T^3 radY / (ρ cV) -> dimensionless
  IdefixArray4D<real> Vc = data->hydro->Vc;
  const real gamma = data->hydro->eos->GetGamma();
  const real mu = this->mu;
  const real cV = this->cV;
  const real dt = this->dt;

  const int haveIrradiation = data->haveIrradiation;
  IdefixArray4D<real> Sirrad;
  if (haveIrradiation) Sirrad = data->irradiation->irradiationHeating;
  int num_species = this->num_species;

  idefix_for(
      "UpdatePressure", data->beg[KDIR], data->end[KDIR], data->beg[JDIR], data->end[JDIR],
      data->beg[IDIR], data->end[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
        real rho = Vc(RHO, k, j, i);
        real prs = Vc(PRS, k, j, i);
        real tmp = mu * prs / rho;
        real x = radX(0, k, j, i);
        real y = radY(0, k, j, i);

        real num = tmp * (1 + 3 * x) + y * Erad(k, j, i) / (rho * cV);
        if (haveIrradiation) {
          for (int s = 0; s < num_species; s++) num += Sirrad(s, k, j, i) * dt / cV;
        }
        real denom = 1 + 4 * x;
        real newtmp = num / denom;
        Vc(PRS, k, j, i) = newtmp * rho / mu;
      });
}

// done
void TwoTemperatureFLD::FillUtils() {
  idfx::RegionWrapper region("FLD::FillUtils");

  IdefixArray3D<real> Erad = this->Erad;     // Radiation energy
  IdefixArray4D<real> radY = this->radY;     // κP ρ c dt -> dimensionless
  IdefixArray4D<real> radX = this->radX;     // aR T^3 radY / (ρ cV) -> dimensionless
  IdefixArray3D<real> nu = this->nu;   // λ c / (kR ρ) -> diffusivity
  IdefixArray3D<real> rhs = this->rhs; // Right hand side -> same units as Erad

  const int haveIrradiation = data->haveIrradiation;
  IdefixArray4D<real> Sirrad;
  if (haveIrradiation) Sirrad = data->irradiation->irradiationHeating;

  const real code_aR = this->code_aR;
  const real code_c = this->code_c;
  const real u_opac = this->unit_opacity;
  const real dt = this->dt;

  int kbeg = data->beg[KDIR];
  int kend = data->end[KDIR];
  int jbeg = data->beg[JDIR];
  int jend = data->end[JDIR];
  int ibeg = data->beg[IDIR];
  int iend = data->end[IDIR];

  // see FillFluxLimiter for explanation: we need nu in the ghost cells
  const int pad = FLD_matrix->havePreconditioner + 1;
  D_EXPAND(ibeg -= pad; iend += pad; , jbeg -= pad; jend += pad; , kbeg -= pad; kend += pad;)

  // Aliases for temporary arrays
  IdefixArray4D<real> _kappaR = radX;
  IdefixArray4D<real> _kappaP = radY;
  IdefixArray3D<real> _lambda = nu;

  // Fill the Rosseland and Planck opacities
  if (haveUserDefOpacity) {
    for (int s = 0; s < num_species; s++) {
      IdefixArray3D<real> kappaP = subview(_kappaP, s, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
      IdefixArray3D<real> kappaR = subview(_kappaR, s, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
      data->radiation->UserOpacityFuncCGS(data, s, kappaP, kappaR);
    }
  } else {
    auto kappaR = data->radiation->constkappaR;
    auto kappaP = data->radiation->constkappaP;

    idefix_for(
        "FillKappa", 0, num_species, kbeg, kend, jbeg, jend, ibeg, iend,
        KOKKOS_LAMBDA(int s, int k, int j, int i) {
          _kappaR(s, k, j, i) = kappaR[s];
          _kappaP(s, k, j, i) = kappaP[s];
        });
  }

  // Compute total Rosseland opacity:
  for (int s = 0; s < num_species; s++) {
    IdefixArray4D<real> Vc;
    if (s == 0)
      Vc = data->hydro->Vc;
    else
      Vc = data->dust[s - 1]->Vc;

    idefix_for(
        "rho_kappa", kbeg, kend, jbeg, jend, ibeg, iend, KOKKOS_LAMBDA(int k, int j, int i) {
          if (s == 0) {
            _kappaR(0, k, j, i) = _kappaR(s, k, j, i) * Vc(RHO, k, j, i) / u_opac;
            _kappaP(0, k, j, i) = _kappaP(s, k, j, i) * Vc(RHO, k, j, i) / u_opac;
          } else {
            _kappaR(0, k, j, i) += _kappaR(s, k, j, i) * Vc(RHO, k, j, i) / u_opac;
            _kappaP(0, k, j, i) += _kappaP(s, k, j, i) * Vc(RHO, k, j, i) / u_opac;
          }
        });
  }

  // Compute the flux limiter
  FillFluxLimiter(_kappaR, _lambda);

  // Fill the final radiation arrays
  IdefixArray4D<real> Vc = data->hydro->Vc;
  const real mu = this->mu;
  const real cV = this->cV;

  for (int s = 0; s < num_species; s++) {
    idefix_for(
        "FillUtils", kbeg, kend, jbeg, jend, ibeg, iend, KOKKOS_LAMBDA(int k, int j, int i) {
          real rho = Vc(RHO, k, j, i);
          real prs = Vc(PRS, k, j, i);  // Works for dust too if tracers are used.
          real tmp = mu * prs / rho;

          if (s == 0) {
            nu(k, j, i) = _lambda(k, j, i) * code_c / _kappaR(0, k, j, i);
            rhs(k, j, i) = Erad(k, j, i);

            radY(0, k, j, i) = _kappaP(0, k, j, i) * code_c * dt;
            radX(0, k, j, i) = code_aR * pow(tmp, 3) * radY(0, k, j, i) / (rho * cV);
            rhs(k, j, i) += code_aR * pow(tmp, 4) * radY(0, k, j, i) / (1 + 4 * radX(0, k, j, i));
          }

          if (haveIrradiation) {
            rhs(k, j, i) +=
                Sirrad(s, k, j, i) * rho * dt * 4 * radX(0, k, j, i) / (1 + 4 * radX(0, k, j, i));
          }
        });
  }
}

void TwoTemperatureFLD::FillMatrixCouplingTerms() {
  idfx::RegionWrapper region("FLD::FillMatrixCouplingTerms");

  int ibeg, iend, jbeg, jend, kbeg, kend;
  ibeg = data->beg[IDIR];
  iend = data->end[IDIR];
  jbeg = data->beg[JDIR];
  jend = data->end[JDIR];
  kbeg = data->beg[KDIR];
  kend = data->end[KDIR];

  // If preconditioning, we need the diagonal matrix elements in the ghost cells.
  // See FillFluxLimiter for explanation.
  const int pad = FLD_matrix->havePreconditioner;
  D_EXPAND(ibeg -= pad; iend += pad; , jbeg -= pad; jend += pad; , kbeg -= pad; kend += pad;)


  IdefixArray3D<real> diag = this->diag;
  IdefixArray4D<real> radY = this->radY;
  IdefixArray4D<real> radX = this->radX;
  IdefixArray3D<real> dV = data->dV;


  idefix_for(
      "FiniteDifference", kbeg, kend, jbeg, jend, ibeg, iend, KOKKOS_LAMBDA(int k, int j, int i) {
        // Coupling coefficients and matrix elements
        diag(k, j, i) = (1 + radY(0, k, j, i)) / (1 + 4 * radX(0, k, j, i));
      });
}

// done
void FluxLimitedDiffusion::ShowConfig() {
  idfx::cout << "FLD: Using ";
  switch (solver) {
    case BICGSTAB:
      idfx::cout << "unpreconditioned BICGSTAB";
      break;
    case PBICGSTAB:
      idfx::cout << "preconditioned BICGSTAB";
      break;
    case PCG:
      idfx::cout << "preconditioned CG";
      break;
    case CG:
      idfx::cout << "unpreconditioned CG";
      break;
    case MINRES:
      idfx::cout << "unpreconditioned MinRes";
      break;
    case PMINRES:
      idfx::cout << "preconditioned MinRes";
      break;
    default:
      IDEFIX_ERROR("FLD:: Unknown solver");
  }
  idfx::cout << " solver." << std::endl;

  idfx::cout << "FLD: Flux limiter is ";
  switch (fluxLimiterType) {
    case one_third:
      idfx::cout << "1/3";
      break;
    case Kley1989:
      idfx::cout << "Kley 1989";
      break;
    case Minerbo1978:
      idfx::cout << "Minerbo 1978";
      break;
    case LevermorePomraning1981:
      idfx::cout << "Levermore-Pomraning 1981";
      break;
    case userdeffluxlimiter:
      idfx::cout << "user defined";
      break;
    default:
      IDEFIX_ERROR("FLD:: Unknown flux limiter");
  }
  idfx::cout << "." << std::endl;

  idfx::cout << "FLD: Opacity is ";
  switch (opacityType) {
    case constantkappa:
      idfx::cout << "constant";
      break;
    case userdefkappa:
      idfx::cout << "user defined";
      break;
    default:
      IDEFIX_ERROR("FLD:: Unknown opacity");
  }
  idfx::cout << "." << std::endl;

  iterativeSolver->ShowConfig();

  if (haveMultiGrid) {
    multigrid->ShowConfig();
  }
}

void TwoTemperatureFLD::ShowConfig() {
  idfx::cout << "RT: TwoTemperatureFLD with " << num_species << " species.\n";
  FluxLimitedDiffusion::ShowConfig();
}
