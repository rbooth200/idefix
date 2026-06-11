// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// Module contributed by Alex Ziampras, then at Queen Mary University of London
// ***********************************************************************************

#include "FLD.hpp"

#include <memory>
#include <string>
#include <vector>

#include "bicgstab.hpp"
#include "cg.hpp"
#include "dataBlock.hpp"
#include "fluid.hpp"
#include "idefix.hpp"
#include "minres.hpp"
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

FLDMatrix::FLDMatrix(Input& input, DataBlock* datain) {
  data = datain;

  isTwoPi = true;

  this->M = IdefixArray4D<real>("Matrix", 2 * DIMENSIONS + 1, data->np_tot[KDIR],
                                data->np_tot[JDIR], data->np_tot[IDIR]);

  // Get the radiation-related boundary conditions
  for (int dir = 0; dir < 3; dir++) {
    std::string label = std::string("boundary-X") + std::to_string(dir + 1) + std::string("-beg");
    std::string boundary = input.Get<std::string>("Radiation", label, 0);
    this->lvalue[dir] = ZERO_F;

    if (boundary.compare("dirichlet") == 0) {
      lbound[dir] = dirichlet;
      lvalue[dir] = input.GetOrSet<real>("Radiation", label, 1, 1e-20);
      isPeriodic = false;
    } else if (boundary.compare("periodic") == 0) {
      lbound[dir] = periodic;
    } else if (boundary.compare("neumann") == 0) {
      lbound[dir] = neumann;
      lvalue[dir] = input.GetOrSet<real>("Radiation", label, 1, 0);
      isPeriodic = false;
    } else if (boundary.compare("internalradiation") == 0) {
      lbound[dir] = internalradiation;
      isPeriodic = false;
    } else if (boundary.compare("userdef") == 0) {
      lbound[dir] = userdef;
      isPeriodic = false;
    } else if (boundary.compare("axis") == 0) {
      lbound[dir] = axis;
      isPeriodic = false;
    } else {
      std::stringstream msg;
      msg << "FLD:: Unknown boundary type " << boundary;
      IDEFIX_ERROR(msg);
    }

    label = std::string("boundary-X") + std::to_string(dir + 1) + std::string("-end");
    boundary = input.Get<std::string>("Radiation", label, 0);
    this->rvalue[dir] = ZERO_F;

    if (boundary.compare("dirichlet") == 0) {
      this->rbound[dir] = dirichlet;
      this->rvalue[dir] = input.GetOrSet<real>("Radiation", label, 1, 1e-20);
      this->isPeriodic = false;
    } else if (boundary.compare("periodic") == 0) {
      this->rbound[dir] = periodic;
    } else if (boundary.compare("neumann") == 0) {
      this->rbound[dir] = neumann;
      this->rvalue[dir] = input.GetOrSet<real>("Radiation", label, 1, 0);
      this->isPeriodic = false;
    } else if (boundary.compare("internalradiation") == 0) {
      this->rbound[dir] = internalradiation;
      this->isPeriodic = false;
    } else if (boundary.compare("userdef") == 0) {
      this->rbound[dir] = userdef;
      this->isPeriodic = false;
    } else if (boundary.compare("axis") == 0) {
      this->rbound[dir] = axis;
      this->isPeriodic = false;
    } else {
      std::stringstream msg;
      msg << "FLD:: Unknown boundary type " << boundary;
      IDEFIX_ERROR(msg);
    }
  }

// Update internal boundaries in case of domain decomposition
#ifdef WITH_MPI
  for (int dir = 0; dir < DIMENSIONS; dir++) {
    if (data->mygrid->nproc[dir] > 1) {
      if (this->data->lbound[dir] == internal) {
        this->lbound[dir] = internalradiation;
      }
      if (this->data->rbound[dir] == internal) {
        this->rbound[dir] = internalradiation;
      }
    }
  }
#endif

#if GEOMETRY == SPHERICAL
  if ((this->rbound[JDIR] == axis) || (this->lbound[JDIR] == axis)) {
    // Check wether the x3 spherical axis is full two pi
    if (fabs((data->mygrid->xend[KDIR] - data->mygrid->xbeg[KDIR] - 2.0 * M_PI)) < 1e-10) {
      this->isTwoPi = true;
    }

#ifdef WITH_MPI
    // Check that there is no domain decomposition in phi
    if (data->mygrid->nproc[KDIR] > 1) {
      IDEFIX_ERROR(
          "FLD:: Axis boundaries are not compatible with "
          "MPI domain decomposition in X3");
    }
#endif
  }
#endif

// Init MPI stack when needed
#ifdef WITH_MPI
  this->arr4D = IdefixArray4D<real>("WorkingArrayMpi", 1, data->np_tot[KDIR], data->np_tot[JDIR],
                                    data->np_tot[IDIR]);

  int ntarget = 0;
  std::vector<int> mapVars;
  mapVars.push_back(ntarget);

  this->mpi.Init(data->mygrid, mapVars, this->nghost.data(), data->np_int.data());
#endif
}

// done
void FluxLimitedDiffusion::Init(Input& input, DataBlock* datain) {
  idfx::RegionWrapper region("FLD::Init");

  // Save the parents data objects
  this->data = datain;

  // Initialize (default) solver parameters
  this->dt = 0.;
  this->isPeriodic = true;
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

  // Enable preconditioner
  if (this->solver == PBICGSTAB || this->solver == PCG || this->solver == PMINRES) {
    this->havePreconditioner = true;
  }

  FLD_matrix = std::make_unique<FLDMatrix>(input, data);

  // Instantiate the bicgstab solver
  if (solver == BICGSTAB || solver == PBICGSTAB) {
    iterativeSolver = new Bicgstab<FLDMatrix>(*FLD_matrix.get(), targetError, maxiter, data->np_tot,
                                              data->beg, data->end);
  } else if (solver == CG || solver == PCG) {
    iterativeSolver = new Cg<FLDMatrix>(*FLD_matrix.get(), targetError, maxiter, data->np_tot,
                                        data->beg, data->end);
  } else if (solver == MINRES || solver == PMINRES) {
    iterativeSolver = new Minres<FLDMatrix>(*FLD_matrix.get(), targetError, maxiter, data->np_tot,
                                            data->beg, data->end);
  } else {
    IDEFIX_ERROR("FLD iterative solver not defined: should never reach here.");
  }

  // Arrays initialization
  this->rhs = IdefixArray3D<real>("RadiationRHS", data->np_tot[KDIR], data->np_tot[JDIR],
                                  data->np_tot[IDIR]);

  this->Erad =
      IdefixArray3D<real>("Erad", data->np_tot[KDIR], data->np_tot[JDIR], data->np_tot[IDIR]);

  this->nu = IdefixArray3D<real>("RadiationDiffusivity", data->np_tot[KDIR], data->np_tot[JDIR],
                                 data->np_tot[IDIR]);

  // Always allocate - we use this for the Radiation pressure source term, even if not
  // preconditioning.
  this->precond = IdefixArray3D<real>("Preconditioner", data->np_tot[KDIR], data->np_tot[JDIR],
                                      data->np_tot[IDIR]);
  // Fill arrays with 0

  auto Erad = this->Erad;
  auto rhs = this->rhs;
  auto P = this->precond;
  auto nu = this->nu;

  bool havePreconditioner = this->havePreconditioner;
  idefix_for(
      "InitRadiationArrays", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, data->np_tot[IDIR],
      KOKKOS_LAMBDA(int k, int j, int i) {
        Erad(k, j, i) = ZERO_F;
        rhs(k, j, i) = ZERO_F;
        nu(k, j, i) = ZERO_F;
        if (havePreconditioner) P(k, j, i) = ONE_F;
      });

  // Output radiation energy density
  data->dump->RegisterVariable(this->Erad, "ERAD");
  data->vtk->RegisterVariable(this->Erad, "ERAD");

  // Copy Boundary conditions here:
  isPeriodic = FLD_matrix->periodic;
  lbound = FLD_matrix->lbound;
  rbound = FLD_matrix->rbound;
  lvalue = FLD_matrix->lvalue;
  rvalue = FLD_matrix->rvalue;
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
  const int pad = this->havePreconditioner + 1;
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

void FluxLimitedDiffusion::PreconditionMatrix() {
  idfx::RegionWrapper region("Radiation::ApplyPreconditioner");

  int ibeg, iend, jbeg, jend, kbeg, kend;
  ibeg = data->beg[IDIR];
  iend = data->end[IDIR];
  jbeg = data->beg[JDIR];
  jend = data->end[JDIR];
  kbeg = data->beg[KDIR];
  kend = data->end[KDIR];
  IdefixArray4D<real> M = FLD_matrix->M;
  IdefixArray3D<real> P = this->precond;
  IdefixArray3D<real> rhs = this->rhs;

  // If preconditioning, we need Erad in the ghost cells.
  // The rest of the quantities don't matter.
  const int pad = this->havePreconditioner;
  D_EXPAND(ibeg -= pad; iend += pad; , jbeg -= pad; jend += pad; , kbeg -= pad; kend += pad;)

  idefix_for(
      "PreconditionM", kbeg, kend, jbeg, jend, ibeg, iend, KOKKOS_LAMBDA(int k, int j, int i) {
        // can't set M to one because it might be negative!
        M(0, k, j, i) /= P(k, j, i) * P(k, j, i);
        D_EXPAND(M(1, k, j, i) /= P(k, j, i) * P(k, j, i - 1);
                 M(2, k, j, i) /= P(k, j, i) * P(k, j, i + 1);
                 , M(3, k, j, i) /= P(k, j, i) * P(k, j - 1, i);
                 M(4, k, j, i) /= P(k, j, i) * P(k, j + 1, i);
                 , M(5, k, j, i) /= P(k, j, i) * P(k - 1, j, i);
                 M(6, k, j, i) /= P(k, j, i) * P(k + 1, j, i);)
        rhs(k, j, i) /= P(k, j, i);
      });
}

void FluxLimitedDiffusion::PreconditionErad(bool undo) {
  int ibeg, iend, jbeg, jend, kbeg, kend;
  ibeg = data->beg[IDIR];
  iend = data->end[IDIR];
  jbeg = data->beg[JDIR];
  jend = data->end[JDIR];
  kbeg = data->beg[KDIR];
  kend = data->end[KDIR];

  IdefixArray3D<real> P = this->precond;
  auto Erad = this->Erad;

  // If preconditioning, we need Erad in the ghost cells.
  // The rest of the quantities don't matter.
  const int pad = this->havePreconditioner;
  D_EXPAND(ibeg -= pad; iend += pad; , jbeg -= pad; jend += pad; , kbeg -= pad; kend += pad;)

  if (!undo) {
    idefix_for(
        "PreconditionE", kbeg, kend, jbeg, jend, ibeg, iend,
        KOKKOS_LAMBDA(int k, int j, int i) { Erad(k, j, i) *= P(k, j, i); });

  } else {
    idefix_for(
        "UndoPreconditionE", kbeg, kend, jbeg, jend, ibeg, iend,
        KOKKOS_LAMBDA(int k, int j, int i) { Erad(k, j, i) /= P(k, j, i); });
  }
}

// should be done
void FluxLimitedDiffusion::InitSolver() {
  idfx::RegionWrapper region("FLD::InitSolver");

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
  FLD_matrix->SetBoundaries(Erad, true);

  FillUtils();
  FillMatrixCouplingTerms();
  FillMatrixTransportTerms();
}

// done
void FLDMatrix::operator()(IdefixArray3D<real> array, IdefixArray3D<real> laplacian) {
  idfx::RegionWrapper region("FLD::ComputeLaplacian");

  int ibeg, iend, jbeg, jend, kbeg, kend;
  ibeg = data->beg[IDIR];
  iend = data->end[IDIR];
  jbeg = data->beg[JDIR];
  jend = data->end[JDIR];
  kbeg = data->beg[KDIR];
  kend = data->end[KDIR];

  IdefixArray4D<real> M = this->M;

  // Handling boundaries before laplacian calculation
  SetBoundaries(array, false);

  idefix_for(
      "FiniteDifference", kbeg, kend, jbeg, jend, ibeg, iend, KOKKOS_LAMBDA(int k, int j, int i) {
        real Delta =
            M(0, k, j, i) * array(k, j, i) +  // line doesn't end here, keep adding...
            D_EXPAND(+M(1, k, j, i) * array(k, j, i - 1) + M(2, k, j, i) * array(k, j, i + 1),
                     +M(3, k, j, i) * array(k, j - 1, i) + M(4, k, j, i) * array(k, j + 1, i),
                     +M(5, k, j, i) * array(k - 1, j, i) + M(6, k, j, i) * array(k + 1, j, i));

        laplacian(k, j, i) = Delta;
      });
}

// done
void FLDMatrix::EnforceBoundary(int dir, BoundarySide side, RadiationBoundaryType type,
                                IdefixArray3D<real>& arr, bool apply_physical) {
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
  const int ibeg = (dir == IDIR) ? side * (ighost + nxi) : 0;
  const int iend = (dir == IDIR) ? ighost + side * (ighost + nxi) : data->np_tot[IDIR];
  const int jbeg = (dir == JDIR) ? side * (jghost + nxj) : 0;
  const int jend = (dir == JDIR) ? jghost + side * (jghost + nxj) : data->np_tot[JDIR];
  const int kbeg = (dir == KDIR) ? side * (kghost + nxk) : 0;
  const int kend = (dir == KDIR) ? kghost + side * (kghost + nxk) : data->np_tot[KDIR];

  // Alex: we now use the matrix instead for anything except for periodic
  // ... but in the future we should support more (e.g., userdef)
  if (type != periodic && !apply_physical) {
    return;
  }

  switch (type) {
    case internalradiation:
      // internal is used for MPI-enforced boundary conditions. Nothing to be done here.
      break;

    case periodic: {
      if (data->mygrid->nproc[dir] > 1) break;  // Periodicity already enforced by MPI calls

      idefix_for(
          "BoundaryPeriodic", kbeg, kend, jbeg, jend, ibeg, iend,
          KOKKOS_LAMBDA(int k, int j, int i) {
            int iref, jref, kref;
            // This hack takes care of cases where we have more ghost zones than active zones
            if (dir == IDIR)
              iref = ighost + (i + ighost * (nxi - 1)) % nxi;
            else
              iref = i;
            if (dir == JDIR)
              jref = jghost + (j + jghost * (nxj - 1)) % nxj;
            else
              jref = j;
            if (dir == KDIR)
              kref = kghost + (k + kghost * (nxk - 1)) % nxk;
            else
              kref = k;

            localVar(k, j, i) = localVar(kref, jref, iref);
          });
      break;
    }

    case userdef: {
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

    case dirichlet: {
      real val = side == left ? this->lvalue[dir] : this->rvalue[dir];
      idefix_for(
          "BoundaryDirichlet", kbeg, kend, jbeg, jend, ibeg, iend,
          KOKKOS_LAMBDA(int k, int j, int i) { localVar(k, j, i) = val; });
      break;
    }

    case neumann: {
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

    case axis: {
      // Handling specific loop boundaries
      int jref, offset;
      if (side == left) {
        jref = data->beg[JDIR];
        offset = -1;
      }
      if (side == right) {
        jref = data->end[JDIR] - 1;
        offset = 1;
      }

      // NB: we assume no domain decomposition along phi here

      int np_int_k = data->np_int[KDIR];
      int nghost_k = data->nghost[KDIR];

      if (this->isTwoPi) {
        idefix_for(
            "BoundaryAxis", kbeg, kend, jbeg, jend, ibeg, iend, KOKKOS_LAMBDA(int k, int j, int i) {
              int kcomp = nghost_k + ((k - nghost_k + np_int_k / 2) % np_int_k);
              // Assuming sVc=1 for a scalar
              localVar(k, j, i) = localVar(kcomp, 2 * jref - j + offset, i);
            });
      } else {  // not 2pi
        idefix_for(
            "BoundaryAxis", kbeg, kend, jbeg, jend, ibeg, iend, KOKKOS_LAMBDA(int k, int j, int i) {
              // kcomp = k by construction since we're doing a fraction of twopi
              // Assuming sVc=1 for a scalar
              localVar(k, j, i) = localVar(k, 2 * jref - j + offset, i);
            });
      }
      break;
    }

    default: {
      std::stringstream msg("FLD:: Boundary condition type is not yet implemented");
      IDEFIX_ERROR(msg);
    }
  }
}

void FLDMatrix::EnrollUserDefBoundary(UserDefBoundaryFunc myFunc) {
  userDefBoundaryFunc = myFunc;
  haveUserDefBoundary = true;
}

void FluxLimitedDiffusion::EnrollUserDefBoundary(UserDefBoundaryFunc myFunc) {
  FLD_matrix->EnrollUserDefBoundary(myFunc);
  userDefBoundaryFunc = myFunc;
  haveUserDefBoundary = true;
  idfx::cout << "FLD:: User-defined boundary condition has been enrolled" << std::endl;
}

void FluxLimitedDiffusion::EnrollUserDefOpacityCGS(PrototypeOpacityFuncCGS myFunc) {
  if (!haveUserDefOpacity) {
    IDEFIX_ERROR(
        "FLD:: Attempt to enroll opacity function but a constant value has already been set");
  }
  UserOpacityFuncCGS = myFunc;
  idfx::cout << "FLD:: User-defined opacity function has been enrolled" << std::endl;
}

// done
void FLDMatrix::SetBoundaries(IdefixArray3D<real>& arr, bool apply_physical) {
  idfx::RegionWrapper region("FLD::SetBoundaries");

#ifdef WITH_MPI
  arr4D = IdefixArray4D<real>(arr.data(), 1, data->np_tot[KDIR], data->np_tot[JDIR],
                              data->np_tot[IDIR]);
#endif

  for (int dir = 0; dir < DIMENSIONS; dir++) {
// MPI Exchange data when needed
#ifdef WITH_MPI
    if (data->mygrid->nproc[dir] > 1) {
      switch (dir) {
        case 0:
          mpi.ExchangeX1(arr4D);
          break;
        case 1:
          mpi.ExchangeX2(arr4D);
          break;
        case 2:
          mpi.ExchangeX3(arr4D);
          break;
      }
    }
#endif

    EnforceBoundary(dir, left, lbound[dir], arr, apply_physical);
    EnforceBoundary(dir, right, rbound[dir], arr, apply_physical);
  }
}

// done
void FluxLimitedDiffusion::SolveSystem() {
  idfx::RegionWrapper region("FLD::SolveSystem");

  IdefixArray3D<real> Erad = this->Erad;
  Kokkos::Timer timer;

  elapsedTime -= timer.seconds();

  InitSolver();  // (Re)initialise the solver

  // if we're "preconditioning" (i.e., rescaling), do it before solving
  if (havePreconditioner) {
    PreconditionMatrix();
    PreconditionErad(false);
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
    if (havePreconditioner) PreconditionErad(false);

    // Try again !
    this->nsteps = iterativeSolver->Solve(this->Erad, rhs);
    if (this->nsteps < 0) {
      IDEFIX_ERROR("FLD:: BICGSTAB failed despite restart");
    }
  }

  currentError = iterativeSolver->GetError();

  // if we were "preconditioning", rescale back to normal
  if (havePreconditioner) {
    PreconditionErad(true);
  }

  FLD_matrix->SetBoundaries(Erad, true);
  UpdatePressure();
  elapsedTime += timer.seconds();
}

void FluxLimitedDiffusion::FillMatrixTransportTerms() {
  idfx::RegionWrapper region("FLD::FillMatrixTransportTerms");

  int ibeg, iend, jbeg, jend, kbeg, kend;
  ibeg = data->beg[IDIR];
  iend = data->end[IDIR];
  jbeg = data->beg[JDIR];
  jend = data->end[JDIR];
  kbeg = data->beg[KDIR];
  kend = data->end[KDIR];

  // If preconditioning, we need the diagonal matrix elements in the ghost cells.
  // See FillFluxLimiter for explanation.
  const int pad = this->havePreconditioner;
  D_EXPAND(ibeg -= pad; iend += pad; , jbeg -= pad; jend += pad; , kbeg -= pad; kend += pad;)

  IdefixArray4D<real> M = FLD_matrix->M;
  IdefixArray3D<real> rhs = this->rhs;
  IdefixArray3D<real> P = this->precond;
  IdefixArray3D<real> Erad = this->Erad;
  IdefixArray3D<real> nu = this->nu;
  IdefixArray3D<real> dV = data->dV;

  const real dt = this->dt;

  bool havePreconditioner = this->havePreconditioner;

  D_EXPAND(  // dimensions = 0, 1, 2
      IdefixArray1D<real> x1 = data->x[IDIR]; IdefixArray1D<real> x1l = data->xl[IDIR];
      IdefixArray1D<real> x1r = data->xr[IDIR]; IdefixArray1D<real> dx1 = data->dx[IDIR];
      IdefixArray3D<real> Ax1 = data->A[IDIR]; , IdefixArray1D<real> x2 = data->x[JDIR];
      IdefixArray1D<real> x2l = data->xl[JDIR]; IdefixArray1D<real> x2r = data->xr[JDIR];
      IdefixArray1D<real> dx2 = data->dx[JDIR]; IdefixArray3D<real> Ax2 = data->A[JDIR];
      , IdefixArray1D<real> x3 = data->x[KDIR]; IdefixArray1D<real> x3l = data->xl[KDIR];
      IdefixArray1D<real> x3r = data->xr[KDIR]; IdefixArray1D<real> dx3 = data->dx[KDIR];
      IdefixArray3D<real> Ax3 = data->A[KDIR];
#if GEOMETRY == SPHERICAL
      IdefixArray1D<real> sinth = data->sinx2;
#endif
  )

  real lvalue[3], rvalue[3];
  FLDMatrix::RadiationBoundaryType lbound[3], rbound[3];

  for (int dir = 0; dir < DIMENSIONS; dir++) {
    lbound[dir] = this->lbound[dir];
    rbound[dir] = this->rbound[dir];
    lvalue[dir] = this->lvalue[dir];
    rvalue[dir] = this->rvalue[dir];
  }

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

#ifndef SECOND_ORDER_FLUXES
        D_EXPAND(
            const real nu1m = lageval1(x1(i - 1), x1(i), nu(k, j, i - 1), nu(k, j, i), x1l(i));
            const real nu1p = lageval1(x1(i), x1(i + 1), nu(k, j, i), nu(k, j, i + 1), x1r(i));
            M(1, k, j, i) =
                -dt * 2. / h1 * nu1m * Ax1(k, j, i) / (dx1(i) + dx1(i - 1)) / dV(k, j, i);
            M(2, k, j, i) =
                -dt * 2. / h1 * nu1p * Ax1(k, j, i + 1) / (dx1(i) + dx1(i + 1)) / dV(k, j, i);
            M(0, k, j, i) -= M(1, k, j, i) + M(2, k, j, i);
            ,
            const real nu2m = lageval1(x2(j - 1), x2(j), nu(k, j - 1, i), nu(k, j, i), x2l(j));
            const real nu2p = lageval1(x2(j), x2(j + 1), nu(k, j, i), nu(k, j + 1, i), x2r(j));
            M(3, k, j, i) =
                -dt * 2. / h2 * nu2m * Ax2(k, j, i) / (dx2(j) + dx2(j - 1)) / dV(k, j, i);
            M(4, k, j, i) =
                -dt * 2. / h2 * nu2p * Ax2(k, j + 1, i) / (dx2(j) + dx2(j + 1)) / dV(k, j, i);
            M(0, k, j, i) -= M(3, k, j, i) + M(4, k, j, i);
            ,
            const real nu3m = lageval1(x3(k - 1), x3(k), nu(k - 1, j, i), nu(k, j, i), x3l(k));
            const real nu3p = lageval1(x3(k), x3(k + 1), nu(k, j, i), nu(k + 1, j, i), x3r(k));
            M(5, k, j, i) =
                -dt * 2. / h3 * nu3m * Ax3(k, j, i) / (dx3(k) + dx3(k - 1)) / dV(k, j, i);
            M(6, k, j, i) =
                -dt * 2. / h3 * nu3p * Ax3(k + 1, j, i) / (dx3(k) + dx3(k + 1)) / dV(k, j, i);
            M(0, k, j, i) -= M(5, k, j, i) + M(6, k, j, i);
          )
#else
D_EXPAND(
        real nudx1m = (2./h1) / (dx1(i)/nu(k,j,i) + dx1(i-1)/nu(k,j,i-1));
        real nudx1p = (2./h1) / (dx1(i)/nu(k,j,i) + dx1(i+1)/nu(k,j,i+1));
        M(1,k,j,i) = -dt * nudx1m * Ax1(k,j,i) / dV(k,j,i);
        M(2,k,j,i) = -dt * nudx1p * Ax1(k,j,i+1) / dV(k,j,i);
        M(0,k,j,i) -= M(1,k,j,i) + M(2,k,j,i);
      ,
        real nudx2m = (2./h2) / (dx2(j)/nu(k,j,i) + dx2(j-1)/nu(k,j-1,i));
        real nudx2p = (2./h2) / (dx2(j)/nu(k,j,i) + dx2(j+1)/nu(k,j+1,i));
        M(3,k,j,i) = -dt * nudx2m * Ax2(k,j,i) / dV(k,j,i);
        M(4,k,j,i) = -dt * nudx2p * Ax2(k,j+1,i) / dV(k,j,i);
        M(0,k,j,i) -= M(3,k,j,i) + M(4,k,j,i);
      ,
        real nudx3m = (2./h3) / (dx3(k)/nu(k,j,i) + dx3(k-1)/nu(k-1,j,i));
        real nudx3p = (2./h3) / (dx3(k)/nu(k,j,i) + dx3(k+1)/nu(k+1,j,i));
        M(5,k,j,i) = -dt * nudx3m * Ax3(k,j,i) / dV(k,j,i);
        M(6,k,j,i) = -dt * nudx3p * Ax3(k+1,j,i) / dV(k,j,i);
        M(0,k,j,i) -= M(5,k,j,i) + M(6,k,j,i);
      )
#endif
        // now handle boundary conditions
        real delta;
        D_EXPAND(
            if (i == ibeg + pad) {
              switch (lbound[0]) {
                case FLDMatrix::RadiationBoundaryType::dirichlet:
                  delta = dx1(i - 1) * nu(k, j, i) / (dx1(i) * nu(k, j, i - 1));
                  M(0, k, j, i) -= M(1, k, j, i) * delta;
                  rhs(k, j, i) -= Erad(k, j, i - 1) * M(1, k, j, i) * (1 + delta);
                  M(1, k, j, i) = 0;
                  break;
                case FLDMatrix::RadiationBoundaryType::neumann:
                  M(0, k, j, i) += M(1, k, j, i);
                  real dx_nu = 0.5 * dx1(i - 1) / nu(k, j, i - 1) + 0.5 * dx1(i) / nu(k, j, i);
                  rhs(k, j, i) += lvalue[0] * M(1, k, j, i) * dx_nu;
                  M(1, k, j, i) = 0;
                  break;
              }
            } else if (i == iend - 1 - pad) {
              switch (rbound[0]) {
                case FLDMatrix::RadiationBoundaryType::dirichlet:
                  delta = dx1(i + 1) * nu(k, j, i) / (dx1(i) * nu(k, j, i + 1));
                  M(0, k, j, i) -= M(2, k, j, i) * delta;
                  rhs(k, j, i) -= Erad(k, j, i + 1) * M(2, k, j, i) * (1 + delta);
                  M(2, k, j, i) = 0;
                  break;
                case FLDMatrix::RadiationBoundaryType::neumann:
                  M(0, k, j, i) += M(2, k, j, i);
                  real dx_nu = 0.5 * dx1(i + 1) / nu(k, j, i + 1) + 0.5 * dx1(i) / nu(k, j, i);
                  rhs(k, j, i) += rvalue[0] * M(2, k, j, i) * dx_nu;
                  M(2, k, j, i) = 0;
                  break;
              }
            },
            if (j == jbeg + pad) {
              switch (lbound[1]) {
                case FLDMatrix::RadiationBoundaryType::dirichlet:
                  delta = dx2(j - 1) * nu(k, j, i) / (dx2(j) * nu(k, j - 1, i));
                  M(0, k, j, i) -= M(3, k, j, i) * delta;
                  rhs(k, j, i) -= Erad(k, j - 1, i) * M(3, k, j, i) * (1 + delta);
                  M(3, k, j, i) = 0;
                  break;
                case FLDMatrix::RadiationBoundaryType::neumann:
                  M(0, k, j, i) += M(3, k, j, i);
                  const real dx_nu =
                      0.5 * dx2(j - 1) / nu(k, j - 1, i) + 0.5 * dx2(j) / nu(k, j, i);
                  rhs(k, j, i) += lvalue[1] * M(3, k, j, i) * dx_nu;
                  M(3, k, j, i) = 0;
                  break;
              }
            } else if (j == jend - 1 - pad) {
              switch (rbound[1]) {
                case FLDMatrix::RadiationBoundaryType::dirichlet:
                  delta = dx2(j + 1) * nu(k, j, i) / (dx2(j) * nu(k, j + 1, i));
                  M(0, k, j, i) -= M(4, k, j, i) * delta;
                  rhs(k, j, i) -= Erad(k, j + 1, i) * M(4, k, j, i) * (1 + delta);
                  M(4, k, j, i) = 0;
                  break;
                case FLDMatrix::RadiationBoundaryType::neumann:
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
                case FLDMatrix::RadiationBoundaryType::dirichlet:
                  delta = dx3(k - 1) * nu(k, j, i) / (dx3(k) * nu(k - 1, j, i));
                  M(0, k, j, i) -= M(5, k, j, i) * delta;
                  rhs(k, j, i) -= Erad(k - 1, j, i) * M(5, k, j, i) * (1 + delta);
                  M(5, k, j, i) = 0;
                  break;
                case FLDMatrix::RadiationBoundaryType::neumann:
                  const real dx_nu =
                      0.5 * dx3(k - 1) / nu(k - 1, j, i) + 0.5 * dx3(k) / nu(k, j, i);
                  rhs(k, j, i) += lvalue[2] * M(5, k, j, i) * dx_nu;
                  M(0, k, j, i) += M(5, k, j, i);
                  M(5, k, j, i) = 0;
                  break;
              }
            } else if (k == kend - 1 - pad) {
              switch (rbound[2]) {
                case FLDMatrix::RadiationBoundaryType::dirichlet:
                  delta = dx3(k + 1) * nu(k, j, i) / (dx3(k) * nu(k + 1, j, i));
                  M(0, k, j, i) -= M(6, k, j, i) * delta;
                  rhs(k, j, i) -= Erad(k + 1, j, i) * M(6, k, j, i) * (1 + delta);
                  M(6, k, j, i) = 0;
                  break;
                case FLDMatrix::RadiationBoundaryType::neumann:
                  M(0, k, j, i) += M(6, k, j, i);
                  const real dx_nu =
                      0.5 * dx3(k + 1) / nu(k + 1, j, i) + 0.5 * dx3(k) / nu(k, j, i);
                  rhs(k, j, i) += rvalue[2] * M(6, k, j, i) * dx_nu;
                  M(6, k, j, i) = 0;
                  break;
              }
            })

        if (havePreconditioner) P(k, j, i) = sqrt(fabs(M(0, k, j, i)));
      });
}

void FluxLimitedDiffusion::_ComputeRadiationPressureSourceTerm(real dt, int species) {
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
    data->radiation->UserOpacityFuncCGS(data, species, this->precond, kappaR);
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

      D_EXPAND(
        Uc(VX1, k, j, i) += dt_over_c * (flux1m + flux1p) * Vc(RHO, k, j, i) * kappaR(k, j, i); ,
        Uc(VX2, k, j, i) += dt_over_c * (flux2m + flux2p) * Vc(RHO, k, j, i) * kappaR(k, j, i); ,
        Uc(VX3, k, j, i) += dt_over_c * (flux3m + flux3p) * Vc(RHO, k, j, i) * kappaR(k, j, i);
      )
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
  auto nu = this->nu;
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

  IdefixArray3D<real> Erad = this->Erad;  // Radiation energy
  IdefixArray4D<real> radY = this->radY;  // κP ρ c dt -> dimensionless
  IdefixArray4D<real> radX = this->radX;  // aR T^3 radY / (ρ cV) -> dimensionless
  IdefixArray3D<real> nu = this->nu;      // λ c / (kR ρ) -> diffusivity
  IdefixArray3D<real> rhs = this->rhs;    // Right hand side -> same units as Erad

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
  const int pad = this->havePreconditioner + 1;
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
  const int pad = this->havePreconditioner;
  D_EXPAND(ibeg -= pad; iend += pad; , jbeg -= pad; jend += pad; , kbeg -= pad; kend += pad;)

  IdefixArray4D<real> M = FLD_matrix->M;
  IdefixArray4D<real> radY = this->radY;
  IdefixArray4D<real> radX = this->radX;

  idefix_for(
      "FiniteDifference", kbeg, kend, jbeg, jend, ibeg, iend, KOKKOS_LAMBDA(int k, int j, int i) {
        // Coupling coefficients and matrix elements
        M(0, k, j, i) = 1 + radY(0, k, j, i) / (1 + 4 * radX(0, k, j, i));
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

  // idfx::cout << "FLD: target L2 norm error=" << targetError << "." << std::endl;

  // The setup is periodic if it passes the previous boundary loading
  if (this->isPeriodic == true) {
    idfx::cout << "FLD: Setup is periodic, doing nothing." << std::endl;
  }

  iterativeSolver->ShowConfig();
}

void TwoTemperatureFLD::ShowConfig() {
  idfx::cout << "RT: TwoTemperatureFLD with " << num_species << " species.\n";
  FluxLimitedDiffusion::ShowConfig();
}
