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

#ifndef RADIATION_FLD_HPP_
#define RADIATION_FLD_HPP_

#include <memory>
#include <vector>

#include "fluid_defs.hpp"
#include "grid.hpp"
#include "idefix.hpp"
#include "input.hpp"
#include "iterativesolver.hpp"

#ifdef WITH_MPI
#include "mpi.hpp"
#endif

// Forward class declarations
class DataBlock;
class FluxLimitedDiffusion;
class TwoTemperatureFLD;
class MultiSpeciesFLD;

class FLDMatrix {
 public:
  friend class FluxLimitedDiffusion;
  friend class TwoTemperatureFLD;
  friend class MultiSpeciesFLD;

  // Types of boundary which can be treated
  enum RadiationBoundaryType { periodic, dirichlet, neumann, userdef, internalradiation, axis };

  // userdef boundary.
  using UserDefBoundaryFunc = void (*)(DataBlock&, int dir, BoundarySide side, const real t,
                                       IdefixArray3D<real>& arr);

  FLDMatrix() = default;
  FLDMatrix(Input& input, DataBlock*);

  // Enforce boundary conditions in a specific dir, side, following boundary type
  // and for the specified array
  void EnforceBoundary(int dir, BoundarySide side, RadiationBoundaryType type, IdefixArray3D<real>&,
                       bool apply_physical);
  void SetBoundaries(IdefixArray3D<real>&,
                     bool apply_physical = false);  // Set the proper boundaries for the given array

  void EnrollUserDefBoundary(UserDefBoundaryFunc);  // Enroll user-defined boundary conditions

  // User defined Boundary conditions
  bool haveUserDefBoundary{false};
  // User defined opacities and flux limiter
  UserDefBoundaryFunc userDefBoundaryFunc{NULL};
  bool haveUserDefOpacity{false};

  // The main laplacian operator
  void operator()(IdefixArray3D<real> in, IdefixArray3D<real> laplacian);

 private:
  DataBlock* data;        // My parent data object
  IdefixArray4D<real> M;  // Matrix for the radiation solver

  bool isPeriodic;  // Periodicity status of the density distribution // Alex: What do I need this
                    // for?
  std::array<RadiationBoundaryType, 3> lbound;  // Boundary condition to the left
  std::array<RadiationBoundaryType, 3> rbound;  // Boundary condition to the right
  std::array<real, 3> lvalue;                   // Boundary value to the left
  std::array<real, 3> rvalue;                   // Boundary value to the right

  bool isTwoPi{false};

#ifdef WITH_MPI
  MPI_Comm originComm;        ///< MPI communicator used by the origin boundary condition
  IdefixArray4D<real> arr4D;  // Intermediate array for boundary handling
#endif
};

class FluxLimitedDiffusion {
 public:
  enum RadiationSolver { JACOBI, BICGSTAB, PBICGSTAB, PCG, CG, PMINRES, MINRES };
  enum OpacityType { constantkappa, userdefkappa };
  enum FluxLimiterType {
    one_third,
    Minerbo1978,
    LevermorePomraning1981,
    Kley1989,
    userdeffluxlimiter
  };

  // Handling userdef boundary & opacities
  using UserDefBoundaryFunc = void (*)(DataBlock&, int dir, BoundarySide side, const real t,
                                       IdefixArray3D<real>& arr);
  using PrototypeOpacityFuncCGS = void (*)(DataBlock* data, int species, IdefixArray3D<real> kappaP,
                                           IdefixArray3D<real> kappaR);

  FluxLimitedDiffusion(Input&, DataBlock*);

  void Init(Input&, DataBlock*);  // Initialisation of the class attributes
  virtual void ShowConfig();      // display current configuration
  void InitSolver();  // (Re)initialisation of the solver for a given density distribution
  virtual void FillUtils() = 0;
  virtual void FillMatrixCouplingTerms() = 0;

  void PreconditionMatrix();  // For preconditioning versions
  void PreconditionErad(bool undo = false);

  void FillFluxLimiter(IdefixArray4D<real> rho_kappaR, IdefixArray3D<real> lambda);
  void FillMatrixTransportTerms();

  virtual void UpdatePressure() = 0;  // Update pressure with new radiation field
  void SolveSystem();                 // Solve Radiation equation

  template <class Phys, bool update_Vc = false>
  void ComputeRadiationPressureSourceTerm(Fluid<Phys>* fluid, real t, real dt) {
    if (!Phys::dust) {
      _ComputeRadiationPressureSourceTerm(dt, 0, update_Vc);
    } else {
      _ComputeRadiationPressureSourceTerm(dt, fluid->instanceNumber + 1, update_Vc);
    }
  }

  void EnrollUserDefBoundary(UserDefBoundaryFunc);  // Enroll user-defined boundary conditions

  // User defined Boundary conditions
  UserDefBoundaryFunc userDefBoundaryFunc{NULL};
  bool haveUserDefBoundary{false};

  // User defined opacities and flux limiter
  void EnrollUserDefOpacityCGS(PrototypeOpacityFuncCGS);
  PrototypeOpacityFuncCGS UserOpacityFuncCGS{NULL};
  bool haveUserDefOpacity{false};

  IdefixArray1D<real> constkappaR;  // constant Rosseland opacity
  IdefixArray1D<real> constkappaP;  // constant Planck opacity

  IdefixArray3D<real> Erad;  // Radiation energy

#ifdef WITH_MPI
  Mpi mpi;  // Mpi object when WITH_MPI is set
#endif

  real currentError{0};  // last error of the iterative solver
  int nsteps{0};         // # of steps of the latest iteration
  double elapsedTime;    // time spent solving radiation

  real mu{2.353};         // mean molecular weight
  real gamma{5.0 / 3.0};  // adiabatic index
  real code_c;            // speed of light in code units
  real code_aR;           // radiation constant in code units
  real cV;
  // some units
  real unit_opacity;

  void _ComputeRadiationPressureSourceTerm(real dt, int species, bool update_Vc);

 protected:
  DataBlock* data;              // My parent data object
  IdefixArray3D<real> precond;  // Diagonal preconditioner
  IdefixArray3D<real> rhs;      // Right hand side -> same units as Erad
  IdefixArray3D<real> nu;       // λ c dt / (kR ρ) -> diffusivity

  real dt;  // CFL timestep

  IterativeSolver<FLDMatrix>* iterativeSolver;
  std::unique_ptr<FLDMatrix> FLD_matrix;

  bool isPeriodic;  // Periodicity status of the density distribution // Alex: What do I need this
                    // for?
  std::array<FLDMatrix::RadiationBoundaryType, 3> lbound;  // Boundary condition to the left
  std::array<FLDMatrix::RadiationBoundaryType, 3> rbound;  // Boundary condition to the right
  std::array<real, 3> lvalue;                              // Boundary value to the left
  std::array<real, 3> rvalue;                              // Boundary value to the right
                               // Warning : might differ from (M)HD solver !

  RadiationSolver solver;           // The solver  used to solve Poisson
  OpacityType opacityType;          // Type of opacity
  FluxLimiterType fluxLimiterType;  // Type of flux limiter

  bool havePreconditioner{false};        // Use of preconditioner (or not)
  bool haveInitialisedRadiation{false};  // whether the radiation field has already been initialised
  int num_species{1};
};

class TwoTemperatureFLD : public FluxLimitedDiffusion {
 public:
  TwoTemperatureFLD(Input&, DataBlock*);

  void Init(Input&, DataBlock*);  // Initialisation of the class attributes
  void ShowConfig();              // display current configuration

  void FillUtils();                // fill utility arrays and rhs
  void FillMatrixCouplingTerms();  // fill matrix for the radiation solver

  void UpdatePressure();  // Update pressure with new radiation field

 private:
  IdefixArray4D<real> radY;  // κP ρ c dt -> dimensionless
  IdefixArray4D<real> radX;  // aR T^3 radY / (ρ cV) -> dimensionless
};

class MultiSpeciesFLD : public FluxLimitedDiffusion {
 public:
  MultiSpeciesFLD(Input&, DataBlock*);

  void Init(Input&, DataBlock*);  // Initialisation of the class attributes
  void ShowConfig();              // display current configuration

  void FillUtils();                // fill utility arrays and rhs
  void FillMatrixCouplingTerms();  // fill matrix for the radiation solver

  void UpdatePressure();  // Update pressure with new radiation field

 private:
  IdefixArray4D<real> radZ, radGamma;               //
  IdefixArray4D<real> kappaP, kappaR;               //
  IdefixArray3D<real> k_eff, cV_eff, u_eff, l_eff;  //
};

#endif  // RADIATION_FLD_HPP_
