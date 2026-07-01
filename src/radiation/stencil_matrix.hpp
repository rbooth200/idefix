// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

// Module contributed by Alex Ziampras, then at Queen Mary University of London
// Modified by Richard Booth, then at University of Leeds

#ifndef RADIATION_STENCIL_MATRIX_HPP_
#define RADIATION_STENCIL_MATRIX_HPP_

#include <array>

#include "idefix.hpp"

class DataBlock; // Fwd declaration

/* class StencilMatrix
  Wrapper class for a matrix operator with a +-shaped stencil.

  The matrix is stored in a 4D array, with the first dimension corresponding to the
  diagonal and off-diagonal elements of the matrix. The remaining three dimensions
  correspond to the spatial dimensions of the grid. The matrix is applied to a 3D
  array using the operator() function, which computes the matrix-vector product.

  Physical boundary conditions (neumann, dirichlet, etc) must be set via the matrix elements.
  SetBoundaries applies internal (MPI) exchanges plus periodic/axis boundary fills, and
  zeroes physical boundaries - appropriate for residual equations.
  ApplyPeriodicAxisBoundariesOnly keeps physical boundaries unchanged while still applying
  internal and periodic/axis boundaries, which is useful creating multigrid operators.

  The matrix can be scaled (preconditioned) to unit diagonal using the ScaleMatrix function, and
  the right-hand side and solution can be scaled accordingly using the ScaleSolution and
  ScaleSolAndRhs functions.
*/
class StencilMatrix {
 public:
  StencilMatrix() = default;
  StencilMatrix(DataBlock* data, std::array<bool, 3> isPeriodic={false, false, false});

  void SetBoundaries(IdefixArray3D<real> arr);
  void ApplyPeriodicAxisBoundariesOnly(IdefixArray3D<real> arr);
  void operator()(IdefixArray3D<real> in, IdefixArray3D<real> out);
  void ComputeResidual(const IdefixArray3D<real> x, const IdefixArray3D<real> b,
                       IdefixArray3D<real> r);

  void Reset(); // Call before setting a new matrix.

  void enablePreconditioner();  // Enable preconditioning to unit diagonal
  void ScaleMatrix();
  void ScaleSolution(IdefixArray3D<real> sol, bool undo=false);
  void ScaleSolAndRhs(IdefixArray3D<real> sol, IdefixArray3D<real> rhs, bool undo=false);

  void PrintMatrix();  // For debugging purposes

  IdefixArray4D<real> M;   // matrix

  bool havePreconditioner{false};  // Whether to precondition the matrix to unit diagonal
  IdefixArray3D<real> precond;     // Preconditioner

  void SetPeriodic(std::array<bool, 3> isPeriodic) { this->isPeriodic = isPeriodic; }
  std::array<bool, 3> isPeriodic;  // Whether the boundary is periodic in each direction


 protected:
  DataBlock* data;                 // My parent data object


 private:
  void ApplyBoundariesInternal(IdefixArray3D<real> arr, bool zeroPhysical);
  bool isScaled{false};                // Whether the matrix has been scaled to unit diagonal

#ifdef WITH_MPI
  Mpi mpi;        ///< MPI communicatior
#endif
};

#endif  // RADIATION_STENCIL_MATRIX_HPP_
