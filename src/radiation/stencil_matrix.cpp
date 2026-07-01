// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

// Module contributed by Alex Ziampras, then at Queen Mary University of London
// Modified by Richard Booth, then at University of Leeds

#include "stencil_matrix.hpp"

#include "boundary_utils.hpp"
#include "dataBlock.hpp"

void EnforcePeriodicBoundary(DataBlock*, int, BoundarySide, bool, bool, IdefixArray3D<real>);

StencilMatrix::StencilMatrix(DataBlock* data, std::array<bool, 3> isPeriodic)
    : data(data), isPeriodic(isPeriodic) {

  // Fill the memory:
  M = IdefixArray4D<real>("M", 1 + 2 * DIMENSIONS,
                              data->np_tot[KDIR], data->np_tot[JDIR], data->np_tot[IDIR]);

  idefix_for("Zero Matrix", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, data->np_tot[IDIR],
    KOKKOS_LAMBDA(int k, int j, int i) {
      for (int d = 0; d < 1 + 2 * DIMENSIONS; d++) {
        M(d, k, j, i) = ZERO_F;
      }
  });

#ifdef WITH_MPI
  mpi.Init(data, {0}, data->nghost.data(), data->np_int.data());
#endif
}

void StencilMatrix::operator()(IdefixArray3D<real> in, IdefixArray3D<real> out) {
  idfx::RegionWrapper region("StencilMatrix::operator()");

  int ibeg = data->beg[IDIR];
  int iend = data->end[IDIR];
  int jbeg = data->beg[JDIR];
  int jend = data->end[JDIR];
  int kbeg = data->beg[KDIR];
  int kend = data->end[KDIR];

  IdefixArray4D<real> M = this->M;

  SetBoundaries(in);

  idefix_for("operator()", kbeg, kend, jbeg, jend, ibeg, iend,
      KOKKOS_LAMBDA(int k, int j, int i) {
          out(k,j,i) = M(0, k, j, i) * in(k,j,i)
              D_EXPAND(
                  + M(1, k, j, i)* in(k, j, i - 1) + M(2, k, j, i) * in(k, j, i + 1),
                  + M(3, k, j, i)* in(k, j - 1, i) + M(4, k, j, i) * in(k, j + 1, i),
                  + M(5, k, j, i)* in(k - 1, j, i) + M(6, k, j, i) * in(k + 1, j, i));
      });
}


void StencilMatrix::ComputeResidual(const IdefixArray3D<real> x, const IdefixArray3D<real> b,
                                    IdefixArray3D<real> r) {
  idfx::RegionWrapper region("StencilMatrix::ComputeResidual");

  int ibeg = data->beg[IDIR];
  int iend = data->end[IDIR];
  int jbeg = data->beg[JDIR];
  int jend = data->end[JDIR];
  int kbeg = data->beg[KDIR];
  int kend = data->end[KDIR];

  IdefixArray4D<real> M = this->M;

  idefix_for("residual", kbeg, kend, jbeg, jend, ibeg, iend,
      KOKKOS_LAMBDA(int k, int j, int i) {
          r(k,j,i) = b(k,j,i) - M(0, k, j, i) * x(k,j,i)
              D_EXPAND(
                  - M(1, k, j, i)* x(k, j, i - 1) - M(2, k, j, i) * x(k, j, i + 1),
                  - M(3, k, j, i)* x(k, j - 1, i) - M(4, k, j, i) * x(k, j + 1, i),
                  - M(5, k, j, i)* x(k - 1, j, i) - M(6, k, j, i) * x(k + 1, j, i));
      });

  SetBoundaries(r);
}


void StencilMatrix::Reset() {
    idfx::RegionWrapper region("StencilMatrix::Reset");

    isScaled = false;
}

void StencilMatrix::enablePreconditioner() {
  idfx::RegionWrapper region("StencilMatrix::enablePreconditioner");

  if (!havePreconditioner) {
    havePreconditioner = true;
    precond = IdefixArray3D<real>("precond",
                                  data->np_tot[KDIR], data->np_tot[JDIR], data->np_tot[IDIR]);
  }
}

void StencilMatrix::ScaleMatrix() {
  idfx::RegionWrapper region("StencilMatrix::ScaleMatrix");

  if (isScaled) {
    IDEFIX_ERROR("StencilMatrix::ScaleMatrix: Matrix is already scaled to unit diagonal\n");
  }

  if (!havePreconditioner) {
    enablePreconditioner();
  }

  int ibeg = data->beg[IDIR];
  int iend = data->end[IDIR];
  int jbeg = data->beg[JDIR];
  int jend = data->end[JDIR];
  int kbeg = data->beg[KDIR];
  int kend = data->end[KDIR];

  IdefixArray4D<real> M = this->M;
  IdefixArray3D<real> P = this->precond;

  // If preconditioning, we need Erad in the ghost cells.
  // The rest of the quantities don't matter.
  const int pad = havePreconditioner;
  D_EXPAND(ibeg -= pad; iend += pad; , jbeg -= pad; jend += pad; , kbeg -= pad; kend += pad;)

  idefix_for("SetPreconditioner", kbeg, kend, jbeg, jend, ibeg, iend,
    KOKKOS_LAMBDA(int k, int j, int i) {
      P(k, j, i) = sqrt(fabs(M(0, k, j, i)));
    });

  idefix_for(
    "PreconditionM", kbeg, kend, jbeg, jend, ibeg, iend,
    KOKKOS_LAMBDA(int k, int j, int i) {
      M(0, k, j, i) /= P(k, j, i) * P(k, j, i);   // M(0,:) might be negative!
      D_EXPAND(M(1, k, j, i) /= P(k, j, i) * P(k, j, i - 1);
               M(2, k, j, i) /= P(k, j, i) * P(k, j, i + 1); ,
               M(3, k, j, i) /= P(k, j, i) * P(k, j - 1, i);
               M(4, k, j, i) /= P(k, j, i) * P(k, j + 1, i); ,
               M(5, k, j, i) /= P(k, j, i) * P(k - 1, j, i);
               M(6, k, j, i) /= P(k, j, i) * P(k + 1, j, i);)
    });

  isScaled = true;
}

void StencilMatrix::ScaleSolution(IdefixArray3D<real> sol, bool undo) {
  idfx::RegionWrapper region("StencilMatrix::ScaleSolution");

  if (!isScaled) {
    return; // Nothing to do
  }

  int ibeg = data->beg[IDIR];
  int iend = data->end[IDIR];
  int jbeg = data->beg[JDIR];
  int jend = data->end[JDIR];
  int kbeg = data->beg[KDIR];
  int kend = data->end[KDIR];

  IdefixArray3D<real> P = this->precond;

  // If preconditioning, we need Erad in the ghost cells.
  // The rest of the quantities don't matter.
  const int pad = havePreconditioner;
  D_EXPAND(ibeg -= pad; iend += pad; , jbeg -= pad; jend += pad; , kbeg -= pad; kend += pad;)

  idefix_for(
    "ScaleSol", kbeg, kend, jbeg, jend, ibeg, iend,
    KOKKOS_LAMBDA(int k, int j, int i) {
      if (!undo) {
        sol(k,j,i) *= P(k,j,i);
      } else {
        sol(k,j,i) /= P(k,j,i);
      }
  });
}

void StencilMatrix::ScaleSolAndRhs(IdefixArray3D<real> sol, IdefixArray3D<real> rhs, bool undo) {
  idfx::RegionWrapper region("StencilMatrix::ScaleSolAndRhs");

  if (!isScaled) {
    return; // Nothing to do
  }

  int ibeg = data->beg[IDIR];
  int iend = data->end[IDIR];
  int jbeg = data->beg[JDIR];
  int jend = data->end[JDIR];
  int kbeg = data->beg[KDIR];
  int kend = data->end[KDIR];

  IdefixArray3D<real> P = this->precond;

  // If preconditioning, we need Erad in the ghost cells.
  // The rest of the quantities don't matter.
  const int pad = havePreconditioner;
  D_EXPAND(ibeg -= pad; iend += pad; , jbeg -= pad; jend += pad; , kbeg -= pad; kend += pad;)

  idefix_for(
    "ScaleSolAndRhs", kbeg, kend, jbeg, jend, ibeg, iend,
    KOKKOS_LAMBDA(int k, int j, int i) {
      if (!undo) {
        sol(k,j,i) *= P(k,j,i);
        rhs(k,j,i) /= P(k,j,i);
      } else {
        sol(k,j,i) /= P(k,j,i);
        rhs(k,j,i) *= P(k,j,i);
      }
  });
}

void StencilMatrix::SetBoundaries(IdefixArray3D<real> arr) {
  ApplyBoundariesInternal(arr, true);
}

void StencilMatrix::ApplyPeriodicAxisBoundariesOnly(IdefixArray3D<real> arr) {
  ApplyBoundariesInternal(arr, false);
}

void StencilMatrix::ApplyBoundariesInternal(IdefixArray3D<real> arr, bool zeroPhysical) {
  idfx::RegionWrapper region("MG_Grid::SetBoundaries");

#ifdef WITH_MPI
  IdefixArray4D<real> arr4D = IdefixArray4D<real>(
    arr.data(), 1, data->np_tot[KDIR], data->np_tot[JDIR],data->np_tot[IDIR]
  );
#endif

  ApplyBoundaryByDirection(
      data,
      [&](int dir) {
        // MPI Exchange data when needed
  #ifdef WITH_MPI
    MPIBoundaryExchange(data, mpi, arr4D, dir);
  #else
    (void)dir;
  #endif
      },
      [&](int dir, BoundarySide side) {
        // Enforce periodic boundaries if not handled by MPI.
        // Zero's out non-periodic boundaries when requested.
        EnforcePeriodicBoundary(data, dir, side, isPeriodic[dir], zeroPhysical, arr);
      });
}

void StencilMatrix::PrintMatrix() {
  idfx::RegionWrapper region("StencilMatrix::PrintMatrix");

  int ibeg = data->beg[IDIR];
  int iend = data->end[IDIR];
  int jbeg = data->beg[JDIR];
  int jend = data->end[JDIR];
  int kbeg = data->beg[KDIR];
  int kend = data->end[KDIR];

  idfx::cout << "Matrix:" << "\n";
  for (int k = kbeg; k < kend; k++) {
    for (int j = jbeg; j < jend; j++) {
      for (int i = ibeg; i < iend; i++) {
        idfx::cout << "   " << i << " " << j << " " << k << ",";
        real tot = 0;
        for (int l=0; l < 1 + 2*DIMENSIONS; l++) {
          idfx::cout << " " << M(l, k, j, i);
          tot += M(l, k, j, i);
        }
        idfx::cout << ", " << tot << "\n";
      }
    }
  }
}


void EnforcePeriodicBoundary(DataBlock* data, int dir, BoundarySide side,
                             bool isPeriodic, bool zeroPhysical, IdefixArray3D<real> arr) {
  idfx::RegionWrapper region("EnforcePeriodicBoundary");

  IdefixArray3D<real> localVar = arr;

  BoundaryLoopRange range = ComputeBoundaryLoopRange(data, dir, side);
  const int ibeg = range.ibeg;
  const int iend = range.iend;
  const int jbeg = range.jbeg;
  const int jend = range.jend;
  const int kbeg = range.kbeg;
  const int kend = range.kend;

  const bool isAxisBoundary =
      (side == left) ? (data->lbound[dir] == BoundaryType::axis)
                     : (data->rbound[dir] == BoundaryType::axis);


  if (isPeriodic) {
    if (data->mygrid->nproc[dir] > 1) {
      return;  // Periodicity already enforced by MPI calls
    }
    ApplyPeriodicBoundary(data, dir, side, arr);
  } else if (isAxisBoundary && dir == JDIR) {
    // Mirror data across the spherical axis (same convention as FLD boundaries).
    bool isTwoPi = IsFullTwoPiDomain(data);
    ApplyAxisBoundary(data, side, range, arr, isTwoPi);
  } else {
    // Zero-out physical, non-periodic boundaries
    if (zeroPhysical &&
        ((data->mygrid->xproc[dir] == 0 && side == left) ||
         (data->mygrid->xproc[dir] == data->mygrid->nproc[dir] - 1 && side == right))) {
      idefix_for("BoundaryZero", kbeg, kend, jbeg, jend, ibeg, iend,
        KOKKOS_LAMBDA(int k, int j, int i) {
          localVar(k, j, i) = 0;
        });
    }
  }
}
