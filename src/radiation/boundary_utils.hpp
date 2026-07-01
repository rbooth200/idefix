// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef RADIATION_BOUNDARY_UTILS_HPP_
#define RADIATION_BOUNDARY_UTILS_HPP_

#include "dataBlock.hpp"

struct BoundaryLoopRange {
  int ibeg;
  int iend;
  int jbeg;
  int jend;
  int kbeg;
  int kend;
};

template <typename ExchangeDirFunc, typename ApplySideFunc>
inline void ApplyBoundaryByDirection(DataBlock* data, ExchangeDirFunc exchangeDir,
                                     ApplySideFunc applySide) {
  for (int dir = 0; dir < DIMENSIONS; dir++) {
    exchangeDir(dir);
    applySide(dir, left);
    applySide(dir, right);
  }
}

inline BoundaryLoopRange ComputeBoundaryLoopRange(DataBlock* data, int dir, BoundarySide side) {
  const int nxi = data->np_int[IDIR];
  const int nxj = data->np_int[JDIR];
  const int nxk = data->np_int[KDIR];

  const int ighost = data->nghost[IDIR];
  const int jghost = data->nghost[JDIR];
  const int kghost = data->nghost[KDIR];

  BoundaryLoopRange range;
  range.ibeg = (dir == IDIR) ? side * (ighost + nxi) : 0;
  range.iend = (dir == IDIR) ? ighost + side * (ighost + nxi) : data->np_tot[IDIR];
  range.jbeg = (dir == JDIR) ? side * (jghost + nxj) : 0;
  range.jend = (dir == JDIR) ? jghost + side * (jghost + nxj) : data->np_tot[JDIR];
  range.kbeg = (dir == KDIR) ? side * (kghost + nxk) : 0;
  range.kend = (dir == KDIR) ? kghost + side * (kghost + nxk) : data->np_tot[KDIR];

  return range;
}

inline bool IsFullTwoPiDomain(DataBlock* data) {
#if GEOMETRY == SPHERICAL
  return fabs((data->mygrid->xend[KDIR] - data->mygrid->xbeg[KDIR] - 2.0 * M_PI)) < 1e-10;
#else
  (void)data;
  return false;
#endif
}

inline void ApplyPeriodicBoundary(DataBlock* data, int dir, BoundarySide side,
                                  IdefixArray3D<real> arr) {
  IdefixArray3D<real> localVar = arr;

  const int nxi = data->np_int[IDIR];
  const int nxj = data->np_int[JDIR];
  const int nxk = data->np_int[KDIR];

  const int ighost = data->nghost[IDIR];
  const int jghost = data->nghost[JDIR];
  const int kghost = data->nghost[KDIR];

  BoundaryLoopRange range = ComputeBoundaryLoopRange(data, dir, side);

  idefix_for("BoundaryPeriodic",
    range.kbeg, range.kend, range.jbeg, range.jend, range.ibeg, range.iend,
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
}

inline void ApplyAxisBoundary(DataBlock* data, BoundarySide side,
                              const BoundaryLoopRange& range, IdefixArray3D<real> arr,
                              bool isTwoPi) {
  IdefixArray3D<real> localVar = arr;

  const int jref = (side == left) ? data->beg[JDIR] : data->end[JDIR] - 1;
  const int offset = (side == left) ? -1 : 1;

  const int np_int_k = data->np_int[KDIR];
  const int nghost_k = data->nghost[KDIR];

  if (isTwoPi) {
    idefix_for("BoundaryAxis", range.kbeg, range.kend, range.jbeg, range.jend, range.ibeg,
               range.iend, KOKKOS_LAMBDA(int k, int j, int i) {
                 int kcomp = nghost_k + ((k - nghost_k + np_int_k / 2) % np_int_k);
                 localVar(k, j, i) = localVar(kcomp, 2 * jref - j + offset, i);
               });
  } else {
    idefix_for("BoundaryAxis", range.kbeg, range.kend, range.jbeg, range.jend, range.ibeg,
               range.iend, KOKKOS_LAMBDA(int k, int j, int i) {
                 localVar(k, j, i) = localVar(k, 2 * jref - j + offset, i);
               });
  }
}

template <typename TMpi>
inline void MPIBoundaryExchange(DataBlock* data, TMpi& mpi, IdefixArray4D<real> arr4D, int dir) {
#ifdef WITH_MPI
  if (data->mygrid->nproc[dir] > 1) {
    switch (dir) {
      case IDIR:
        mpi.ExchangeX1(arr4D);
        break;
      case JDIR:
        mpi.ExchangeX2(arr4D);
        break;
      case KDIR:
        mpi.ExchangeX3(arr4D);
        break;
    }
  }
#else
  (void)data;
  (void)mpi;
  (void)arr4D;
  (void)dir;
#endif
}

#endif  // RADIATION_BOUNDARY_UTILS_HPP_
