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
// Module contributed by Alex Ziampras and Richard Booth.
// ***********************************************************************************

#ifndef RADIATION_IRRADIATION_HPP_
#define RADIATION_IRRADIATION_HPP_

#include <memory>
#include <vector>

#include "fluid_defs.hpp"
#include "grid.hpp"
#include "idefix.hpp"
#include "input.hpp"

#ifdef WITH_MPI
#include "mpi.hpp"
#endif

// Forward class hydro declaration
class DataBlock;
class Column;

class Irradiation {
 protected:
  enum OpacityType { constant, userconst };
  using ColumnBoundaryFunc = void (*)(DataBlock*, IdefixArray3D<real>);
  using UserOpacityFunc = void (*)(DataBlock*, IdefixArray2D<real>);
  using UserRadiationFunc = void (*)(DataBlock*, IdefixArray1D<real>);

 public:
  Irradiation(Input&, DataBlock*);

  virtual void ShowConfig();
  virtual void ComputeIrradiation() = 0;
  virtual void GetBoundaryFlux(int dir, int side, IdefixArray2D<real>) = 0;

  template <typename Phys>
  void ComputeRadiationPressureSourceTerm(Fluid<Phys>* fluid, real t, real dt) {
    if (!Phys::dust) {
      _ComputeRadiationPressureSourceTerm(dt, 0);
    } else {
      _ComputeRadiationPressureSourceTerm(dt, fluid->instanceNumber + 1);
    }
  }

  void UpdatePressure();

  IdefixArray3D<real> boundaryColumn;  ///< Column Density at the inner edge of the disc.
  void InitBoundaryColumnToZero();
  void EnrollUserColumnBoundary(ColumnBoundaryFunc);
  void EnrollUserOpacity(UserOpacityFunc);
  void EnrollUserRadiationField(UserRadiationFunc);

  IdefixArray4D<real>
      irradiationHeating;  ///< Heating rate per unit volume due to stellar irradiation
  int num_species;
  int num_bands;

  double elapsedTime;  // time spent solving radiation

  virtual void _ComputeRadiationPressureSourceTerm(real dt, int species) = 0;
 protected:
  DataBlock* data;

  IdefixArray2D<real> kappa;
  IdefixArray1D<real> radiation_field;  // Luminosity/flux dependening on geometry.

  real unit_luminosity, unit_opacity;

  bool haveUserColumnBoundary;
  ColumnBoundaryFunc UserDefColumnBoundary;

  bool haveUserOpacity;
  UserOpacityFunc user_opacity;
  OpacityType opacityType;

  bool haveUserRadiationField;
  UserRadiationFunc user_radiation;
  OpacityType radiationType;
};

// Long characteristic irradiation from a point source at the origin
class LongCharIrradiation : public Irradiation {
  using Irradiation::ColumnBoundaryFunc;
  using Irradiation::OpacityType;

 public:
  LongCharIrradiation(Input&, DataBlock*);
  ~LongCharIrradiation();

  void ShowConfig();

  void ComputeIrradiation();
  void GetBoundaryFlux(int dir, int side, IdefixArray2D<real>);

  IdefixArray3D<real> localOffset;
  IdefixArray3D<real> globalOffset;

  int radialRank;  // Rank in the radial communicator, default to 0
  int radialSize;  // Size of the radial communicator, default to 1
#ifdef WITH_MPI
  MPI_Comm RadialComm;  // Radial communicator
#endif

 void _ComputeRadiationPressureSourceTerm(real dt, int species);

 private:
  IdefixArray4D<real> column;
  Column* column_sum = NULL;
};

// Short Characteristic radiation in a plane-parallel geometry originating from -z.

/* Bag-of-data class for information about short-characteristics rays
 */
class RayInfo {
 public:
  enum class RayType { RADIAL = 0, POLOIDAL = 1, CORE = 2, EDGE = 3, UNSPECIFIED = -1 };
  explicit RayInfo(int dim_[2]);

  void dump(std::ostream& out, DataBlock* data) const;

  int dim[2];
  IdefixArray2D<RayType> type;
  IdefixArray2D<int> index_i, index_j, cell_i, cell_j;
  IdefixArray2D<double> start, length;
  IdefixArray2D<int> order;
  IdefixArray2D<double> weights;
};
class HostRayInfo;

class SphericalShortChar : public Irradiation {
 public:
  SphericalShortChar(Input&, DataBlock* db);

  void compute_optical_depths();
  void compute_heating_rate();

  void dump_rays(std::ostream& f) const;
  void dump_heating(std::ostream& f) const;
  void dump_tau(std::ostream& f) const;

  void ComputeIrradiation() {
    compute_optical_depths();
    compute_heating_rate();
  }
  void GetBoundaryFlux(int dir, int side, IdefixArray2D<real>);

  void ShowConfig();

  void _ComputeRadiationPressureSourceTerm(real dt, int species);

 public:
  void _build_ray_weights();
 private:
  void _build_rays(HostRayInfo&);
  void _sort_ray_walk(HostRayInfo&);
  int _get_cell_order(int i, int j, HostRayInfo&);

  int shape[2];
  std::unique_ptr<RayInfo> rays;

  // Info for ordering the ray-walk
  std::vector<int> _cells_per_order;
  IdefixArray2D<int> _cells_in_order;
  int _max_order;

  // Store boundary optical depth
  IdefixArray3D<real> _tau_boundary;  ///< Optical Depth at the inner edge of the domain.

 public:
  IdefixArray4D<double> exp_tau, kapp;

 private:
  IdefixArray4D<double> rho_all;
};

#endif // RADIATION_IRRADIATION_HPP_
