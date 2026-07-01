// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "dataBlock.hpp"
#include "fluid.hpp"
#include "irradiation.hpp"
#include "units.hpp"
#include "vector.hpp"

RayInfo::RayInfo(int dim_[2])
    : dim{dim_[0], dim_[1]},
      type("type", dim_[1], dim_[0]),
      index_i("index_i", dim_[1], dim_[0]),
      index_j("index_j", dim_[1], dim_[0]),
      cell_i("cell_i", dim_[1], dim_[0]),
      cell_j("cell_j", dim_[1], dim_[0]),
      start("start", dim_[1], dim_[0]),
      length("length", dim_[1], dim_[0]),
      order("order", dim_[1], dim_[0]),
      weights("weights", dim_[1], dim_[0]) {}

class HostRayInfo {
 public:
  explicit HostRayInfo(const RayInfo& ray);

  void copy_to_device(RayInfo& ray);
  void copy_from_device(const RayInfo& ray);

  IdefixHostArray2D<RayInfo::RayType> type;
  IdefixHostArray2D<int> index_i, index_j, cell_i, cell_j;
  IdefixHostArray2D<double> start, length;
  IdefixHostArray2D<int> order;
  IdefixHostArray2D<double> weights;
};

HostRayInfo::HostRayInfo(const RayInfo& ray) {
  type = Kokkos::create_mirror_view(ray.type);
  order = Kokkos::create_mirror_view(ray.order);

  index_i = Kokkos::create_mirror_view(ray.index_i);
  index_j = Kokkos::create_mirror_view(ray.index_j);
  cell_i = Kokkos::create_mirror_view(ray.cell_i);
  cell_j = Kokkos::create_mirror_view(ray.cell_j);

  start = Kokkos::create_mirror_view(ray.start);
  length = Kokkos::create_mirror_view(ray.length);
  weights = Kokkos::create_mirror_view(ray.weights);
}

void HostRayInfo::copy_from_device(const RayInfo& ray) {
  Kokkos::deep_copy(type, ray.type);
  Kokkos::deep_copy(order, ray.order);

  Kokkos::deep_copy(index_i, ray.index_i);
  Kokkos::deep_copy(index_j, ray.index_j);
  Kokkos::deep_copy(cell_i, ray.cell_i);
  Kokkos::deep_copy(cell_j, ray.cell_j);

  Kokkos::deep_copy(start, ray.start);
  Kokkos::deep_copy(length, ray.length);
  Kokkos::deep_copy(weights, ray.weights);
}

void HostRayInfo::copy_to_device(RayInfo& ray) {
  Kokkos::deep_copy(ray.type, type);
  Kokkos::deep_copy(ray.order, order);

  Kokkos::deep_copy(ray.index_i, index_i);
  Kokkos::deep_copy(ray.index_j, index_j);
  Kokkos::deep_copy(ray.cell_i, cell_i);
  Kokkos::deep_copy(ray.cell_j, cell_j);

  Kokkos::deep_copy(ray.start, start);
  Kokkos::deep_copy(ray.length, length);
  Kokkos::deep_copy(ray.weights, weights);
}

void RayInfo::dump(std::ostream& out, DataBlock* data) const {
  HostRayInfo rays(*this);
  rays.copy_from_device(*this);

  DataBlockHost d(*data);

  IdefixHostArray1D<double> r("Re", dim[0]), th("the", dim[1]);
  for (int i = 0; i < dim[0]; i++) {
    if (i + 1 < dim[0])
      r[i] = d.xl[IDIR][i];
    else
      r[i] = d.xr[IDIR][i - 1];
  }
  for (int i = 0; i < dim[1]; i++) {
    if (i + 1 < dim[1])
      th[i] = d.xl[JDIR][i];
    else
      th[i] = d.xr[JDIR][i - 1];
  }

  out << "# i j r th type index_i index_j start length order weights\n";

  for (int j = 0; j < dim[1]; j++)
    for (int i = 0; i < dim[0]; i++)
      out << i << " " << j << " " << r[i] << " " << th[j] << " " << int(rays.type(j, i)) << " "
          << rays.index_i(j, i) << " " << rays.index_j(j, i) << " " << rays.start(j, i) << " "
          << rays.length(j, i) << " " << rays.order(j, i) << " " << rays.weights(j, i) << "\n";
}

SphericalShortChar::SphericalShortChar(Input& input, DataBlock* datain)
    : Irradiation(input, datain),
      shape{data->np_tot[IDIR] + 1, data->np_tot[JDIR] + 1},
      _tau_boundary("Boundary optical depth", data->np_tot[KDIR], data->np_tot[JDIR], num_bands),
      exp_tau("Optical Depth", data->np_tot[KDIR], data->np_tot[JDIR] + 1, data->np_tot[IDIR] + 1,
              num_bands),
      kapp("Opacity", data->np_tot[KDIR], data->np_tot[JDIR] + 1, data->np_tot[IDIR] + 1,
           num_bands),
      rho_all("Densities", num_species, data->np_tot[KDIR], data->np_tot[JDIR] + 1,
              data->np_tot[IDIR] + 1) {
#if GEOMETRY != SPHERICAL
  std::stringstream msg;
  msg << "SphericalShortChar:: only works in SPHERICAL geometry";
  IDEFIX_ERROR(msg);
#endif
#if WITH_MPI
  std::stringstream msg;
  msg << "SphericalShortChar:: does not work with MPI";
  IDEFIX_ERROR(msg);
#endif

  // Setup Luminosity
  if (input.CheckEntry("Irradiation", "flux") >= 0) {
    this->radiation_field = IdefixArray1D<real>("irrad_field", this->num_bands);

    std::string irrad = input.Get<std::string>("Irradiation", "flux", 0);

    if (irrad.compare("constant") == 0) {
      radiationType = OpacityType::constant;

      IdefixArray1D<real>::HostMirror f_host = Kokkos::create_mirror_view(this->radiation_field);
      for (int i = 0; i < this->num_bands; i++)
        f_host(i) = input.Get<real>("Irradiation", "flux", i + 1);  // in cgs units

      Kokkos::deep_copy(radiation_field, f_host);
    } else if (irrad.compare("userconst") == 0) {
      haveUserRadiationField = true;
      radiationType = OpacityType::userconst;
    } else {
      std::stringstream msg;
      msg << "Irradiation:: luminosity must be either 'constant' or 'userconst'.";
      IDEFIX_ERROR(msg);
    }
  } else {
    std::stringstream msg;
    msg << "Irradiation:: luminosity must be specified.";
    IDEFIX_ERROR(msg);
  }

  rays = std::make_unique<RayInfo>(shape);

  HostRayInfo host_rays(*rays);

  _build_rays(host_rays);
  _sort_ray_walk(host_rays);
  host_rays.copy_to_device(*rays);

  _build_ray_weights();
}

void SphericalShortChar::_build_rays(HostRayInfo& host_rays) {
  int ibeg = data->beg[IDIR];
  int jbeg = data->beg[JDIR];

  int iend = data->end[IDIR];
  int jend = data->end[JDIR];

  DataBlockHost d(*data);


  auto& re = d.xl[IDIR];
  auto& the = d.xl[JDIR];

  for (int j = jbeg; j <= jend; j++)
    for (int i = ibeg; i <= iend; i++) {
      // Step 0:
      //  Get the ray's end point and direction of travel
      //  Assumes the ray cannot cross r = rl[i] before we terminate it.
      double r = re[i], th = the[j];

      double R = r * sin(th);
      double Z = r * cos(th);

      // Is the ray towards or away from the planet?
      double sin_jm1 = sin(the[j - 1]);
      bool out = (j == jbeg) || R > r * sin_jm1;

      // Step 1: Is the cell corner on a radial boundary:
      if (i == iend && out) {
        host_rays.type(j, i) = RayInfo::RayType::EDGE;
        continue;
      }
      if (i == ibeg && (!out)) {
        host_rays.type(j, i) = RayInfo::RayType::CORE;
        continue;
      }

      // Step 2: Handle th=0/pi
      if (j == jbeg) {
        host_rays.type(j, i) = RayInfo::RayType::RADIAL;
        host_rays.index_i(j, i) = i + 1;
        host_rays.index_j(j, i) = j;
        host_rays.start(j, i) = the[j];
        host_rays.length(j, i) = re[i + 1] * cos(th) - Z;
        host_rays.cell_i(j, i) = i;
        host_rays.cell_j(j, i) = 0;
        continue;
      } else if (j == jend) {
        host_rays.type(j, i) = RayInfo::RayType::RADIAL;
        host_rays.index_i(j, i) = i - 1;
        host_rays.index_j(j, i) = j;
        host_rays.start(j, i) = the[j];
        host_rays.length(j, i) = re[i - 1] * cos(th) - Z;
        host_rays.cell_i(j, i) = i - 1;
        host_rays.cell_j(j, i) = jend - 1;
        continue;
      }

      // Step 3: Handle the normal rays.
      //  They are 4 cases depending which edge the ray strikes
      if (out) {
        if (R < re[i + 1] * sin_jm1) {
          host_rays.type(j, i) = RayInfo::RayType::POLOIDAL;
          host_rays.index_i(j, i) = i + 1;
          host_rays.index_j(j, i) = j - 1;
          host_rays.start(j, i) = R / sin_jm1;
          host_rays.length(j, i) = host_rays.start(j, i) * cos(the[j - 1]) - Z;
          host_rays.cell_i(j, i) = i;
          host_rays.cell_j(j, i) = j - 1;
          continue;
        } else {
          host_rays.type(j, i) = RayInfo::RayType::RADIAL;
          host_rays.index_i(j, i) = i + 1;
          host_rays.index_j(j, i) = j;
          host_rays.start(j, i) = asin(R / re[i + 1]);
          host_rays.length(j, i) = re[i + 1] * cos(host_rays.start(j, i)) - Z;
          host_rays.cell_i(j, i) = i;
          host_rays.cell_j(j, i) = j - 1;
        }
      } else {  // not out
        if (R >= re[i - 1] * sin_jm1) {
          host_rays.type(j, i) = RayInfo::RayType::POLOIDAL;
          host_rays.index_i(j, i) = i;
          host_rays.index_j(j, i) = j - 1;
          host_rays.start(j, i) = R / sin_jm1;
          host_rays.length(j, i) = host_rays.start(j, i) * cos(the[j - 1]) - Z;
          host_rays.cell_i(j, i) = i - 1;
          host_rays.cell_j(j, i) = j - 1;
          continue;
        } else {
          host_rays.type(j, i) = RayInfo::RayType::RADIAL;
          host_rays.index_i(j, i) = i - 1;
          host_rays.index_j(j, i) = j;
          host_rays.start(j, i) = M_PI - asin(R / re[i - 1]);
          host_rays.length(j, i) = re[i - 1] * cos(host_rays.start(j, i)) - Z;
          host_rays.cell_i(j, i) = i - 1;
          host_rays.cell_j(j, i) = j - 1;
        }
      }
    }
}

void SphericalShortChar::_build_ray_weights() {
  idfx::RegionWrapper region("SphericalShortChar::_build_ray_weights");

  int ibeg = data->beg[IDIR];
  int jbeg = data->beg[JDIR];

  int iend = data->end[IDIR];
  int jend = data->end[JDIR];

  IdefixArray1D<double> re = data->xl[IDIR];
  IdefixArray1D<double> the = data->xl[JDIR];

  IdefixArray2D<RayInfo::RayType> type = rays->type;
  IdefixArray2D<int> index_i = rays->index_i;
  IdefixArray2D<int> index_j = rays->index_j;
  IdefixArray2D<double> start = rays->start;
  IdefixArray2D<double> weights = rays->weights;

  idefix_for("BuildRayWeights", jbeg, jend + 1, ibeg, iend + 1,
      KOKKOS_LAMBDA(int j, int i) {
        if (type(j, i) == RayInfo::RayType::RADIAL) {
          double th = start(j, i);

          // Find neighbouring azimuthal points
          int jj = index_j(j, i);

          // Compute the weights
          double w;
          if (jj == jbeg)
            w = 1;
          else
            w = (cos(th) - cos(the[jj - 1])) / (cos(the[jj]) - cos(the[jj - 1]));

          weights(j, i) = w;
        } else if (type(j, i) == RayInfo::RayType::POLOIDAL) {
          double r = start(j, i);

          // Find neighbouring radial points
          int ii = index_i(j, i);

          // Compute the weights
          double w;
          if (ii == ibeg)
            w = 1;
          else
            w = (r - re[ii - 1]) / (re[ii] - re[ii - 1]);

          weights(j, i) = w;
        }
      });
}

int SphericalShortChar::_get_cell_order(int i, int j, HostRayInfo& host_rays) {
  // Recursively walk the list of rays to the boundary,
  // storing the order of any other cells we discover along
  // the way.
  int order = host_rays.order(j, i);
  if (order < 0) {
    if (host_rays.type(j, i) == RayInfo::RayType::RADIAL ||
        host_rays.type(j, i) == RayInfo::RayType::POLOIDAL) {
      int ii = host_rays.index_i(j, i);
      int jj = host_rays.index_j(j, i);

      int o1 = _get_cell_order(ii, jj, host_rays);
      int o2 = 0;
      if (host_rays.type(j, i) == RayInfo::RayType::RADIAL) {
        if (jj > data->beg[JDIR]) o2 = _get_cell_order(ii, jj - 1, host_rays);
      } else if (host_rays.type(j, i) == RayInfo::RayType::POLOIDAL) {
        if (ii > data->beg[IDIR]) o2 = _get_cell_order(ii - 1, jj, host_rays);
      }
      order = std::max(o1, o2) + 1;
    } else {
      order = 0;
    }
    host_rays.order(j, i) = order;
  }

  return order;
}

void SphericalShortChar::_sort_ray_walk(HostRayInfo& host_rays) {
  idfx::RegionWrapper region("SphericalShortChar::_sort_ray_walk");

  int ibeg = data->beg[IDIR];
  int jbeg = data->beg[JDIR];

  int iend = data->end[IDIR];
  int jend = data->end[JDIR];

  // Step 0:
  //  Initialize order
  for (int j = jbeg; j <= jend; j++)
    for (int i = ibeg; i <= iend; i++) host_rays.order(j, i) = -1;

  // Step 1:
  //  Compute the ordering of the cell updates and find the maximum
  _max_order = 0;
  for (int j = jbeg; j <= jend; j++)
    for (int i = ibeg; i <= iend; i++) {
      int order = _get_cell_order(i, j, host_rays);
      _max_order = std::max(order, _max_order);
    }
  _max_order += 1;

  // Step 2: Create the storage for the order:
  _cells_per_order = std::vector<int>(_max_order, 0);
  _cells_in_order = IdefixArray2D<int>("cell_ids", shape[1] * shape[0], 2);

  IdefixArray2D<int>::HostMirror cells_in_order = Kokkos::create_mirror_view(_cells_in_order);

  // Step 3: Store the cells in order:
  int count = 0;
  for (int order = 0; order < _max_order; order++) {
    for (int j = jbeg; j <= jend; j++)
      for (int i = ibeg; i <= iend; i++) {
        if (host_rays.order(j, i) == order) {
          cells_in_order(count, 0) = i;
          cells_in_order(count, 1) = j;

          count++;
        }
      }
  _cells_per_order[order] = count;
  }
  Kokkos::deep_copy(_cells_in_order, cells_in_order);
}

void SphericalShortChar::compute_optical_depths() {
  idfx::RegionWrapper region("SphericalShortChar::compute_optical_depths");

  Kokkos::Timer timer;
  elapsedTime -= timer.seconds();

  if (haveUserOpacity) user_opacity(data, kappa);

  if (haveUserColumnBoundary) UserDefColumnBoundary(data, boundaryColumn);

  int ibeg = data->beg[IDIR];
  int jbeg = data->beg[JDIR];
  int kbeg = data->beg[KDIR];

  int iend = data->end[IDIR];
  int jend = data->end[JDIR];
  int kend = data->end[KDIR];

  IdefixArray4D<double> kapp = this->kapp;
  IdefixArray4D<double> rho_all = this->rho_all;
  IdefixArray3D<double> boundaryColumn = this->boundaryColumn;
  IdefixArray3D<double> _tau_boundary = this->_tau_boundary;

  const real u_opac = unit_opacity;
  int num_bands = this->num_bands;

  // Fill boundaries as we need the density.
  data->SetBoundaries();
  for (int s = 0; s < num_species; s++) {
    IdefixArray4D<double> Vc;
    if (s == 0)
      Vc = data->hydro->Vc;
    else
      Vc = data->dust[s - 1]->Vc;

    if (this->opacityType == OpacityType::constant || this->opacityType == OpacityType::userconst) {
      IdefixArray2D<double> kappa = this->kappa;

      idefix_for("CornerOpacity", kbeg, kend, jbeg, jend + 1, ibeg, iend + 1,
        KOKKOS_LAMBDA(int k, int j, int i) {
          double rho = 0.25 * (Vc(RHO, k, j, i) + Vc(RHO, k, j, i - 1) + Vc(RHO, k, j - 1, i) +
                                Vc(RHO, k, j - 1, i - 1));

          for (int l = 0; l < num_bands; l++) {
            if (s == 0) kapp(k, j, i, l) = 0;
            kapp(k, j, i, l) += kappa(s, l) * rho / u_opac;
          }

          // Save the density in the cell center
          rho_all(s, k, j, i) = Vc(RHO, k, j, i);
        });

      idefix_for("BoundaryOpacity", kbeg, kend, jbeg, jend,
        KOKKOS_LAMBDA(int k, int j) {
            for (int l = 0; l < num_bands; l++) {
              if (s == 0) _tau_boundary(k, j, l) = 0;
              _tau_boundary(k, j, l) += kappa(s, l) * boundaryColumn(s, k, j) / u_opac;
            }
          });

    } else {
      std::stringstream msg;
      msg << "Irradiation:: Opacity (kappa) type must be 'constant' or 'userconst'";
      IDEFIX_ERROR(msg);
    }
  }

  int cend = 0;

  for (int order = 0; order < _max_order; order++) {
    int cbeg = cend;
    cend = _cells_per_order[order];

    IdefixArray2D<int> cells_in_order = _cells_in_order;

    IdefixArray2D<RayInfo::RayType> type = rays->type;
    IdefixArray2D<int> index_i = rays->index_i;
    IdefixArray2D<int> index_j = rays->index_j;
    IdefixArray2D<double> weights = rays->weights;
    IdefixArray2D<double> length = rays->length;

    IdefixArray4D<double> exp_tau = this->exp_tau;

    idefix_for("ComputeTau", kbeg, kend, cbeg, cend, 0, num_bands,
      KOKKOS_LAMBDA(int k, int cell, int l) {
        int i = cells_in_order(cell, 0);
        int j = cells_in_order(cell, 1);

        if (order == 0) {  // Start with boundary
          if (type(j, i) == RayInfo::RayType::CORE)
            exp_tau(k, j, i, l) = 0;
          else if (type(j, i) == RayInfo::RayType::EDGE)
            exp_tau(k, j, i, l) = exp(-_tau_boundary(k, j, l));
        } else {  // Not boundary
          int ii = index_i(j, i);
          int jj = index_j(j, i);
          double w = weights(j, i);

          // Interpolate depth and opacity to the start of the ray
          double f = exp_tau(k, jj, ii, l) * w;
          double kappa = kapp(k, jj, ii, l) * w;
          if (type(j, i) == RayInfo::RayType::RADIAL) {
            if (jj > jbeg) {
              f += exp_tau(k, jj - 1, ii, l) * (1 - w);
              kappa += kapp(k, jj - 1, ii, l) * (1 - w);
            }
          } else if (type(j, i) == RayInfo::RayType::POLOIDAL) {
            if (ii > ibeg) {
              f += exp_tau(k, jj, ii - 1, l) * (1 - w);
              kappa += kapp(k, jj, ii - 1, l) * (1 - w);
            }
          }
          exp_tau(k, j, i, l) = f * exp(-0.5 * (kappa + kapp(k, j, i, l)) * length(j, i));
        }
      });
  }

  elapsedTime += timer.seconds();
}

void SphericalShortChar::compute_heating_rate() {
  idfx::RegionWrapper region("SphericalShortChar::compute_heating_rate");

  Kokkos::Timer timer;
  elapsedTime -= timer.seconds();

  if (haveUserRadiationField) user_radiation(data, radiation_field);

  int ibeg = data->beg[IDIR];
  int jbeg = data->beg[JDIR];
  int kbeg = data->beg[KDIR];

  int iend = data->end[IDIR];
  int jend = data->end[JDIR];
  int kend = data->end[KDIR];

  IdefixArray1D<double> re = data->xl[IDIR];
  IdefixArray1D<double> the = data->xl[JDIR];
  IdefixArray1D<double> phie = data->xl[KDIR];

  IdefixArray3D<double> dV = data->dV;

  IdefixArray1D<double> F0 = this->radiation_field;
  IdefixArray2D<double> kappa = this->kappa;

  IdefixArray4D<double> Vc = data->hydro->Vc;

  IdefixArray4D<double> rho_all = this->rho_all;
  IdefixArray4D<double> exp_tau = this->exp_tau;
  IdefixArray4D<double> heating = this->irradiationHeating;
  int num_species = this->num_species;
  int num_bands = this->num_bands;

  // Step 1:
  //   Compute the heating rate from div-F.
  idefix_for(
      "IrradiationHeating", kbeg, kend, jbeg, jend, ibeg, iend, KOKKOS_LAMBDA(int k, int j, int i) {
        double cm = cos(the[j]), cp = cos(the[j + 1]);

        double wP = (2 * re[i] + re[i + 1]) / (3 * (re[i] + re[i + 1]));
        double wR = (2 * cm + cp) / (3 * (cm + cp));

        double sm2 = sin(the[j]) * sin(the[j]), sp2 = sin(the[j + 1]) * sin(the[j + 1]);
        double AR[2] = {0.5 * re[i] * re[i] * (phie[k + 1] - phie[k]) * (sm2 - sp2),
                        0.5 * re[i + 1] * re[i + 1] * (phie[k + 1] - phie[k]) * (sm2 - sp2)};
        double AP[2] = {
            0.5 * (re[i + 1] * re[i + 1] - re[i] * re[i]) * (phie[k + 1] - phie[k]) * sm2,
            0.5 * (re[i + 1] * re[i + 1] - re[i] * re[i]) * (phie[k + 1] - phie[k]) * sp2};

        for (int s = 0; s < num_species; s++) heating(s, k, j, i) = 0;

        for (int l = 0; l < num_bands; l++) {
          double FR[2] = {0, 0}, FP[2] = {0, 0};

          // Compute fluxes
          FP[0] = exp_tau(k, j, i + 1, l) * wP + exp_tau(k, j, i, l) * (1 - wP);
          FP[1] = exp_tau(k, j + 1, i + 1, l) * wP + exp_tau(k, j + 1, i, l) * (1 - wP);

          FR[0] = exp_tau(k, j + 1, i, l) * wR + exp_tau(k, j, i, l) * (1 - wR);
          FR[1] = exp_tau(k, j + 1, i + 1, l) * wR + exp_tau(k, j, i + 1, l) * (1 - wR);

          // Integrate F.dS over the area (assumes F points in the z direction)
          double heat_l = -F0[l] *
                          ((FR[1] * AR[1] - FR[0] * AR[0]) + (FP[1] * AP[1] - FP[0] * AP[0])) /
                          dV(k, j, i);

          // Divide the flux over the different species
          double ktot = 0;
          for (int s = 0; s < num_species; s++) ktot += rho_all(s, k, j, i) * kappa(s, l);

          if (ktot > 0) {
            for (int s = 0; s < num_species; s++) {
              heating(s, k, j, i) += heat_l * (kappa(s, l) / ktot);
            }
          }
        }
      });
  // Step 2:
  //  Limit the heating rate based on the fluxes at the corners
  const real u_opac = unit_opacity;
  const real u_flux = unit_luminosity / pow(idfx::units.GetLength(), 2);

  idefix_for("FixIrradiationHeating", 0, num_species, kbeg, kend, jbeg, jend, ibeg, iend,
      KOKKOS_LAMBDA(int s, int k, int j, int i) {
        double hc[2][2] = {{0, 0}, {0, 0}};
        for (int l = 0; l < num_bands; l++) {
          hc[0][0] += F0[l] * exp_tau(k, j, i, l) * kappa(s, l) / u_opac;
          hc[0][1] += F0[l] * exp_tau(k, j, i + 1, l) * kappa(s, l) / u_opac;

          hc[1][0] += F0[l] * exp_tau(k, j + 1, i, l) * kappa(s, l) / u_opac;
          hc[1][1] += F0[l] * exp_tau(k, j + 1, i + 1, l) * kappa(s, l) / u_opac;
        }
        double hmin, hmax;
        hmin = Kokkos::min(hc[0][0], Kokkos::min(hc[0][1], Kokkos::min(hc[1][0], hc[1][1])));
        hmax = Kokkos::max(hc[0][0], Kokkos::max(hc[0][1], Kokkos::max(hc[1][0], hc[1][1])));

        heating(s, k, j, i) = Kokkos::max(hmin, Kokkos::min(hmax, heating(s, k, j, i))) / u_flux;
      });

  // Add the heating to the total energy if we don't have FLD.
  if (!data->haveRadiation) UpdatePressure();
  elapsedTime += timer.seconds();
}

void SphericalShortChar::GetBoundaryFlux(int dir, int side, IdefixArray2D<real> flux) {
  idfx::RegionWrapper region("SphericalShortChar::GetBoundaryFlux");

  const real u_flux = unit_luminosity / pow(idfx::units.GetLength(), 2);

  if (dir == IDIR) {
    int ibeg;
    if (side == left)
      ibeg = data->beg[IDIR];
    else
      ibeg = data->end[IDIR];
    int jbeg = data->beg[JDIR];
    int kbeg = data->beg[KDIR];

    int jend = data->end[JDIR];
    int kend = data->end[KDIR];

    IdefixArray1D<double> the = data->xl[JDIR];

    IdefixArray1D<double> F0 = this->radiation_field;
    IdefixArray4D<double> exp_tau = this->exp_tau;
    int num_bands = this->num_bands;

    idefix_for("IrradBoundaryFlux", kbeg, kend, jbeg, jend,
      KOKKOS_LAMBDA(int k, int j) {
          double cm = cos(the[j]), cp = cos(the[j + 1]);
          double wR = (2 * cm + cp) / (3 * (cm + cp));

          flux(k, j) = 0;
          for (int l = 0; l < num_bands; l++) {
            double FR = exp_tau(k, j + 1, ibeg, l) * wR + exp_tau(k, j, ibeg, l) * (1 - wR);
            flux(k, j) -= F0[l] * Kokkos::max(FR * (cm + cp) / 2, 0.0) / u_flux;
          }
        });
    return;
  } else {
    IDEFIX_ERROR("SphericalShortChar::GetBoundaryFlux is not implemented for this boundary");
  }
}

void SphericalShortChar::dump_rays(std::ostream& f) const { rays->dump(f, data); }

void SphericalShortChar::dump_heating(std::ostream& out) const {
  throw std::runtime_error("Need to implement device->host transfer");

  int dim[3] = {data->np_tot[IDIR], data->np_tot[JDIR], data->np_tot[KDIR]};

  IdefixArray1D<double>& r = data->x[IDIR];
  IdefixArray1D<double>& th = data->x[JDIR];
  IdefixArray1D<double>& phi = data->x[KDIR];

  out << "# r th phi heating\n";

  for (int k = 0; k < dim[2]; k++)
    for (int j = 0; j < dim[1]; j++)
      for (int i = 0; i < dim[0]; i++)
        out << r[i] << " " << th[j] << " " << phi[k] << " " << irradiationHeating(0, k, j, i)
            << "\n";
}

void SphericalShortChar::dump_tau(std::ostream& out) const {
  throw std::runtime_error("Need to implement device->host transfer");

  int dim[3] = {data->np_tot[IDIR] + 1, data->np_tot[JDIR] + 1, data->np_tot[KDIR]};

  IdefixArray1D<double> r("Re", dim[0]), th("the", dim[1]);
  IdefixArray1D<double>& phi = data->x[KDIR];

  for (int i = 0; i < dim[0]; i++) {
    if (i + 1 < dim[0])
      r[i] = data->xl[IDIR][i];
    else
      r[i] = data->xr[IDIR][i - 1];
  }
  for (int i = 0; i < dim[1]; i++) {
    if (i + 1 < dim[1])
      th[i] = data->xl[JDIR][i];
    else
      th[i] = data->xr[JDIR][i - 1];
  }

  out << "# r th phi";
  for (int l = 0; l < num_bands; l++) out << " tau_" << l;
  out << "\n";

  for (int k = 0; k < dim[2]; k++)
    for (int j = 0; j < dim[1]; j++)
      for (int i = 0; i < dim[0]; i++) {
        out << r[i] << " " << th[j] << " " << phi[k];
        for (int l = 0; l < num_bands; l++) out << " " << log(1 / exp_tau(k, j, i, l));
        out << "\n";
      }
}

void SphericalShortChar::ShowConfig() {
  idfx::cout << "Irradiation: using Spherical Short Characteristics.\n";
  Irradiation::ShowConfig();
}

void SphericalShortChar::_ComputeRadiationPressureSourceTerm(real dt, int s, bool update_Vc) {
  idfx::RegionWrapper region("SphericalShortChar::ComputeRadiationPressureSourceTerm");

  IdefixArray4D<real> heating = this->irradiationHeating;
  real code_c = idfx::units.c / idfx::units.GetVelocity();
  real dt_over_c = dt / code_c;

  IdefixArray1D<double> th = data->x[JDIR];

  IdefixArray4D<real> Uc, Vc;
  if (s == 0) {
    Uc = data->hydro->Uc;
    Vc = data->hydro->Vc;
  }
  else {
    Uc = data->dust[s - 1]->Uc;
    Vc = data->dust[s - 1]->Vc;
  }

  idefix_for("IrradiationPressureSource",
                data->beg[KDIR], data->end[KDIR],
                data->beg[JDIR], data->end[JDIR],
                data->beg[IDIR], data->end[IDIR],
    KOKKOS_LAMBDA(int k, int j, int i) {
      if (update_Vc) {
          EXPAND(
            Vc(VX1, k, j, i) += -cos(th[j]) * heating(s, k, j, i) * dt_over_c; ,
            Vc(VX2, k, j, i) += +sin(th[j]) * heating(s, k, j, i) * dt_over_c; ,
            {}
          )
        }
      else {
        EXPAND(
          Uc(VX1, k, j, i) += -cos(th[j]) * heating(s, k, j, i) * Vc(RHO, k, j, i) * dt_over_c; ,
          Uc(VX2, k, j, i) += +sin(th[j]) * heating(s, k, j, i) * Vc(RHO, k, j, i) * dt_over_c; ,
          {}
        )
      }
  });
}
