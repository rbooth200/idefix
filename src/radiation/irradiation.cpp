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

#include "irradiation.hpp"

#include <sstream>
#include <string>
#include <vector>

#include "column.hpp"
#include "dataBlock.hpp"
#include "fluid.hpp"
#include "idefix.hpp"
#include "units.hpp"
#include "vector.hpp"

Irradiation::Irradiation(Input& input, DataBlock* datain) {
  this->data = datain;
  elapsedTime = 0.0;

  // Setup units
  this->unit_opacity = 1.0 / (idfx::units.GetLength() * idfx::units.GetDensity());
  this->unit_luminosity =
      pow(idfx::units.GetLength(), 3) * idfx::units.GetEnergy() / idfx::units.GetTime();

  // Do we treat the heating separately for each species?
  num_species = 1;
  if (input.CheckBlock("Dust")) {
    num_species = 1 + input.Get<int>("Dust", "nSpecies", 0);
  }

  // Setup opacity
  haveUserOpacity = false;
  if (input.CheckEntry("Irradiation", "kappa") >= 0) {
    this->num_bands = input.Get<int>("Irradiation", "kappa", 1);
    this->kappa = IdefixArray2D<real>("kappa", num_species, this->num_bands);

    std::string opac = input.Get<std::string>("Irradiation", "kappa", 0);

    if (opac.compare("constant") == 0) {
      opacityType = OpacityType::constant;

      IdefixHostArray2D<real> kappa_host = Kokkos::create_mirror_view(this->kappa);
      for (int s = 0; s < num_species; s++)
        for (int i = 0; i < num_bands; i++)
          kappa_host(s, i) = input.Get<real>("Irradiation", "kappa", 2 + (s * num_bands + i));
      Kokkos::deep_copy(this->kappa, kappa_host);

    } else if (opac.compare("userconst") == 0) {
      opacityType = OpacityType::userconst;
      haveUserOpacity = true;
    } else {
      std::stringstream msg;
      msg << "Irradiation:: Opacity (kappa) type must be 'constant' or 'userconst" << opac;
      IDEFIX_ERROR(msg);
    }
  } else {
    std::stringstream msg;
    msg << "Irradiation:: Opacity (kappa) must be specified.";
    IDEFIX_ERROR(msg);
  }

  // Setup the internal workspace arrays
  this->irradiationHeating =
      IdefixArray4D<real>("IrradiationHeating", num_species, data->np_tot[KDIR], data->np_tot[JDIR],
                          data->np_tot[IDIR]);

  // Setup the boundary for the inner column
  InitBoundaryColumnToZero();
}

void Irradiation::InitBoundaryColumnToZero() {
  this->haveUserColumnBoundary = false;
  this->boundaryColumn =
      IdefixArray3D<real>("boundaryColumn", num_species, data->np_tot[KDIR], data->np_tot[JDIR]);

  IdefixArray3D<real> bC = this->boundaryColumn;
  idefix_for(
      "InitBoundaryColumn", 0, num_species, 0, data->np_tot[KDIR], 0, data->np_tot[JDIR],
      KOKKOS_LAMBDA(int s, int k, int j) { bC(s, k, j) = ZERO_F; });
}

void Irradiation::EnrollUserColumnBoundary(ColumnBoundaryFunc func) {
  this->haveUserColumnBoundary = true;
  UserDefColumnBoundary = func;
}

void Irradiation::EnrollUserOpacity(UserOpacityFunc func) {
  if (!haveUserOpacity) {
    IDEFIX_ERROR(
        "Trying to enroll a user opacity function for irradiation but haveUserOpacity=false");
  }

  user_opacity = func;
}

void Irradiation::EnrollUserRadiationField(UserRadiationFunc func) {
  if (!haveUserRadiationField) {
    IDEFIX_ERROR("Trying to enroll a user irradiation field but haveUserRadiationField=false");
  }

  user_radiation = func;
}

void Irradiation::UpdatePressure() {
  idfx::RegionWrapper region("Irradiation::UpdatePressure");

  IdefixArray4D<real> heating = this->irradiationHeating;
  const real dt = data->dt;

  {
    IdefixArray4D<real> Vc = data->hydro->Vc;
    const real gamma = data->hydro->eos->GetGamma();

    idefix_for(
        "UpdatePressure", data->beg[KDIR], data->end[KDIR], data->beg[JDIR], data->end[JDIR],
        data->beg[IDIR], data->end[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
          Vc(PRS, k, j, i) += heating(0, k, j, i) * Vc(RHO, k, j, i) * dt * (gamma - 1);
        });
  }

  // Update the dust internal energy (stored as a tracer)
  for (int d = 1; d < num_species; d++) {
    if (data->dust[d - 1]->drag->have_energy) {
      IdefixArray4D<real> Vc = data->dust[d - 1]->Vc;
      real cV = data->dust[d - 1]->drag->cV;

      idefix_for(
          "DustHeating", data->beg[KDIR], data->end[KDIR], data->beg[JDIR], data->end[JDIR],
          data->beg[IDIR], data->end[IDIR],

          KOKKOS_LAMBDA(int k, int j, int i) {
            Vc(TRD, k, j, i) += heating(d, k, j, i) * dt / cV;
          });
    }
  }
  data->SetBoundaries();
}

template <typename RadiationGeometry>
void _ComputeRadiationPressureSourceTerm(DataBlock* data, RadiationGeometry geometry,
                                         real dt_over_c, int species) {}

LongCharIrradiation::LongCharIrradiation(Input& input, DataBlock* datain)
    : Irradiation(input, datain) {
#if GEOMETRY != SPHERICAL
  std::stringstream msg;
  msg << "LongCharIrradiation:: only works in SPHERICAL geometry";
  IDEFIX_ERROR(msg);
#endif

  // Setup Luminosity
  if (input.CheckEntry("Irradiation", "luminosity") >= 0) {
    this->radiation_field = IdefixArray1D<real>("irrad_field", this->num_bands);

    std::string irrad = input.Get<std::string>("Irradiation", "luminosity", 0);

    if (irrad.compare("constant") == 0) {
      radiationType = OpacityType::constant;

      IdefixArray1D<real>::HostMirror L_host = Kokkos::create_mirror_view(this->radiation_field);
      for (int i = 0; i < this->num_bands; i++)
        L_host(i) = idfx::units.L_sun * input.Get<real>("Irradiation", "luminosity", i + 1);

      Kokkos::deep_copy(this->radiation_field, L_host);

    } else {
        if (irrad.compare("userconst") == 0) {
        haveUserRadiationField = true;
        radiationType = OpacityType::userconst;
      } else {
        std::stringstream msg;
        msg << "Irradiation:: luminosity must be either 'constant' or 'userconst'.";
        IDEFIX_ERROR(msg);
      }
    }
  } else {
    std::stringstream msg;
    msg << "Irradiation:: luminosity must be specified.";
    IDEFIX_ERROR(msg);
  }

  // Setup the internal workspace arrays
  this->column = IdefixArray4D<real>("ColumnDensity", num_species, data->np_tot[KDIR],
                                     data->np_tot[JDIR], data->np_tot[IDIR]);

  column_sum = new Column(IDIR, 1, data);

  this->radialRank = 0;
  this->radialSize = 1;
#if WITH_MPI
  // Set up the radial communicator
  int remainDims[3] = {true, false, false};  // Keep the radial direction only
  MPI_SAFE_CALL(MPI_Cart_sub(data->mygrid->CartComm, remainDims, &this->RadialComm));
  MPI_SAFE_CALL(MPI_Comm_rank(this->RadialComm, &this->radialRank));
  MPI_SAFE_CALL(MPI_Comm_size(this->RadialComm, &this->radialSize));

  this->localOffset =
      IdefixArray3D<real>("localOffset", this->radialSize, data->np_tot[KDIR], data->np_tot[JDIR]);
  this->globalOffset =
      IdefixArray3D<real>("globalOffset", this->radialSize, data->np_tot[KDIR], data->np_tot[JDIR]);
#endif
}

LongCharIrradiation::~LongCharIrradiation() {
  if (column_sum != NULL) delete column_sum;
}

void LongCharIrradiation::ComputeIrradiation() {
  idfx::RegionWrapper region("Irradiation::ComputeIrradiation");

  Kokkos::Timer timer;
  elapsedTime -= timer.seconds();

  if (haveUserOpacity) user_opacity(data, kappa);

  if (haveUserRadiationField) user_radiation(data, radiation_field);

  // Start by filling the per-cell column
  IdefixArray4D<real> Vc = data->hydro->Vc;
  IdefixArray1D<real> dx = data->dx[IDIR];
  IdefixArray1D<real> xl = data->xl[IDIR];
  IdefixArray3D<real> A = data->A[IDIR];
  IdefixArray3D<real> dV = data->dV;

  int ibeg = data->beg[IDIR];
  int iend = data->end[IDIR];
  int jbeg = data->beg[JDIR];
  int jend = data->end[JDIR];
  int kbeg = data->beg[KDIR];
  int kend = data->end[KDIR];

  IdefixArray4D<real> column = this->column;
  IdefixArray3D<real> boundaryColumn = this->boundaryColumn;
  IdefixArray3D<real> loffset = this->localOffset;
  IdefixArray3D<real> goffset = this->globalOffset;
  int rank = this->radialRank;

  if (haveUserColumnBoundary) UserDefColumnBoundary(data, boundaryColumn);

  // RAB: Now uses idefix's column routines
  IdefixArray4D<real> dcol = IdefixArray4D<real>("dcolumn", num_species, data->np_tot[KDIR],
                                                 data->np_tot[JDIR], data->np_tot[IDIR]);

  for (int s = 0; s < num_species; s++) {
    IdefixArray4D<real> Vc;
    if (s == 0)
      Vc = data->hydro->Vc;
    else
      Vc = data->dust[s - 1]->Vc;

    idefix_for(
        "ConstructColumn", kbeg, kend, jbeg, jend, ibeg, iend, KOKKOS_LAMBDA(int k, int j, int i) {
          dcol(s, k, j, i) = Vc(RHO, k, j, i);

          if (rank == 0 && i == ibeg) dcol(s, k, j, i) += boundaryColumn(s, k, j);
        });

    column_sum->ComputeColumn(dcol, s);
    IdefixArray3D<real> col_result = column_sum->GetColumn();

    idefix_for(
        "SaveColumn", kbeg, kend, jbeg, jend, ibeg, iend, KOKKOS_LAMBDA(int k, int j, int i) {
          column(s, k, j, i) = col_result(k, j, i);

          // Save the partial column again for later
          dcol(s, k, j, i) = Vc(RHO, k, j, i) * dV(k, j, i) / (0.5 * (A(k, j, i) + A(k, j, i + 1)));
        });
  }

  int num_bands = this->num_bands;
  int num_spec = num_species;
  IdefixArray2D<real> kappa = this->kappa;
  IdefixArray1D<real> lum = this->radiation_field;

  IdefixArray4D<real> heating = this->irradiationHeating;

  real u_lum = this->unit_luminosity;
  real u_opac = this->unit_opacity;

  idefix_for(
      "ComputeHeating", kbeg, kend, jbeg, jend, ibeg, iend, KOKKOS_LAMBDA(int k, int j, int i) {
        real solid_angle = 1 / (4 * M_PI * xl(i) * xl(i));
        real norm = (1 / u_lum) * solid_angle * A(k, j, i) / dV(k, j, i);

        real dl = dV(k, j, i) / (0.5 * (A(k, j, i) + A(k, j, i + 1)));

        for (int l = 0; l < num_bands; l++) {
          real tau = 0, dtau = 1e-300;
          for (int s = 0; s < num_spec; s++) {
            tau += column(s, k, j, i) * kappa(s, l);
            dtau += dcol(s, k, j, i) * kappa(s, l);
          }

          real heat_l = norm * lum(l) * exp(-tau / u_opac) * (-expm1(-dtau / u_opac));

          for (int s = 0; s < num_spec; s++) {
            if (l == 0) heating(s, k, j, i) = 0;
            heating(s, k, j, i) += heat_l * (dl * kappa(s, l) / dtau);
          }
        }
      });

  // Add the heating to the total energy if we don't have FLD.
  if (!data->haveRadiation) UpdatePressure();

  elapsedTime += timer.seconds();
}

void LongCharIrradiation::GetBoundaryFlux(int dir, int side, IdefixArray2D<real>) {
  IDEFIX_ERROR("LongCharIrradiation::GetBoundaryFlux is not implemented");
}

void Irradiation::ShowConfig() {
  if (haveUserOpacity)
    idfx::cout << "Irradiation: opacity type: " << "userconst" << ".\n";
  else
    idfx::cout << "Irradiation: opacity type: " << "constant" << ".\n";

  idfx::cout << "Irradiation: number of species " << num_species << ".\n";
}

void LongCharIrradiation::ShowConfig() {
  idfx::cout << "Irradiation: using Spherical Long Characteristics.\n";
  Irradiation::ShowConfig();
}

void LongCharIrradiation::_ComputeRadiationPressureSourceTerm(real dt, int species) {
  idfx::RegionWrapper region("LongCharIrradiation::ComputeRadiationPressureSourceTerm");

  IdefixArray4D<real> heating = this->irradiationHeating;
  real code_c = idfx::units.c / idfx::units.GetVelocity();
  real dt_over_c = dt / code_c;

  if (species == 0) {
    IdefixArray4D<real> Uc = data->hydro->Uc;
    IdefixArray4D<real> Vc = data->hydro->Vc;

    idefix_for(
        "IrradiationPressureSource", data->beg[KDIR], data->end[KDIR], data->beg[JDIR],
        data->end[JDIR], data->beg[IDIR], data->end[IDIR],

        KOKKOS_LAMBDA(int k, int j, int i) {
          Uc(VX1, k, j, i) += heating(0, k, j, i) * Vc(RHO, k, j, i) * dt_over_c;
        });
  } else {
    IdefixArray4D<real> Uc = data->dust[species - 1]->Uc;
    IdefixArray4D<real> Vc = data->dust[species - 1]->Vc;

    idefix_for(
        "DustPressureSource", data->beg[KDIR], data->end[KDIR], data->beg[JDIR], data->end[JDIR],
        data->beg[IDIR], data->end[IDIR],

        KOKKOS_LAMBDA(int k, int j, int i) {
          Uc(VX1, k, j, i) += heating(species, k, j, i) * Vc(RHO, k, j, i) * dt_over_c;
        });
  }
}
