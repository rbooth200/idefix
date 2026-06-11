
// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#include <string>
#include <vector>

#include "FLD.hpp"
#include "dataBlock.hpp"
#include "fluid.hpp"
#include "idefix.hpp"
#include "units.hpp"
#include "vector.hpp"

MultiSpeciesFLD::MultiSpeciesFLD(Input& input, DataBlock* datain)
    : FluxLimitedDiffusion(input, datain) {
  Init(input, datain);
}

void MultiSpeciesFLD::Init(Input& input, DataBlock* datain) {
  idfx::RegionWrapper region("MultiSpeciesFLD::Init");

  auto data = this->data;

  if (!input.Get<bool>("Dust", "have_energy", 0)) {
    IDEFIX_ERROR("Dust must have an energy equation for MultiSpeciesFLD");
  }

  // Initialize our workspace arrays
  this->kappaP = IdefixArray4D<real>("kappaP", num_species, data->np_tot[KDIR], data->np_tot[JDIR],
                                     data->np_tot[IDIR]);

  this->kappaR = IdefixArray4D<real>("kappaR", num_species, data->np_tot[KDIR], data->np_tot[JDIR],
                                     data->np_tot[IDIR]);
  this->radZ = IdefixArray4D<real>("RadiationZ", num_species, data->np_tot[KDIR],
                                   data->np_tot[JDIR], data->np_tot[IDIR]);

  this->radGamma = IdefixArray4D<real>("RadiationGamma", num_species, data->np_tot[KDIR],
                                       data->np_tot[JDIR], data->np_tot[IDIR]);

  this->k_eff =
      IdefixArray3D<real>("kappa_eff", data->np_tot[KDIR], data->np_tot[JDIR], data->np_tot[IDIR]);

  this->cV_eff =
      IdefixArray3D<real>("cV_eff", data->np_tot[KDIR], data->np_tot[JDIR], data->np_tot[IDIR]);

  this->u_eff =
      IdefixArray3D<real>("u_eff", data->np_tot[KDIR], data->np_tot[JDIR], data->np_tot[IDIR]);

  this->l_eff =
      IdefixArray3D<real>("l_eff", data->np_tot[KDIR], data->np_tot[JDIR], data->np_tot[IDIR]);

  // Fill arrays with 0
  auto nu = this->nu;
  auto radZ = this->radZ;
  auto radG = this->radGamma;
  auto k_eff = this->k_eff;
  auto cV_eff = this->cV_eff;
  auto u_eff = this->u_eff;
  auto l_eff = this->l_eff;
  int num_species = this->num_species;
  bool havePreconditioner = this->havePreconditioner;

  idefix_for(
      "InitRadiationArrays", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, data->np_tot[IDIR],
      KOKKOS_LAMBDA(int k, int j, int i) {
        k_eff(k, j, i) = ZERO_F;
        cV_eff(k, j, i) = ZERO_F;
        u_eff(k, j, i) = ZERO_F;
        l_eff(k, j, i) = ZERO_F;
        for (int s = 0; s < num_species; s++) {
          radZ(s, k, j, i) = ZERO_F;
          radG(s, k, j, i) = ZERO_F;
        }
      });
}

void MultiSpeciesFLD::UpdatePressure() {
  idfx::RegionWrapper region("FLD::UpdatePressure");

  IdefixArray3D<real> Erad = this->Erad;  // Radiation energy
  IdefixArray4D<real> Z = this->radZ;
  IdefixArray3D<real> k_eff = this->k_eff;
  IdefixArray3D<real> u_eff = this->u_eff;

  const real cV = this->cV;
  const real dt = this->dt;
  const real code_c = this->code_c;

  const int haveIrradiation = data->haveIrradiation;
  IdefixArray4D<real> Sirrad;
  if (haveIrradiation) Sirrad = data->irradiation->irradiationHeating;
  int num_species = this->num_species;

  {
    IdefixArray4D<real> Vc = data->hydro->Vc;
    const real mu = this->mu;

    idefix_for(
        "UpdatePressureGas", data->beg[KDIR], data->end[KDIR], data->beg[JDIR], data->end[JDIR],
        data->beg[IDIR], data->end[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
          real rho = Vc(RHO, k, j, i);

          real newtmp =
              (u_eff(k, j, i) + dt * code_c * k_eff(k, j, i) * Erad(k, j, i)) / Z(0, k, j, i);

          Vc(PRS, k, j, i) = newtmp * rho / mu;
        });
  }

  for (int s = 1; s < num_species; s++) {
    IdefixArray4D<real> Vc_gas = data->hydro->Vc;
    IdefixArray4D<real> Vc = data->dust[s - 1]->Vc;
    IdefixArray4D<real> Gamma = this->radGamma;

    real cV = data->dust[s - 1]->drag->cV;

    idefix_for(
        "UpdatePressureDust", data->beg[KDIR], data->end[KDIR], data->beg[JDIR], data->end[JDIR],
        data->beg[IDIR], data->end[IDIR], KOKKOS_LAMBDA(int k, int j, int i) {
          real rho = Vc(RHO, k, j, i);
          real tmp = Vc(TRD, k, j, i);
          real tgas = mu * Vc_gas(PRS, k, j, i) / Vc_gas(RHO, k, j, i);

          real gamma = Gamma(s, k, j, i);
          real l = dt * code_c * kappaP(s, k, j, i) * code_aR * tmp * tmp * tmp;

          real newtmp = (rho * cV * tmp + 3 * l * tmp +
                         dt * code_c * kappaP(s, k, j, i) * Erad(k, j, i) + gamma * tgas) /
                        Z(s, k, j, i);

          if (haveIrradiation) {
            newtmp += dt * rho * Sirrad(s, k, j, i) / Z(s, k, j, i);
          }

          Vc(TRD, k, j, i) = newtmp;
        });
  }
}

void MultiSpeciesFLD::FillUtils() {
  idfx::RegionWrapper region("MultiSpeciesFLD::FillUtils");

  IdefixArray3D<real> Erad = this->Erad;  // Radiation energy
  IdefixArray3D<real> nu = this->nu;
  IdefixArray4D<real> Z = this->radZ;
  IdefixArray3D<real> k_eff = this->k_eff;
  IdefixArray3D<real> cV_eff = this->cV_eff;
  IdefixArray3D<real> u_eff = this->u_eff;
  IdefixArray3D<real> l_eff = this->l_eff;
  IdefixArray3D<real> rhs = this->rhs;  // Right hand side -> same units as Erad

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
  IdefixArray4D<real> kappaR = this->kappaR;
  IdefixArray4D<real> kappaP = this->kappaP;
  IdefixArray3D<real> lambda = nu;

  // Fill the Rosseland and Planck opacities
  if (haveUserDefOpacity) {
    for (int s = 0; s < num_species; s++) {
      IdefixArray3D<real> _kappaP = subview(kappaP, s, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
      IdefixArray3D<real> _kappaR = subview(kappaR, s, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
      data->radiation->UserOpacityFuncCGS(data, s, _kappaP, _kappaR);
    }
  } else {
    auto _kappaR = data->radiation->constkappaR;
    auto _kappaP = data->radiation->constkappaP;

    idefix_for(
        "FillKappa", 0, num_species, kbeg, kend, jbeg, jend, ibeg, iend,
        KOKKOS_LAMBDA(int s, int k, int j, int i) {
          kappaR(s, k, j, i) = _kappaR[s];
          kappaP(s, k, j, i) = _kappaP[s];
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
            kappaR(0, k, j, i) = kappaR(s, k, j, i) * Vc(RHO, k, j, i) / u_opac;
            kappaP(s, k, j, i) = kappaP(s, k, j, i) * Vc(RHO, k, j, i) / u_opac;
          } else {
            kappaR(0, k, j, i) += kappaR(s, k, j, i) * Vc(RHO, k, j, i) / u_opac;
            kappaP(s, k, j, i) = kappaP(s, k, j, i) * Vc(RHO, k, j, i) / u_opac;
          }
        });
  }

  // Compute the flux limiter
  FillFluxLimiter(kappaR, lambda);

  // Fill the final radiation arrays
  {  // Gas
    IdefixArray4D<real> Vc = data->hydro->Vc;
    const real mu = this->mu;
    const real cV = this->cV;

    idefix_for(
        "FillUtilsGas", kbeg, kend, jbeg, jend, ibeg, iend, KOKKOS_LAMBDA(int k, int j, int i) {
          real rho = Vc(RHO, k, j, i);
          real prs = Vc(PRS, k, j, i);  // Works for dust too if tracers are used.
          real tmp = mu * prs / rho;

          real l = dt * code_c * kappaP(0, k, j, i) * code_aR * tmp * tmp * tmp;

          nu(k, j, i) = lambda(k, j, i) * code_c / kappaR(0, k, j, i);

          rhs(k, j, i) = Erad(k, j, i) - 3 * l * tmp;

          k_eff(k, j, i) = kappaP(0, k, j, i);
          cV_eff(k, j, i) = rho * cV;
          u_eff(k, j, i) = rho * cV * tmp + 3 * l * tmp;
          l_eff(k, j, i) = l;

          Z(0, k, j, i) = rho * cV + 4 * l;

          if (haveIrradiation) {
            u_eff(k, j, i) += dt * rho * Sirrad(0, k, j, i);
          }
        });
  }

  IdefixArray4D<real> Gamma = this->radGamma;

  for (int s = 1; s < num_species; s++) {
    IdefixArray4D<real> Vc_gas = data->hydro->Vc;
    IdefixArray4D<real> Vc = data->dust[s - 1]->Vc;

    real cV = data->dust[s - 1]->drag->cV;

    real alpha = data->dust[s - 1]->drag->alpha_coll * this->cV;
    auto gammaDrag = data->dust[s - 1]->drag->gammaDrag;
    gammaDrag.RefreshUserDrag(data);

    idefix_for(
        "FillUtilsDust", kbeg, kend, jbeg, jend, ibeg, iend, KOKKOS_LAMBDA(int k, int j, int i) {
          real rho = Vc(RHO, k, j, i);
          real tmp = Vc(TRD, k, j, i);

          real gamma = Gamma(s, k, j, i) =
              dt * gammaDrag.GetGamma(k, j, i) * alpha * rho * Vc_gas(RHO, k, j, i);
          real l = dt * code_c * kappaP(s, k, j, i) * code_aR * tmp * tmp * tmp;
          real Zd = rho * cV + 4 * l + gamma;

          Z(s, k, j, i) = Zd;
          Z(0, k, j, i) += (rho * cV + 4 * l) * (gamma / Zd);

          k_eff(k, j, i) += kappaP(s, k, j, i) * (gamma / Zd);
          cV_eff(k, j, i) += rho * cV * (gamma / Zd);
          u_eff(k, j, i) += (rho * cV * tmp + 3 * l * tmp) * (gamma / Zd);
          l_eff(k, j, i) += l * (gamma / Zd);

          rhs(k, j, i) += -3 * l * tmp + 4 * l * (rho * cV * tmp + 3 * l * tmp) / Zd;

          if (haveIrradiation) {
            u_eff(k, j, i) += dt * rho * Sirrad(s, k, j, i) * (gamma / Zd);
            rhs(k, j, i) += dt * rho * Sirrad(s, k, j, i) * (4 * l / Zd);
          }
        });
  }

  idefix_for(
      "FinishRHS", kbeg, kend, jbeg, jend, ibeg, iend, KOKKOS_LAMBDA(int k, int j, int i) {
        rhs(k, j, i) += 4 * l_eff(k, j, i) * u_eff(k, j, i) / Z(0, k, j, i);
      });
}
void MultiSpeciesFLD::FillMatrixCouplingTerms() {
  idfx::RegionWrapper region("MultiSpeciesFLD::FillMatrixCouplingTerms");

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
  IdefixArray4D<real> Z = this->radZ;
  IdefixArray3D<real> cV_eff = this->cV_eff;
  IdefixArray3D<real> u_eff = this->u_eff;

  IdefixArray4D<real> kappaP = this->kappaP;

  const real code_c = this->code_c;
  const real dt = this->dt;

  idefix_for(
      "CouplingGas", kbeg, kend, jbeg, jend, ibeg, iend, KOKKOS_LAMBDA(int k, int j, int i) {
        M(0, k, j, i) = 1 + dt * code_c * k_eff(k, j, i) * cV_eff(k, j, i) / Z(0, k, j, i);
      });

  for (int s = 1; s < num_species; s++) {
    IdefixArray4D<real> Vc = data->dust[s - 1]->Vc;
    real cV = data->dust[s - 1]->drag->cV;

    idefix_for(
        "CouplingDust", kbeg, kend, jbeg, jend, ibeg, iend, KOKKOS_LAMBDA(int k, int j, int i) {
          real rho = Vc(RHO, k, j, i);

          M(0, k, j, i) += dt * code_c * rho * cV * kappaP(s, k, j, i) / Z(s, k, j, i);
        });
  }

  auto rhs = this->rhs;
  auto Erad = this->Erad;
  auto Vc = this->data->hydro->Vc;
  auto Vc2 = this->data->dust[0]->Vc;

  const real code_aR = this->code_aR;
  const real mu = this->mu;
}
void MultiSpeciesFLD::ShowConfig() {
  idfx::cout << "RT: MultiSpeciesFLD with " << num_species << "species\n";
  FluxLimitedDiffusion::ShowConfig();
}
