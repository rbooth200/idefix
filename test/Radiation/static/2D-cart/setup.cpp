#include "idefix.hpp"
#include "setup.hpp"

void MyRadiationBoundary(DataBlock &data, int dir, BoundarySide side, real t, IdefixArray3D<real> &Erad) {

    if (dir==IDIR) {
        IdefixArray4D<real> Vc = data.hydro->Vc;
        IdefixArray1D<real> x1 = data.x[IDIR];

        int ibeg,iend;
        if(side == left) {
            ibeg = 0;
            iend = data.beg[IDIR];
        }
        else if(side==right) {
            ibeg = data.end[IDIR];
            iend = data.np_tot[IDIR];
        }

        const real mu = data.radiation->mu;
        const real rho0 = 1.0;
        const real prs0 = 1.0;
        const real tmp0 = mu * prs0/rho0;
        const real Er0 = data.radiation->code_aR * pow(tmp0, 4);

        idefix_for("RadiationBoundary",
          0, data.np_tot[KDIR],
          0, data.np_tot[JDIR],
          ibeg, iend,
                    KOKKOS_LAMBDA (int k, int j, int i) {
                        real x = x1(i);
                        Erad(k,j,i) = Er0 * (1+x);
                    });
    }
}

// User-defined boundaries
void UserdefBoundary(Hydro *hydro, int dir, BoundarySide side, real t) {
   auto *data = hydro->data;

    if (dir==IDIR) {
        IdefixArray4D<real> Vc = hydro->Vc;
        IdefixArray1D<real> x1 = data->x[IDIR];

        int ibeg,iend;
        if(side == left) {
            ibeg = 0;
            iend = data->beg[IDIR];
        }
        else if(side==right) {
            ibeg = data->end[IDIR];
            iend = data->np_tot[IDIR];
        }

        const double aR = data->radiation->code_aR;
        const double mu = data->radiation->mu;
        const double Er0 = aR * pow(mu, 4);

        idefix_for("UserDefBoundary",
          0, data->np_tot[KDIR],
          0, data->np_tot[JDIR],
          ibeg, iend,
                    KOKKOS_LAMBDA (int k, int j, int i) {
                        real x = x1(i);
                        Vc(RHO,k,j,i) = 1.0;
                        Vc(VX1,k,j,i) = 0.0;
                        Vc(VX2,k,j,i) = 0.0;
                        const real Er = Er0 * (1+x);
                        const real tmp = pow(Er/aR, 0.25);
                        // Vc(PRS,k,j,i) = 1.0;// * tmp / mu;
                    });
    }
}


void ResetQuantities(Hydro *hydro, const real t, const real dtin) {

    auto *data = hydro->data;
    real gamma = hydro->eos->GetGamma();
    // If radiation is needed, update it
    // if(data->haveRadiation) data->radiation->SolveSystem();
    
    IdefixArray4D<real> Vc = hydro->Vc;
    IdefixArray4D<real> Uc = hydro->Uc;

    idefix_for("ResetQuantities",
          0, data->np_tot[KDIR],
          0, data->np_tot[JDIR],
          0, data->np_tot[IDIR],
                    KOKKOS_LAMBDA (int k, int j, int i) {
                        Uc(RHO,k,j,i) = 1.0;
                        Uc(VX1,k,j,i) = 0.0;
                        Uc(VX2,k,j,i) = 0.0;
                        Vc(RHO,k,j,i) = 1.0;
                        Vc(VX1,k,j,i) = 0.0;
                        Vc(VX2,k,j,i) = 0.0;
                        // Uc(ENG,k,j,i) = 1.0 / (gamma - 1.0);
    });
}

Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
  data.hydro->EnrollUserDefBoundary(&UserdefBoundary);
  if (data.haveRadiation) data.radiation->EnrollUserDefBoundary(&MyRadiationBoundary);
  data.hydro->EnrollUserSourceTerm(&ResetQuantities);
}


void Setup::InitFlow(DataBlock &data) {
    DataBlockHost d(data);
    
    const real mu = data.radiation->mu;
    const real aR = data.radiation->code_aR;
    const real Er0 = aR * pow(mu, 4);
    
    for(int k = 0; k < d.np_tot[KDIR] ; k++) {
        for(int j = 0; j < d.np_tot[JDIR] ; j++) {
            for(int i = 0; i < d.np_tot[IDIR] ; i++) {

                real x = d.x[IDIR](i);
                real y = d.x[JDIR](j);

                d.Vc(RHO,k,j,i) = 1.0;
                d.Vc(VX1,k,j,i) = 0.0;
                d.Vc(VX2,k,j,i) = 0.0;
                // d.Vc(VX3,k,j,i) = 0.0;
                d.Erad(k,j,i) = Er0 * (1+x);
                const real tmp = pow(d.Erad(k,j,i)/aR, 0.25);
                d.Vc(PRS,k,j,i) = 1.0;// * tmp / mu;
        
            }
        }
    }

    // Send it all, if needed
    d.SyncToDevice();
}

void MakeAnalysis(DataBlock & data) {}
