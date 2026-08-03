#include "idefix.hpp"
#include "setup.hpp"
#include "units.hpp"

void MyRadiationBoundary(DataBlock &data, int dir, BoundarySide side, real t, IdefixArray3D<real> &Erad) {

}

void DiscColumnBoundary(DataBlock* data, IdefixArray3D<real> column) {


    real R0 = data->xbeg[0] ;
    IdefixArray4D<real> Vc = data->hydro->Vc ;
    int i = data->beg[0] ;

   idefix_for("SetBoundaryColumn", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR],
    KOKKOS_LAMBDA (int k, int j)  {
      column(0, k, j) = 3*R0*Vc(RHO, k,j,i) ;
    });
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
                        Vc(PRS,k,j,i) += 1.0;
                    });
    }
}

real Sigma0, p, h0, R0, kflare, d2g[2];
real u_Sig ;

Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {

  // Store parameters
  Sigma0 = input.Get<real>("Setup", "Sigma0", 0) ;
  p = input.Get<real>("Setup", "SigmaSlope", 0) ;

  h0 = input.Get<real>("Setup", "h0", 0) ;
  R0 = input.Get<real>("Setup", "r0", 0) ;
  kflare =  input.Get<real>("Setup", "kflare", 0) ;
  d2g[0] = input.Get<real>("Setup", "d2g", 0) ;
  d2g[1] = input.Get<real>("Setup", "d2g", 1) ;

  // Read units
  u_Sig = idfx::units.GetDensity() * idfx::units.GetLength() ;

  data.irradiation->EnrollUserColumnBoundary(&DiscColumnBoundary) ;

 //  data.hydro->EnrollUserDefBoundary(&UserdefBoundary);
 //  if (data.haveRadiation) data.radiation->EnrollUserDefBoundary(&MyRadiationBoundary);
}


void Setup::InitFlow(DataBlock &data) {
    DataBlockHost d(data);

    const real mu = data.radiation->mu;
    const real aR = data.radiation->code_aR;
    const real Er0 = aR * pow(mu, 4);

    int nDust = d.dustVc.size();

    for(int k = 0; k < d.np_tot[KDIR] ; k++) {
        for(int j = 0; j < d.np_tot[JDIR] ; j++) {
            for(int i = 0; i < d.np_tot[IDIR] ; i++) {

                real R = d.x[IDIR](i) * sin(d.x[JDIR](j)) ;
                real z = d.x[IDIR](i) * cos(d.x[JDIR](j)) ;
                real c_phi = cos(d.x[KDIR](k)) ;

                real H = h0 * R * pow(R/R0, kflare) ;
                real rho = (Sigma0 / u_Sig) * pow(R/R0, -p) * exp(-0.5*z*z/(H*H)) / (H * sqrt(2*M_PI)) ;
                real T =  mu * (H*H) / (R*R*R) ;

                d.Vc(RHO,k,j,i) = rho ;
                d.Vc(VX1,k,j,i) = 0.0;
                d.Vc(VX2,k,j,i) = 0.0;
                d.Vc(VX3,k,j,i) = 0.0;

                d.Vc(PRS, k,j,i) = rho * T / mu * (1 + 0.1*c_phi) ;
                d.Erad(k,j,i) = aR * pow(T, 4) ;

                for (int s=0; s < nDust; s++) {
                    d.dustVc[s](RHO,k,j,i) = rho*d2g[s];
                    d.dustVc[s](VX1,k,j,i) = 0.0;
                    d.dustVc[s](VX2,k,j,i) = 0.0;
                    d.dustVc[s](VX3,k,j,i) = 0.0;

                    d.dustVc[s](TRD, k,j,i) = T;
                }
            }
        }
    }


    // Send it all, if needed
    d.SyncToDevice();
}

void MakeAnalysis(DataBlock & data) {}
