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
      column(0, k, j) = 0 ;
    });
}

// User-defined boundaries
void UserdefBoundary(Hydro *hydro, int dir, BoundarySide side, real t) {
   auto *data = hydro->data;
    throw std::runtime_error("User boundary not implemented");
}

static real rho0, p, T0;

Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {

  real k0 = input.Get<real>("Irradiation", "kappa", 2) / data.radiation->unit_opacity;
  // Store parameters
  rho0 = input.Get<real>("Setup", "tau0", 0) / k0 ;
  p = input.Get<real>("Setup", "rhoSlope", 0) ;
  T0 =  input.Get<real>("Setup", "T0", 0) ;

  data.irradiation->EnrollUserColumnBoundary(&DiscColumnBoundary) ;

 //  data.hydro->EnrollUserDefBoundary(&UserdefBoundary);
 //  if (data.haveRadiation) data.radiation->EnrollUserDefBoundary(&MyRadiationBoundary);
}


void Setup::InitFlow(DataBlock &data) {
    DataBlockHost d(data);

    const real mu = data.radiation->mu;
    const real aR = data.radiation->code_aR;


    for(int k = 0; k < d.np_tot[KDIR] ; k++) {
        for(int j = 0; j < d.np_tot[JDIR] ; j++) {
            for(int i = 0; i < d.np_tot[IDIR] ; i++) {

                real R = d.x[IDIR](i) ;
                real rho = rho0 * pow(R, -p) ;

                d.Vc(RHO,k,j,i) = rho;
                d.Vc(VX1,k,j,i) = 0.0;
                d.Vc(VX2,k,j,i) = 0.0;
                d.Vc(VX3,k,j,i) = 0.0;

                d.Vc(PRS, k,j,i) = rho * T0 / mu ;
                d.Erad(k,j,i) = aR * pow(T0, 4) ;
            }
        }
    }

    // Send it all, if needed
    d.SyncToDevice();
}

void MakeAnalysis(DataBlock & data) {}
