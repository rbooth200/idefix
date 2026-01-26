#include "idefix.hpp"
#include "setup.hpp"
#include "units.hpp"

void MyRadiationBoundary(DataBlock &data, int dir, BoundarySide side, real t, IdefixArray3D<real> &Erad) {

}

// User-defined boundaries
void UserdefBoundary(Hydro *hydro, int dir, BoundarySide side, real t) {
   auto *data = hydro->data;
    throw std::runtime_error("User boundary not implemented");
}


static real Tg, Td, Tr, rho0, d2g;
static std::string out_file;

void WriteState(DataBlock& data, const real t, const real dt) {
  std::ofstream f(out_file, std::ios_base::app | std::ios_base::out);

  int i = data.beg[IDIR], j = data.beg[JDIR], k = data.beg[KDIR];
  const real mu = data.radiation->mu;
  const real aR = data.radiation->code_aR;

  DataBlockHost d(data);
  d.SyncFromDevice();

  f << t << " " 
    << d.Vc(RHO,k,j,i) << " " << d.dustVc[0](RHO,k,j,i) << " "
    << mu * d.Vc(PRS,k,j,i) / d.Vc(RHO,k,j,i) << " "
    << d.dustVc[0](TRD,k,j,i) / d.dustVc[0](RHO,k,j,i) << " "
    << pow(d.Erad(k,j,i)/ aR, 0.25) << "\n" ;
     
}


Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
  
  Tg =  input.Get<real>("Setup", "Tgas", 0) / idfx::units.GetKelvin();
  Td =  input.Get<real>("Setup", "Tdust", 0) / idfx::units.GetKelvin();
  Tr =  input.Get<real>("Setup", "Trad", 0) / idfx::units.GetKelvin();

  rho0 = input.Get<real>("Setup", "rho0", 0) /  idfx::units.GetDensity();
  d2g =  input.Get<real>("Setup", "d2g", 0) ;

  out_file = input.Get<std::string>("Setup", "monitoring", 0);

  std::ofstream f(out_file);
  f << "# t rho_g rho_d Tg Td Tr\n";


  data.EnrollUserStepLast(WriteState);

  //data.irradiation->EnrollUserColumnBoundary(&MyColumnBoundary) ;

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
                real rho = rho0 ;

                d.Vc(RHO,k,j,i) = rho;
                d.Vc(VX1,k,j,i) = 0.0;

                d.Vc(PRS, k,j,i) = rho * Tg / mu ;
                d.Erad(k,j,i) = aR * pow(Tr, 4) ;        
            }
        }
    }

    int nDust = d.dustVc.size();
    for(int k = 0; k < d.np_tot[KDIR] ; k++) {
        for(int j = 0; j < d.np_tot[JDIR] ; j++) {
            for(int i = 0; i < d.np_tot[IDIR] ; i++) {

                real R = d.x[IDIR](i) ;
                real rho = rho0 * d2g ;

                for (int s=0; s < nDust; s++) {
                    d.dustVc[s](RHO,k,j,i) = rho;
                    d.dustVc[s](VX1,k,j,i) = 0.0;
                    d.dustVc[s](TRD, k,j,i) = rho * Td ;
                }
            }
        }
    }

    // Send it all, if needed
    d.SyncToDevice();
}


void MakeAnalysis(DataBlock & data) {}
