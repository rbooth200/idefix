#include "idefix.hpp"
#include "setup.hpp"
#include "units.hpp"

void MakeAnalysis(DataBlock & data) ;

static real T, rho0, d2g, amp, drho[4], dvel[4], dT[4], dEr[2];
static std::string out_file;

Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
  
  T =  input.Get<real>("Setup", "T", 0) / idfx::units.GetKelvin();
  rho0 = input.Get<real>("Setup", "rho", 0);
  d2g = input.Get<real>("Setup", "d2g", 0);

  for (int i=0; i < 4; i++) {
    drho[i] = input.Get<real>("Setup", "drho", i);
    dvel[i] = input.Get<real>("Setup", "dvel", i);
    dT[i]   = input.Get<real>("Setup", "dTmp", i);
    if (i < 2)
        dEr[i] = input.Get<real>("Setup", "dEr", i);
  }
  amp = input.Get<real>("Setup", "amp", 0);
  
  output.EnrollAnalysis(&MakeAnalysis);
  out_file = input.Get<std::string>("Output", "dmp_dir", 0) + "/analysis.txt";

  std::ofstream f(out_file);
  f << "# t rho_g v_g Tg rho_d v_d Td Er\n";
}


void Setup::InitFlow(DataBlock &data) {
    DataBlockHost d(data);
    
    const real mu = data.radiation->mu;
    const real aR = data.radiation->code_aR;

    const real cs = std::sqrt(T/mu) ;
    const real kx = 2*M_PI ;
    
    for(int k = 0; k < d.np_tot[KDIR]; k++) {
        for(int j = 0; j < d.np_tot[JDIR]; j++) {
            for(int i = 0; i < d.np_tot[IDIR]; i++) {

                real c = std::cos(kx*d.x[IDIR](i)), s = std::sin(kx*d.x[IDIR](i));
                {
                    real rho = rho0*(1 + amp*(drho[0]*c - drho[1]*s));
                    real  vg =      cs * amp*(dvel[0]*c - dvel[1]*s);
                    real  Tg =    T*(1 + amp*(  dT[0]*c -   dT[1]*s));

                    d.Vc(RHO,k,j,i) = rho;
                    d.Vc(VX1,k,j,i) = vg;
                    d.Vc(PRS, k,j,i) = rho * Tg / mu;
                }
                {
                    real rho = rho0*(1 + amp*(drho[2]*c - drho[3]*s)) * d2g;
                    real  vd =      cs * amp*(dvel[2]*c - dvel[3]*s);
                    real  Td =    T*(1 + amp*(  dT[2]*c -   dT[3]*s));

                    d.dustVc[0](RHO,k,j,i) = rho;
                    d.dustVc[0](VX1,k,j,i) = vd;
                    d.dustVc[0](TRD,k,j,i) = rho * Td;
                }
                d.Erad(k,j,i) = aR * pow(T, 4)*(1 + amp*(dEr[0]*c - dEr[1]*s));
            }
        }
    }

    // Send it all, if needed
    d.SyncToDevice();
    MakeAnalysis(data);
}


void MakeAnalysis(DataBlock & data) {
  std::ofstream f(out_file, std::ios_base::app | std::ios_base::out);

  int i = data.beg[IDIR], j = data.beg[JDIR], k = data.beg[KDIR];
  const real mu = data.radiation->mu;
  const real aR = data.radiation->code_aR;

  DataBlockHost d(data);
  d.SyncFromDevice();

  f << std::setprecision(12);
  f << data.t << " " 
    << d.Vc(RHO,k,j,i) << " " << d.Vc(VX1,k,j,i) << " " << mu * d.Vc(PRS,k,j,i) / d.Vc(RHO,k,j,i) << " "
    << d.dustVc[0](RHO,k,j,i) << " " << d.dustVc[0](VX1,k,j,i) << " " << d.dustVc[0](TRD,k,j,i) / d.dustVc[0](RHO,k,j,i) << " "
    << d.Erad(k,j,i) << "\n";
}
