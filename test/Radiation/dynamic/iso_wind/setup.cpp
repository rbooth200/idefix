#include "idefix.hpp"
#include "setup.hpp"
#include "units.hpp"
#include "potential.hpp"


void MyRadiationBoundary(DataBlock &data, int dir, BoundarySide side, real t, IdefixArray3D<real> &Erad);
void Analysis(DataBlock & data);


static real T0, R_Hill, R_Bondi, R_p, mu ;
static const real amp = 0.25;
TidalPotential tidal_potential(0,0,1);

void SoundSpeed(DataBlock &data, const real t, IdefixArray3D<real> &cs) {

  real cs_val = sqrt(T0 / mu) ;

  auto th = data.x[JDIR];
  
  idefix_for("cs", 0, data.np_tot[KDIR], 0, data.np_tot[JDIR], 0, data.np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
      cs(k,j,i) = cs_val  * (1 + amp*cos(th(j))) / (1+amp) ;
  });
}


void MyPotential(DataBlock& data, real t, IdefixArray1D<real> &x, IdefixArray1D<real> &y, IdefixArray1D<real> &z, IdefixArray3D<real> &phiP) {    

  idefix_for("ComputePotential", 0, data.np_tot[KDIR],
                                 0, data.np_tot[JDIR],
                                 0, data.np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
      real xpos = x(i) * cos(y(j));
      real ypos = x(i) * sin(y(j)) * sin(z(k));
      real zpos = x(i) * sin(y(j)) * cos(z(k));

      phiP(k,j,i) = tidal_potential(xpos, ypos, zpos);
  });

}

template<typename Phys>
void UserBoundary(Fluid<Phys> *hydro, int dir, BoundarySide side, real t) {

  IdefixArray4D<real> Vc = hydro->Vc;
  const int ighost = hydro->data->nghost[IDIR];

  auto R = hydro->data->x[IDIR];
  auto th = hydro->data->x[JDIR];

  if(dir==IDIR && side == BoundarySide::left) {
    hydro->boundary->BoundaryFor("UserDefBoundary", dir, side,
      KOKKOS_LAMBDA (int k, int j, int i) {
          
        const int iref = ighost;

        for (int n=0; n<Phys::nvar; n++){
          if (n == RHO) {
            real f = pow((1 + amp*cos(th(j))) / (1+amp), -2);
            Vc(n,k,j,i) = f * exp(R_Bondi * (1.0/R(i) - 1.0/R_p));
          }
          else if(n == (VX1+dir)) 
            Vc(n,k,j,i) = fabs(Vc(n,k,j,iref));
          else
            Vc(n,k,j,i) = Vc(n,k,j,iref);
        }

      });
  } else {
    IDEFIX_ERROR("UserBoundary not implemented for this side/dir");
  }
}

template<typename Phys>
void FluxBoundary(Fluid<Phys> *hydro, int dir, BoundarySide side, real t) {

  IdefixArray4D<real> Flux = hydro->FluxRiemann;
  IdefixArray4D<real> Vc = hydro->Vc;

  DataBlock* data = hydro->data;


  if(dir==IDIR && side == BoundarySide::left) {
    int i = data->beg[IDIR];  
    auto th = data->x[JDIR];
    IdefixArray3D<real> A = data->A[dir];
  
    idefix_for("UserFluxBoundary", data->beg[KDIR],data->end[KDIR],
                                   data->beg[JDIR],data->end[JDIR],
      KOKKOS_LAMBDA (int k, int j) {
          
        if (Flux(RHO, k,j,i) < ZERO_F) {
          for (int n=0; n<Phys::nvar; n++){
              Flux(n,k,j,i) = 0;
              if (n == VX1+dir)
                Flux(n,k,j,i) = A(k,j,i) * Vc(RHO, k,j,i) * (T0/mu) * pow((1 + amp*cos(th(j))) / (1+amp), 2);
          }
        }

      });
    }
}


Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
  
  // Store parameters
  T0 =  input.Get<real>("Setup", "T0", 0) / idfx::units.GetKelvin();
  mu = input.Get<real>("Setup", "mu", 0) ;

  // Potential
  real GM = idfx::units.G / (idfx::units.GetLength() * pow(idfx::units.GetVelocity(), 2));

  real a_p = input.Get<real>("Setup", "a_p", 0) * idfx::units.au / idfx::units.GetLength();
  real GMstar = GM * input.Get<real>("Setup", "Mstar", 0) * idfx::units.M_sun;
  real GMplanet = GM * input.GetOrSet<real>("Setup", "M_p", 0, 0) * idfx::units.M_earth;
  
  R_p = grid.xbeg[0];
  R_Hill = pow(GMplanet/(3*GMstar), 1./3.) * a_p ;
  R_Bondi = GMplanet * mu / T0 ;
  idfx::cout << "Hill Radius: " << R_Hill / R_p << " R_p\n";
  idfx::cout << "Bondi Radius: " << R_Bondi / R_p << " R_p\n";
  idfx::cout << "Sound speed: " << sqrt(T0 / mu) << "\n";


  tidal_potential = TidalPotential(GMstar, GMplanet, a_p);



  data.hydro->EnrollUserDefBoundary(&UserBoundary<DefaultPhysics>);
  data.hydro->EnrollFluxBoundary(&FluxBoundary<DefaultPhysics>);

  data.hydro->EnrollIsoSoundSpeed(&SoundSpeed);
  data.gravity->EnrollPotential(&MyPotential);

}


void Setup::InitFlow(DataBlock &data) {
    DataBlockHost d(data);

    for(int k = 0; k < d.np_tot[KDIR]; k++) {
        for(int j = 0; j < d.np_tot[JDIR]; j++) {
            for(int i = 0; i < d.np_tot[IDIR]; i++) {

                real R = d.x[IDIR](i);
                real th = d.x[JDIR](j);
                real f = pow((1 + amp*cos(th)) / (1+amp), -2);
                real rho = exp(1.5 * f * R_Bondi * (1./R - 1./R_p)) ;

                d.Vc(RHO,k,j,i) = rho*f;
                d.Vc(VX1,k,j,i) = 0.0;
                d.Vc(VX2,k,j,i) = 0.0;
                d.Vc(VX3,k,j,i) = 0.0;
            }
        }
    }

    // Send it all, if needed
    d.SyncToDevice();
}

Setup::~Setup() {

}


void Analysis(DataBlock & data) {
}
