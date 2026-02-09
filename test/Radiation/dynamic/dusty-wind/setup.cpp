#include "idefix.hpp"
#include "setup.hpp"
#include "condensation.hpp"
#include "units.hpp"
#include "opacity.hpp"
#include "potential.hpp"


/* TODO: 
 - Radiation Pressure
 - Coriolis Force
*/

void MyRadiationBoundary(DataBlock &data, int dir, BoundarySide side, real t, IdefixArray3D<real> &Erad);
void Analysis(DataBlock & data);

static bool surface_condensation = true;

static real R_Hill, R_Bondi, R_p;
static real T0, d2g, C_surf;
static IdefixArray2D<real> T_surf, F_surf;
std::string surf_base;
std::array<int, 3> n;

// Setup the condensation parameters
//    Perez-Becker & Chiang (2013)
Condensible silicate(169, 3.21e10, 6.72e14, 0.1);

std::unique_ptr<GasOpacity> gas_opac;
std::unique_ptr<DustOpacity> dust_opac;
std::unique_ptr<StellarProperties> star;

TidalPotential tidal_potential(0,0,1);

void MyPotential(DataBlock& data, real t, IdefixArray1D<real> &x, IdefixArray1D<real> &y, IdefixArray1D<real> &z, IdefixArray3D<real> &phiP) {    

  idefix_for("ComputePotential", 0, data.np_tot[KDIR],
                                 0, data.np_tot[JDIR],
                                 0, data.np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
      real xpos = x(i) * cos(y(j));
      real ypos = x(i) * sin(y(j)) ;// * sin(z(k));
      real zpos = 0; //x(i) * sin(y(j)) * cos(z(k));

      phiP(k,j,i) = tidal_potential(xpos, ypos, zpos);
  });

}

template<typename Phys>
void UserBoundary(Fluid<Phys> *hydro, int dir, BoundarySide side, real t) {

  IdefixArray4D<real> Vc = hydro->Vc;
  const int ighost = hydro->data->nghost[IDIR];

  real unit_density = idfx::units.GetDensity();
  real unit_temp = idfx::units.GetKelvin();

  if(dir==IDIR && side == BoundarySide::left) {
    hydro->boundary->BoundaryFor("UserDefBoundary", dir, side,
      KOKKOS_LAMBDA (int k, int j, int i) {
          
        const int iref = 2*(ighost) - i - 1;
        const int sign = (Phys::dust && (Vc(VX1+dir,k,j,iref) < 0))? 1 : -1;

        for (int n=0; n<Phys::nvar; n++) {
          if(n == (VX1+dir)) 
            Vc(n,k,j,i) = sign * Vc(n,k,j,iref);
          else
            Vc(n,k,j,i) = Vc(n,k,j,iref);
        }

      });
  } else {
    IDEFIX_ERROR("UserBoundary not implemented for this side/dir");
  }
}


template<typename Phys, int sign=1>
void ZeroFluxBoundary(Fluid<Phys> *hydro, int dir, BoundarySide side, real t) {

  IdefixArray4D<real> Flux = hydro->FluxRiemann;
  IdefixArray4D<real> Vc = hydro->Vc;

  DataBlock* data = hydro->data;


  if(dir==IDIR && side == BoundarySide::left) {
    int i = data->beg[IDIR];  
    auto th = data->x[JDIR];
    IdefixArray3D<real> A = data->A[dir];
  
    idefix_for("UserFluxBoundary", 0, data->np_tot[KDIR],
                                   0, data->np_tot[JDIR],
      KOKKOS_LAMBDA (int k, int j) {

        real P = Phys::dust ? 0.0 : Vc(PRS,k,j,i);
          
        if (sign*Flux(RHO, k,j,i) <= ZERO_F) {
          for (int n=0; n<Phys::nvar; n++){
              Flux(n,k,j,i) = 0;
              if (n == VX1+dir)
                Flux(n,k,j,i) = A(k,j,i) * P;
          }
        }

      });
    }
}



void ApplyCondensation(DataBlock& data, const real t, const real dt) {

  real u_den = idfx::units.GetDensity();
  real u_temp = idfx::units.GetKelvin();
  real mu = data.radiation->mu;


  if (data.dust.size() != 1) {
    IDEFIX_ERROR("Condensation only works with one species");
  }

  IdefixArray4D<real> Gas = data.hydro->Vc;
  IdefixArray4D<real> Dust = data.dust[0]->Vc;
  auto gammaDrag = data.dust[0]->drag->gammaDrag;

  idefix_for("ApplyCondensation", 0, data.np_tot[KDIR], 0, data.np_tot[JDIR], 0, data.np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i)  {
      real T_g = Gas(PRS,k,j,i) / Gas(RHO, k,j,i) * mu;
      real T_d = Dust(TRD,k,j,i);
      
      // Save total density and momentum
      real rho_t = Gas(RHO, k,j,i) + Dust(RHO, k,j,i);
      real mt1, mt2, mt3;
      EXPAND(
        mt1 = Dust(RHO, k,j,i)*Dust(VX1, k,j,i) + Gas(RHO, k,j,i)*Gas(VX1, k,j,i); ,
        mt2 = Dust(RHO, k,j,i)*Dust(VX2, k,j,i) + Gas(RHO, k,j,i)*Gas(VX2, k,j,i); ,
        mt3 = Dust(RHO, k,j,i)*Dust(VX3, k,j,i) + Gas(RHO, k,j,i)*Gas(VX3, k,j,i);  
      );

      // Equilibrium vapour density
      real rho_v = silicate.rho_vap(T_d*u_temp) * sqrt(T_g/T_d) / u_den;

      real K = 0.75 * gammaDrag.GetGamma(k,j,i) * silicate.P_stick * dt;
      real rho_p = rho_t - rho_v;


      // Compute the new gas/dust density.
      real rho_d, rho_g;
      real x = K * rho_p;
      if (FABS(x) > 1e-6) {
        rho_d = Dust(RHO, k,j,i) / (exp(-x) - Dust(RHO, k,j,i)/rho_p * expm1(-x));
        real t = 1/(expm1(x));
        rho_g = (Dust(RHO, k,j,i)*rho_v +  Gas(RHO, k,j,i)*rho_p*t)/(Dust(RHO, k,j,i) + rho_p*t);
      }
      else {
        rho_d = Dust(RHO, k,j,i) / (1 - x + K*Dust(RHO, k,j,i));
        rho_g = rho_t - rho_d;
      }
      rho_d = fmax(rho_d, 1e-100*rho_t); // prevent 0

      // Update density/temperature
      Dust(RHO, k,j,i) = rho_d;
      Gas(RHO, k,j,i)  = rho_g; 

      Gas(PRS, k,j,i)  = rho_g * T_g / mu ;


      // Update the velocities
      real f = exp(- K * (rho_g + rho_d*rho_v/rho_g));

      real dv1, dv2, dv3;
      EXPAND(
        dv1 = f*(Dust(VX1, k,j,i) - Gas(VX1, k,j,i)); ,
        dv2 = f*(Dust(VX2, k,j,i) - Gas(VX2, k,j,i)); ,
        dv3 = f*(Dust(VX3, k,j,i) - Gas(VX3, k,j,i));
      );
      EXPAND(
        Dust(VX1, k,j,i) = (mt1 + rho_g*dv1)/rho_t; ,
        Dust(VX2, k,j,i) = (mt2 + rho_g*dv2)/rho_t; ,
        Dust(VX3, k,j,i) = (mt3 + rho_g*dv3)/rho_t; 
      );
      EXPAND(
        Gas(VX1, k,j,i) = Dust(VX1, k,j,i) - dv1; ,
        Gas(VX2, k,j,i) = Dust(VX2, k,j,i) - dv2; ,
        Gas(VX3, k,j,i) = Dust(VX3, k,j,i) - dv3;
      );

  }); 

  // Apply condensation at the surface
  if (surface_condensation) {
    int ibeg = data.beg[IDIR];
    IdefixArray3D<real> Ax1 = data.A[IDIR];
    IdefixArray3D<real> dV = data.dV;
    real k0 = sqrt(8/M_PI);

    idefix_for("SurfaceCondensation", 0, data.np_tot[KDIR], 0, data.np_tot[JDIR],
      KOKKOS_LAMBDA (int k, int j)  {
        real T_g = Gas(PRS,k,j,ibeg) / Gas(RHO, k,j,ibeg) * mu;
        real T_s = T_surf(k, j);

        real v_t = k0 * sqrt(Gas(PRS,k,j,ibeg) / Gas(RHO, k,j,ibeg));
        
        // Equilibrium vapour density
        real rho_v = silicate.rho_vap(T_s*u_temp) * sqrt(T_g/T_s) / u_den;

        real K = v_t * silicate.P_stick * Ax1(k,j,ibeg) / dV(k,j,ibeg);
        real x = K*dt; 

        // Compute the new gas density.
        Gas(RHO, k,j,ibeg) = Gas(RHO, k,j,ibeg)*exp(-x) - rho_v*expm1(-x);

        // For now let's leave the temperature and velocity at the boundary.
    }); 
  }

  /*
  int imax[2] = {0,0}, jmax[2] = {0,0}, kmax[2] = {0,0};
  real max[2] = {0,0};
  for (int k=0; k < data.np_tot[KDIR]; k++) 
   for (int j=0; j < data.np_tot[JDIR]; j++) 
     for (int i=0; i < data.np_tot[IDIR]; i++) {

      real x = fabs(Gas(VX1, k,j,i)) + Gas(PRS, k,j,i)/Gas(RHO, k,j,i);
      if (x > max[0]) {
        max[0] = x;
        imax[0] = i;
        jmax[0] = j;
        kmax[0] = k;
      }
      x = fabs(Dust(VX1, k,j,i));
      if (x > max[1]) {
        max[1] = x;
        imax[1] = i;
        jmax[1] = j;
        kmax[1] = k;
      }
     }
  std::cout << " Max velocities after condensation: Gas " << max[0] << " at (" << imax[0] << "," << jmax[0] << "," << kmax[0] << ")"
            << " Dust " << max[1] << " at (" << imax[1] << "," << jmax[1] << "," << kmax[1] << ")" <<std::endl;
  */
}

void MyColumnBoundary(DataBlock* data, IdefixArray3D<real> column) {

  real R0 = data->xend[0];
  IdefixArray4D<real> Vc = data->hydro->Vc;
  int i = data->end[0];
  
  int num_species = data->dust.size()+1 ;
  
  real u_opac = data->radiation->unit_opacity;

  idefix_for("SetBoundaryColumn", 0, num_species, 0, data->np_tot[KDIR], 0, data->np_tot[JDIR],
  KOKKOS_LAMBDA (int s, int k, int j)  {
    if (s == 0) 
      // Use non-zero because strong Fe lines produce high T.
      column(s, k, j) = 1e-4 * u_opac; 
    else
      column(s, k, j) = 0;
  }); 
}


void UpdateSurface(DataBlock& data, const real t, const real dt) {

 // Get the irradiation at the surface
 if (data.haveIrradiation)
    data.irradiation->GetBoundaryFlux(IDIR, left, F_surf);

 
  const real code_aR = data.radiation->code_aR;
  const real code_cdt = data.radiation->code_c*dt;
  auto Er = data.radiation->Erad;

  const int i = data.beg[IDIR];


  idefix_for("SurfaceTemp", 0, data.np_tot[KDIR], 0, data.np_tot[JDIR],
    KOKKOS_LAMBDA (int k, int j)  {
      // Note - currently assumes F << cE_rad
      real X = 0.25*code_cdt*code_aR*pow(T_surf(k, j), 3);
      real Z = C_surf + 4*X;     
      real u_0 = C_surf*T_surf(k, j) + 3*X*T_surf(k, j) + 0.25*code_cdt*Er(k,j,i) - dt*F_surf(k,j);

      T_surf(k, j) = u_0 / Z;
  });     

  // Do dust condensation/evaporation
  ApplyCondensation(data, t, dt);

}

void MeanOpacity(DataBlock* data, IdefixArray4D<real> kappaP, IdefixArray4D<real> kappaR) {

    // Set the gas opacity
    gas_opac->evaluate_mean_opacity(data, kappaP, kappaR);

    // Set the dust opacity
    for (int s=0; s < data->dust.size(); s++)
       dust_opac->evaluate_mean_opacity(data, kappaP, kappaR, s+1);
}

void IrradiationOpacity(DataBlock* data, IdefixArray2D<real> kappa) {

  // Check the shape matches:
  if (data->irradiation->num_bands != gas_opac->num_wle_bins)
    IDEFIX_ERROR("Number of bands expected by irradiation does not match gas opacity");    

  // Set the gas opacity
  gas_opac->evaluate_opacity(kappa);

  // Set the dust opacity
  for (int s=0; s < data->dust.size(); s++)
    dust_opac->evaluate_opacity(gas_opac->num_wle_bins, gas_opac->wle_micron, kappa, s+1);
}

void RadiationField(DataBlock* data, IdefixArray1D<real> flux) {
  star->get_radiation_field(flux);
}




Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
  
  // Store parameters
  T0 =  input.Get<real>("Setup", "T0", 0) / idfx::units.GetKelvin();
  d2g =  input.Get<real>("Setup", "d2g", 0);
  real mu = input.Get<real>("Radiation", "mu", 0) ;

  // Total heat capacity of the surface
  C_surf = input.GetOrSet<real>("Setup", "surface_heat_capacity", 0, 3.0) / idfx::units.GetLength();
  surface_condensation = input.GetOrSet<bool>("Setup", "surface_condensation", 0, true);

  surf_base = input.Get<std::string>("Output", "dmp_dir", 0) + "/surface_";

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

  gas_opac = std::make_unique<GasOpacity>();
  dust_opac = std::make_unique<DustOpacity>(input.Get<real>("Setup", "grain_size", 0));

  star = std::make_unique<StellarProperties>(
    input.Get<real>("Setup", "Mstar", 0),
    input.Get<real>("Setup", "Lstar", 0),
    input.Get<real>("Setup", "Tstar", 0),
    input.Get<real>("Setup", "a_p", 0)
  );
  star->compute_radiation_field(gas_opac->num_wle_bins,  gas_opac->wle_micron);


  data.hydro->EnrollUserDefBoundary(&UserBoundary<DefaultPhysics>);
  data.dust[0]->EnrollUserDefBoundary(&UserBoundary<DustPhysics>);

  data.hydro->EnrollFluxBoundary(&ZeroFluxBoundary<DefaultPhysics,0>);
  data.dust[0]->EnrollFluxBoundary(&ZeroFluxBoundary<DustPhysics,-1>);


  data.gravity->EnrollPotential(&MyPotential);

  data.irradiation->EnrollUserColumnBoundary(&MyColumnBoundary);
  data.irradiation->EnrollUserOpacity(&IrradiationOpacity);
  data.irradiation->EnrollUserRadiationField(&RadiationField);

  data.radiation->EnrollUserDefOpacityCGS(&MeanOpacity);
  data.EnrollUserStepLast(UpdateSurface);
  output.EnrollAnalysis(&Analysis);

}


void Setup::InitFlow(DataBlock &data) {
    DataBlockHost d(data);
    
    const real mu = data.radiation->mu;
    const real aR = data.radiation->code_aR;

    n = d.np_tot;

    // Set the surface temperature and irradiation
    T_surf = IdefixArray2D<real> ("Surface_temperature",  d.np_tot[KDIR],
                                                          d.np_tot[JDIR]);

    F_surf = IdefixArray2D<real> ("Surface_irradiation",  d.np_tot[KDIR],
                                                          d.np_tot[JDIR]);


    IdefixHostArray2D<real> T_host = Kokkos::create_mirror_view(T_surf);
    IdefixHostArray2D<real> F_host = Kokkos::create_mirror_view(F_surf);
    for(int k = 0; k < d.np_tot[KDIR]; k++) {
        for(int j = 0; j < d.np_tot[JDIR]; j++) {
            T_host(k, j) = T0;
            F_host(k, j) = 0;

        }
    }
    Kokkos::deep_copy(T_surf,T_host);
    Kokkos::deep_copy(F_surf,F_host);

    int nDust = d.dustVc.size();

    real rho0 = 1.1 * silicate.rho_vap(T0*idfx::units.GetKelvin()) / idfx::units.GetDensity();

    for(int k = 0; k < d.np_tot[KDIR]; k++) {
        for(int j = 0; j < d.np_tot[JDIR]; j++) {
            for(int i = 0; i < d.np_tot[IDIR]; i++) {

                real R = d.x[IDIR](i);
                real rho = rho0 * exp(1.2 * R_Bondi * (1./R - 1./R_p));

                d.Vc(RHO,k,j,i) = rho;
                EXPAND(
                  d.Vc(VX1,k,j,i) = 0.0;,
                  d.Vc(VX2,k,j,i) = 0.0;,
                  d.Vc(VX3,k,j,i) = 0.0;
                )

                d.Vc(PRS, k,j,i) = rho * T0 / mu;
                d.Erad(k,j,i) = aR * pow(T0, 4);        

                for (int s=0; s < nDust; s++) {
                    d.dustVc[s](RHO,k,j,i) = rho * d2g;
                    EXPAND(
                      d.dustVc[s](VX1,k,j,i) = 0.0;,
                      d.dustVc[s](VX2,k,j,i) = 0.0;,
                      d.dustVc[s](VX3,k,j,i) = 0.0;
                    );

                    d.dustVc[s](TRD, k,j,i) = T0;
                }
            }
        }
    }

    // Send it all, if needed
    d.SyncToDevice();
}

Setup::~Setup() {
    // Free our local memory
    T_surf = IdefixArray2D<real>();
    F_surf = IdefixArray2D<real>();

    // Free opacity etc
    gas_opac.reset();
    dust_opac.reset();
    star.reset();
}


static int surf_num = 0;
void Analysis(DataBlock & data) {
    std::stringstream surf_file ;
    surf_file << surf_base << surf_num++ << ".txt";    
    std::ofstream f(surf_file.str());
    
    f << "# j k T_surf F_surf" << "\n";

    IdefixHostArray2D<real> T_host = Kokkos::create_mirror_view(T_surf);
    IdefixHostArray2D<real> F_host = Kokkos::create_mirror_view(F_surf);

    Kokkos::deep_copy(T_host, T_surf);
    Kokkos::deep_copy(F_host, F_surf);


    for(int k = data.beg[KDIR]; k < data.end[KDIR]; k++) {
        for(int j = data.beg[JDIR]; j < data.end[JDIR]; j++) {
            f << j << " " << k << " " << T_host(k,j) << " " << F_host(k,j) << "\n";
      }
    }
}
