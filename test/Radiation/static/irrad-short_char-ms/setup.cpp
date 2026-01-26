#include "idefix.hpp"
#include "setup.hpp"
#include "units.hpp"

void MyRadiationBoundary(DataBlock &data, int dir, BoundarySide side, real t, IdefixArray3D<real> &Erad) ;
void Analysis(DataBlock & data) ;

void MyColumnBoundary(DataBlock* data, IdefixArray3D<real> column) {

    
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

static real rho0, p, T0, d2g, C_surf;
static IdefixArray2D<real> T_surf, F_surf; ;
std::string surf_file;
std::array<int, 3> n;

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

      T_surf(k, j) = u_0 / Z ;
  });     
}

Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
  
  real k0 = input.Get<real>("Irradiation", "kappa", 2) / data.radiation->unit_opacity;
  // Store parameters
  rho0 = input.Get<real>("Setup", "tau0", 0) / k0 ;
  p = input.Get<real>("Setup", "rhoSlope", 0) ;
  T0 =  input.Get<real>("Setup", "T0", 0) / idfx::units.GetKelvin();
  d2g =  input.Get<real>("Setup", "d2g", 0) ;

  // Total heat capacity of the surface
  C_surf = input.GetOrSet<real>("Setup", "surface_heat_capacity", 0, 3.0) / idfx::units.GetLength() ;


  surf_file = input.Get<std::string>("Output", "dmp_dir", 0) + "/surface.txt" ;


  data.irradiation->EnrollUserColumnBoundary(&MyColumnBoundary) ;
  data.EnrollUserStepLast(UpdateSurface);
  output.EnrollAnalysis(&Analysis);

 //  data.hydro->EnrollUserDefBoundary(&UserdefBoundary);
 //  if (data.haveRadiation) data.radiation->EnrollUserDefBoundary(&MyRadiationBoundary);
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


    IdefixHostArray2D<real> T_host = Kokkos::create_mirror_view(T_surf) ;
    IdefixHostArray2D<real> F_host = Kokkos::create_mirror_view(F_surf) ;
    for(int k = 0; k < d.np_tot[KDIR] ; k++) {
        for(int j = 0; j < d.np_tot[JDIR] ; j++) {
            T_host(k, j) = T0 ;
            F_host(k, j) = 0 ;

        }
    }
    Kokkos::deep_copy(T_surf,T_host);
    Kokkos::deep_copy(F_surf,F_host);


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

    int nDust = d.dustVc.size();
    for(int k = 0; k < d.np_tot[KDIR] ; k++) {
        for(int j = 0; j < d.np_tot[JDIR] ; j++) {
            for(int i = 0; i < d.np_tot[IDIR] ; i++) {

                real R = d.x[IDIR](i) ;
                real rho = d2g * rho0 * pow(R, -p) ;

                for (int s=0; s < nDust; s++) {
                    d.dustVc[s](RHO,k,j,i) = rho;
                    d.dustVc[s](VX1,k,j,i) = 0.0;
                    d.dustVc[s](VX2,k,j,i) = 0.0;
                    d.dustVc[s](VX3,k,j,i) = 0.0;

                    d.dustVc[s](TRD, k,j,i) = rho * T0 ;
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
}


void Analysis(DataBlock & data) {
    std::ofstream f(surf_file) ;
    
    f << "# j k T_surf F_surf" << "\n";

    IdefixHostArray2D<real> T_host = Kokkos::create_mirror_view(T_surf) ;
    IdefixHostArray2D<real> F_host = Kokkos::create_mirror_view(F_surf) ;

    Kokkos::deep_copy(T_host, T_surf);
    Kokkos::deep_copy(F_host, F_surf);


    for(int k = data.beg[KDIR]; k < data.end[KDIR] ; k++) {
        for(int j = data.beg[JDIR]; j < data.end[JDIR] ; j++) {
            f << j << " " << k << " " << T_host(k,j) << " " << F_host(k,j) << "\n";
      }
    }
}
