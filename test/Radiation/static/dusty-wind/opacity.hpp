#ifndef _PROBLEM_OPAC_H_
#define _PROBLEM_OPAC_H_

#include "dataBlock.hpp"
#include "idefix.hpp"
#include "planck.hpp"

class StellarProperties {
 public:
  StellarProperties(double Mstar, double Lstar, double Tstar, double a_p);

  void compute_radiation_field(int num_bands, IdefixArray1D<real> wle_micron);
  void get_radiation_field(IdefixArray1D<real> flux);

 private:
  real _M, _L, _T, _a;
  int num_bands;
  IdefixArray1D<real> _flux;
  PlanckIntegral _planck;
};

class DustOpacity {
 public:
  DustOpacity(double size_cm, double beta = 1, double rho_s = 3)
   : _k0(3 / (4 * rho_s * size_cm)),
     _a(size_cm),
     _T0(1 / (2 * M_PI * 3.475 * size_cm)),
     _beta(beta) {}


  void evaluate_opacity(int num_wle, IdefixArray1D<real> wle_micron, IdefixArray2D<real> kappa, int spec);
  void evaluate_mean_opacity(DataBlock* data, IdefixArray4D<real> kP, IdefixArray4D<real> kR, int spec);

 private:
  real _k0, _a, _T0, _beta;
};

class GasOpacity {
 public:
  static const int num_wle_bins = 41;
  IdefixArray1D<real> wle_micron, kappa_2000;

  GasOpacity();

  void evaluate_opacity(IdefixArray2D<real> kappa);
  void evaluate_mean_opacity(DataBlock* data, IdefixArray4D<real> kP, IdefixArray4D<real> kR);

 private:


  static const double _wle[];
  static const double _kappa_2000[];

  static const int num_pl = 300;
  static constexpr double logT0 = 5.703782475;
  static constexpr double logT1 = 8.699514748;

  IdefixArray1D<real> logKpl, logKRoss;
  static const double _logKpl[];
  static const double _logKRoss[];

  bool have_old_opac=false;
  IdefixArray3D<real> _kP_old, _kR_old;

};

#endif// _PROBLEM_OPAC_H_
