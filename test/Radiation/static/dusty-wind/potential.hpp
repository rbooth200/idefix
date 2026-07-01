#include "idefix.hpp"

class TidalPotential {
 public:
  TidalPotential(real GMstar=1, real GMplanet=0, real a_p=0)
    : _GMstar(GMstar), _GMplanet(GMplanet), _a_p(a_p), _omega(sqrt((GMstar+GMplanet)/(a_p*a_p*a_p)))
  { }

 KOKKOS_INLINE_FUNCTION
 real operator()(real x, real y, real z) const {
    real r_s = sqrt(x*x + y*y + z*z);
    real r_p = sqrt((x - _a_p)*(x - _a_p) + y*y + z*z);

    real x_COM  =  - _a_p * _GMstar / (_GMstar + _GMplanet);

    real phi_s = - _GMstar / r_s;
    real phi_p = - _GMplanet / r_p;
    real phi_c = + 0.5 * _omega*_omega*( (x - x_COM)*(x - x_COM) + y*y );

    real phi = phi_s + phi_p + phi_c;

    return phi;
  }

  real get_semi_major() const {
    return _a_p;
  }

 private:
    real _GMstar, _GMplanet, _a_p, _omega;
} ;
