// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef RADIATION_BOUNDARY_PARSER_HPP_
#define RADIATION_BOUNDARY_PARSER_HPP_

#include <array>
#include <string>

#include "input.hpp"

// Types of radiation boundary which can be treated.
// "internal" denotes non-physical MPI internal interfaces, not user input boundaries.
enum class RadiationBoundaryType { periodic, dirichlet, neumann, userdef, internal, axis };

struct ParsedRadiationBoundary {
  RadiationBoundaryType type;
  real value;
};

inline RadiationBoundaryType ParseRadiationPhysicalBoundaryTypeToken(const std::string& boundary,
                                                                     bool allowUserDef,
                                                                     const std::string& caller) {
  if (boundary.compare("dirichlet") == 0) {
    return RadiationBoundaryType::dirichlet;
  }
  if (boundary.compare("periodic") == 0) {
    return RadiationBoundaryType::periodic;
  }
  if (boundary.compare("neumann") == 0) {
    return RadiationBoundaryType::neumann;
  }
  if (boundary.compare("axis") == 0) {
    return RadiationBoundaryType::axis;
  }
  if (boundary.compare("userdef") == 0) {
    if (!allowUserDef) {
      IDEFIX_ERROR(caller + ":: boundary type userdef is not supported");
    }
    return RadiationBoundaryType::userdef;
  }
  if (boundary.compare("internalradiation") == 0 || boundary.compare("internal") == 0) {
    IDEFIX_ERROR(caller + ":: boundary type " + boundary +
                 " is not a physical boundary input and cannot be set by parser");
  }

  IDEFIX_ERROR(caller + ":: Unknown boundary type " + boundary);

  // Keeps compilers happy when IDEFIX_ERROR is not seen as noreturn.
  return RadiationBoundaryType::periodic;
}

inline ParsedRadiationBoundary ParseRadiationPhysicalBoundary(Input& input,
                                                              const std::string& label,
                                                              bool allowUserDef,
                                                              const std::string& caller) {
  ParsedRadiationBoundary parsed;
  parsed.value = ZERO_F;

  const std::string boundary = input.Get<std::string>("Radiation", label, 0);
  parsed.type = ParseRadiationPhysicalBoundaryTypeToken(boundary, allowUserDef, caller);

  if (parsed.type == RadiationBoundaryType::dirichlet) {
    parsed.value = input.GetOrSet<real>("Radiation", label, 1, 1e-20);
  } else if (parsed.type == RadiationBoundaryType::neumann) {
    parsed.value = input.GetOrSet<real>("Radiation", label, 1, 0);
  }

  return parsed;
}

inline void ValidateRadiationAxisBoundaries(const std::array<RadiationBoundaryType, 3>& lbound,
                                            const std::array<RadiationBoundaryType, 3>& rbound,
                                            int nproc_k, const std::string& caller) {
  for (int dir = 0; dir < DIMENSIONS; dir++) {
    if ((lbound[dir] == RadiationBoundaryType::axis ||
         rbound[dir] == RadiationBoundaryType::axis) &&
        dir != JDIR) {
      IDEFIX_ERROR(caller + ":: Axis boundaries are only applicable to X2");
    }
  }

  if ((lbound[JDIR] == RadiationBoundaryType::axis) ||
      (rbound[JDIR] == RadiationBoundaryType::axis)) {
    if (nproc_k > 1) {
      IDEFIX_ERROR(caller +
                   ":: Axis boundaries are not compatible with MPI domain decomposition in X3");
    }
  }
}

#endif  // RADIATION_BOUNDARY_PARSER_HPP_
