// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef RADIATION_MULTIGRID_MODE_HPP_
#define RADIATION_MULTIGRID_MODE_HPP_

#include <algorithm>
#include <cctype>
#include <string>

#include "idefix.hpp"

enum class MultigridMode { V, F, NONE };

inline std::string ToUpperString(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
  return s;
}

inline MultigridMode ParseMultigridModeToken(const std::string& token, bool allowNone,
                                             const std::string& caller) {
  if (token.empty()) {
    IDEFIX_ERROR(caller + ": multigrid cycle type is missing. Expected V or F as first value"
                 " (or N/None to disable multigrid)");
  }

  std::string mode = ToUpperString(token);
  if (mode == "V") {
    return MultigridMode::V;
  }
  if (mode == "F") {
    return MultigridMode::F;
  }
  if (allowNone && (mode == "N" || mode == "NONE")) {
    return MultigridMode::NONE;
  }

  std::string expected = allowNone ? "V, F, N or None" : "V or F";
  IDEFIX_ERROR(caller + ": Invalid multigrid cycle type. Expected " + expected +
               " as first value");

  // Keeps compilers happy when IDEFIX_ERROR is not seen as noreturn.
  return MultigridMode::V;
}

#endif  // RADIATION_MULTIGRID_MODE_HPP_
