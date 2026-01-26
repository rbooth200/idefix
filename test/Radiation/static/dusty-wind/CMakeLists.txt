target_sources(idefix
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/interpolate.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/planck.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/condensation.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/opacity.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/opacity.cpp
)

replace_idefix_source(timeIntegrator.cpp timeIntegrator.cpp)
