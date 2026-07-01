target_sources(idefix
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/stencil_matrix.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/stencil_matrix.cpp

  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/FLD.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/FLD.cpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/multispecies_FLD.cpp

  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/irradiation.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/irradiation.cpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/short_char.cpp

  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/multigrid.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/multigrid_mode.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/boundary_utils.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/boundary_parser.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/multigrid.cpp
  )
