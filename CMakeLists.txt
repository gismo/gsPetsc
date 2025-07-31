######################################################################
## CMakeLists.txt --- gsPetsc
## This file is part of the G+Smo library.
##
## Author: H. Honnerova
######################################################################

## gsPetsc module
project(gsPetsc)

# Apply G+Smo config
include(gsConfig)

## Collect files
aux_header_directory     (${CMAKE_CURRENT_SOURCE_DIR}/src ${PROJECT_NAME}_H  )

if( (NOT GISMO_BUILD_LIB) )
  aux_instance_directory (${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_NAME}_INS)
  if(${PROJECT_NAME}_INS)
    LIST( REMOVE_ITEM ${PROJECT_NAME}_CPP ${${PROJECT_NAME}_INS})
  endif()
endif()

if(NOT GISMO_WITH_MPI)
  message(WARNING "Please enable MPI by setting -D GISMO_WITH_MPI=ON in order to use PETSc")
endif(NOT GISMO_WITH_MPI)

execute_process( COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/FindPETSc.cmake ${CMAKE_SOURCE_DIR}/cmake/)
find_package(PETSc REQUIRED)

# Add object library
add_library(${PROJECT_NAME} OBJECT
  ${${PROJECT_NAME}_H}
  ${${PROJECT_NAME}_HPP}
  ${${PROJECT_NAME}_CPP} )

set_target_properties(${PROJECT_NAME} PROPERTIES
  COMPILE_DEFINITIONS gismo_EXPORTS
  POSITION_INDEPENDENT_CODE ON
  LINKER_LANGUAGE CXX
  FOLDER "G+Smo modules" )

set(gismo_MODULES ${gismo_MODULES} $<TARGET_OBJECTS:${PROJECT_NAME}>
  CACHE INTERNAL "G+Smo modules" )

#Symlink include dir (in case your headers are in /src)
execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_BINARY_DIR}/${PROJECT_NAME})

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}"
  DESTINATION include/gismo
  FILES_MATCHING PATTERN "*.h" )

# Include directory
set (GISMO_INCLUDE_DIRS ${GISMO_INCLUDE_DIRS} ${PETSC_INCLUDES}
  CACHE INTERNAL "Gismo include directories" FORCE)
  
include_directories(${GISMO_INCLUDE_DIRS})

# Linker
set(gismo_LINKER ${gismo_LINKER} ${PETSC_LIBRARY}
  CACHE INTERNAL "${PROJECT_NAME} extra linker objects" FORCE)
  
# add example files
aux_cpp_directory(${CMAKE_CURRENT_SOURCE_DIR}/examples FILES)
foreach(file ${FILES})
    add_gismo_executable(${file})
    get_filename_component(tarname ${file} NAME_WE) # name without extension
    set_property(TEST ${tarname} PROPERTY LABELS "${PROJECT_NAME}")
    set_target_properties(${tarname} PROPERTIES FOLDER "${PROJECT_NAME}")
    # install the example executables (optionally)
    install(TARGETS ${tarname} DESTINATION "${BIN_INSTALL_DIR}" COMPONENT exe OPTIONAL)
endforeach(file ${FILES})

# add unittests
aux_gs_cpp_directory(${PROJECT_SOURCE_DIR}/unittests unittests_SRCS)
set(gismo_UNITTESTS ${gismo_UNITTESTS} ${unittests_SRCS}
  CACHE INTERNAL "gismo list of unittests")
