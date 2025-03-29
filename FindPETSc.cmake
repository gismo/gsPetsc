######################################################################
## FindPETSc.cmake
## This file is part of the G+Smo library.
##
## Author: H. Honnerova
######################################################################

if(NOT DEFINED PETSC_DIR OR PETSC_DIR STREQUAL "")
    set(PETSC_DIR "PETSC_DIR-NOTFOUND" CACHE STRING "Path to PETSc root directory")
    message(WARNING "PETSC_DIR is not set! Please specify the path to PETSc root directory.")
endif()

if(NOT DEFINED PETSC_ARCH OR PETSC_ARCH STREQUAL "")
    set(PETSC_ARCH "PETSC_ARCH-NOTFOUND" CACHE STRING "PETSc architecture (e.g., arch-linux-c-debug)")
    message(WARNING "PETSC_ARCH is not set! Please specify the correct PETSc build architecture.")
endif()

unset(PETSC_INCLUDES CACHE)
unset(PETSC_LIBRARY CACHE)

find_path(PETSC_INCLUDE_SRC
    NAMES petsc.h
    PATHS ${PETSC_DIR}/include
    ${INCLUDE_INSTALL_DIR}
    )

find_path(PETSC_INCLUDE_ARCH
    NAMES petscconf.h
    PATHS ${PETSC_DIR}/${PETSC_ARCH}/include ${INCLUDE_INSTALL_DIR}
    )

set(PETSC_INCLUDES ${PETSC_INCLUDE_SRC} ${PETSC_INCLUDE_ARCH} CACHE PATH "")

MESSAGE(STATUS "Found PETSc include: ${PETSC_INCLUDE_ARCH}")
MESSAGE(STATUS "Found PETSc include: ${PETSC_INCLUDE_SRC}")
MESSAGE(STATUS "Found PETSc include: ${PETSC_INCLUDES}")

  
find_library(PETSC_LIBRARY
    NAMES petsc
    PATHS ${PETSC_DIR}/${PETSC_ARCH}/lib ${LIB_INSTALL_DIR}
    )
  
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PETSc DEFAULT_MSG PETSC_LIBRARY PETSC_INCLUDES)

mark_as_advanced(PETSC_INCLUDES PETSC_LIBRARY)

if(PETSC_FOUND)
    MESSAGE(STATUS "Found PETSc: ${PETSC_LIBRARY}")
endif()
