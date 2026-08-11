######################################################################
## FindPETSc.cmake
## This file is part of the G+Smo library.
##
## Author: H. Honnerova
######################################################################

# PETSc comes in two layouts and both must work:
#
#   in-place build   PETSC_DIR/PETSC_ARCH/{include,lib}  + PETSC_DIR/include
#                    (what `./configure && make` in a PETSc source tree gives)
#   prefix install   PETSC_DIR/{include,lib}, no PETSC_ARCH at all
#                    (what a package manager or an HPC module gives, e.g.
#                     EasyBuild's PETSc/x.y.z-foss-YYYYa on SURF Snellius)
#
# So PETSC_ARCH is OPTIONAL: when it is absent we simply look in PETSC_DIR
# itself. Requiring it used to make every module-provided PETSc fail the
# find_package(PETSc REQUIRED) in gsPetsc/CMakeLists.txt.
if(NOT DEFINED PETSC_DIR OR PETSC_DIR STREQUAL "")
    # $PETSC_DIR is set by a source build; $EBROOTPETSC by an EasyBuild module.
    if(DEFINED ENV{PETSC_DIR} AND NOT "$ENV{PETSC_DIR}" STREQUAL "")
        set(PETSC_DIR "$ENV{PETSC_DIR}" CACHE STRING "Path to PETSc root directory")
    elseif(DEFINED ENV{EBROOTPETSC} AND NOT "$ENV{EBROOTPETSC}" STREQUAL "")
        set(PETSC_DIR "$ENV{EBROOTPETSC}" CACHE STRING "Path to PETSc root directory")
    else()
        set(PETSC_DIR "PETSC_DIR-NOTFOUND" CACHE STRING "Path to PETSc root directory")
        message(WARNING "PETSC_DIR is not set! Please specify the path to PETSc root directory.")
    endif()
endif()

if(NOT DEFINED PETSC_ARCH)
    set(PETSC_ARCH "$ENV{PETSC_ARCH}" CACHE STRING "PETSc architecture (empty for a prefix install)")
endif()

unset(PETSC_INCLUDES CACHE)
unset(PETSC_LIBRARY CACHE)
unset(PETSC_INCLUDE_SRC CACHE)
unset(PETSC_INCLUDE_ARCH CACHE)

find_path(PETSC_INCLUDE_SRC
    NAMES petsc.h
    PATHS ${PETSC_DIR}/include
    ${INCLUDE_INSTALL_DIR}
    )

# petscconf.h lives under PETSC_ARCH for an in-place build and directly in
# include/ for a prefix install -- try both, in that order.
find_path(PETSC_INCLUDE_ARCH
    NAMES petscconf.h
    PATHS ${PETSC_DIR}/${PETSC_ARCH}/include
    ${PETSC_DIR}/include
    ${INCLUDE_INSTALL_DIR}
    )

set(PETSC_INCLUDES ${PETSC_INCLUDE_SRC} ${PETSC_INCLUDE_ARCH} CACHE PATH "")

MESSAGE(STATUS "Found PETSc include: ${PETSC_INCLUDE_ARCH}")
MESSAGE(STATUS "Found PETSc include: ${PETSC_INCLUDE_SRC}")
MESSAGE(STATUS "Found PETSc include: ${PETSC_INCLUDES}")

  
find_library(PETSC_LIBRARY
    NAMES petsc
    PATHS ${PETSC_DIR}/${PETSC_ARCH}/lib ${PETSC_DIR}/lib
    ${PETSC_DIR}/${PETSC_ARCH}/lib64 ${PETSC_DIR}/lib64
    ${LIB_INSTALL_DIR}
    )
  
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PETSc DEFAULT_MSG PETSC_LIBRARY PETSC_INCLUDES)

mark_as_advanced(PETSC_INCLUDES PETSC_LIBRARY)

if(PETSC_FOUND)
    MESSAGE(STATUS "Found PETSc: ${PETSC_LIBRARY}")
endif()
