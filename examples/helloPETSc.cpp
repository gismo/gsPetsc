/** @file helloSubmodule.cpp

    @brief First example of submodule

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s):
*/

//! [Include namespace]
#include <gismo.h>
#include <gsPetsc/PETScSupport.h>

using namespace gismo;
//! [Include namespace]

int main(int argc, char *argv[])
{
    // Initialize the MPI environment
    const gsMpi & mpi = gsMpi::init(argc, argv);

    // Get the world communicator
    gsMpiComm comm = mpi.worldComm();
    
    //Get size and rank of the processor
    int _rank = comm.rank();
    int _size = comm.size();

    if (0==_rank)
    {
        gsInfo << "Hello PETSc!\n";
        gsInfo<<"Running on "<<_size<<" processes.\n";
    }

    // PetscInitialize(&argc, &argv, (char *)0, "");

    gsEigen::PetscKSP<gsSparseMatrix<real_t,RowMajor> > solver;
    solver.initialize(comm);

    index_t mat_size = 10;

    auto localGlobal = solver.computeLayout(mat_size, comm);

    // Create a linear system
    gsSparseMatrix<real_t, RowMajor>  Q(mat_size,mat_size);
    gsMatrix<>        b(localGlobal.first,1), x;

    Q.reserve( gsVector<int>::Constant(mat_size,1) ); // Reserve memory for 1 non-zero entry per column
    for (index_t i = 0; i!=localGlobal.first; ++i)
    {
        Q(i + localGlobal.second, i + localGlobal.second) = 1.0;
        b(i) = i+localGlobal.second+1;
    }

    Q.makeCompressed(); // always call makeCompressed after sparse matrix has been filled

    solver.compute(Q, comm);
    x = solver.solve(b);

    if (0==_rank)
        gsInfo <<"Solution: "<< x.transpose() <<"\n";

    //PetscFinalize();
    return EXIT_SUCCESS;
}
