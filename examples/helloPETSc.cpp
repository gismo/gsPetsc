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

    gsInfo << "Hello PETSc!\n";

    index_t mat_size = 10;

    // Create a linear system
    gsSparseMatrix<real_t, RowMajor>  Q(mat_size,mat_size);
    gsMatrix<>        b(mat_size,1), x(mat_size,1), x0(mat_size,1);
    x0.setOnes();
    Q.reserve( gsVector<int>::Constant(mat_size,1) ); // Reserve memory for 1 non-zero entry per column
    for (index_t i = 0; i!=mat_size; ++i)
        Q(i,i) = b.at(i) = i+1;
    
    Q.makeCompressed(); // always call makeCompressed after sparse matrix has been filled

    // PETSc here

    // Initialize the MPI environment
    const gsMpi & mpi = gsMpi::init(argc, argv);
    // Get the world communicator
    gsMpiComm comm = mpi.worldComm();
    int _rank = comm.rank();

    //Get size and rank of the processor
    int _size = comm.size();
    if (0==_rank)
        gsInfo<<"Running on "<<_size<<" processes.\n";

    gsEigen::PetscKSP<gsSparseMatrix<real_t,RowMajor> > solver;
    solver.compute(Q, comm);
    x = solver.solve(b);

    gsInfo <<"Solution: "<< x.transpose() <<"\n";
    
    return EXIT_SUCCESS;
}
