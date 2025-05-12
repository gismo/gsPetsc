/** @file helloSubmodule.cpp

    @brief First example of submodule

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): H. Honnerova, A. Mantzaflaris

    Example run: 
    mpirun -np 6 ./bin/helloPETSc -n 2000
*/

//! [Include namespace]
#include <gismo.h>
#include <gsPetsc/PETScSupport.h>

using namespace gismo;
//! [Include namespace]

int main(int argc, char *argv[])
{
    // Size of global sparse matrix
    index_t mat_size = 10;

    gsCmdLine cmd("Testing the use of sparse linear solvers.");
    cmd.addInt("n", "size", "Size of the matrices", mat_size);

    try { cmd.getValues(argc,argv); } catch (int rv) { return rv; }

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

    // Initialize PETSc solver with the desired communicator
    gsEigen::PetscKSP<gsSparseMatrix<real_t,RowMajor> > solver(comm);

    // Get local size and offset for the node
    std::pair<index_t, index_t> localGlobal = solver.computeLayout(mat_size);

    // Assemble the linear system
    // Each node fills in their local part of the matrix (a set of rows)
    // On a sparse matrix of global size
    gsSparseMatrix<real_t, RowMajor>  Q(mat_size,mat_size);
    gsMatrix<>        b(localGlobal.first,1), x;

    Q.reserve( gsVector<int>::Constant(mat_size,1) ); // Reserve memory for 1 non-zero entry per column
    for (index_t i = 0; i!=localGlobal.first; ++i)
    {
        Q(i + localGlobal.second, i + localGlobal.second) = 1.0;
        b(i) = i+localGlobal.second+1;
    }
    Q.makeCompressed(); // always call makeCompressed after sparse matrix has been filled

    solver.compute(Q);
    x = solver.solve(b);

    if (0==_rank)
        gsInfo <<"Solution: "<< x.transpose() <<"\n";

    //PetscFinalize();
    return EXIT_SUCCESS;
}
