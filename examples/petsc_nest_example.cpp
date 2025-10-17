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


gsOptionList setOptions(std::string slv)
{
    gsOptionList result;
    if (slv == "gmres")
    {
        result.addString("-ksp_type", "Type of Krylov solver", "fgmres");
        result.addString("-ksp_initial_guess_nonzero", "", "true"); // how does this work ?
        result.addString("-pc_type", "", "jacobi");
    }

    if (slv == "svd")
    {
        result.addString("-pc_type", "", "svd");
        result.addString("-pc_svd_monitor", "", "true");
    }

        if (slv == "eigenvalue")
    {
        result.addString("-ksp_type", "Type of Krylov solver", "gmres");
        result.addString("-ksp_monitor_singular_value", "", ""); // switches with no value should be empty string
        result.addString("-ksp_gmres_restart", "", "1000");
        result.addString("-pc_type", "", "none");
    }
       
    return result;
}



int main(int argc, char *argv[])
{
    // Size of global sparse matrix
    index_t mat_size = 10;
    std::string slv("gmres");
    std::string spm(""); // sparse matrix from a file

    gsCmdLine cmd("Testing the use of sparse linear solvers.");
    cmd.addInt("n", "size", "Size of the matrices", mat_size);
    cmd.addString("s", "setup", "Setup options for PETSc", slv);
    cmd.addString("m", "matrix", "Filename to read sparse matrix and right hand side", spm);

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
    gsEigen::PetscNestKSP<gsSparseMatrix<real_t,RowMajor> > solver(comm);

    solver.options() = setOptions(slv);
    if (0==_rank)
        gsInfo << solver.options() <<"\n";

    gsSparseMatrix<real_t, RowMajor>  Q;
    gsMatrix<>        b, x;

    if (!spm.empty())
    {
        if (0==_rank)
        {
            gsFileData<> fd(spm);
            gsSparseMatrix<> Qcm;
            fd.getFirst(Qcm);
            Q = Qcm;
            fd.getFirst(b);
            mat_size = Q.rows();

            //Communicate mat_size to all processes

            //Compute layout and distribute matrix.. (make: petsc_distributeSparseMat)
        }
    }
    else
    {
        // Get local size and offset for the node
        // localGlobal.first : local size at this node
        // localGlobal.second : offset of the global index
        std::pair<index_t, index_t> localGlobal = solver.computeLayout(mat_size);

        // Assemble the linear system
        // Each node fills in their local part of the matrix (a set of rows)
        // On a sparse matrix of global size
        Q = gsSparseMatrix<real_t, RowMajor>(mat_size,mat_size);
        b.resize(localGlobal.first,1);

        Q.reserve( gsVector<int>::Constant(mat_size,1) ); // Reserve memory for 1 non-zero entry per column
        for (index_t i = 0; i!=localGlobal.first; ++i)
        {
            Q(i + localGlobal.second, i + localGlobal.second) = 1.0;
            b(i) = i+localGlobal.second+1;
        }
        Q.makeCompressed(); // always call makeCompressed after sparse matrix has been filled
    }

    // Block system [F B^T; B 0]
    gsMatrix<gsSparseMatrix<real_t, RowMajor>, 2, 2> BMat;
    BMat(0,0) = BMat(1,1) = Q;
    gsVector<gsMatrix<real_t>, 2> Bx, BVec;
    BVec[0] = BVec[1] = b;

    solver.compute(BMat);

    Bx[0].setOnes(2,1);
    Bx[1].setOnes(2,1);
    //Bx = // TO DO
        solver.solve(BVec);
    solver.print();


    if (0==_rank && mat_size < 200)
        gsInfo <<"Solution: "<< x.transpose() <<"\n";

    comm.barrier(); // does this work ?

    //std::pair<index_t, index_t> localGlobal = solver.computeLayout(mat_size);
    //gsInfo <<"Check ("<<_rank<<"): "<< ( (b-x.middleRows(localGlobal.second,localGlobal.first) ).squaredNorm()<1e-8 ) <<"\n";

   
    return EXIT_SUCCESS;
}
