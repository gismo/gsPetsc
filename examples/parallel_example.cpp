/** @file phase7_helper_class.cpp
 
    @brief PHASE 7: PETSc helper class for parallel assembly
    
    This example demonstrates:
    1. Using a C++ helper class to encapsulate PETSc objects and calls
    2. Simplifying the main application logic
    3. Solving a distributed problem with the helper class
    
    This file is part of the G+Smo library.
    
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
    
    Author(s): H. Verhelst
    
    Example run: 
    ./bin/phase7_helper_class
    mpirun -np 2 ./bin/phase7_helper_class -x 20 -y 20
*/

#include <gismo.h>
#include <gsPetsc/PETScSupport.h>
#include <numeric>
#include <vector>

#include <gsPetsc/gsPetscLinearSolver.h>



using namespace gismo;

int main(int argc, char *argv[])
{
    // Initialize PETSc
    PetscInitialize(0, NULL, NULL, 
                    "PHASE 7: PETSc Helper Class for Parallel Assembly\n");
    MPI_Comm comm = PETSC_COMM_WORLD;
    
    int rank = 0, nProc = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nProc);
    
#ifdef _OPENMP
    if (rank==0) gsInfo << "Using OpenMP with " << omp_get_max_threads() << " threads\n";
#endif
    
    // Command line arguments
    index_t numKnotsX = 10;
    index_t degreeX = 2;
    index_t numKnotsY = 10;
    index_t degreeY = 2;
    
    gsCmdLine cmd("PHASE 7: PETSc Helper Class for Parallel Assembly.");
    cmd.addInt("x", "numKnotsX", "Number of knots in X direction", numKnotsX);
    cmd.addInt("p", "degreeX", "Degree in X direction", degreeX);
    cmd.addInt("y", "numKnotsY", "Number of knots in Y direction", numKnotsY);
    cmd.addInt("q", "degreeY", "Degree in Y direction", degreeY);
    try { cmd.getValues(argc,argv); } catch (int rv) { return rv; }
    
    if (rank == 0) {
        gsInfo << "\n========================================\n";
        gsInfo << "PHASE 7: PETSc Helper Class Example\n";
        gsInfo << "========================================\n";
        gsInfo << "Problem: 2D Poisson with Dirichlet BCs\n";
        gsInfo << "Domain: [0,1] x [0,1]\n";
        gsInfo << "Knots: (" << numKnotsX << " x " << numKnotsY << ")\n";
        gsInfo << "Degrees: (" << degreeX << " x " << degreeY << ")\n";
        gsInfo << "Decomposition into " << nProc << " pieces\n\n";
    }
    
    // Create basis
    gsKnotVector<> kvX(0.0, 1.0, numKnotsX, degreeX+1);
    gsKnotVector<> kvY(0.0, 1.0, numKnotsY, degreeY+1);
    gsTensorBSplineBasis<2, real_t> basis(kvX, kvY);
    
    if (rank == 0) {
        gsInfo << "kvX numElements: " << kvX.numElements() << "\n";
        gsInfo << "kvY numElements: " << kvY.numElements() << "\n";
        gsInfo << "Basis size (number of DoFs): " << basis.size() << "\n";
    }
    
    gsGeometry<real_t>::uPtr geometry = basis.makeGeometry(basis.anchors().transpose());
    gsMultiPatch<real_t> mp(*geometry);
    mp.computeTopology();
    gsMultiBasis<real_t> mb(mp, true);
    
    // Right-hand side function
    gsFunctionExpr<real_t> f("sin(pi*x)*sin(pi*y)", 2);
    
    gsBoundaryConditions<real_t> bcInfo;
    bcInfo.addCondition(boundary::west,  condition_type::dirichlet, 0, 0, false, -1);
    bcInfo.addCondition(boundary::east,  condition_type::dirichlet, 0, 0, false, -1);
    bcInfo.addCondition(boundary::north, condition_type::dirichlet, 0, 0, false, -1);
    bcInfo.addCondition(boundary::south, condition_type::dirichlet, 0, 0, false, -1);
    bcInfo.setGeoMap(mp);

    // Decompose domain
    if (rank == 0)
        gsInfo << "Decomposing domain into " << nProc << " pieces...\n";
    
    auto basisDomain = basis.domain(); 
    auto decomposed = basisDomain->decompose(nProc);
    
    if (rank == 0)
        gsInfo << "Successfully decomposed into subdomains\n\n";    

    gsStopwatch timer;
    
    gsExprAssembler<real_t,RowMajor> assembler(1,1);
    assembler.setIntegrationDomain(decomposed->subdomain(rank));
    
    auto u = assembler.getSpace(mb);
    auto G = assembler.getMap(mp);
    auto F = assembler.getCoeff(f,G);
    u.setup(bcInfo,dirichlet::l2Projection);  

    assembler.initSystem();
    gsInfo<<"Rank " << rank << " number of elements: " << decomposed->subdomain(rank)->numElements() << "\n";
    assembler.assemble(u*u.tr()*meas(G),u*F*meas(G));
    
    gsSparseMatrix<real_t,RowMajor> A_local(assembler.numDofs(), assembler.numDofs());
    gsMatrix<real_t> rhs_matrix_local(assembler.numDofs(), 1);
    assembler.matrix_into(A_local);
    assembler.rhs_into(rhs_matrix_local);
    gsVector<real_t> b_local = rhs_matrix_local.col(0);

    real_t assemblyTime = timer.stop();
    
    if (rank == 0) {
        gsInfo << "G+Smo Local Assembly time: " << assemblyTime << " seconds\n";
        gsInfo << "G+Smo Local Matrix size (global DoFs): " << A_local.rows() << " x " << A_local.cols() << "\n";
        gsInfo << "G+Smo Local Matrix nonzeros: " << A_local.nonZeros() << "\n\n";
    }

    // Instantiate and use PetscLinearSolver
    PetscLinearSolver<real_t> petscSolver(comm);
    petscSolver.compute(A_local);

    real_t petsc_A_norm = petscSolver.getMatrixNorm();

    if (rank == 0) {
        gsInfo << "Norm of PETSc matrix A: " << petsc_A_norm << "\n";
    }
    
    petscSolver.setupSolver();
    gsVector<real_t> solution;
    petscSolver.solve(b_local, solution);

    real_t petsc_norm = petscSolver.getSolutionNorm();
    real_t petsc_b_norm = petscSolver.getRHSNorm();
    if (rank == 0) {
        gsInfo << "Norm of PETSc RHS vector b: " << petsc_b_norm << "\n";
        gsInfo << "Norm of PETSc solution vector: " << petsc_norm << "\n";
    }

    if (rank == 0) {
        gsInfo << "\nSolution (first 10 DoFs):\n";
        for (index_t i = 0; i < std::min((index_t)10, solution.size()); ++i) {
            gsInfo << solution[i] << " ";
        }
        gsInfo << "\n\nSolution norm: " << solution.norm() << "\n";
    }

    petscSolver.destroy();

    if (rank == 0) {
        gsInfo << "\n========================================\n";
        gsInfo << "PHASE 7: Complete\n";
        gsInfo << "========================================\n";
    }
    
    PetscFinalize();
    return EXIT_SUCCESS;
}
