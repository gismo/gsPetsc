/** @file phase6_expression_assembly.cpp
 
    @brief PHASE 6: Direct PETSc matrix assembly with domain iteration
    
    This example demonstrates:
    1. Looping over domain elements
    2. Assembling element contributions directly into PETSc matrix/vector
    3. Solving with PETSc solvers
    
    Combines domain iteration from simple_assembly.cpp with PETSc matrix wrapping
    from PHASE 1.
    
    This file is part of the G+Smo library.
    
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
    
    Author(s): H. Verhelst
    
    Example run: 
    ./bin/phase6_expression_assembly
    mpirun -np 2 ./bin/phase6_expression_assembly -x 20 -y 20
*/

#include <gismo.h>
#include <gsPetsc/gsPetscLinearSolver.h>
#include <numeric>

using namespace gismo;

int main(int argc, char *argv[])
{
    // Initialize PETSc
    PetscInitialize(0, NULL, NULL, 
                    "Serial Solver Example\nUses the gsPetscLinearSolver class.\n");
    MPI_Comm comm = PETSC_COMM_WORLD;
    
    int rank = 0, nProc = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nProc);

    if (nProc > 1)
    {
        if (rank == 0)
            gsInfo << "This is a serial example, but it is being run in parallel.\n"
                   << "For a parallel example, please see 'parallel_example'.\n";
    }
    
#ifdef _OPENMP
    if (rank==0) gsInfo << "Using OpenMP with " << omp_get_max_threads() << " threads\n";
#endif
    
    // Command line arguments
    index_t numKnotsX = 10;
    index_t degreeX = 2;
    index_t numKnotsY = 10;
    index_t degreeY = 2;
    
    gsCmdLine cmd("Serial solver example using gsPetscLinearSolver.");
    cmd.addInt("x", "numKnotsX", "Number of knots in X direction", numKnotsX);
    cmd.addInt("p", "degreeX", "Degree in X direction", degreeX);
    cmd.addInt("y", "numKnotsY", "Number of knots in Y direction", numKnotsY);
    cmd.addInt("q", "degreeY", "Degree in Y direction", degreeY);
    try { cmd.getValues(argc,argv); } catch (int rv) { return rv; }
    
    if (rank == 0) {
        gsInfo << "\n========================================\n";
        gsInfo << "Serial Solver Example\n";
        gsInfo << "========================================\n";
        gsInfo << "Problem: 2D Poisson with Dirichlet BCs\n";
        gsInfo << "Domain: [0,1] x [0,1]\n";
        gsInfo << "Knots: (" << numKnotsX << " x " << numKnotsY << ")\n";
        gsInfo << "Degrees: (" << degreeX << " x " << degreeY << ")\n\n";
    }
    
    // Create basis
    gsKnotVector<> kvX(0.0, 1.0, numKnotsX, degreeX+1);
    gsKnotVector<> kvY(0.0, 1.0, numKnotsY, degreeY+1);
    gsTensorBSplineBasis<2, real_t> basis(kvX, kvY);
    
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

    gsStopwatch timer;
    
    gsExprAssembler<real_t,RowMajor> assembler(1,1);
    auto domain = basis.domain();
    domain->setPatchId(0);
    assembler.setIntegrationDomain(domain);
    auto u = assembler.getSpace(mb);
    auto G = assembler.getMap(mp);
    auto F = assembler.getCoeff(f,G);
    u.setup(bcInfo,dirichlet::l2Projection);  

    assembler.initSystem();
    assembler.assemble(u*u.tr()*meas(G),u*F*meas(G));
    gsSparseMatrix<real_t,RowMajor> A(assembler.numDofs(), assembler.numDofs());
    gsMatrix<real_t> rhs(assembler.numDofs(), 1);
    assembler.matrix_into(A);
    assembler.rhs_into(rhs);
    gsVector<real_t> b = rhs.col(0);

    real_t assemblyTime = timer.stop();
    
    if (rank == 0) {
        gsInfo << "Assembly time: " << assemblyTime << " seconds\n";
        gsInfo << "Matrix size: " << A.rows() << " x " << A.cols() << "\n";
        gsInfo << "Matrix nonzeros: " << A.nonZeros() << "\n\n";
    }
    
    // Use the generic gsPetscLinearSolver
    PetscLinearSolver<real_t> petscSolver(comm);
    petscSolver.compute(A);
    petscSolver.setupSolver();
    
    gsVector<real_t> solution;
    petscSolver.solve(b, solution);

    if (rank == 0) {
        gsInfo << "\nSolution norm: " << solution.norm() << "\n";
        
        gsInfo << "\nFirst 10 DoFs of solution: ";
        for (index_t i = 0; i < std::min((index_t)10, solution.size()); i++)
            gsInfo << solution[i] << " ";
        gsInfo << "\n";
    }
    
    // Cleanup
    petscSolver.destroy();
    
    if (rank == 0) {
        gsInfo << "\n========================================\n";
        gsInfo << "Serial Example: Complete\n";
        gsInfo << "========================================\n";
    }
    
    PetscFinalize();
    return EXIT_SUCCESS;
}
