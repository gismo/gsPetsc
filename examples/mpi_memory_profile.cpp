/** @file mpi_memory_profile.cpp

    @brief Toy Poisson problem assembled element-partitioned over MPI ranks,
    with per-rank memory accounting of every G+Smo / PETSc object involved.

    Each rank
      1. builds the full multipatch geometry and multibasis (replicated),
      2. assembles only its own block of elements with gsExprAssembler
         (via gsElementRangeDomain),
      3. sends its contributions to a distributed PETSc matrix (COO),
      4. solves with KSP,
      5. gathers the solution for gsFeSolution and computes the error on its
         own elements.

    The heap growth of every stage and the size of the main objects is
    reported as min/max/sum over the ranks. Compare runs with different
    numbers of ranks at fixed N (see --csv and scripts/memory_tables.py):
    the max of a replicated object stays constant, the max of a
    distributed one decreases like 1/P.

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Example run:
    mpirun -np 4 ./bin/mpi_memory_profile -r 7 -p 2 -s 1
*/

#include <gismo.h>
#include <gsPetsc/partitioned/gsElementRangeDomain.h>
#include <gsPetsc/partitioned/gsMemoryProbe.h>
#include <gsPetsc/partitioned/gsPetscCOO.h>
#include <petscksp.h>

using namespace gismo;

int main(int argc, char *argv[])
{
    index_t dim = 2, degree = 2, numRefine = 5, numSplit = 1;
    bool refineGeometry = false, legacy = false, csv = false, noReserve = false;
    std::string petscOpts("-ksp_type cg -pc_type gamg -ksp_rtol 1e-10");

    gsCmdLine cmd("Memory profile of element-partitioned assembly with gsExprAssembler + PETSc.");
    cmd.addInt   ("d", "dim",     "Spatial dimension (2 or 3)", dim);
    cmd.addInt   ("p", "degree",  "Spline degree", degree);
    cmd.addInt   ("r", "refine",  "Uniform h-refinement steps", numRefine);
    cmd.addInt   ("s", "split",   "Patch grid: 2^s patches per direction", numSplit);
    cmd.addSwitch("geo", "Refine the geometry together with the basis (fine CAD / isoparametric geometry)", refineGeometry);
    cmd.addSwitch("legacy", "Also measure the conversion path of PETScSupport.h (global gsSparseMatrix + RowMajor copy)", legacy);
    cmd.addSwitch("noreserve", "Do not reserve fiber storage in initSystem (bdA=bdB=bdO=0)", noReserve);
    cmd.addSwitch("csv", "Print CSV lines (prefix CSV,) in addition to the table", csv);
    cmd.addString("o", "petsc", "PETSc options", petscOpts);
    try { cmd.getValues(argc,argv); } catch (int rv) { return rv; }

    const gsMpi & mpi = gsMpi::init(argc, argv);
    gsMpiComm comm = mpi.worldComm();
    const int rank = comm.rank(), nproc = comm.size();
    PetscCall( PetscInitializeNoArguments() );
    PetscCall( PetscOptionsInsertString(NULL, petscOpts.c_str()) );

    memprobe::Ledger L(comm);
    gsStopwatch clock;

    // ------------------------------------------------------------------
    // 1. Geometry and discretization (built identically on every rank)
    // ------------------------------------------------------------------
    const int np = 1 << numSplit;
    gsMultiPatch<> mp = (2 == dim) ? gsNurbsCreator<>::BSplineSquareGrid(np, np, 1.0)
                                   : gsNurbsCreator<>::BSplineCubeGrid(np, np, np, 1.0);
    if (refineGeometry)
        for (size_t k = 0; k != mp.nPatches(); ++k)
        {
            mp.patch(k).degreeElevate(degree - mp.patch(k).degree(0));
            for (index_t r = 0; r != numRefine; ++r) mp.patch(k).uniformRefine();
        }
    L.stage("geometry gsMultiPatch");

    gsMultiBasis<> mb(mp, true);
    mb.setDegree(degree);
    if (!refineGeometry)
        for (index_t r = 0; r != numRefine; ++r) mb.uniformRefine();
    L.stage("gsMultiBasis (refined)");

    const std::string sx = (2 == dim) ? "sin(pi*x)*sin(pi*y)" : "sin(pi*x)*sin(pi*y)*sin(pi*z)";
    gsFunctionExpr<> ms(sx, dim);
    gsFunctionExpr<> f(util::to_string(dim) + "*pi^2*" + sx, dim);
    gsBoundaryConditions<> bc;
    for (gsMultiPatch<>::const_biterator bit = mp.bBegin(); bit != mp.bEnd(); ++bit)
        bc.addCondition(*bit, condition_type::dirichlet, &ms);
    bc.setGeoMap(mp);
    L.stage("gsBoundaryConditions");

    // ------------------------------------------------------------------
    // 2. Element partition and assembler
    // ------------------------------------------------------------------
    gsDomain<>::Ptr fullDomain = mb.domain();
    gsDomain<>::Ptr myDomain = gsElementRangeDomain<real_t>::blockPartition(fullDomain, rank, nproc);
    const index_t numElemGlobal = fullDomain->numElements();
    L.stage("element partition (gsDomain)");

    gsExprAssembler<> A(1,1);
    if (noReserve)
    {
        A.options().setReal("bdA", 0);
        A.options().setInt ("bdB", 0);
        A.options().setReal("bdO", 0);
    }
    A.setIntegrationDomain(myDomain);
    gsExprEvaluator<> ev(A);
    gsExprAssembler<>::geometryMap G = A.getMap(mp);
    gsExprAssembler<>::space u = A.getSpace(mb);
    auto ff = A.getCoeff(f, G);
    auto u_ex = ev.getVariable(ms, G);

    u.setup(bc, dirichlet::interpolation, 0);
    L.stage("space setup (gsDofMapper + Dirichlet)");
    L.object("  gsDofMapper", memprobe::bytesOf(u.mapper()));
    L.object("  fixed (Dirichlet) dofs", memprobe::bytesOf(u.fixedPart()));

    A.initSystem();
    const index_t N = A.numDofs();
    L.stage("initSystem (fiber matrix + rhs)");
    L.object("  gsFiberMatrix after initSystem", memprobe::bytesOf(A.fiberMatrix()));
    L.object("  rhs vector", memprobe::bytesOf(A.rhs()));

    A.computePattern(igrad(u, G) * igrad(u, G).tr());
    L.stage("computePattern (local elements)");

    A.assemble(igrad(u, G) * igrad(u, G).tr() * meas(G), u * ff * meas(G));
    L.stage("assemble (local elements)");
    L.object("  gsFiberMatrix after assembly", memprobe::bytesOf(A.fiberMatrix()));
    const long long localNnz = A.fiberMatrix().nonZeros();
    const double tAssemble = clock.stop();

    // ------------------------------------------------------------------
    // 3. Transfer to PETSc and solve
    // ------------------------------------------------------------------
    clock.restart();
    Mat PA; Vec Pb, Px; KSP ksp;
    PetscCount nOff = 0;
    PetscCall( petsc_matFromLocalFibers(A.fiberMatrix(), comm, PA, &nOff) );
    PetscCall( petsc_vecFromLocalContributions(A.rhs(), PA, Pb) );
    L.stage("PETSc Mat/Vec (COO, distributed)");

    if (legacy)
    {
        // What PETScSupport.h::petsc_copySparseMat needs on every rank
        gsSparseMatrix<> csc;
        A.matrix_into(csc);
        gsSparseMatrix<real_t, RowMajor> csr = csc;
        L.object("  global gsSparseMatrix (CSC)", memprobe::bytesOf(csc));
        L.object("  global gsSparseMatrix (CSR)", memprobe::bytesOf(csr));
        L.object("  petsc_copySparseMat maps (2 x N)", 2 * static_cast<long long>(N) * sizeof(index_t));
        L.stage("legacy conversion (CSC + CSR)");
        csc.resize(0,0); csc.data().squeeze();
        csr.resize(0,0); csr.data().squeeze();
        L.stage("legacy conversion freed");
    }

    // The assembler's storage is no longer needed once PETSc owns the system
    A.giveFiberMatrix();
    gsMatrix<> rhsMoved; A.rhs_into(rhsMoved); rhsMoved.resize(0,0);
    L.stage("release assembler matrix/rhs");

    PetscCall( MatCreateVecs(PA, &Px, nullptr) );
    PetscCall( KSPCreate(comm, &ksp) );
    PetscCall( KSPSetOperators(ksp, PA, PA) );
    PetscCall( KSPSetFromOptions(ksp) );
    PetscCall( KSPSolve(ksp, Pb, Px) );
    PetscInt its;
    PetscCall( KSPGetIterationNumber(ksp, &its) );
    L.stage("KSP setup + solve");
    const double tSolve = clock.stop();

    // ------------------------------------------------------------------
    // 4. Back to G+Smo
    // ------------------------------------------------------------------
    clock.restart();
    // Dofs actually needed by the local elements ("owned + ghost")
    std::vector<PetscInt> needed;
    {
        gsMatrix<index_t> act;
        for (auto it = myDomain->beginAll(); it != myDomain->endAll(); ++it)
        {
            const index_t p = it.patchIndex();
            mb.basis(p).active_into(it.centerPoint(), act);
            for (index_t i = 0; i != act.rows(); ++i)
            {
                const index_t ii = u.mapper().index(act(i), p);
                if (u.mapper().is_free_index(ii)) needed.push_back(ii);
            }
        }
        std::sort(needed.begin(), needed.end());
        needed.erase(std::unique(needed.begin(), needed.end()), needed.end());
        needed.shrink_to_fit();
    }
    gsMatrix<> ghosted;
    PetscCall( petsc_gatherEntries(Px, needed, ghosted) );
    L.stage("gather local+ghost solution entries");
    L.object("  local+ghost solution (what is needed)", memprobe::bytesOf(ghosted) + needed.size() * sizeof(PetscInt));

    gsMatrix<> solVector;
    PetscCall( petsc_gatherAll(Px, solVector) );
    L.stage("gather full solution (gsFeSolution)");
    L.object("  global solution vector", memprobe::bytesOf(solVector));

    gsExprAssembler<>::solution u_sol = A.getSolution(u, solVector);
    gsMultiPatch<> mpSol;
    u_sol.extract(mpSol);
    L.stage("solution as gsMultiPatch (extract)");
    L.object("  solution gsMultiPatch", memprobe::bytesOf(mpSol));
    mpSol.clear();
    L.stage("free solution gsMultiPatch");

    real_t l2loc = ev.integral((u_ex - u_sol).sqNorm() * meas(G)), l2 = 0;
    MPI_Allreduce(&l2loc, &l2, 1, MPI_DOUBLE, MPI_SUM, comm);
    l2 = math::sqrt(l2);
    L.stage("error evaluation (local elements)");
    const double tPost = clock.stop();

    long long offSum = nOff, nnzSum = localNnz, ghostMax = needed.size();
    MPI_Allreduce(MPI_IN_PLACE, &offSum, 1, MPI_LONG_LONG, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, &nnzSum, 1, MPI_LONG_LONG, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, &ghostMax, 1, MPI_LONG_LONG, MPI_MAX, comm);

    if (0 == rank)
    {
        gsInfo << "ranks " << nproc << ", dim " << dim << ", degree " << degree
               << ", patches " << mp.nPatches() << ", elements " << numElemGlobal
               << ", dofs " << N << (refineGeometry ? " (refined geometry)" : "") << "\n"
               << "KSP iterations " << its << ", L2 error " << l2 << "\n"
               << "assembled entries (sum over ranks) " << nnzSum
               << ", of which in rows owned by another rank " << offSum << "\n"
               << "max local+ghost dofs per rank " << ghostMax << " (" << 100.*ghostMax/N << "% of N)\n"
               << "time: assemble " << tAssemble << "s, petsc+solve " << tSolve
               << "s, post " << tPost << "s\n";
    }
    std::ostringstream tag;
    tag << "P=" << nproc << " d=" << dim << " p=" << degree << " N=" << N
        << " patches=" << mp.nPatches() << (refineGeometry ? " geo" : "")
        << (noReserve ? " noreserve" : "");
    L.report(gsInfo, tag.str());
    if (csv)
    {
        std::ostringstream ctag;
        ctag << "CSV," << nproc << "," << dim << "," << degree << "," << N << ","
             << mp.nPatches() << "," << refineGeometry << "," << noReserve;
        L.csv(gsInfo, ctag.str());
    }

    PetscCall( KSPDestroy(&ksp) );
    PetscCall( VecDestroy(&Px) );
    PetscCall( VecDestroy(&Pb) );
    PetscCall( MatDestroy(&PA) );
    PetscCall( PetscFinalize() );
    return EXIT_SUCCESS;
}
