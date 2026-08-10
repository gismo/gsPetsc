/** @file gsPoissonScaling_example.cpp

    @brief Strong/weak scaling driver for the Poisson equation, solved with a
    METIS domain decomposition and a distributed PETSc matrix.

    This is the performance counterpart of gsMetisPetscAssembly_example.cpp:
    that file verifies the partitioned assembly against a serial reference,
    this one measures it. Every stage is timed separately and reduced across
    ranks, and one CSV row per run is appended to --csv so that a SLURM sweep
    produces a directly plottable table.

    Problem
    -------
    -Laplacian(u) = f on the unit box (or any multipatch given with -f),
    with the manufactured solution

        u_ex = sin(pi x) sin(pi y)            (2D)
        u_ex = sin(pi x) sin(pi y) sin(pi z)  (3D)

    imposed as Dirichlet data on the ENTIRE exterior boundary and
    f = -Laplacian(u_ex). The solution is stated in PHYSICAL coordinates, so
    it stays consistent for any geometry, including one read from XML.

    Parallel architecture (inherited from gsMetisPetscAssembly_example)
    ------------------------------------------------------------------
    Geometry, basis, DOF mapper, element graph and METIS labelling are
    replicated on every rank. METIS runs once (rank 0) and the labels are
    broadcast, so ownership and the global permutation are then recomputed
    identically everywhere without communication. Each rank assembles ONE
    combined subdomain (the union of its partitions) with gismo's global DOF
    indexing untouched, and insertion through the permutation is what places
    the entries into the partitioned PETSc row layout.

    Consequently per-rank memory shrinks only sublinearly with rank count
    under strong scaling, and GROWS under weak scaling -- see the MEMORY
    CAVEAT in gsScalingCommon.h. rss_hwm_mb (the true peak) is the quantity
    to size a batch allocation from.

    Scaling recipes
    ---------------
    Strong scaling: fix the discretisation, vary the rank count.
      for np in 1 2 4 8 16; do
        mpirun -np $np ./bin/gsPoissonScaling_example -d 2 --npx 8 --npy 8 \
               -r 4 -p 2 --csv strong2d.csv --tag strong2d
      done

    Weak scaling: grow the patch grid with the rank count so that nDofs/rank
    stays roughly constant (the patch grid takes arbitrary integers, so any
    rank count is reachable):
      mpirun -np  1 ... --npx 4 --npy 4  -r 4
      mpirun -np  4 ... --npx 8 --npy 4  -r 4
      mpirun -np 16 ... --npx 8 --npy 8  -r 4    # etc.
    Run once with -r ... and read "nDofs" from the output to calibrate.

    Assembly-only scaling (no solver in the picture): add --nosolve.

    Usage:
      mpirun -np 4 ./bin/gsPoissonScaling_example -d 2 --npx 4 --npy 4 -r 4 -p 2

    All PETSc options are honoured, e.g.
      -ksp_type cg -pc_type gamg -ksp_monitor -log_view

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): H.M. Verhelst
*/

#include "gsScalingCommon.h"

using namespace gismo;
using namespace gismo::scaling;

int main(int argc, char* argv[])
{
    // Options are registered before PetscInitialize so that --help/--version
    // can be serviced without entering PETSc/MPI at all (see
    // handleHelpBeforeMPI).
    ScalingOptions o;
    gsCmdLine cmd("METIS + PETSc scaling driver for the Poisson equation.");
    registerCommonOptions(cmd, o);

    int helpCode = 0;
    if (handleHelpBeforeMPI(cmd, argc, argv, helpCode)) return helpCode;

    PetscInitialize(&argc, &argv, NULL, NULL);
    PetscCall( PetscOptionsSetValue(NULL, "-options_left", "false") );

    int rank = 0, nranks = 1;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &nranks);
    const MPI_Comm comm = PETSC_COMM_WORLD;

    try { cmd.getValues(argc, argv); } catch (int ret) { PetscFinalize(); return ret; }

    // NOTE: --plot deliberately does NOT imply --check. Plotting needs the
    // gathered solution; the error norms need a full-domain integration on
    // top of that, which is far more expensive and, for elasticity, only
    // defined in --manufactured mode.

#ifdef _OPENMP
    if (o.threads > 0) omp_set_num_threads(static_cast<int>(o.threads));
    const index_t nthreads = omp_get_max_threads();
#else
    const index_t nthreads = 1;
#endif

    // Baseline RSS, sampled before ANY problem data exists. On a cluster this
    // is dominated by MPI_Init (fabric buffer registration -- commonly
    // 100-300 MB/rank, vs ~19 MB for a shared-memory-only laptop MPI), so it
    // is a large CONSTANT that would otherwise swamp the weak-scaling memory
    // curve. rss_delta_mb below subtracts it out.
    double rssBaseMB = 0, rssBaseSumMB = 0;
    reduceRss(comm, rssBaseMB, rssBaseSumMB);

    PhaseTimer timer(comm);

    // -----------------------------------------------------------------------
    // Geometry and basis (replicated on every rank)
    // -----------------------------------------------------------------------
    timer.tic("geometry");
    gsMultiPatch<real_t> mp = buildGeometry(o);
    const short_t dim  = mp.parDim();
    const short_t gdim = mp.geoDim(); // may exceed dim: a surface embedded in 3D

    gsMultiBasis<real_t> mb(mp, true); // true: poly-splines, not NURBS
    mb.setDegree(o.degree);
    for (index_t i = 0; i != o.nref; ++i)
        mb.uniformRefine();
    timer.toc();

    // -----------------------------------------------------------------------
    // Manufactured solution, stated in physical coordinates so that it stays
    // valid for any geometry (box grid or XML): Dirichlet data on the whole
    // exterior boundary is u_ex itself, and f = -Laplacian(u_ex).
    // -----------------------------------------------------------------------
    //
    // Both expressions are built over gdim (= geoDim) variables, not parDim:
    // they are evaluated at PHYSICAL points, so a surface embedded in 3D
    // (parDim 2, geoDim 3) needs a z variable or gsFunctionExpr asserts on
    // the point dimension. On such a surface, though, f = -Laplacian(u_ex)
    // is the AMBIENT Laplacian while the PDE discretises the
    // Laplace-Beltrami operator, so u_ex is no longer the exact solution --
    // the run is still a perfectly valid workload, but --check reports a
    // model error, not a discretisation error. Warned about below.
    const std::string sx = "sin(pi*x)", sy = "sin(pi*y)", sz = "sin(pi*z)";
    std::string uexStr, fStr;
    if (2 == gdim)
    {
        uexStr = sx + "*" + sy;
        fStr   = "2*pi*pi*" + uexStr;
    }
    else
    {
        uexStr = sx + "*" + sy + "*" + sz;
        fStr   = "3*pi*pi*" + uexStr;
    }
    gsFunctionExpr<real_t> u_ex(uexStr, gdim);
    gsFunctionExpr<real_t> f   (fStr,   gdim);

    if (dim != gdim && o.check && 0 == rank)
        gsWarn << "parDim ("<<dim<<") != geoDim ("<<gdim<<"): u_ex is not the "
                  "exact solution of the surface problem, so --check reports a "
                  "model error rather than a discretisation error.\n";

    gsBoundaryConditions<real_t> bc;
    bc.setGeoMap(mp);
    for (auto& bs : mp.boundaries())
        bc.addCondition(bs.patch, bs.side(), condition_type::dirichlet, &u_ex);

    // -----------------------------------------------------------------------
    // Assembler, space, DOF mapper
    // -----------------------------------------------------------------------
    gsExprAssembler<real_t> A(1, 1);
    A.setIntegrationElements(mb);
    // Lazy is the DEFAULT here (opposite of gsMetisPetscAssembly_example): a
    // rank only ever touches the fraction of the nDofs columns its own
    // partition spans, so reserving all of them eagerly is pure waste at
    // scale. --eagerMatrix restores the old behaviour for comparison.
    A.options().setSwitch("lazyMatrix", !o.eagerMatrix);

    auto u = A.getSpace(mb);

    timer.tic("dofmapper");
    u.setup(bc, o.dirInterp ? dirichlet::interpolation : dirichlet::l2Projection, 0);
    timer.toc();

    auto G  = A.getMap(mp);
    auto ff = A.getCoeff(f, G);

    // numDofs() only reads the (already finalized) mapper -- deliberately no
    // full-domain initSystem(), which would reserve the entire
    // freeSize x freeSize pattern just to obtain one integer.
    const index_t nDofs  = A.numDofs();
    const index_t nElems = static_cast<index_t>(mb.domain()->numElements());

    // -----------------------------------------------------------------------
    // METIS partitioning: rank 0 runs it, everyone else only builds the graph.
    // -----------------------------------------------------------------------
    const index_t nparts = (o.nparts > 0) ? o.nparts
                                          : math::max((index_t)1, o.partsPerRank * (index_t)nranks);
    GISMO_ENSURE(nparts <= nElems,
        "nparts ("<<nparts<<") exceeds the element count ("<<nElems<<"): "
        "refine further (-r) or use fewer ranks/partitions.");

    gsMetisPartitioner<real_t>::Options partOpts;
    partOpts.storeElementDofs = true;   // required by makeDofMapper()
    partOpts.contiguous       = o.metisContig;
    partOpts.weightByDofs     = o.metisWeightDofs;
    partOpts.imbalance        = o.metisImbalance;

    timer.tic("metis");
    gsMetisPartitioner<real_t> partitioner(mb, u.mapper(), nparts, partOpts);
    partitionAndBroadcast(partitioner, nElems, rank, comm);
    timer.toc();

    // DOF ownership + global permutation, recomputed identically on every rank.
    timer.tic("ownership");
    const gsPartitionedDofMapper dofMap = partitioner.makeDofMapper(nranks);
    const gsVector<index_t>&     perm   = dofMap.permutation();
    typename gsDomain<real_t>::Ptr rankDomain = partitioner.subdomainForRank(rank, nranks);
    timer.toc();

    const index_t nOwned    = dofMap.numOwnedDofs(rank);
    const index_t nRankElem = static_cast<index_t>(rankDomain->numElements());

#if PETSC_VERSION_GE(3,18,0)
    const bool useCoo = !o.noCoo;
#else
    const bool useCoo = false;
    GISMO_UNUSED(o.noCoo);
#endif

    // -----------------------------------------------------------------------
    // Distributed PETSc matrix and vectors, partitioned row layout: rank r
    // owns exactly dofMap.numOwnedDofs(r) rows.
    // -----------------------------------------------------------------------
    timer.tic("petsc_setup");
    Mat K;
    petsc_setupMatrixPartitioned(K, nDofs, nOwned, comm);
    if (!useCoo)
        PetscCall( MatSetOption(K, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE) );
    Vec b, x;
    PetscCall( MatCreateVecs(K, &x, &b) );
    timer.toc();

    // -----------------------------------------------------------------------
    // Local assembly: ONE setIntegrationDomain/initSystem/assemble per rank,
    // over the union of that rank's partitions. The matrix stays
    // freeSize x freeSize with global DOF indices; the permutation applied at
    // insertion is what maps it into the distributed layout.
    // -----------------------------------------------------------------------
    timer.tic("assemble");
    A.setIntegrationDomain(rankDomain);
    A.initSystem();
    A.assemble( igrad(u, G) * igrad(u, G).tr() * meas(G),   // matrix
                u * ff * meas(G) );                         // rhs (incl. -K_ib*g)
    timer.toc();

    const index_t localNnz = A.matrix().nonZeros();

    timer.tic("insert");
#if PETSC_VERSION_GE(3,18,0)
    if (useCoo) petsc_insertSparseMatrixCOO(K, A.matrix(), perm);
    else        petsc_insertSparseMatrixPermuted(K, A.matrix(), perm);
#else
    petsc_insertSparseMatrixPermuted(K, A.matrix(), perm);
#endif
    petsc_insertVectorPermuted(b, A.rhs(), perm);
    timer.toc();

    timer.tic("mat_assembly");
    PetscCall( MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY) );
    PetscCall( MatAssemblyEnd  (K, MAT_FINAL_ASSEMBLY) );
    PetscCall( VecAssemblyBegin(b) );
    PetscCall( VecAssemblyEnd  (b) );
    timer.toc();

    // Peak memory is measured here: everything replicated (basis, mapper,
    // graph, local matrix) plus the distributed PETSc matrix is now live.
    double rssMaxMB = 0, rssSumMB = 0;
    reduceRss(comm, rssMaxMB, rssSumMB);

    // -----------------------------------------------------------------------
    // Solve. CG + GAMG unless overridden on the PETSc command line. Poisson
    // needs no near-null-space beyond the constant vector, which GAMG assumes
    // by default.
    // -----------------------------------------------------------------------
    SolveStats st;
    if (!o.noSolve)
        solveSystem(K, b, x, o.rtol, o.maxIts, st, timer, comm);
    else
    { timer.skip("ksp_setup"); timer.skip("ksp_solve"); } // keep the CSV schema fixed

    // -----------------------------------------------------------------------
    // Optional verification (--check): gathers the global solution on EVERY
    // rank and integrates over the FULL domain, neither of which scales.
    // -----------------------------------------------------------------------
    // --plot needs only the gathered solution, NOT the error norms: it must
    // not silently drag in the (expensive, non-scaling) integration of
    // ||u-u_ex|| over the full domain.
    const bool wantErr  = o.check && !o.noSolve;
    const bool wantPlot = o.plot  && !o.noSolve;

    real_t l2err = -1, h1err = -1;
    if (!wantErr && !wantPlot) timer.skip("postproc");
    else
    {
        // Both paths need the global solution gathered on every rank, so they
        // share one phase. Neither scales -- this is verification only.
        timer.tic("postproc");
        gsMatrix<real_t> solPermuted;
        petsc_copyVecToGismo(x, solPermuted, comm);
        gsMatrix<real_t> solVector(nDofs, 1);
        for (index_t g = 0; g != nDofs; ++g)
            solVector(g, 0) = solPermuted(perm(g), 0);

        // Norms/plot are over the whole domain, not this rank's subdomain:
        // the evaluator shares the assembler's expression data, so the
        // integration domain must be reset explicitly first.
        A.setIntegrationElements(mb);
        gsExprEvaluator<real_t> ev(A);
        auto u_sol = A.getSolution(u, solVector);

        if (wantErr)
        {
            auto u_ex_var = ev.getVariable(u_ex, G);
            l2err = math::sqrt( ev.integral( (u_ex_var - u_sol).sqNorm() * meas(G) ) );
            h1err = l2err + math::sqrt(
                ev.integral( (igrad(u_ex_var) - igrad(u_sol, G)).sqNorm() * meas(G) ) );
        }

        if (wantPlot && 0 == rank)
        {
            gsParaviewCollection collection("ParaviewOutput/poisson_scaling", &ev);
            collection.options().setSwitch("plotElements", true);
            collection.newTimeStep(&mp);
            collection.addField(u_sol, "numerical solution");
            collection.saveTimeStep();
            collection.save();
        }
        timer.toc();
    }

    // -----------------------------------------------------------------------
    // Reduce and report
    // -----------------------------------------------------------------------
    // True peak (VmHWM), sampled at the very end so it also covers the AMG
    // hierarchy built in KSPSetUp and the transient COO triplet arrays --
    // neither of which the instantaneous rss_max_mb sample above can see.
    // This is the number to size a scheduler's --mem-per-cpu from.
    const double rssHwmMB = reduceRssPeak(comm);

    std::vector<double> tMax, tMin;
    timer.reduce(tMax, tMin);

    index_t ownedMin = 0, ownedMax = 0, elemMin = 0, elemMax = 0, nnzMin = 0, nnzMax = 0;
    reduceMinMax(nOwned,    ownedMin, ownedMax, comm);
    reduceMinMax(nRankElem, elemMin,  elemMax,  comm);
    reduceMinMax(localNnz,  nnzMin,   nnzMax,   comm);

    index_t nInterface = 0;
    for (index_t g = 0; g != nDofs; ++g)
        if (dofMap.isInterfaceDof(g)) ++nInterface;

    if (0 == rank)
    {
        Record rec;
        rec.add("tag",      o.tag.empty() ? std::string("-") : o.tag)
           .add("problem",  std::string("poisson"))
           .add("dim",      (index_t)dim)
           .add("nranks",   (index_t)nranks)
           .add("threads",  nthreads)
           .add("nparts",   nparts)
           .add("npatches", (index_t)mp.nPatches())
           .add("nref",     o.nref)
           .add("degree",   (index_t)mb.minCwiseDegree())
           .add("nelems",   nElems)
           .add("ndofs",    nDofs)
           .add("dofs_per_rank", (index_t)(nDofs / nranks))
           .add("edgecut",  partitioner.edgeCut())
           .add("interface_dofs", nInterface)
           .add("owned_min", ownedMin)
           .add("owned_max", ownedMax)
           .add("elem_min",  elemMin)
           .add("elem_max",  elemMax)
           .add("nnz_local_min", nnzMin)
           .add("nnz_local_max", nnzMax)
           .add("lazy",     (index_t)(o.eagerMatrix ? 0 : 1))
           .add("coo",      (index_t)(useCoo ? 1 : 0));

        addPhases(rec, timer.names(), tMax, tMin);

        rec.add("ksp_its",     (index_t)st.its)
           .add("ksp_reason",  (index_t)st.reason)
           .add("rel_residual", (double)st.relRes)
           .add("rss_base_mb",  rssBaseMB)
           .add("rss_max_mb",   rssMaxMB)
           .add("rss_delta_mb", rssMaxMB - rssBaseMB)
           .add("rss_sum_mb",   rssSumMB)
           .add("rss_hwm_mb",   rssHwmMB)
           .add("l2_error",     (double)l2err)
           .add("h1_error",     (double)h1err);

        gsInfo << "\n=== Poisson METIS+PETSc scaling run ===\n";
        rec.print(gsInfo);
        gsInfo << "\n";

        if (!o.csv.empty()) rec.writeCsv(o.csv);
    }

    int exitCode = 0;
    if (!o.noSolve && st.reason <= 0)
    {
        if (0 == rank)
            gsWarn << "KSP did NOT converge (reason " << st.reason << ").\n";
        exitCode = 1;
    }

    PetscCall( MatDestroy(&K) );
    PetscCall( VecDestroy(&b) );
    PetscCall( VecDestroy(&x) );
    PetscFinalize();
    return exitCode;
}
