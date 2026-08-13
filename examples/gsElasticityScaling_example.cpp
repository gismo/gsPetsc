/** @file gsElasticityScaling_example.cpp

    @brief Strong/weak scaling driver for linear elasticity, solved with a
    METIS or graph-free geometric domain decomposition (--partitioner) and a
    distributed PETSc matrix.

    Same architecture, timing instrumentation and CSV output as
    gsPoissonScaling_example.cpp -- see that file and gsScalingCommon.h for
    the parallel design and its memory caveat. What differs here:

      * a vector-valued space (dim components), so nDofs is dim times larger
        and the matrix has dim^2 times the block density,
      * a rigid-body near-null-space attached to the matrix (--noRBM to
        disable). PCGAMG's default near-null-space is the constant vector,
        which for elasticity captures the translations but NOT the rotations.
        Without the rotational modes the coarse space is deficient and the
        AMG iteration count grows with the mesh -- which would show up in a
        scaling plot as bad solver scaling that is really a missing-nullspace
        artefact.

    Problem
    -------
    -div sigma(u) = f on the unit box (or any multipatch given with -f), with
    sigma(u) = lambda tr(eps(u)) I + 2 mu eps(u).

    DEFAULT load case -- a clamped block under surface traction:
      * u = 0 on the MINIMUM face along --loadAxis (default x),
      * traction g = (T,T[,T]) N/mm^2 on the MAXIMUM face (--traction T,
        default 10),
      * every other face traction-free, zero body force.
    Faces are picked geometrically (gsScalingCommon.h::classifyFaces), not by
    hard-coded patch indices, so this works for the generated patch grid and
    for a multipatch read from XML alike. Defaults lambda = mu = 80000 N/mm^2
    (steel) on a 1 mm box, giving displacements of order T/mu ~ 1e-4 mm.

    Because only ONE face is clamped, the rotational rigid-body modes are not
    pinned by the boundary conditions -- this is the load case in which the
    near-null-space handed to PCGAMG actually matters.

    Reported QoI: the compliance b^T x (= twice the strain energy). It is one
    distributed dot product, so unlike --check it costs nothing and works at
    any size; and since it must be identical across rank counts it is also the
    sharpest cheap check that the partitioned assembly (volume AND traction)
    neither double-counts nor drops a contribution.

    --manufactured selects the alternative case: Dirichlet data on the ENTIRE
    boundary from

        u_ex = ( s, s )      with s = sin(pi x) sin(pi y)              (2D)
        u_ex = ( s, s, s )   with s = sin(pi x) sin(pi y) sin(pi z)    (3D)

    plus the matching body force

        f = -(lambda + mu) grad(div u_ex) - mu Laplacian(u_ex)

    stated in physical coordinates, so it stays consistent for any geometry.
    This is the only mode with an exact solution and hence the only one
    --check can grade (measured: L2 ~ h^(p+1), H1 ~ h^p in 2D and 3D); it is
    unphysical, though, since clamping everything also pins the rotations.

    The weak form assembled below is the one from
    examples/linear_elasticity_example.cpp:

        a(u,v) = lambda (div u, div v) + mu ((grad u + grad u^T) : grad v)
               = lambda (div u, div v) + 2 mu (eps(u) : eps(v))

    so mu is the shear modulus (second Lame constant), consistent with the
    forcing above.

    Scaling recipes
    ---------------
    Strong scaling: fix the discretisation, vary the rank count.
      for np in 1 2 4 8 16; do
        mpirun -np $np ./bin/gsElasticityScaling_example -d 3 --npx 2 --npy 2 --npz 2 \
               -r 3 -p 2 --csv strong3d.csv --tag strong3d
      done

    Weak scaling: grow the patch grid with the rank count so that nDofs/rank
    stays roughly constant.
      mpirun -np  1 ... --npx 2 --npy 2 --npz 2 -r 3
      mpirun -np  2 ... --npx 4 --npy 2 --npz 2 -r 3
      mpirun -np  4 ... --npx 4 --npy 4 --npz 2 -r 3   # etc.

    Assembly-only scaling (no solver): add --nosolve.

    Usage:
      mpirun -np 4 ./bin/gsElasticityScaling_example -d 3 --npx 2 --npy 2 --npz 2 -r 3

    All PETSc options are honoured, e.g.
      -pc_type gamg -pc_gamg_threshold 0.02 -ksp_monitor -log_view

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): H.M. Verhelst
*/

#include "gsScalingCommon.h"

using namespace gismo;
using namespace gismo::scaling;

namespace {

/// Formats \a v with enough digits to round-trip through gsFunctionExpr.
std::string num(real_t v)
{
    std::ostringstream ss;
    ss << std::setprecision(16) << v;
    return ss.str();
}

} // anonymous namespace

int main(int argc, char* argv[])
{
    ScalingOptions o;
    real_t lambda = 80000.0, mu = 80000.0;

    real_t  traction     = 10.0;   // N/mm^2 on every loaded component
    index_t loadAxis     = 0;      // clamp min face / load max face along this axis
    bool    manufactured = false;

    gsCmdLine cmd("METIS + PETSc scaling driver for linear elasticity.");
    registerCommonOptions(cmd, o);
    cmd.addReal("L", "firstLame",  "First Lame constant lambda [N/mm^2]", lambda);
    cmd.addReal("M", "secondLame", "Second Lame constant mu (shear modulus) [N/mm^2]", mu);
    cmd.addReal("T", "traction",
        "Surface traction magnitude [N/mm^2] applied to EVERY component on the "
        "loaded face, i.e. g = (T,T[,T])", traction);
    cmd.addInt("a", "loadAxis",
        "Axis along which the domain is clamped (minimum face) and loaded "
        "(maximum face): 0=x, 1=y, 2=z", loadAxis);
    cmd.addSwitch("manufactured",
        "Use the manufactured-solution load case instead of the traction one: "
        "Dirichlet data on the WHOLE boundary plus the matching body force. "
        "Needed for --check (it is the only mode with an exact solution), but "
        "unphysical -- clamping everything also pins the rotations, which "
        "hides what the rigid-body near-null-space is for.", manufactured);

    int helpCode = 0;
    if (handleHelpBeforeMPI(cmd, argc, argv, helpCode)) return helpCode;

    PetscInitialize(&argc, &argv, NULL, NULL);
    PetscCall( PetscOptionsSetValue(NULL, "-options_left", "false") );
    // Prerequisite for PetscMemoryGetMaximumUsage() (the GAMG bracket printed
    // under --memReport): without this call that function silently reads 0,
    // which would make the solver look free instead of reporting "not
    // measured". Called unconditionally (cheap, no /proc or MPI cost) since
    // --memReport is not parsed until cmd.getValues() below.
    PetscCall( PetscMemorySetGetMaximumUsage() );

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

    PhaseTimer timer(comm, o.memReport);

    // -----------------------------------------------------------------------
    // Geometry and basis (replicated on every rank)
    // -----------------------------------------------------------------------
    timer.tic("geometry");
    gsMultiPatch<real_t> mp = buildGeometry(o);
    const short_t dim = mp.parDim();
    GISMO_ENSURE(dim == mp.geoDim(),
        "This driver assumes a full-dimensional solid (parDim == geoDim), got "
        <<dim<<" and "<<mp.geoDim()<<".");

    gsMultiBasis<real_t> mb(mp, true);
    mb.setDegree(o.degree);
    for (index_t i = 0; i != o.nref; ++i)
        mb.uniformRefine();
    timer.toc();

    // -----------------------------------------------------------------------
    // Load case.
    //
    // DEFAULT (physical): a clamped-one-face block under surface traction.
    //   * u = 0 on the minimum face along --loadAxis (all components),
    //   * traction g = (T,T[,T]) [N/mm^2] on the maximum face,
    //   * every other face traction-free (natural), zero body force.
    // With only one face clamped the rotational rigid-body modes are NOT
    // pinned by the boundary conditions, so this is also the load case in
    // which the near-null-space handed to PCGAMG earns its keep.
    //
    // --manufactured: the previous case -- Dirichlet data on the WHOLE
    // boundary plus the matching body force
    //   u_ex = (s,...,s),  s = prod_i sin(pi x_i)
    //   f    = -(lambda+mu) grad(div u_ex) - mu Laplacian(u_ex)
    //   2D:  d_i(div u) = pi^2 (cx cy - sx sy)                    for i = 1,2
    //        Laplacian(u_i) = -2 pi^2 sx sy
    //   3D:  d_1(div u) = pi^2 (-sx sy sz + cx cy sz + cx sy cz)  (cyclic)
    //        Laplacian(u_i) = -3 pi^2 sx sy sz
    // It is the only mode with an exact solution, hence the only one --check
    // can grade; it is kept for exactly that reason.
    //
    // Both function objects live at main() scope: gsBoundaryConditions stores
    // non-owning pointers to them.
    // -----------------------------------------------------------------------
    GISMO_ENSURE(loadAxis >= 0 && loadAxis < dim,
        "--loadAxis must be in [0,"<<dim-1<<"] for a "<<dim<<"D domain, got "<<loadAxis<<".");

    const std::string sx = "sin(pi*x)", cx = "cos(pi*x)";
    const std::string sy = "sin(pi*y)", cy = "cos(pi*y)";
    const std::string sz = "sin(pi*z)", cz = "cos(pi*z)";
    const std::string lm = num(lambda + mu), m2 = num(mu);

    gsFunctionExpr<real_t>  u_ex, fManuf;
    gsConstantFunction<real_t> fZero(gsVector<real_t>::Zero(dim), dim);
    gsConstantFunction<real_t> uClamp(gsVector<real_t>::Zero(dim), dim);
    gsConstantFunction<real_t> gTrac(gsVector<real_t>::Constant(dim, traction), dim);

    if (2 == dim)
    {
        const std::string s = sx + "*" + sy;
        // -(lambda+mu)*pi^2*(cx*cy - s) + 2*mu*pi^2*s
        const std::string fi =
            "-" + lm + "*pi*pi*(" + cx + "*" + cy + " - " + s + ")"
            " + 2*" + m2 + "*pi*pi*" + s;
        u_ex   = gsFunctionExpr<real_t>(s, s, dim);
        fManuf = gsFunctionExpr<real_t>(fi, fi, dim);
    }
    else
    {
        const std::string s = sx + "*" + sy + "*" + sz;
        const std::string d1 = "(-" + s + " + " + cx+"*"+cy+"*"+sz + " + " + cx+"*"+sy+"*"+cz + ")";
        const std::string d2 = "("  + cx+"*"+cy+"*"+sz + " - " + s + " + " + sx+"*"+cy+"*"+cz + ")";
        const std::string d3 = "("  + cx+"*"+sy+"*"+cz + " + " + sx+"*"+cy+"*"+cz + " - " + s + ")";
        const std::string tail = " + 3*" + m2 + "*pi*pi*" + s;
        u_ex   = gsFunctionExpr<real_t>(s, s, s, dim);
        fManuf = gsFunctionExpr<real_t>("-" + lm + "*pi*pi*" + d1 + tail,
                                        "-" + lm + "*pi*pi*" + d2 + tail,
                                        "-" + lm + "*pi*pi*" + d3 + tail, dim);
    }

    gsBoundaryConditions<real_t> bc;
    bc.setGeoMap(mp);

    BoxFaces faces;
    if (manufactured)
    {
        for (auto& bs : mp.boundaries())
            bc.addCondition(bs.patch, bs.side(), condition_type::dirichlet, &u_ex);
    }
    else
    {
        faces = classifyFaces(mp, loadAxis);
        GISMO_ENSURE(!faces.lo.empty(),
            "No boundary side found on the minimum face along axis "<<loadAxis
            <<" -- nothing to clamp, the system would be singular.");
        GISMO_ENSURE(!faces.hi.empty(),
            "No boundary side found on the maximum face along axis "<<loadAxis
            <<" -- nothing to load, the right-hand side would be zero.");
        for (size_t k = 0; k != faces.lo.size(); ++k)
            bc.addCondition(faces.lo[k], condition_type::dirichlet, &uClamp);
        for (size_t k = 0; k != faces.hi.size(); ++k)
            bc.addCondition(faces.hi[k], condition_type::neumann,   &gTrac);
    }

    // Body force: zero in the traction case. The two-argument assemble() is
    // used in BOTH modes so the assembly phase measures the same code path
    // (matrix + load vector + Dirichlet elimination term) either way.
    gsFunctionSet<real_t>& fBody = manufactured
        ? static_cast<gsFunctionSet<real_t>&>(fManuf)
        : static_cast<gsFunctionSet<real_t>&>(fZero);

    // -----------------------------------------------------------------------
    // Assembler, vector-valued space, DOF mapper
    // -----------------------------------------------------------------------
    gsExprAssembler<real_t> A(1, 1);
    A.setIntegrationElements(mb);
    A.options().setSwitch("lazyMatrix", !o.eagerMatrix);

    auto u = A.getSpace(mb, dim); // dim components

    timer.tic("dofmapper");
    u.setup(bc, o.dirInterp ? dirichlet::interpolation : dirichlet::l2Projection, 0);
    timer.toc();

    auto G  = A.getMap(mp);
    auto ff = A.getCoeff(fBody, G);

    const index_t nDofs  = A.numDofs();
    const index_t nElems = static_cast<index_t>(mb.domain()->numElements());

    // -----------------------------------------------------------------------
    // Partitioning (--partitioner) + DOF ownership. See buildPartition().
    // -----------------------------------------------------------------------
    const index_t nparts = (o.nparts > 0) ? o.nparts
                                          : math::max((index_t)1, o.partsPerRank * (index_t)nranks);
    GISMO_ENSURE(nparts <= nElems,
        "nparts ("<<nparts<<") exceeds the element count ("<<nElems<<"): "
        "refine further (-r) or use fewer ranks/partitions.");

    const PartitionResult part = buildPartition(o, mp, mb, u.mapper(), nparts,
                                                nElems, rank, (index_t)nranks,
                                                timer, comm);

    const gsPartitionedDofMapper&  dofMap     = part.dofMap;
    const gsVector<index_t>&       perm       = dofMap.permutation();
    typename gsDomain<real_t>::Ptr rankDomain = part.rankDomain;

    const index_t nOwned    = dofMap.numOwnedDofs(rank);
    const index_t nRankElem = static_cast<index_t>(rankDomain->numElements());

#if PETSC_VERSION_GE(3,18,0)
    const bool useCoo = !o.noCoo;
#else
    const bool useCoo = false;
    GISMO_UNUSED(o.noCoo);
#endif

    // -----------------------------------------------------------------------
    // Distributed PETSc matrix and vectors
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
    // Local assembly over this rank's combined subdomain.
    // Same weak form as examples/linear_elasticity_example.cpp.
    // -----------------------------------------------------------------------
    timer.tic("assemble");
    A.setIntegrationDomain(rankDomain);
    A.initSystem();

    auto phys_jacobian = ijac(u, G);
    auto bilin_lambda  = lambda * idiv(u, G) * idiv(u, G).tr() * meas(G);
    auto bilin_mu      = mu * ((phys_jacobian.cwisetr() + phys_jacobian) % phys_jacobian.tr()) * meas(G);

    A.assemble( bilin_lambda + bilin_mu,   // matrix
                u * ff * meas(G) );        // rhs (incl. -K_ib*g)

    // Surface traction, accumulated into the SAME rhs (no initSystem() in
    // between). nv(G).norm() is the boundary measure, so this is the
    // int_Gamma v . g ds term. assembleBdr respects setIntegrationDomain, so
    // each rank integrates only the loaded sides of its own elements and the
    // ADD_VALUES sum over ranks is the full load -- the rank-invariance of
    // "compliance" below is what actually verifies that.
    if (!manufactured)
    {
        auto g_N = A.getBdrFunction(G);
        A.assembleBdr(bc.get("Neumann"), u * g_N * nv(G).norm());
    }
    timer.toc();

    const index_t localNnz = A.matrix().nonZeros();

    timer.tic("insert");
    // Fresh for every measured call: the insertion helpers never reset it, so
    // a reused instance would silently report a previous call's numbers.
    // Only measured (non-NULL) under --memReport -- the helpers skip all
    // internal sampling when memOut is NULL, so this costs nothing otherwise.
    gsPetscInsertMemory insMem;
    gsPetscInsertMemory* insMemOut = o.memReport ? &insMem : NULL;
#if PETSC_VERSION_GE(3,18,0)
    if (useCoo) petsc_insertSparseMatrixCOO(K, A.matrix(), perm, insMemOut);
    else        petsc_insertSparseMatrixPermuted(K, A.matrix(), perm, ADD_VALUES, insMemOut);
#else
    petsc_insertSparseMatrixPermuted(K, A.matrix(), perm, ADD_VALUES, insMemOut);
#endif
    petsc_insertVectorPermuted(b, A.rhs(), perm);
    timer.toc();

    timer.tic("mat_assembly");
    PetscCall( MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY) );
    PetscCall( MatAssemblyEnd  (K, MAT_FINAL_ASSEMBLY) );
    PetscCall( VecAssemblyBegin(b) );
    PetscCall( VecAssemblyEnd  (b) );
    timer.toc();

    double rssMaxMB = 0, rssSumMB = 0;
    reduceRss(comm, rssMaxMB, rssSumMB);

    // --memReport block (B): the mat_assembly boundary is the "everything
    // live" point (K/b/x freshly assembled, fiber matrix + CSC copy both
    // still live), so this is where the named byte budget is filled first.
    if (o.memReport)
    {
        MemBudget budget;
        budget.add("fiber_ptrs", (double)A.fiberMatrix().fiberPointerBytes());
        budget.add("fiber_data", (double)A.fiberMatrix().fiberDataBytes());
        budget.add("csc_matrix", sparseMatrixBytes(A.matrix()));

        addPetscMatBudget(budget, K, nOwned, comm);

        PetscInt bLoc = 0, xLoc = 0;
        PetscCall( VecGetLocalSize(b, &bLoc) );
        PetscCall( VecGetLocalSize(x, &xLoc) );
        budget.add("petsc_vecs", (double)(bLoc + xLoc) * (double)sizeof(PetscScalar));

        budget.add("dofmapper",           (double)u.mapper().nBytes());
        budget.add("partitioned_mapper",  (double)dofMap.nBytes());

        // insert_transient reports the CAPACITY of whichever insertion
        // path's transient buffers ran (COO triplets or the RowMajor copy).
        // insert_peak_rise is the separately OBSERVED VmHWM rise while those
        // buffers were alive -- on the permuted (--no-coo) path this is the
        // only figure that also sees Eigen's storage-order-conversion
        // allocations, which transientBytes cannot; see gsPetscInsertMemory's
        // doc. A 0 rise is a legitimate result (transients fit under an
        // earlier peak), not a missing measurement.
        budget.addTransient("insert_transient", insMem.transientBytes);
        budget.addTransient("insert_peak_rise (observed VmHWM rise during insert)",
                            insMem.peakRssBytes - insMem.entryPeakRssBytes);
        budget.addTransient("partitioner_lb", part.partitionerBytesLB);

        budget.print(gsInfo, "mat_assembly", nDofs, localNnz, comm);
        if (0 == rank)
            gsInfo << "  touched fiber columns: " << touchedFiberColumns(A.matrix())
                   << " (bounds allocator overhead: count x ~16-32 B per `new Fiber`)\n";
    }

    // -----------------------------------------------------------------------
    // Rigid-body near-null-space for PCGAMG (see the file header).
    //
    // Scoped so the replicated dim x nDofs coordinate array is released as
    // soon as the (distributed) modes have been built from it.
    // -----------------------------------------------------------------------
    // dof_geometry (--memReport only): dg is scoped INSIDE this block and
    // does not exist at the mat_assembly boundary above, so its bytes are
    // captured here into a variable that outlives the block. Stays exactly 0
    // under --noRBM / --nosolve (the block below does not run) and 0 for
    // Poisson (which never calls computeDofGeometry).
    double dofGeometryBytes = 0.0;
    const bool useRBM = !o.noRBM && !o.noSolve;
    if (!useRBM) timer.skip("nullspace"); // keep the CSV schema fixed
    else
    {
        timer.tic("nullspace");
        gsVector<index_t> invPerm(nDofs);
        for (index_t g = 0; g != nDofs; ++g)
            invPerm(perm(g)) = g;

        const DofGeometry dg = computeDofGeometry(mp, mb, u.mapper(), dim);
        dofGeometryBytes = (double)dg.coords.rows() * (double)dg.coords.cols() * (double)sizeof(real_t)
                          + (double)dg.comp.capacity() * (double)sizeof(index_t);
        attachRigidBodyNullSpace(K, dg, dim, invPerm, comm);
        timer.toc();
    }

    // -----------------------------------------------------------------------
    // Solve
    // -----------------------------------------------------------------------
    // GAMG bracket (--memReport only): see gsPoissonScaling_example.cpp for
    // the rationale -- only the delta is meaningful, sampled before/after
    // regardless of --nosolve so the delta is honestly 0 there.
    double gamgPreBytes = 0.0, gamgPostBytes = 0.0;
    if (o.memReport)
    {
        PetscLogDouble m = 0;
        PetscCall( PetscMemoryGetMaximumUsage(&m) );
        gamgPreBytes = (double)m;
    }

    SolveStats st;
    real_t compliance = 0;
    if (!o.noSolve)
    {
        solveSystem(K, b, x, o.rtol, o.maxIts, st, timer, comm);

        // Compliance b^T x (= 2x the strain energy). One distributed dot
        // product: O(1) memory, no gather, so unlike --check it is usable at
        // any problem size. It is the load case's physical scalar QoI AND the
        // sharpest cheap correctness check available here -- it must be
        // identical across rank counts, which catches any double-counted or
        // dropped contribution in the partitioned assembly (volume OR
        // traction) that the residual norm alone would not.
        PetscScalar dot = 0;
        PetscCall( VecDot(b, x, &dot) );
        compliance = static_cast<real_t>(PetscRealPart(dot));
    }
    else
    { timer.skip("ksp_setup"); timer.skip("ksp_solve"); } // keep the CSV schema fixed

    if (o.memReport)
    {
        PetscLogDouble m = 0;
        PetscCall( PetscMemoryGetMaximumUsage(&m) );
        gamgPostBytes = (double)m;
    }

    // --memReport block (B), post-solve: same named lines again (K/b/x are
    // still live), plus dof_geometry (elasticity-only transient) and the GAMG
    // bracket delta -- exactly 0 under --nosolve, > 0 otherwise.
    if (o.memReport)
    {
        MemBudget budget;
        budget.add("fiber_ptrs", (double)A.fiberMatrix().fiberPointerBytes());
        budget.add("fiber_data", (double)A.fiberMatrix().fiberDataBytes());
        budget.add("csc_matrix", sparseMatrixBytes(A.matrix()));

        addPetscMatBudget(budget, K, nOwned, comm);

        PetscInt bLoc = 0, xLoc = 0;
        PetscCall( VecGetLocalSize(b, &bLoc) );
        PetscCall( VecGetLocalSize(x, &xLoc) );
        budget.add("petsc_vecs", (double)(bLoc + xLoc) * (double)sizeof(PetscScalar));

        budget.add("dofmapper",          (double)u.mapper().nBytes());
        budget.add("partitioned_mapper", (double)dofMap.nBytes());

        budget.addTransient("dof_geometry", dofGeometryBytes);

        budget.print(gsInfo, "post-solve", nDofs, localNnz, comm);

        const double gamgLocalBytes = gamgPostBytes - gamgPreBytes;
        double gamgMaxMB = 0.0;
        MPI_Reduce(&gamgLocalBytes, &gamgMaxMB, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        gamgMaxMB /= (1024.0*1024.0);
        if (0 == rank)
            gsInfo << "  gamg_bracket (delta, PetscMemoryGetMaximumUsage post-pre): "
                   << std::fixed << std::setprecision(2) << gamgMaxMB << " MB\n";
    }

    // -----------------------------------------------------------------------
    // Optional verification (--check): does NOT scale, see gsScalingCommon.h.
    // -----------------------------------------------------------------------
    // The traction load case has no closed-form solution, so --check has
    // nothing to grade there; use "compliance" (rank-invariant) instead, or
    // re-run with --manufactured for a graded convergence check. Plotting,
    // however, only needs the solution -- it is NOT gated on --check.
    if (o.check && !manufactured && 0 == rank)
        gsWarn << "--check needs an exact solution: pass --manufactured, or "
                  "use the rank-invariance of 'compliance' as the check for "
                  "the traction load case. Skipping error computation.\n";

    const bool wantErr  = o.check && manufactured && !o.noSolve;
    const bool wantPlot = o.plot && !o.noSolve;

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

        A.setIntegrationElements(mb); // norms/plot are over the FULL domain
        gsExprEvaluator<real_t> ev(A);
        auto u_sol = A.getSolution(u, solVector);

        if (wantErr)
        {
            auto u_ex_var = ev.getVariable(u_ex, G);
            // jac(.) rather than igrad(.) on the vector-valued reference:
            // grad_expr asserts a 1D variable ("use jac(.) instead"), so
            // igrad(u_ex_var) is invalid for a displacement field. Both
            // jac(u_ex_var) (the variable carries the geometry map, so its
            // derivatives are already physical) and ijac(u_sol,G) are dim x dim
            // physical Jacobians.
            l2err = math::sqrt( ev.integral( (u_ex_var - u_sol).sqNorm() * meas(G) ) );
            h1err = l2err + math::sqrt(
                ev.integral( (jac(u_ex_var) - ijac(u_sol, G)).sqNorm() * meas(G) ) );
        }

        if (wantPlot && 0 == rank)
        {
            gsParaviewCollection collection("ParaviewOutput/elasticity_scaling", &ev);
            collection.options().setSwitch("plotElements", true);
            collection.newTimeStep(&mp);
            collection.addField(u_sol, "displacement");
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

    // --memReport block (A): collective (no-op, no collective at all, when
    // the flag is off -- see PhaseTimer::reduceMem()).
    std::vector<double> hwmMaxMB, rssMaxMB_trace;
    timer.reduceMem(hwmMaxMB, rssMaxMB_trace);

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
           .add("problem",  std::string("elasticity"))
           .add("dim",      (index_t)dim)
           .add("nranks",   (index_t)nranks)
           .add("threads",  nthreads)
           .add("nparts",   nparts)
           .add("partitioner", o.partitioner)
           .add("part_weight", part.partWeight)
           .add("npatches", (index_t)mp.nPatches())
           .add("nref",     o.nref)
           .add("degree",   (index_t)mb.minCwiseDegree())
           .add("nelems",   nElems)
           .add("ndofs",    nDofs)
           .add("dofs_per_rank", (index_t)(nDofs / nranks))
           .add("edgecut",  part.edgeCut)
           .add("interface_dofs", nInterface)
           .add("owned_min", ownedMin)
           .add("owned_max", ownedMax)
           .add("elem_min",  elemMin)
           .add("elem_max",  elemMax)
           .add("nnz_local_min", nnzMin)
           .add("nnz_local_max", nnzMax)
           .add("load_case", std::string(manufactured ? "manufactured" : "traction"))
           .add("load_axis", loadAxis)
           .add("traction",  (double)traction)
           .add("clamped_sides", (index_t)faces.lo.size())
           .add("loaded_sides",  (index_t)faces.hi.size())
           .add("lambda",   (double)lambda)
           .add("mu",       (double)mu)
           .add("rbm",      (index_t)(useRBM ? 1 : 0))
           .add("lazy",     (index_t)(o.eagerMatrix ? 0 : 1))
           .add("coo",      (index_t)(useCoo ? 1 : 0));

        addPhases(rec, timer.names(), tMax, tMin);

        rec.add("ksp_its",     (index_t)st.its)
           .add("ksp_reason",  (index_t)st.reason)
           .add("rel_residual", (double)st.relRes)
           .add("compliance",   (double)compliance, 12) // rank-invariant QoI
           .add("rss_base_mb",  rssBaseMB)
           .add("rss_max_mb",   rssMaxMB)
           .add("rss_delta_mb", rssMaxMB - rssBaseMB)
           .add("rss_sum_mb",   rssSumMB)
           .add("rss_hwm_mb",   rssHwmMB)
           .add("l2_error",     (double)l2err)
           .add("h1_error",     (double)h1err);

        gsInfo << "\n=== Linear elasticity METIS+PETSc scaling run ===\n";
        rec.print(gsInfo);
        gsInfo << "\n";

        if (o.memReport) printMemTrace(gsInfo, timer.names(), hwmMaxMB, rssMaxMB_trace);

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
