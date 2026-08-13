/** @file gsMetisPetscAssembly_example.cpp

    @brief MPI-parallel Poisson assembly via element partitioning and PETSc,
    with a partitioned PETSc row layout and DOF-permuted insertion (Phase 4).

    Two partitioners are selectable with --partitioner: "metis" (default,
    element dual graph + METIS, gsMetisPartitioner) and the graph-free
    geometric strategies "rcb" | "hilbert" | "morton" (gsGeometricPartitioner).
    Every verification gate below is partitioner-AGNOSTIC -- each one compares
    the partitioned parallel result against a full serial reference assembled
    on every rank, and none of them inspects labels, edge cuts or part counts.
    That makes this driver the correctness oracle for any partitioner plugged
    into it: a new labelling strategy is validated end-to-end by running it
    here. Note the gates are scalar (single-component space), so they cannot
    detect component-related DOF-ownership bugs.

    All ranks replicate the geometry, basis, and the partitioning (serial,
    identical on every rank). Partitions are assigned to ranks cyclically:
    rank r owns partitions { r, r+nranks, r+2*nranks, ... }. gismo's
    gsExprAssembler stays global-indexed throughout (single
    setIntegrationDomain/initSystem/assemble per rank, over one combined
    subdomain = the union of the rank's owned partitions); a
    gsPartitionedDofMapper computes, without communication, each free DOF's
    owning rank and a global permutation that groups PETSc rows contiguously
    by owner. The PETSc matrix/vector are sized via
    petsc_setupMatrixPartitioned to that ownership (not an even nDofs/nranks
    split), and insertion goes through the permutation
    (petsc_insertSparseMatrixPermuted/petsc_insertVectorPermuted, or the COO
    API petsc_insertSparseMatrixCOO on PETSc >= 3.18, --no-coo to force the
    fallback). Pass --lazyMatrix to exercise it: each rank's assembler now
    only ever touches the fraction of the nDofs matrix columns its own DOFs
    span, so lazy columns are the interesting mode here (default stays off,
    same as Phase 4.1, so both remain reachable from the CLI).

    Correctness is verified by comparing K_parallel * p against K_serial * p
    (probe p = cos(i+1)), all reads/writes going through the permutation.
    Pass criterion: ||K_par*p - K_ser*p|| / ||K_ser*p|| < 1e-10.

    The problem carries a manufactured solution (forcing f + Dirichlet data
    on the whole exterior boundary), so the load term and the -K_ib*g
    Dirichlet-elimination term both contribute to A.rhs(). Each rank's RHS
    is inserted into the distributed, permuted PETSc Vec, verified directly
    against the serial reference RHS, and then solved with KSPSolve; the
    discrete solution is compared against the manufactured exact solution.

    Usage:
      mpirun -np 4 ./bin/gsMetisPetscAssembly_example -n 4 -r 3

    Options:
      -n <int>     Number of partitions       (default: 4)
      -r <int>     Global refinement levels   (default: 2)
      --partitioner <metis|rcb|hilbert|morton>
                   Element partitioner (default: metis). metis builds the
                   element dual graph and calls METIS; rcb/hilbert/morton are
                   graph-free geometric strategies (gsGeometricPartitioner)
                   which every rank recomputes locally, without a broadcast.
      --lazyMatrix Lazy fiber-matrix columns  (default: off = eager)
      --no-coo     Force the MatSetValue fallback instead of PETSc's COO
                   insertion API (default: COO, when PETSc >= 3.18)

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): H.M. Verhelst
*/

#include <gismo.h>
#include <gsMetis/gsMetis.h>
#include <gsPetsc/PETScSupport.h>
#include <gsPetsc/gsPetscLocalToGlobal.h>

#include <iomanip>

using namespace gismo;

// ---------------------------------------------------------------------------
// Memory diagnostics.
//
// Two complementary measurements:
//   * reportRSS()  — ground-truth resident set size (RSS) of the process at a
//     checkpoint, reduced across ranks (max over ranks = worst single rank,
//     sum = whole job). Deltas between checkpoints attribute memory to the
//     phase that ran in between (mesh/basis, mapper, partition, assembly).
//     This captures *everything*, including per-rank replication and PETSc's
//     internal/stash allocations.
//   * the analytic/measured per-object sizes printed in main() (dof mapper,
//     basis, local matrix, PETSc Mat) — to break a phase's delta into objects.
// ---------------------------------------------------------------------------
static double mb_(double bytes) { return bytes / (1024.0 * 1024.0); }

static void reportRSS(const std::string& tag, int rank)
{
    PetscLogDouble rss = 0;
    PetscMemoryGetCurrentUsage(&rss); // current process RSS in bytes (reads /proc on Linux)
    double local = static_cast<double>(rss), maxr = 0, sumr = 0;
    MPI_Reduce(&local, &maxr, 1, MPI_DOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD);
    MPI_Reduce(&local, &sumr, 1, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
    if (rank == 0)
        gsInfo << "[mem] " << std::left << std::setw(26) << tag << std::right
               << " RSS  max/rank=" << std::setw(8) << std::fixed << std::setprecision(1)
               << mb_(maxr) << " MB   job total=" << std::setw(8) << mb_(sumr) << " MB\n";
}

// Bytes held by a gsSparseMatrix (CSC: values + inner indices + outer pointers).
static double sparseBytes(const gsSparseMatrix<real_t>& M)
{
    return double(M.nonZeros()) * (sizeof(real_t) + sizeof(index_t))
         + double(M.outerSize() + 1) * sizeof(index_t);
}

// ---------------------------------------------------------------------------
// 7-patch conforming H-domain (same geometry as gsMetisDD_example)
//
//  y=3  +---+       +---+
//       | p0|       | p1|
//  y=2  +---+---+---+---+
//       | p2| p3| p4|
//  y=1  +---+---+---+---+
//       | p5|       | p6|
//  y=0  +---+       +---+
//       x=0 x=1   x=2 x=3
// ---------------------------------------------------------------------------
gsMultiPatch<real_t> makeHDomain7()
{
    gsKnotVector<real_t> kv(0.0, 1.0, 0, 2); // degree 1, no interior knots

    auto makeRect = [&](real_t x0, real_t x1, real_t y0, real_t y1)
    {
        gsTensorBSplineBasis<2, real_t> basis(kv, kv);
        gsMatrix<real_t> c(4, 2);
        c << x0,y0, x1,y0, x0,y1, x1,y1;
        return gsTensorBSpline<2, real_t>(basis, c);
    };

    gsMultiPatch<real_t> mp;
    mp.addPatch(makeRect(0,1, 2,3)); // p0: top-left arm
    mp.addPatch(makeRect(2,3, 2,3)); // p1: top-right arm
    mp.addPatch(makeRect(0,1, 1,2)); // p2: middle-left
    mp.addPatch(makeRect(1,2, 1,2)); // p3: centre connector
    mp.addPatch(makeRect(2,3, 1,2)); // p4: middle-right
    mp.addPatch(makeRect(0,1, 0,1)); // p5: bottom-left arm
    mp.addPatch(makeRect(2,3, 0,1)); // p6: bottom-right arm
    mp.computeTopology();
    return mp;
}

int main(int argc, char* argv[])
{
    // The options are registered before PetscInitialize so that --help/
    // --version can be serviced without entering PETSc/MPI at all: a binary
    // launched directly (not via mpirun) has no MPI runtime to initialise,
    // and on hosts where the singleton MPI runtime cannot start, calling
    // PetscInitialize first would hang "-h" before any output is printed.
    index_t nparts = 4;
    index_t nref   = 2;
    bool    noVerify = false;
    // Bound by reference into cmd below, so it must outlive cmd.getValues().
    std::string partitionerName = "metis";

    gsCmdLine cmd("Partitioned (METIS or graph-free) PETSc parallel assembly verification.");
    cmd.addInt("n", "nparts", "Number of partitions", nparts);
    cmd.addInt("r", "nref",   "Global refinement levels",   nref);
    // Long-form only: -n and -r are taken, and an empty flag string is the
    // documented way to register an option with no short flag.
    cmd.addString("", "partitioner",
        "Element partitioner: metis (default, element dual graph + METIS) | "
        "rcb | hilbert | morton (graph-free geometric, gsGeometricPartitioner).",
        partitionerName);
    cmd.addSwitch("noverify",
        "Skip the serial verification. The check assembles the full serial "
        "stiffness K_ser and gathers the global solution vector on EVERY rank "
        "(neither scales with the number of ranks), so disable it at high "
        "refinement to avoid running out of memory.", noVerify);
    // Default off, same as before -- gsCmdLine switches are one-directional
    // (TCLAP::SwitchArg: presence sets true, there is no CLI way to force
    // false back), so a default-true here would make eager mode
    // unreachable from the command line. Pass --lazyMatrix explicitly: with
    // one combined subdomain per rank (below), a rank only ever touches a
    // fraction of the nDofs matrix columns, so lazy is the interesting mode
    // for this partitioned architecture, but both must stay reachable for
    // the eager-vs-lazy comparison the 4.1 checkpoint relies on.
    bool lazyMatrix = false;
    cmd.addSwitch("lazyMatrix",
        "Allocate gsExprAssembler fiber-matrix columns on first use "
        "(memory-lean for subdomain assembly). Default off = eager.", lazyMatrix);
    bool noCoo = false;
    cmd.addSwitch("no-coo",
        "Use the MatSetValue-per-entry fallback insertion path instead of "
        "PETSc's COO API (MatSetPreallocationCOO/MatSetValuesCOO, PETSc >= "
        "3.18). No effect if built against an older PETSc, where the "
        "fallback is the only path.", noCoo);

    // Pre-MPI fast path: if the user only asked for help/version, honour it
    // on a synthetic argv (the options above are registered exactly once and
    // render the real usage text), so unrelated PETSc options such as
    // -ksp_rtol can never fail this parse. Stop at "--" like TCLAP's
    // ignore_rest switch does.
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);
        if (arg == "--") break;
        if (arg == "-h" || arg == "--help" || arg == "--version")
        {
            char *hargv[] = { argv[0], argv[i], NULL };
            try { cmd.getValues(2, hargv); }
            catch (int ret) { return ret; }
            return 0;
        }
    }

    // PetscInitialize must be first for the actual run: it also initialises
    // MPI and strips PETSc-specific options from argv before the full
    // gsCmdLine parse below.
    PetscInitialize(&argc, &argv, NULL, NULL);
    // Suppress "unused options" warnings for flags we parse ourselves via gsCmdLine.
    PetscCall( PetscOptionsSetValue(NULL, "-options_left", "false") );

    int rank = 0, nranks = 1;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &nranks);

    try { cmd.getValues(argc, argv); }
    catch (int ret) { PetscFinalize(); return ret; }

    reportRSS("startup", rank);

    // -----------------------------------------------------------------------
    // Geometry, basis — replicated identically on every rank.
    // -----------------------------------------------------------------------
    gsMultiPatch<real_t> mp = makeHDomain7();
    gsMultiBasis<real_t> mb(mp);
    for (index_t i = 0; i < nref; ++i)
        mb.uniformRefine();

    reportRSS("after mesh+basis", rank);

    // -----------------------------------------------------------------------
    // Expression assembler setup — space and geometry map are shared across
    // all per-partition assemblies below.
    // -----------------------------------------------------------------------
    gsExprAssembler<real_t> A(1, 1);
    A.setIntegrationElements(mb);
    A.options().setSwitch("lazyMatrix", lazyMatrix);

    auto u = A.getSpace(mb);

    // Manufactured solution u_ex = sin(pi/3 x) sin(pi/3 y) on the [0,3]x[0,3]
    // bounding box, with matching forcing f = -Laplacian(u_ex). Dirichlet
    // data is imposed on the ENTIRE exterior boundary (not just one side):
    // u_ex has nonzero normal flux on most of the H-domain's exterior edges,
    // so leaving those sides "natural" (homogeneous Neumann) would assemble
    // a PDE inconsistent with u_ex -- the KSPSolve-vs-exact check would then
    // fail for a real physics reason, not a solver-tolerance one (see
    // poisson2d_bvp.xml, which pairs its manufactured solution with an
    // explicit Neumann flux function on the non-Dirichlet sides for the same
    // reason). Pure Dirichlet everywhere sidesteps needing that flux
    // function, and is a *stronger* test of the -K_ib*g elimination-term RHS
    // path since every partition touching any exterior boundary now
    // contributes an elimination term that must sum correctly.
    gsFunctionExpr<real_t> u_ex("sin(pi/3*x)*sin(pi/3*y)", 2);
    gsFunctionExpr<real_t> f("2*pi*pi/9*sin(pi/3*x)*sin(pi/3*y)", 2);

    gsBoundaryConditions<real_t> bc;
    bc.setGeoMap(mp);
    for (auto & bs : mp.boundaries())
        bc.addCondition(bs.patch, bs.side(), condition_type::dirichlet, &u_ex);
    u.setup(bc, dirichlet::l2Projection, 0);

    auto G = A.getMap(mp);
    auto ff = A.getCoeff(f, G);

    // nDofs is available straight from the finalized mapper (u.setup() called
    // mapper.finalize()), so we do NOT initSystem() on the full domain here:
    // that would reserve the entire freeSize x freeSize sparsity pattern just
    // to read a single integer (~1.6 GB/rank at r10). numDofs() only reads the
    // mapper.
    const index_t nDofs = A.numDofs();

    reportRSS("after mapper (no full initSystem)", rank);

    if (rank == 0)
    {
        gsInfo << "nDofs=" << nDofs << "  nparts=" << nparts
               << "  nranks=" << nranks << "\n";

        // Per-object estimates (per rank; these structures are replicated on
        // every rank). The dof mapper's dominant store is m_dofs: one index
        // per local (patch, basis-function) pair = mb.totalSize() entries.
        const double mapperBytes = double(mb.totalSize()) * sizeof(index_t);
        gsInfo << "[mem] dof mapper (est)      " << std::fixed << std::setprecision(1)
               << mb_(mapperBytes) << " MB   (" << mb.totalSize()
               << " local dofs x " << sizeof(index_t) << " B)\n";
        gsInfo << "[mem] basis: " << mb.totalSize()
               << " functions over " << mb.nBases()
               << " patches (object stores only knot vectors -> small)\n";
    }

    // -----------------------------------------------------------------------
    // Partitioning. Two paths behind --partitioner:
    //
    //  * metis (default): unchanged. Fix 5: METIS itself only needs to run
    //    ONCE per job, not once per rank -- rank 0 calls partition() (builds
    //    the element graph, then runs METIS); every other rank calls
    //    buildGraph() only (needed for makeDofMapper()/setPartLabels() below,
    //    both of which require the element graph to exist, but not for METIS's
    //    result, which arrives via the broadcast+setPartLabels() instead).
    //    storeElementDofs=true keeps the per-element free-DOF lists the DOF
    //    mapper below needs (gsElementGraph::elementFreeDofsCSR()).
    //
    //  * rcb | hilbert | morton: graph-free geometric partitioning. NO element
    //    graph and NO broadcast: every rank runs partition() and recomputes an
    //    identical partition locally. gsGeometricPartitioner's orderings are
    //    total (centroid coordinate, then element id), so rank-local
    //    recomputation agrees exactly; there is no setPartLabels() path into
    //    that class, and re-adding a broadcast would defeat the purpose.
    //
    // Only subdomains(), makeDofMapper() and subdomainForRank() are used
    // downstream, and all three live on gsPartitionerBase<real_t>, so the rest
    // of the driver -- including every verification gate -- is
    // partitioner-agnostic and runs unchanged on both paths.
    // -----------------------------------------------------------------------
    memory::unique_ptr< gsMetisPartitioner<real_t> >     metisPart;   // metis path only
    memory::unique_ptr< gsGeometricPartitioner<real_t> > geoPart;     // graph-free path only
    gsPartitionerBase<real_t> * partitioner = NULL;                   // non-owning view
    index_t edgeCutValue = -1;

    if (partitionerName == "metis")
    {
        gsMetisPartitioner<real_t>::Options partOpts;
        partOpts.storeElementDofs = true;
        metisPart.reset(new gsMetisPartitioner<real_t>(mb, u.mapper(), nparts, partOpts));

        if (rank == 0)
            metisPart->partition();
        else
            metisPart->buildGraph();

        {
            // partLabels()/edgeCut() require m_partitioned (i.e. a prior
            // partition() call), which only rank 0 has done above -- size the
            // buffer from the (buildGraph()-populated, on every rank) element
            // count instead of reading partLabels() unconditionally.
            std::vector<idx_t> labels(static_cast<size_t>(mb.domain()->numElements()));
            if (rank == 0)
                labels.assign(metisPart->partLabels().begin(), metisPart->partLabels().end());
            index_t edgeCutBcast = (rank == 0) ? metisPart->edgeCut() : 0;

            MPI_Bcast(labels.data(), static_cast<int>(labels.size() * sizeof(idx_t)),
                      MPI_BYTE, 0, PETSC_COMM_WORLD);
            MPI_Bcast(&edgeCutBcast, static_cast<int>(sizeof(index_t)),
                      MPI_BYTE, 0, PETSC_COMM_WORLD);

            metisPart->setPartLabels(give(labels), edgeCutBcast);
        }

        edgeCutValue = metisPart->edgeCut();
        partitioner  = metisPart.get();
    }
    else
    {
        // strategyFromString validates the name (and throws, identically on
        // every rank, on anything but rcb/hilbert/morton) -- no local string
        // table here.
        gsGeometricPartitioner<real_t>::Options geoOpts;
        geoOpts.strategy = gsGeometricPartitioner<real_t>::strategyFromString(partitionerName);
        geoPart.reset(new gsGeometricPartitioner<real_t>(mp, mb, u.mapper(), nparts, geoOpts));
        geoPart->partition();                // every rank, identical result -- no broadcast
        edgeCutValue = geoPart->edgeCut();   // -1: graph-free, there is no edge cut
        partitioner  = geoPart.get();
    }

    if (rank == 0)
        gsInfo << "partitioner: " << partitionerName
               << "   edge cut: " << edgeCutValue << "  (-1 = graph-free)\n";
    const auto subdomains = partitioner->subdomains(); // one per partition

    // DOF ownership + global permutation: computed identically on every rank
    // from the replicated partition labels (broadcast on the metis path,
    // recomputed rank-locally on the graph-free path) -- no communication
    // needed. perm(g) groups PETSc rows contiguously by owning rank, matching
    // petsc_setupMatrixPartitioned's row layout below.
    const gsPartitionedDofMapper dofMap = partitioner->makeDofMapper(nranks);
    const gsVector<index_t>&     perm   = dofMap.permutation();

    // One combined subdomain per rank -- the union of every partition
    // gsPartitionedDofMapper::rankOfPart cyclically assigns to it (Fix 6:
    // subdomainForRank() replaces the old hand-rolled ownedIndices loop +
    // gsSubDomain downcast; the part->rank convention now lives in exactly
    // one place, gsPartitionedDofMapper::rankOfPart, shared with
    // makeDofMapper() above). Replaces the old per-partition assembly loop:
    // one setIntegrationDomain/initSystem/assemble/insert per rank instead
    // of one per owned partition.
    typename gsDomain<real_t>::Ptr rankDomain = partitioner->subdomainForRank(rank, nranks);

    reportRSS("after partition", rank);

#if PETSC_VERSION_GE(3,18,0)
    const bool useCoo = !noCoo;
#else
    const bool useCoo = false;
    GISMO_UNUSED(noCoo);
#endif

    // Inserts a global-indexed gismo sparse matrix into a partitioned PETSc
    // Mat through dofMap's permutation, via COO (default, PETSc >= 3.18) or
    // the MatSetValue fallback (--no-coo, or an older PETSc).
    auto insertMatrix = [&](Mat& M, const gsSparseMatrix<real_t>& mat)
    {
#if PETSC_VERSION_GE(3,18,0)
        if (useCoo) { petsc_insertSparseMatrixCOO(M, mat, perm); return; }
#endif
        petsc_insertSparseMatrixPermuted(M, mat, perm);
    };

    // -----------------------------------------------------------------------
    // Distributed PETSc MPIAIJ matrix + RHS vector, partitioned layout: rank
    // r owns exactly dofMap.numOwnedDofs(r) rows (not an even nDofs/nranks
    // split), matching the METIS-partition-derived DOF assignment.
    // -----------------------------------------------------------------------
    Mat petscMat;
    petsc_setupMatrixPartitioned(petscMat, nDofs, dofMap.numOwnedDofs(rank), PETSC_COMM_WORLD);
    if (!useCoo)
        // Fallback path only: partition matrices have entries in arbitrary
        // rows (ADD_VALUES off-process is fine, just slower without
        // preallocation). The COO path preallocates exactly, so it doesn't
        // need this.
        PetscCall( MatSetOption(petscMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE) );

    Vec petscRhs, petscSol;
    PetscCall( MatCreateVecs(petscMat, &petscSol, &petscRhs) );

#ifndef NDEBUG
    // Sanity check (Phase-3 review's petsc_computeMatLayout bug class, now
    // checked instead of assumed): PETSc's row ownership for the sizes we
    // gave MatSetSizes must be an exclusive prefix sum in rank order,
    // exactly matching dofMap.rankOffset(rank).
    {
        PetscInt loChk, hiChk;
        PetscCall( MatGetOwnershipRange(petscMat, &loChk, &hiChk) );
        GISMO_ASSERT(static_cast<index_t>(loChk) == dofMap.rankOffset(rank),
            "PETSc row ownership start ("<<loChk<<") does not match "
            "gsPartitionedDofMapper::rankOffset("<<rank<<") = "
            <<dofMap.rankOffset(rank)<<".");
        GISMO_ASSERT(static_cast<index_t>(hiChk - loChk) == dofMap.numOwnedDofs(rank),
            "PETSc owned row count does not match numOwnedDofs("<<rank<<").");
    }
#endif

    // -----------------------------------------------------------------------
    // Parallel local assembly: ONE setIntegrationDomain/initSystem/assemble
    // per rank, over rankDomain (all owned partitions combined). The
    // matrix/rhs remain freeSize (x freeSize) with global DOF indices --
    // insertion is what maps them into the partitioned/permuted PETSc
    // layout, not assembly. The two-argument assemble() also builds
    // A.rhs(): the load term u*ff*meas(G) plus, where rankDomain touches
    // any Dirichlet (exterior boundary) element, the -K_ib*g
    // elimination-term contribution.
    // -----------------------------------------------------------------------
    A.setIntegrationDomain(rankDomain);
    A.initSystem();
    A.assemble(igrad(u, G) * igrad(u, G).tr() * meas(G), u * ff * meas(G));
    const double localMatBytes = sparseBytes(A.matrix()); // freeSize x freeSize, sparse
    const index_t localMatNnz  = A.matrix().nonZeros();
    insertMatrix(petscMat, A.matrix());
    petsc_insertVectorPermuted(petscRhs, A.rhs(), perm);

    reportRSS("after local assembly+insert", rank);

    PetscCall( MatAssemblyBegin(petscMat, MAT_FINAL_ASSEMBLY) );
    PetscCall( MatAssemblyEnd  (petscMat, MAT_FINAL_ASSEMBLY) );
    PetscCall( VecAssemblyBegin(petscRhs) );
    PetscCall( VecAssemblyEnd  (petscRhs) );

    reportRSS("after PETSc MatAssembly", rank);

    // Size of one per-partition gsSparseMatrix. NOTE: this is the nnz-based
    // *logical* size after fill; the reservation during initSystem() is larger
    // (reservePerColumn runs over all numDofs columns regardless of
    // setIntegrationDomain). The PETSc matrix cost is not printed here:
    // MatInfo.memory is unreliable in this build, and the RSS deltas around
    // "local assembly+insert" / "PETSc MatAssembly" already quantify it.
    {
        double lmMax = 0, lmSum = 0;
        MPI_Reduce(&localMatBytes, &lmMax, 1, MPI_DOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD);
        MPI_Reduce(&localMatBytes, &lmSum, 1, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
        if (rank == 0)
            gsInfo << "[mem] local partition Mat   max/rank=" << std::fixed << std::setprecision(1)
                   << mb_(lmMax) << " MB nnz-logical   (" << localMatNnz
                   << " nnz; reservation is larger)\n";
    }

    // -----------------------------------------------------------------------
    // Verification (optional, --noverify to skip).
    //
    // This is the memory-heavy part: it assembles the full serial stiffness
    // K_ser AND gathers the global solution vector, both replicated on EVERY
    // rank (neither scales down with the number of ranks). At high refinement
    // skip it with --noverify to keep per-rank memory bounded.
    // -----------------------------------------------------------------------
    int exitCode = 0;
    if (!noVerify)
    {
        // Serial reference — full domain, same space/map, all ranks. Also
        // builds the serial reference RHS (load term + Dirichlet elimination)
        // used by the RHS-only probe below.
        A.setIntegrationElements(mb);
        A.initSystem();
        A.assemble(igrad(u, G) * igrad(u, G).tr() * meas(G), u * ff * meas(G));
        const gsSparseMatrix<real_t>& K_ser = A.matrix();
        const gsMatrix<real_t>&       b_ser_manuf = A.rhs();

        reportRSS("after serial K_ser", rank);
        if (rank == 0)
            gsInfo << "[mem] serial K_ser (meas)   " << std::fixed << std::setprecision(1)
                   << mb_(sparseBytes(K_ser)) << " MB/rank   (" << K_ser.nonZeros()
                   << " nnz, full matrix replicated on every rank)\n";

        // Verification: K_par * probe  vs  K_ser * probe
        //
        // The Poisson stiffness has K*1 = 0 (B-spline partition of unity), so
        // the all-ones vector is in the null space and cannot detect assembly
        // errors.  Use a deterministic non-constant probe cos(i+1) instead:
        // it is identical on all ranks, is not in the null space, and exposes
        // any missing, duplicated, or misplaced partition contribution.
        gsVector<real_t> probe(nDofs);
        for (index_t i = 0; i < nDofs; ++i)
            probe(i) = std::cos(static_cast<real_t>(i) + 1.0);

        // Inverse permutation: invPerm(i) is the global DOF g with
        // perm(g) == i, i.e. the DOF that occupies PETSc row i. Needed to
        // fill a PETSc vector "by global DOF value" while only touching
        // this rank's owned rows (perm itself maps the other way, global
        // DOF -> PETSc row).
        gsVector<index_t> invPerm(nDofs);
        for (index_t g = 0; g < nDofs; ++g)
            invPerm(perm(g)) = g;

        Vec x_petsc, b_petsc;
        PetscCall( MatCreateVecs(petscMat, &x_petsc, &b_petsc) );

        // Fill x_petsc(i) = probe(invPerm(i)) over the ownership range, i.e.
        // x_petsc(perm(g)) = probe(g) for every g owned by this rank.
        PetscInt lo, hi;
        PetscCall( VecGetOwnershipRange(x_petsc, &lo, &hi) );
        for (PetscInt i = lo; i < hi; ++i)
            PetscCall( VecSetValue(x_petsc, i,
                static_cast<PetscScalar>(probe(invPerm(static_cast<index_t>(i)))), INSERT_VALUES) );
        PetscCall( VecAssemblyBegin(x_petsc) );
        PetscCall( VecAssemblyEnd  (x_petsc) );

        PetscCall( MatMult(petscMat, x_petsc, b_petsc) );

        // Gather (still PETSc-row order) then un-permute back to global DOF
        // order: b_par(g) = gathered(perm(g)).
        gsVector<real_t> b_par_permuted;
        petsc_copyVecToGismo(b_petsc, b_par_permuted, PETSC_COMM_WORLD);
        gsVector<real_t> b_par(nDofs);
        for (index_t g = 0; g < nDofs; ++g)
            b_par(g) = b_par_permuted(perm(g));

        gsVector<real_t> b_ser = K_ser * probe;

        const real_t err    = (b_par - b_ser).norm();
        const real_t ref    = b_ser.norm();
        const real_t relerr = (ref > 0) ? err / ref : err;

        if (rank == 0)
        {
            gsInfo << "||K_par*p - K_ser*p|| / ||K_ser*p|| = " << relerr << "\n";
            gsInfo << (relerr < 1e-10 ? "PASSED\n" : "FAILED (tol 1e-10)\n");
        }

        PetscCall( VecDestroy(&x_petsc) );
        PetscCall( VecDestroy(&b_petsc) );
        exitCode = (relerr < 1e-10) ? 0 : 1;

        // -------------------------------------------------------------------
        // Extension (Phase 2 / review #4): RHS-only probe, checked BEFORE any
        // solver enters the picture. This is the tightest test of
        // petsc_insertVectorPermuted itself: gather the distributed
        // petscRhs (summed per-rank via ADD_VALUES in the main assembly
        // above), un-permute back to global DOF order, and compare
        // directly against the serial reference RHS. Must hold regardless
        // of nranks/nparts, and specifically exercises DOFs on the
        // Dirichlet-adjacent partition boundary, where the load term and the
        // -K_ib*g elimination term from neighboring partitions must sum
        // exactly once each (no double-count, no drop).
        // -------------------------------------------------------------------
        {
            gsVector<real_t> b_par_rhs_permuted;
            petsc_copyVecToGismo(petscRhs, b_par_rhs_permuted, PETSC_COMM_WORLD);
            gsVector<real_t> b_par_rhs(nDofs);
            for (index_t g = 0; g < nDofs; ++g)
                b_par_rhs(g) = b_par_rhs_permuted(perm(g));

            const real_t errRhs    = (b_par_rhs - b_ser_manuf).norm();
            const real_t refRhs    = b_ser_manuf.norm();
            const real_t relerrRhs = (refRhs > 0) ? errRhs / refRhs : errRhs;

            if (rank == 0)
            {
                gsInfo << std::scientific << std::setprecision(3)
                       << "[rhs] ||b_par - b_ser|| / ||b_ser|| = " << relerrRhs
                       << std::defaultfloat << "\n";
                gsInfo << (relerrRhs < 1e-10 ? "PASSED (rhs insertion)\n"
                                              : "FAILED (rhs insertion, tol 1e-10)\n");
            }
            if (relerrRhs >= 1e-10) exitCode = 1;
        }

        // -------------------------------------------------------------------
        // Extension (Phase 2 / review #4): KSPSolve integration check.
        //
        // Two separate comparisons, deliberately not conflated:
        //  - TIGHT gate: distributed KSPSolve vs. a serial solve of the SAME
        //    discrete system (K_ser, b_ser_manuf). Both solve identical
        //    linear algebra, so this is discretization-independent and
        //    directly validates the Phase-2 pipeline (RHS insertion +
        //    distributed solve == serial), unaffected by mesh/degree choice.
        //  - INFORMATIONAL only: distributed solution vs. the manufactured
        //    exact u_ex. This carries real O(h^2) discretization error (this
        //    example uses degree-1 elements) on top of solver tolerance, so
        //    it is reported but not gated -- gating on it would make the
        //    check fail under a coarser mesh/degree even though the solve
        //    pipeline is correct.
        // -------------------------------------------------------------------
        {
            KSP ksp;
            PetscCall( KSPCreate(PETSC_COMM_WORLD, &ksp) );
            PetscCall( KSPSetOperators(ksp, petscMat, petscMat) );
            // PETSc's built-in default relative tolerance (1e-5) is too loose
            // for the 1e-6 gate below -- tighten it here, before
            // KSPSetFromOptions() so a user-supplied -ksp_rtol still wins.
            PetscCall( KSPSetTolerances(ksp, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT) );
            PetscCall( KSPSetFromOptions(ksp) );
            PetscCall( KSPSolve(ksp, petscRhs, petscSol) );

            KSPConvergedReason reason;
            PetscCall( KSPGetConvergedReason(ksp, &reason) );

            // Gather the (global, replicated) solution -- petscSol has
            // exactly nDofs rows, but in PETSc/permuted row order -- then
            // un-permute into the global-DOF-ordered vector A.getSolution()
            // expects: solVector(g) = gathered(perm(g)).
            gsMatrix<real_t> solVectorPermuted;
            petsc_copyVecToGismo(petscSol, solVectorPermuted, PETSC_COMM_WORLD);
            gsMatrix<real_t> solVector(nDofs, 1);
            for (index_t g = 0; g < nDofs; ++g)
                solVector(g, 0) = solVectorPermuted(perm(g), 0);

            // Tight gate: same discrete system, serial solve.
            gsSparseSolver<real_t>::CGDiagonal serialSolver;
            serialSolver.compute(K_ser);
            const gsMatrix<real_t> x_ser = serialSolver.solve(b_ser_manuf);

            const real_t errSolve    = (solVector - x_ser).norm();
            const real_t refSolve    = x_ser.norm();
            const real_t relerrSolve = (refSolve > 0) ? errSolve / refSolve : errSolve;

            // Informational: vs. manufactured exact solution.
            gsExprEvaluator<real_t> ev(A);
            auto u_ex_var = ev.getVariable(u_ex, G);
            auto u_sol    = A.getSolution(u, solVector);
            const real_t l2err = math::sqrt(
                ev.integral((u_ex_var - u_sol).sqNorm() * meas(G)));

            if (rank == 0)
            {
                // Explicit scientific formatting: earlier reportRSS() calls
                // leave the stream in std::fixed<<setprecision(1), which
                // would otherwise round these small relative errors to "0.0"
                // and hide genuine (if small) mismatches.
                gsInfo << std::scientific << std::setprecision(3)
                       << "[solve] KSP converged reason = " << reason
                       << "   ||x_par - x_ser|| / ||x_ser|| = " << relerrSolve
                       << "   ||u_par - u_ex||_L2 (info) = " << l2err
                       << std::defaultfloat << "\n";
                gsInfo << ((reason > 0 && relerrSolve < 1e-6) ? "PASSED (KSPSolve vs serial solve)\n"
                                                               : "FAILED (KSPSolve vs serial solve)\n");
            }
            if (reason <= 0 || relerrSolve >= 1e-6) exitCode = 1;

            PetscCall( KSPDestroy(&ksp) );
        }

        // -------------------------------------------------------------------
        // Extension (Phase 1.2 / review #3): subdomain-restricted boundary
        // assembly. The stiffness probe above never calls assembleBdr(), so
        // it cannot see the "every rank integrates the full patch boundary"
        // duplication bug at all. Assemble a boundary mass term
        // (sum_bdr u*v) over rankDomain (all owned partitions combined) and
        // compare against the same term assembled once on the full domain:
        // with the subdomain(k) fix, contributions from different ranks
        // must not overlap, so the ADD_VALUES sum must equal the serial
        // reference exactly, regardless of nranks/nparts. Through the same
        // partitioned/permuted path as the main matrix above.
        // -------------------------------------------------------------------
        {
            Mat petscBdrMat;
            petsc_setupMatrixPartitioned(petscBdrMat, nDofs, dofMap.numOwnedDofs(rank), PETSC_COMM_WORLD);
            if (!useCoo)
                PetscCall( MatSetOption(petscBdrMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE) );

            A.setIntegrationDomain(rankDomain);
            A.initSystem();
            A.assembleBdr(mp.boundaries(), u * u.tr() * meas(G));
            insertMatrix(petscBdrMat, A.matrix());
            PetscCall( MatAssemblyBegin(petscBdrMat, MAT_FINAL_ASSEMBLY) );
            PetscCall( MatAssemblyEnd  (petscBdrMat, MAT_FINAL_ASSEMBLY) );

            // Serial reference: the same boundary term over the full domain.
            A.setIntegrationElements(mb);
            A.initSystem();
            A.assembleBdr(mp.boundaries(), u * u.tr() * meas(G));
            const gsSparseMatrix<real_t> Kbdr_ser = A.matrix(); // copy: A is reused below

            Vec xb, bb;
            PetscCall( MatCreateVecs(petscBdrMat, &xb, &bb) );
            PetscInt loB, hiB;
            PetscCall( VecGetOwnershipRange(xb, &loB, &hiB) );
            for (PetscInt i = loB; i < hiB; ++i)
                PetscCall( VecSetValue(xb, i,
                    static_cast<PetscScalar>(probe(invPerm(static_cast<index_t>(i)))), INSERT_VALUES) );
            PetscCall( VecAssemblyBegin(xb) );
            PetscCall( VecAssemblyEnd  (xb) );
            PetscCall( MatMult(petscBdrMat, xb, bb) );

            gsVector<real_t> b_par_bdr_permuted;
            petsc_copyVecToGismo(bb, b_par_bdr_permuted, PETSC_COMM_WORLD);
            gsVector<real_t> b_par_bdr(nDofs);
            for (index_t g = 0; g < nDofs; ++g)
                b_par_bdr(g) = b_par_bdr_permuted(perm(g));
            const gsVector<real_t> b_ser_bdr = Kbdr_ser * probe;

            const real_t errBdr    = (b_par_bdr - b_ser_bdr).norm();
            const real_t refBdr    = b_ser_bdr.norm();
            const real_t relerrBdr = (refBdr > 0) ? errBdr / refBdr : errBdr;

            if (rank == 0)
            {
                gsInfo << "[bdr] ||K_par*p - K_ser*p|| / ||K_ser*p|| = " << relerrBdr << "\n";
                gsInfo << (relerrBdr < 1e-10 ? "PASSED (boundary restriction)\n"
                                              : "FAILED (boundary restriction, tol 1e-10)\n");
            }
            if (relerrBdr >= 1e-10) exitCode = 1;

            PetscCall( VecDestroy(&xb) );
            PetscCall( VecDestroy(&bb) );
            PetscCall( MatDestroy(&petscBdrMat) );
        }

        // -------------------------------------------------------------------
        // Extension (Phase 1.1 / review #2): setIntegrationDomain() must
        // invalidate the sparsity pattern. Switch subdomain and assemble()
        // repeatedly into ONE accumulated matrix without calling
        // initSystem() in between (unlike the main partition loop above,
        // which calls initSystem() every time and so never exercises this
        // path). Without the m_sparsity=0 fix, only the first assemble()
        // call recomputes the sparsity pattern; later calls silently skip
        // _computePattern and can race/drop entries not already present.
        // Purely local (no MPI/PETSc): every rank repeats the same
        // single-process check redundantly, same replication convention as
        // the rest of this example.
        // -------------------------------------------------------------------
        if (nparts >= 2)
        {
            A.setIntegrationDomain(subdomains[0]);
            A.initSystem(); // pattern computed once, for subdomains[0]
            A.assemble(igrad(u, G) * igrad(u, G).tr() * meas(G));
            for (index_t k = 1; k < nparts; ++k)
            {
                A.setIntegrationDomain(subdomains[k]); // no initSystem() in between
                A.assemble(igrad(u, G) * igrad(u, G).tr() * meas(G));
            }
            const gsSparseMatrix<real_t> K_noreset = A.matrix();

            A.setIntegrationElements(mb);
            A.initSystem();
            A.assemble(igrad(u, G) * igrad(u, G).tr() * meas(G));
            const gsSparseMatrix<real_t>& K_ser2 = A.matrix();

            const real_t refNoReset    = K_ser2.norm();
            const real_t errNoReset    = (K_noreset - K_ser2).norm();
            const real_t relerrNoReset = (refNoReset > 0) ? errNoReset / refNoReset : errNoReset;

            if (rank == 0)
            {
                gsInfo << "[sparsity] ||K_noreset - K_ser|| / ||K_ser|| = " << relerrNoReset << "\n";
                gsInfo << (relerrNoReset < 1e-10 ? "PASSED (sparsity invalidation)\n"
                                                  : "FAILED (sparsity invalidation, tol 1e-10)\n");
            }
            if (relerrNoReset >= 1e-10) exitCode = 1;
        }
    }
    else if (rank == 0)
    {
        gsInfo << "Verification skipped (--noverify); parallel assembly completed.\n";
    }

    reportRSS("end of run", rank);

    PetscCall( MatDestroy(&petscMat) );
    PetscCall( VecDestroy(&petscRhs) );
    PetscCall( VecDestroy(&petscSol) );
    PetscFinalize();
    return exitCode;
}
