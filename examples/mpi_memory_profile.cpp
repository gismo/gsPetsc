/** @file mpi_memory_profile.cpp

    @brief Toy Poisson problem assembled element-partitioned over MPI ranks,
    with per-rank memory accounting of every G+Smo / PETSc object involved.

    Each rank
      1. builds the full multipatch geometry and multibasis (replicated),
      2. assembles only its own block of elements with gsExprAssembler
         (via gsElementRangeDomain),
      3. sends its contributions to a distributed PETSc matrix:
         - without --sink: gismo fiber matrix -> PETSc COO;
         - with --sink: computePattern_into into gsPetscPatternSink (exact
           AIJ preallocation), then assemble_into into gsPetscSystemSink
           (MatSetValues),
         optionally after localizing the dof numbering (--local),
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

    Options (defaults in brackets):
      -d, --dim D     spatial dimension, 2 or 3 [2]
      -p, --degree P  spline degree [2]
      -r, --refine R  uniform h-refinement steps [5]
      -s, --split S   patch grid of 2^S patches per direction [1]
      -a, --aspect A  multiply the patches in the last direction by A
               (weak scaling) [1]
      --partition block|rcb|hilbert|morton  element partition [block]; the
               geometric ones use gsGeometricPartitioner and the dof
               ownership of gsPartitionedDofMapper (PETSc rows aligned with
               the elements)
      --geo    refine the geometry together with the basis
      --legacy also measure the conversion path of PETScSupport.h (global
               gsSparseMatrix + RowMajor copy); excludes --sink and --local
      --noreserve  do not reserve fiber storage in initSystem
      --lazy   gsExprAssembler option lazyMatrix (fibers allocated on first use)
      --local  rank-local dof numbering (gsDofMapper::localize): local
               matrix, rhs and solution, no global vector on any rank
      --rendezvous  (needs --local) dof ownership and rows by a distributed
               rendezvous: lowest touching rank owns a dof, no global
               tables (gsPartitionedDofMapper is not built)
      --sink   assemble directly into PETSc (no gismo-side matrix or rhs):
               computePattern_into with gsPetscPatternSink (exact AIJ
               preallocation), assemble_into with gsPetscSystemSink
               (MatSetValues)
      --sparse-mapper  sparse gsDofMapper storage for the space
               (gsFeSpace::setMapperStorage). The nBytes() of the mapper is
               reported at three points: "setup" is a separate, unfinalized
               mapper built by the same createMapper call after the ledger;
               "finalized" and "localized" are the space's own mapper right
               after u.setup and after localization
      --check-mapper  (needs --sparse-mapper) compare the mapper query by
               query with a dense twin built by the same calls, and again
               after an extra localization; aborts on the first difference
      --serial-partition  (needs a geometric --partition) every rank runs the
               whole pass A of gsGeometricPartitioner (serial constructor);
               by default pass A is split over the ranks (gsMpiComm
               constructor). The variant gets the suffix "+serialpart"
      --check-partition  (needs a geometric --partition) every rank also builds
               a serial partitioner with the same options and compares labels.
               The parallel labels must be identical on all ranks, otherwise
               all ranks abort together. Differences from the serial labels
               are printed on rank 0 and are not fatal
      --csv    print CSV lines (prefix CSV,) in addition to the table
      -o, --petsc OPTS  PETSc options
               ["-ksp_type cg -pc_type gamg -ksp_rtol 1e-10"]

    Ledger stages on the --sink path:
    "localize mapper (local -> global rows)" (with --local),
    "computePattern_into (pattern sink)",
    "PETSc Mat/Vec (preallocated)", "assemble_into (MatSetValues, stash)" and
    "PETSc assembly (communication)". With a geometric partition the
    partition is three stages "element partition (<strategy>): construct
    (element domain)", "...: partition() (centroids + labels)" and
    "...: subdomainForRank"; the solve is "KSP setup (incl. preconditioner)"
    and "KSP solve". With --check-partition an extra stage "partition check
    (not part of the run)" follows the partition stages, and "peak RSS (whole
    run)" includes its transient.

    Every stage reports the heap it still holds at its end, VmRSS at its end
    and VmHWM over the stage (transients included). The objects at the top
    are the start-up baseline: VmRSS after MPI_Init, then VmRSS with its
    RssAnon / RssFile / RssShmem parts after PetscInitialize. The counts at
    the bottom (local elements, owned rows, local+ghost dofs, nonzeros and
    off-rank entries per rank) measure load balance and partition quality.

    PETSc's own profile: -log_view and -memory_view must be set before
    PetscInitialize, i.e. through the environment, e.g.
    PETSC_OPTIONS="-log_view :petsc_log.txt -memory_view"; options given
    with -o are inserted after initialization.

    Example run:
    mpirun -np 4 ./bin/mpi_memory_profile -r 7 -p 2 -s 1 --partition rcb --sink
*/

#include <gismo.h>
#include <gsPetsc/partitioned/gsElementRangeDomain.h>
#include <gsPetsc/partitioned/gsMemoryProbe.h>
#include <gsPetsc/partitioned/gsPetscCOO.h>
#include <gsPetsc/partitioned/gsPetscSink.h>
#include <gsPetsc/partitioned/gsLocalDofs.h>
#include <gsPetsc/partitioned/gsRendezvousNumbering.h>
#include <petscksp.h>

using namespace gismo;

int main(int argc, char *argv[])
{
    index_t dim = 2, degree = 2, numRefine = 5, numSplit = 1, aspect = 1;
    bool refineGeometry = false, legacy = false, csv = false, noReserve = false;
    bool lazy = false, sink = false, localNumbering = false, rendezvous = false;
    bool sparseMapper = false, checkMapper = false;
    bool serialPartition = false, checkPartition = false;
    std::string partition("block");
    std::string petscOpts("-ksp_type cg -pc_type gamg -ksp_rtol 1e-10");

    gsCmdLine cmd("Memory profile of element-partitioned assembly with gsExprAssembler + PETSc.");
    cmd.addInt   ("d", "dim",     "Spatial dimension (2 or 3)", dim);
    cmd.addInt   ("p", "degree",  "Spline degree", degree);
    cmd.addInt   ("r", "refine",  "Uniform h-refinement steps", numRefine);
    cmd.addInt   ("s", "split",   "Patch grid: 2^s patches per direction", numSplit);
    cmd.addInt   ("a", "aspect",  "Multiply the patches in the last direction by this (weak scaling)", aspect);
    cmd.addString("", "partition", "Element partition: block, rcb, hilbert, morton", partition);
    cmd.addSwitch("geo", "Refine the geometry together with the basis (fine CAD / isoparametric geometry)", refineGeometry);
    cmd.addSwitch("legacy", "Also measure the conversion path of PETScSupport.h (global gsSparseMatrix + RowMajor copy)", legacy);
    cmd.addSwitch("noreserve", "Do not reserve fiber storage in initSystem (bdA=bdB=bdO=0)", noReserve);
    cmd.addSwitch("lazy", "Allocate fibers on first use (option lazyMatrix)", lazy);
    cmd.addSwitch("local", "Rank-local dof numbering (localized mapper)", localNumbering);
    cmd.addSwitch("rendezvous", "Distributed dof ownership/rows (needs --local)", rendezvous);
    cmd.addSwitch("sink", "Assemble directly into PETSc (no gismo matrix/rhs)", sink);
    cmd.addSwitch("sparse-mapper", "Opt-in sparse gsDofMapper storage (gsFeSpace::setMapperStorage)", sparseMapper);
    cmd.addSwitch("check-mapper", "With --sparse-mapper: after the run, compare the mapper with a dense twin built by the same calls (aborts on the first difference)", checkMapper);
    cmd.addSwitch("serial-partition", "Geometric partition: every rank runs the whole pass A (serial gsGeometricPartitioner constructor) instead of splitting it over the ranks", serialPartition);
    cmd.addSwitch("check-partition", "Geometric partition: also build a serial partitioner on every rank and compare the labels (fatal if they differ across ranks, reported if they differ from serial)", checkPartition);
    cmd.addSwitch("csv", "Print CSV lines (prefix CSV,) in addition to the table", csv);
    cmd.addString("o", "petsc", "PETSc options", petscOpts);
    try { cmd.getValues(argc,argv); } catch (int rv) { return rv; }
    GISMO_ENSURE(!(sink && legacy), "--legacy needs the gismo-side matrix, it cannot be combined with --sink");
    GISMO_ENSURE(!rendezvous || localNumbering, "--rendezvous needs --local");
    GISMO_ENSURE(!checkMapper || sparseMapper, "--check-mapper needs --sparse-mapper");
    GISMO_ENSURE(!checkPartition  || "block" != partition, "--check-partition needs a geometric partition (rcb, hilbert or morton)");
    GISMO_ENSURE(!serialPartition || "block" != partition, "--serial-partition needs a geometric partition (rcb, hilbert or morton)");
    GISMO_ENSURE(!(localNumbering && legacy), "--legacy needs the global numbering, it cannot be combined with --local");

    const gsMpi & mpi = gsMpi::init(argc, argv);
    gsMpiComm comm = mpi.worldComm();
    const int rank = comm.rank(), nproc = comm.size();
    const long long rssAfterMpi = memprobe::rssBytes();
    PetscCall( PetscInitializeNoArguments() );
    PetscCall( PetscOptionsInsertString(NULL, petscOpts.c_str()) );

    memprobe::Ledger L(comm);
    L.object("RSS after MPI init (before PETSc)", rssAfterMpi);
    gsStopwatch clock;

    // ------------------------------------------------------------------
    // 1. Geometry and discretization (built identically on every rank)
    // ------------------------------------------------------------------
    const int np = 1 << numSplit;
    // unit patches; the exact solution vanishes on the boundary of any integer box
    gsMultiPatch<> mp = (2 == dim) ? gsNurbsCreator<>::BSplineSquareGrid(np, np * aspect, 1.0)
                                   : gsNurbsCreator<>::BSplineCubeGrid(np, np, np * aspect, 1.0);
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
    // 2. Spaces, element partition and dof ownership
    // ------------------------------------------------------------------
    gsExprAssembler<> A(1,1);
    if (noReserve)
    {
        A.options().setReal("bdA", 0);
        A.options().setInt ("bdB", 0);
        A.options().setReal("bdO", 0);
    }
    if (lazy) A.options().setSwitch("lazyMatrix", true);
    gsExprEvaluator<> ev(A);
    gsExprAssembler<>::geometryMap G = A.getMap(mp);
    gsExprAssembler<>::space u = A.getSpace(mb);
    auto ff = A.getCoeff(f, G);
    auto u_ex = ev.getVariable(ms, G);

    if (sparseMapper) u.setMapperStorage(gsDofMapper::storage::sparse);
    u.setup(bc, dirichlet::interpolation, 0);
    const index_t N = u.mapper().freeSize();
    L.stage("space setup (gsDofMapper + Dirichlet)");
    L.object("  gsDofMapper", memprobe::bytesOf(u.mapper()));
    L.object("  fixed (Dirichlet) dofs", memprobe::bytesOf(u.fixedPart()));
    const long long mapperBytesFinal = static_cast<long long>(u.mapper().nBytes());

    const index_t numElemGlobal = mb.domain()->numElements();
    gsDomain<>::Ptr myDomain;
    gsVector<index_t> perm;              // global dof -> PETSc row (empty: identity)
    PetscInt nLocal = PETSC_DECIDE;      // PETSc rows owned by this rank
    if ("block" == partition)
    {
        myDomain = gsElementRangeDomain<real_t>::blockPartition(mb.domain(), rank, nproc);
        L.stage("element partition (block)");
    }
    else
    {
        gsGeometricPartitioner<real_t>::Options popt;
        popt.strategy = gsGeometricPartitioner<real_t>::strategyFromString(partition);
        // Pass A is split over the ranks unless --serial-partition. The three
        // sub-stages share the prefix "element partition (<strategy>)"; their
        // sum is the partition cost.
        const std::string pre = "element partition (" + partition + "): ";
        std::unique_ptr<gsGeometricPartitioner<real_t> > partPtr(serialPartition
            ? new gsGeometricPartitioner<real_t>(mp, mb, u.mapper(), nproc, popt)
            : new gsGeometricPartitioner<real_t>(mp, mb, u.mapper(), nproc, comm, popt));
        gsGeometricPartitioner<real_t> & part = *partPtr;
        L.stage(pre + "construct (element domain)");
        part.partition();
        L.stage(pre + "partition() (centroids + labels)");
        myDomain = part.subdomainForRank(rank, nproc);
        L.stage(pre + "subdomainForRank");
        if (checkPartition)
        {
            // Cost: one extra serial partition (pass A O(N) plus labelling,
            // RCB O(N log P), curves O(N log N)) and O(N) hashing/comparison
            // for N elements. Run in its own stage so that none of it is
            // charged to the partition stages.
            long long diff = 0;
            const long long nElem = static_cast<long long>(part.labels().size());
            {
                gsGeometricPartitioner<real_t> ser(mp, mb, u.mapper(), nproc, popt);
                ser.partition();

                // FNV-1a over the labels in element order (order-sensitive)
                std::uint64_t h = 14695981039346656037ull;
                for (index_t l : part.labels())
                {
                    h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(l));
                    h *= 1099511628211ull;
                }
                // The verdict is evaluated on every rank after the reductions,
                // so all ranks throw together or none does.
                const bool sizeMismatch = (ser.labels().size() != part.labels().size());
                std::uint64_t mn[3] = { h, static_cast<std::uint64_t>(part.labels().size()),
                                        static_cast<std::uint64_t>(sizeMismatch) };
                std::uint64_t mx[3] = { mn[0], mn[1], mn[2] };
                MPI_Allreduce(MPI_IN_PLACE, mn, 3, MPI_UINT64_T, MPI_MIN, comm);
                MPI_Allreduce(MPI_IN_PLACE, mx, 3, MPI_UINT64_T, MPI_MAX, comm);
                GISMO_ENSURE(mn[0] == mx[0] && mn[1] == mx[1],
                             "partition check: the parallel labels differ across ranks (hash/count min != max)");
                GISMO_ENSURE(0 == mx[2],
                             "partition check: the serial and parallel partitions have different element counts");

                for (std::size_t e = 0; e < part.labels().size(); ++e)
                    if (part.labels()[e] != ser.labels()[e])
                        ++diff;
                MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_LONG_LONG, MPI_MAX, comm);
            }
            if (0 == rank)
            {
                if (0 == diff)
                    gsInfo << "partition check: labels identical across ranks; parallel vs serial: identical\n";
                else
                    gsInfo << "partition check: labels identical across ranks; parallel vs serial: "
                           << diff << " of " << nElem << " elements differ\n";
            }
            L.stage("partition check (not part of the run)");
        }
        if (!rendezvous)
        {
            gsPartitionedDofMapper pdm = part.makeDofMapper(nproc);
            L.stage("gsPartitionedDofMapper (incl. temporaries)");
            L.object("  gsPartitionedDofMapper", pdm.nBytes());
            perm = pdm.permutation();
            nLocal = pdm.numOwnedDofs(rank);
        }
    }
    // the partitioner (element labels, weights) and the full ownership
    // tables are released here, only the permutation is kept
    L.stage("partitioner released, permutation kept");
    const gsVector<index_t> * permPtr = perm.size() ? &perm : nullptr;
    gsVector<index_t> rowOf;             // local dof -> PETSc row (local numbering)
    if (localNumbering)
    {
        if (rendezvous)
        {
            const std::vector<index_t> l2g = localFreeDofs(*myDomain, mb, u.mapper());
            std::vector<index_t> rows, l2gFree(l2g);
            for (index_t & g : l2gFree) g -= u.mapper().firstIndex();
            nLocal = rendezvousNumbering(comm, N, l2gFree, rows);
            rowOf = gsAsConstVector<index_t>(rows);
            localizeSpace(u, l2g);
        }
        else
            rowOf = localizeSpace(u, *myDomain, permPtr);
        perm.resize(0);                  // the global permutation is not needed anymore
        permPtr = &rowOf;
        L.stage("localize mapper (local -> global rows)");
        L.object("  local-to-global row map", rowOf.size() * sizeof(index_t));
    }
    const long long mapperBytesLocal = static_cast<long long>(u.mapper().nBytes());
    A.setIntegrationDomain(myDomain);

    // ------------------------------------------------------------------
    // 3. Assembly, transfer to PETSc and solve
    // ------------------------------------------------------------------
    Mat PA; Vec Pb, Px; KSP ksp;
    PetscCount nOff = 0;
    long long localNnz = 0;
    long long localSlots = 0, localMallocs = 0;
    double tAssemble = 0;
    if (sink)
    {
        gsPetscPatternSink pattern(comm, N, nLocal, permPtr);
        A.computePattern_into(pattern, igrad(u, G) * igrad(u, G).tr());
        L.stage("computePattern_into (pattern sink)");
        L.object("  pattern sink stored blocks", static_cast<long long>(pattern.storedBytes()));
        PetscCall( pattern.createMatrix(PA) );
        MatInfo pinfo;
        PetscCall( MatGetInfo(PA, MAT_LOCAL, &pinfo) );
        localSlots = static_cast<long long>(pinfo.nz_allocated);
        PetscCall( MatCreateVecs(PA, &Px, &Pb) );
        PetscCall( VecSet(Pb, 0.0) );
        L.stage("PETSc Mat/Vec (preallocated)");

        gsPetscSystemSink system(PA, Pb, permPtr);
        A.assemble_into(system, igrad(u, G) * igrad(u, G).tr() * meas(G), u * ff * meas(G));
        nOff = system.offRankEntries();
        L.stage("assemble_into (MatSetValues, stash)");
        PetscCall( system.assembly() );
        L.stage("PETSc assembly (communication)");
        MatInfo info;
        PetscCall( MatGetInfo(PA, MAT_LOCAL, &info) );
        localNnz = static_cast<long long>(info.nz_used);
        localMallocs = static_cast<long long>(info.mallocs);
        tAssemble = clock.stop();
    }
    else
    {
        A.initSystem();
        L.stage("initSystem (fiber matrix + rhs)");
        L.object("  gsFiberMatrix after initSystem", memprobe::bytesOf(A.fiberMatrix()));
        L.object("  rhs vector", memprobe::bytesOf(A.rhs()));

        A.computePattern(igrad(u, G) * igrad(u, G).tr());
        L.stage("computePattern (local elements)");

        A.assemble(igrad(u, G) * igrad(u, G).tr() * meas(G), u * ff * meas(G));
        L.stage("assemble (local elements)");
        L.object("  gsFiberMatrix after assembly", memprobe::bytesOf(A.fiberMatrix()));
        localNnz = A.fiberMatrix().nonZeros();
        tAssemble = clock.stop();

        PetscCall( petsc_matFromLocalFibers(A.fiberMatrix(), comm, PA, &nOff, permPtr, nLocal, N) );
        PetscCall( petsc_vecFromLocalContributions(A.rhs(), PA, Pb, permPtr) );
        PetscCall( MatCreateVecs(PA, &Px, nullptr) );
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
    }

    clock.restart();
    PetscCall( KSPCreate(comm, &ksp) );
    PetscCall( KSPSetOperators(ksp, PA, PA) );
    PetscCall( KSPSetFromOptions(ksp) );
    PetscCall( KSPSetUp(ksp) );
    L.stage("KSP setup (incl. preconditioner)");
    PetscCall( KSPSolve(ksp, Pb, Px) );
    PetscInt its;
    PetscCall( KSPGetIterationNumber(ksp, &its) );
    L.stage("KSP solve");
    const double tSolve = clock.stop();

    // ------------------------------------------------------------------
    // 4. Back to G+Smo
    // ------------------------------------------------------------------
    clock.restart();
    // Dofs actually needed by the local elements ("owned + ghost")
    std::vector<PetscInt> needed;
    gsMatrix<> solVector;
    if (localNumbering)
    {
        // the local numbering already is the owned + ghost set
        needed.assign(rowOf.data(), rowOf.data() + rowOf.size());
        PetscCall( petsc_gatherEntries(Px, needed, solVector) );
        L.stage("gather local+ghost solution entries");
        L.object("  local+ghost solution (what is needed)", memprobe::bytesOf(solVector) + needed.size() * sizeof(PetscInt));
    }
    else
    {
        gsMatrix<index_t> act;
        for (auto it = myDomain->beginAll(); it != myDomain->endAll(); ++it)
        {
            const index_t p = it.patchIndex();
            mb.basis(p).active_into(it.centerPoint(), act);
            for (index_t i = 0; i != act.rows(); ++i)
            {
                const index_t ii = u.mapper().index(act(i), p);
                if (u.mapper().is_free_index(ii))
                    needed.push_back(permPtr ? perm[ii] : ii);
            }
        }
        std::sort(needed.begin(), needed.end());
        needed.erase(std::unique(needed.begin(), needed.end()), needed.end());
        needed.shrink_to_fit();
        gsMatrix<> ghosted;
        PetscCall( petsc_gatherEntries(Px, needed, ghosted) );
        L.stage("gather local+ghost solution entries");
        L.object("  local+ghost solution (what is needed)", memprobe::bytesOf(ghosted) + needed.size() * sizeof(PetscInt));

        PetscCall( petsc_gatherAll(Px, solVector) );
        if (permPtr)
        {
            gsMatrix<> tmp(N, 1);
            for (index_t g = 0; g != N; ++g) tmp(g) = solVector(perm[g]);
            solVector.swap(tmp);
        }
        L.stage("gather full solution (gsFeSolution)");
        L.object("  global solution vector", memprobe::bytesOf(solVector));
    }

    gsExprAssembler<>::solution u_sol = A.getSolution(u, solVector);
    if (!localNumbering) // extract() visits all patches, i.e. needs the global vector
    {
        gsMultiPatch<> mpSol;
        u_sol.extract(mpSol);
        L.stage("solution as gsMultiPatch (extract)");
        L.object("  solution gsMultiPatch", memprobe::bytesOf(mpSol));
        mpSol.clear();
        L.stage("free solution gsMultiPatch");
    }

    real_t l2loc = ev.integral((u_ex - u_sol).sqNorm() * meas(G)), l2 = 0;
    MPI_Allreduce(&l2loc, &l2, 1, MPI_DOUBLE, MPI_SUM, comm);
    l2 = math::sqrt(l2);
    L.stage("error evaluation (local elements)");
    const double tPost = clock.stop();

    // Per-rank load and partition quality; max/sum = 1/P for a perfect balance.
    PetscInt rowLo, rowHi;
    PetscCall( MatGetOwnershipRange(PA, &rowLo, &rowHi) );
    L.count("local elements", static_cast<long long>(myDomain->numElements()));
    L.count("owned PETSc rows", static_cast<long long>(rowHi - rowLo));
    L.count("local+ghost dofs", static_cast<long long>(needed.size()));
    L.count("matrix nonzeros in owned rows", localNnz);
    L.count("entries in rows of another rank", static_cast<long long>(nOff));

    long long offSum = nOff, nnzSum = localNnz, ghostMax = needed.size();
    MPI_Allreduce(MPI_IN_PLACE, &offSum, 1, MPI_LONG_LONG, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, &nnzSum, 1, MPI_LONG_LONG, MPI_SUM, comm);
    long long slotSum = localSlots, mallocMax = localMallocs;
    MPI_Allreduce(MPI_IN_PLACE, &slotSum, 1, MPI_LONG_LONG, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, &mallocMax, 1, MPI_LONG_LONG, MPI_MAX, comm);
    MPI_Allreduce(MPI_IN_PLACE, &ghostMax, 1, MPI_LONG_LONG, MPI_MAX, comm);

    std::string variant = partition;
    if (serialPartition) variant += "+serialpart";
    if (noReserve) variant += "+noreserve";
    if (lazy)      variant += "+lazy";
    if (localNumbering) variant += "+local";
    if (rendezvous)     variant += "+rv";
    if (sink)      variant += "+sink";
    if (sparseMapper) variant += "+sparsemap";

    if (0 == rank)
    {
        gsInfo << "ranks " << nproc << ", dim " << dim << ", degree " << degree
               << ", patches " << mp.nPatches() << ", elements " << numElemGlobal
               << ", dofs " << N << (refineGeometry ? " (refined geometry)" : "")
               << ", variant " << variant << "\n"
               << "KSP iterations " << its << ", L2 error " << l2 << "\n"
               << (sink ? "matrix nonzeros (sum over ranks) " : "assembled entries (sum over ranks) ") << nnzSum
               << ", " << (sink ? "element-block entries" : "entries") << " in rows owned by another rank " << offSum << "\n"
               << "max local+ghost dofs per rank " << ghostMax << " (" << 100.*ghostMax/N << "% of N)\n"
               << "time: assemble " << tAssemble << "s, solve " << tSolve
               << "s, post " << tPost << "s\n";
        if (sink)
            gsInfo << "preallocated slots (sum over ranks) " << slotSum
                   << ", mallocs (max over ranks) " << mallocMax << "\n";
    }
    std::ostringstream tag;
    tag << "P=" << nproc << " d=" << dim << " p=" << degree << " N=" << N
        << " patches=" << mp.nPatches() << (refineGeometry ? " geo" : "") << " " << variant;
    L.report(gsInfo, tag.str());
    if (csv)
    {
        std::ostringstream ctag;
        ctag << "CSV," << nproc << "," << dim << "," << degree << "," << N << ","
             << mp.nPatches() << "," << refineGeometry << "," << noReserve << ",v=" << variant;
        L.csv(gsInfo, ctag.str());
    }

    // Mapper memory report and dense-twin check; no ledger stage follows.
    const gsDofMapper::storage mapperSt = u.mapperStorage();
    {
        const gsDofMapper probe = createMapper(mb, bc, 1, u.id(), /*conforming=*/true,
                                               /*finalize=*/false, mapperSt);
        long long bytes[3] = { static_cast<long long>(probe.nBytes()),
                               mapperBytesFinal, mapperBytesLocal };
        MPI_Allreduce(MPI_IN_PLACE, bytes, 3, MPI_LONG_LONG, MPI_MAX, comm);
        if (0 == rank)
        {
            gsInfo << "gsDofMapper nBytes (max over ranks, storage "
                   << (gsDofMapper::storage::sparse == mapperSt ? "sparse" : "dense")
                   << "): setup " << bytes[0] << ", finalized " << bytes[1]
                   << ", localized ";
            if (localNumbering) gsInfo << bytes[2]; else gsInfo << "-";
            gsInfo << ", dense table mapSize()*sizeof(index_t) "
                   << u.mapper().mapSize() * sizeof(index_t) << "\n";
        }
    }

    if (checkMapper)
    {
        // Compares every query of the mapper \a s with the dense twin \a d.
        const auto compare = [&](const gsDofMapper & d, const gsDofMapper & s) -> size_t
        {
#define MAPPER_CHECK(cond, what) \
            GISMO_ENSURE(cond, "mapper check, rank " << rank << ": " << what << " differs from the dense twin")
            MAPPER_CHECK(gsDofMapper::storage::dense  == d.storageMode(), "storageMode (twin)");
            MAPPER_CHECK(gsDofMapper::storage::sparse == s.storageMode(), "storageMode");
            MAPPER_CHECK(d.firstIndex() == s.firstIndex(), "firstIndex");
            MAPPER_CHECK(d.freeSize() == s.freeSize(), "freeSize");
            MAPPER_CHECK(d.size() == s.size(), "size");
            MAPPER_CHECK(d.boundarySize() == s.boundarySize(), "boundarySize");
            MAPPER_CHECK(d.coupledSize() == s.coupledSize(), "coupledSize");
            MAPPER_CHECK(d.boundarySizeWithDuplicates() == s.boundarySizeWithDuplicates(),
                         "boundarySizeWithDuplicates");
            MAPPER_CHECK(d.getTagged() == s.getTagged(), "getTagged");
            const auto same = [&](const gsVector<index_t> & x, const gsVector<index_t> & y)
            { return x.size() == y.size() && (0 == x.size() || x == y); };
            size_t positions = 0;
            for (size_t k = 0; k != d.numPatches(); ++k)
            {
                const index_t kk = static_cast<index_t>(k);
                MAPPER_CHECK(d.patchSize(kk) == s.patchSize(kk), "patchSize of patch " << k);
                for (index_t i = 0; i != static_cast<index_t>(d.patchSize(kk)); ++i)
                    MAPPER_CHECK(d.index(i, kk) == s.index(i, kk), "index(" << i << "," << k << ")");
                positions += d.patchSize(kk);
                MAPPER_CHECK(same(d.findBoundary(kk), s.findBoundary(kk)), "findBoundary of patch " << k);
                MAPPER_CHECK(same(d.findFree(kk), s.findFree(kk)), "findFree of patch " << k);
                MAPPER_CHECK(same(d.findFreeUncoupled(kk), s.findFreeUncoupled(kk)),
                             "findFreeUncoupled of patch " << k);
                MAPPER_CHECK(same(d.findCoupled(kk, -1), s.findCoupled(kk, -1)),
                             "findCoupled(" << k << ",-1)");
                for (size_t j = 0; j != d.numPatches(); ++j)
                    MAPPER_CHECK(same(d.findCoupled(kk, static_cast<index_t>(j)),
                                      s.findCoupled(kk, static_cast<index_t>(j))),
                                 "findCoupled(" << k << "," << j << ")");
            }
#undef MAPPER_CHECK
            return positions;
        };

        gsDofMapper twin = createMapper(mb, bc, 1, u.id(), /*conforming=*/true,
                                        /*finalize=*/false, gsDofMapper::storage::dense);
        twin.finalize();
        if (localNumbering)
            twin.localize(localFreeDofs(*myDomain, mb, twin));
        const size_t positions = compare(twin, u.mapper());

        // every other current free dof: localizes a finalized mapper without
        // --local and re-localizes with it
        gsDofMapper a = twin, b = u.mapper();
        std::vector<index_t> l2;
        for (index_t g = a.firstIndex(); g < a.firstIndex() + a.freeSize(); g += 2)
            l2.push_back(g);
        a.localize(l2);
        b.localize(l2);
        compare(a, b);

        if (0 == rank)
            gsInfo << "mapper check vs dense twin: identical (" << positions
                   << " positions on rank 0, " << (localNumbering ? 2 : 1)
                   << (localNumbering ? " localizations)\n" : " localization)\n");
    }

    PetscCall( KSPDestroy(&ksp) );
    PetscCall( VecDestroy(&Px) );
    PetscCall( VecDestroy(&Pb) );
    PetscCall( MatDestroy(&PA) );
    PetscCall( PetscFinalize() );
    return EXIT_SUCCESS;
}
