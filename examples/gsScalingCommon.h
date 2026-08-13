/** @file gsScalingCommon.h

    @brief Shared infrastructure for the METIS+PETSc strong/weak scaling drivers.

    This header is example-scope only (it lives in optional/gsPetsc/examples,
    not in src/, so it is not installed and not part of the gsPetsc module).
    It collects everything the Poisson and linear-elasticity scaling drivers
    have in common, so that each driver file contains only its own PDE:

      - ScalingOptions / registerCommonOptions(): the command-line surface
        shared by both drivers (geometry, refinement, partitioning, solver,
        output).
      - makeBoxGrid() / buildGeometry(): a unit-box tiled by an npx x npy
        (x npz) grid of degree-1 patches, or any multipatch read from XML and
        optionally uniformSplit().
      - PhaseTimer: barrier-synchronised, named wall-clock phases, reduced
        across ranks into (max, min) pairs. max = critical path, max/min =
        load imbalance.
      - Record: an ordered key/value row, pretty-printed to stdout AND
        appended to a CSV file (header written only when the file is new), so
        a SLURM sweep produces one directly plottable table.
      - partitionAndBroadcast(): rank 0 runs METIS, everyone else only builds
        the element graph; labels are broadcast. (METIS here is serial and
        replicated -- see the memory caveat at the bottom of this comment.)
      - buildPartition(): the single entry point both drivers use to honour
        --partitioner (rcb | hilbert | morton | metis; rcb is the default) and
        --partWeight, returning DOF ownership + this rank's subdomain. Only
        the metis value goes through partitionAndBroadcast(); the three
        graph-free strategies build no element graph and need no broadcast,
        since every rank recomputes bit-identical labels from replicated data.
      - computeDofGeometry() / attachRigidBodyNullSpace(): physical
        coordinates + component index of every free DOF, and the resulting
        rigid-body near-null-space handed to PCGAMG. Without this, AMG on
        elasticity loses the rotational modes and its iteration count grows
        with the mesh, which would show up as (spurious) bad solver scaling.
      - solveSystem(): CG+GAMG by default, always overridable from the PETSc
        command line (KSPSetFromOptions is called last), with KSPSetUp and
        KSPSolve timed separately and the TRUE relative residual
        ||b-Ax||/||b|| computed distributedly (no gather).

    MEMORY CAVEAT (matters when choosing weak-scaling sizes): in this
    architecture the geometry, basis and DOF mapper are REPLICATED on every
    rank, whatever --partitioner is set to; only the PETSc Mat/Vec and the
    per-rank share of the assembled entries are distributed. Measured
    consequences:

      * STRONG scaling (fixed problem, more ranks): per-rank memory falls, but
        only sublinearly -- 517 -> 289 MB over 4x the ranks -- because the
        replicated part does not move. The SMALLEST rank count is therefore
        the memory-binding case of a strong-scaling sweep.
      * WEAK scaling (DoFs/rank fixed, more ranks): per-rank memory GROWS --
        80 -> 177 MB over 8x the ranks -- because the replicated structures
        track the GLOBAL problem. This, not communication, is the wall.

    On the --partitioner metis path, and only there, one further structure is
    replicated: the element dual graph, in which two elements are adjacent
    whenever their supports overlap -- ~(2p+1)^3 - 1 = 124 neighbours per
    element for a 3D degree-2 basis -- built on EVERY rank (rank 0 partitions,
    every other rank calls buildGraph()) together with its labelling. Measured
    against a graph-free control at two sizes 8x apart: 171-181 B/element at
    peak, plus ~37 B/element that is still present in the post-assembly RSS
    sample of the p=1 control comparison at 2,097,152 elements (the
    per-element DOF CSR that makeDofMapper() needs). rcb (the default),
    hilbert and morton build no element graph and never pay it; they do still
    allocate O(nelems) per rank while partitioning -- one index_t label per
    element, plus element centroids and weights. These figures were measured
    on a RelWithDebInfo build without -DNDEBUG and are an upper bound for a
    Release cluster build.

    Not paying for the graph is a real, rank-count-independent saving; it is
    NOT an explanation of an out-of-memory job. What dominates the remaining
    per-rank requirement was not characterised, so --mem-per-cpu must still be
    calibrated from a real run rather than derived from this comment.

    The drivers report rss_hwm_mb (VmHWM, the true peak, covering the AMG
    hierarchy and the transient COO arrays) precisely so a batch scheduler's
    --mem-per-cpu can be sized from a real number: the instantaneous
    rss_max_mb sample is measured 1.5-1.9x LOWER. Note that rss_max_mb and
    rss_delta_mb are sampled after buildPartition() has already freed the
    partitioner, so neither can show the 171-181 B/element PEAK above at all
    (only the smaller ~37 B/element term survives into them, and only the p=1
    control comparison at 2,097,152 elements resolved it); and at the p=2 of
    the shipped sweeps the matrix phase peaks higher than the partitioning
    phase, so rss_hwm_mb does not show that peak either (isolating it took a
    p=1 run). --lazyMatrix (default ON here) keeps the assembler from
    reserving all nDofs columns up front and is the single biggest lever on
    the distributed part.

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): H.M. Verhelst
*/

#pragma once

#include <gismo.h>
#include <gsMetis/gsMetis.h>
#include <gsPetsc/PETScSupport.h>
#include <gsPetsc/gsPetscLocalToGlobal.h>

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace gismo
{
namespace scaling
{

// ===========================================================================
// Command-line surface shared by both drivers
// ===========================================================================

struct ScalingOptions
{
    // -- geometry -----------------------------------------------------------
    index_t     dim      = 2;   ///< 2 or 3 (ignored when a file is given)
    index_t     npx      = 2;   ///< patches in x
    index_t     npy      = 2;   ///< patches in y
    index_t     npz      = 2;   ///< patches in z (3D only)
    std::string geoFile;        ///< XML multipatch (id 0); overrides the box grid
    index_t     nsplit   = 0;   ///< uniformSplit() applications after loading

    // -- discretisation -----------------------------------------------------
    index_t     nref     = 3;   ///< uniform h-refinements of the basis
    index_t     degree   = 2;   ///< target spline degree (set, not elevated)
    bool        dirInterp = false; ///< interpolate Dirichlet data instead of L2-projecting

    // -- partitioning -------------------------------------------------------
    index_t     nparts   = -1;  ///< METIS partitions; <0 => partsPerRank*nranks
    index_t     partsPerRank = 1;
    bool        metisContig  = false;
    bool        metisWeightDofs = false;
    real_t      metisImbalance  = -1.0;
    std::string partitioner = "rcb";   ///< rcb | hilbert | morton | metis
    std::string partWeight  = "elements";  ///< elements | dofs (graph-free strategies only)

    // -- assembly / insertion ----------------------------------------------
    bool        eagerMatrix = false; ///< disable the (default) lazy fiber-matrix columns
    bool        noCoo       = false; ///< force the MatSetValues fallback insertion path
    index_t     threads     = -1;    ///< omp_set_num_threads (<0: leave OMP_NUM_THREADS alone)

    // -- solver -------------------------------------------------------------
    real_t      rtol     = 1e-8;
    index_t     maxIts   = 2000;
    bool        noSolve  = false; ///< assembly-only run (skip KSP entirely)
    bool        noRBM    = false; ///< skip the rigid-body near-null-space (elasticity)

    // -- output -------------------------------------------------------------
    std::string csv;            ///< append one row here (header written if new)
    std::string tag;            ///< free-form label copied into the CSV row
    bool        check   = false;///< gather the solution, compute L2/H1 errors
    bool        plot    = false;///< ParaView output (implies --check, rank 0 only)
    bool        memReport = false;///< stdout-only per-phase memory trace + byte budget
};

/// Registers every option in \a o on \a cmd. Driver-specific options
/// (material parameters, ...) are registered by the driver itself.
inline void registerCommonOptions(gsCmdLine& cmd, ScalingOptions& o)
{
    cmd.addInt("d", "dim", "Spatial dimension of the generated box grid (2 or 3)", o.dim);
    cmd.addInt("", "npx", "Patches in x of the generated box grid", o.npx);
    cmd.addInt("", "npy", "Patches in y of the generated box grid", o.npy);
    cmd.addInt("", "npz", "Patches in z of the generated box grid (3D only)", o.npz);
    cmd.addString("f", "file", "XML multipatch file (id 0). Overrides the generated box grid.", o.geoFile);
    cmd.addInt("s", "split", "Apply uniformSplit() this many times after loading/generating "
                             "(multiplies the patch count by 2^dim each time)", o.nsplit);

    cmd.addInt("r", "nref",   "Uniform h-refinement steps applied to the basis", o.nref);
    cmd.addInt("p", "degree", "Spline degree of the discretisation space", o.degree);
    cmd.addSwitch("dirichletInterp",
        "Impose Dirichlet data by interpolation instead of L2 projection. "
        "Cheaper setup (the L2 projection is a serial global solve replicated "
        "on every rank), slightly less accurate.", o.dirInterp);

    cmd.addInt("n", "nparts", "Number of METIS partitions (<0: partsPerRank * nranks)", o.nparts);
    cmd.addInt("", "partsPerRank",
        "Partitions per rank when --nparts is not given. Keep at 1 for scaling "
        "runs: partitions are assigned to ranks cyclically (part p -> rank "
        "p % nranks), so >1 gives every rank a set of unrelated, "
        "geometrically scattered partitions.", o.partsPerRank);
    cmd.addSwitch("metisContig", "METIS_OPTION_CONTIG: force contiguous partitions", o.metisContig);
    cmd.addSwitch("metisWeightDofs", "Weight graph vertices by active DOF count (DOF-balanced partitions)", o.metisWeightDofs);
    cmd.addReal("", "metisImbalance", "Allowed load imbalance above 1.0, e.g. 0.03 = 3% (<0: METIS default)", o.metisImbalance);
    cmd.addString("", "partitioner",
        "Element partitioner: rcb (default, recursive coordinate bisection), "
        "hilbert, morton (space-filling-curve orderings) or metis. The three "
        "graph-free strategies build no element adjacency graph and need no "
        "broadcast: every rank recomputes the same partition.", o.partitioner);
    cmd.addString("", "partWeight",
        "Element weight used to balance the graph-free partitioners: 'elements' "
        "(default, unit weights) or 'dofs' (free-DOF count per element). "
        "Ignored by --partitioner metis, which is controlled by "
        "--metisWeightDofs.", o.partWeight);

    cmd.addSwitch("eagerMatrix",
        "Disable the lazy fiber-matrix columns (ON by default in this driver). "
        "Eager reserves all nDofs columns per rank even though a rank only "
        "touches its own partition's columns -- much heavier, kept reachable "
        "for the eager-vs-lazy memory comparison.", o.eagerMatrix);
    cmd.addSwitch("no-coo",
        "Use the MatSetValues-per-row fallback instead of PETSc's COO "
        "insertion API (MatSetPreallocationCOO, PETSc >= 3.18).", o.noCoo);
    cmd.addInt("t", "threads", "OpenMP threads per rank (<0: leave OMP_NUM_THREADS as-is)", o.threads);

    cmd.addReal("", "rtol", "KSP relative tolerance", o.rtol);
    cmd.addInt("", "maxits", "KSP maximum iterations", o.maxIts);
    cmd.addSwitch("nosolve", "Assembly-only run: skip the linear solve entirely", o.noSolve);
    cmd.addSwitch("noRBM",
        "Do not attach the rigid-body near-null-space to the matrix "
        "(elasticity only; without it AMG iteration counts grow with the mesh)", o.noRBM);

    cmd.addString("", "csv", "Append one result row to this CSV file (header written if new)", o.csv);
    cmd.addString("", "tag", "Free-form label copied into the CSV row (e.g. 'strong-2d')", o.tag);
    cmd.addSwitch("check",
        "Gather the global solution and report L2/H1 errors against the "
        "manufactured solution. The gather is O(nDofs) on EVERY rank and does "
        "not scale -- verification runs only.", o.check);
    cmd.addSwitch("plot", "ParaView output on rank 0 (implies --check)", o.plot);
    cmd.addSwitch("memReport",
        "Print a per-phase VmHWM/RSS trace and an exact byte budget of every named "
        "large structure to stdout (rank 0). Diagnostic only: nothing is written to "
        "--csv and no run without this flag is affected.", o.memReport);
}

// ===========================================================================
// Geometry
// ===========================================================================

/// @brief The unit box [0,1]^dim tiled by an \a nx x \a ny (x \a nz) grid of
/// degree-1 tensor B-spline patches, with topology computed.
///
/// The domain is always the unit box regardless of the patch count, so the
/// manufactured solution, material parameters and boundary data are identical
/// across every point of a scaling sweep -- only the subdivision changes.
inline gsMultiPatch<real_t> makeBoxGrid(short_t dim, index_t nx, index_t ny, index_t nz)
{
    GISMO_ENSURE(2 == dim || 3 == dim, "makeBoxGrid: dim must be 2 or 3.");
    GISMO_ENSURE(nx > 0 && ny > 0 && (2 == dim || nz > 0),
                 "makeBoxGrid: patch counts must be positive.");

    gsKnotVector<real_t> kv(0.0, 1.0, 0, 2); // degree 1, no interior knots
    gsMultiPatch<real_t> mp;

    const real_t hx = (real_t)1 / (real_t)nx;
    const real_t hy = (real_t)1 / (real_t)ny;

    if (2 == dim)
    {
        gsTensorBSplineBasis<2, real_t> basis(kv, kv);
        for (index_t j = 0; j != ny; ++j)
            for (index_t i = 0; i != nx; ++i)
            {
                const real_t x0 = i*hx, x1 = (i+1)*hx, y0 = j*hy, y1 = (j+1)*hy;
                gsMatrix<real_t> c(4, 2); // tensor order: u fastest
                c << x0,y0,  x1,y0,  x0,y1,  x1,y1;
                mp.addPatch(gsTensorBSpline<2, real_t>(basis, c));
            }
    }
    else
    {
        const real_t hz = (real_t)1 / (real_t)nz;
        gsTensorBSplineBasis<3, real_t> basis(kv, kv, kv);
        for (index_t k = 0; k != nz; ++k)
            for (index_t j = 0; j != ny; ++j)
                for (index_t i = 0; i != nx; ++i)
                {
                    const real_t x0 = i*hx, x1 = (i+1)*hx;
                    const real_t y0 = j*hy, y1 = (j+1)*hy;
                    const real_t z0 = k*hz, z1 = (k+1)*hz;
                    gsMatrix<real_t> c(8, 3); // u fastest, then v, then w
                    c << x0,y0,z0,  x1,y0,z0,  x0,y1,z0,  x1,y1,z0,
                         x0,y0,z1,  x1,y0,z1,  x0,y1,z1,  x1,y1,z1;
                    mp.addPatch(gsTensorBSpline<3, real_t>(basis, c));
                }
    }

    // computeTopology()'s tolerance is an absolute distance on corner
    // coordinates: with a fine patch grid the default 1e-4 could exceed the
    // cell size and glue unrelated patches. Scale it below the smallest cell.
    real_t hmin = math::min(hx, hy);
    if (3 == dim) hmin = math::min(hmin, (real_t)1/(real_t)nz);
    mp.computeTopology( math::min( (real_t)1e-4, (real_t)0.05*hmin ) );

    return mp;
}

/// @brief Build the computational domain from \a o: an XML multipatch when
/// \c o.geoFile is set, otherwise the generated box grid; then \c o.nsplit
/// applications of uniformSplit().
inline gsMultiPatch<real_t> buildGeometry(const ScalingOptions& o)
{
    gsMultiPatch<real_t> mp;
    if (!o.geoFile.empty())
    {
        gsFileData<real_t> fd(o.geoFile);
        // Not a bare getId(0, mp): many shipped XML files store loose
        // geometries with no <MultiPatch> wrapper at all, and getId() on
        // those only warns -- leaving mp EMPTY and every later stage to
        // segfault on a zero-patch domain. Accept both layouts and fail
        // loudly if neither is present.
        if (fd.template has< gsMultiPatch<real_t> >())
        {
            fd.template getFirst< gsMultiPatch<real_t> >(mp);
            if (0 == mp.nInterfaces() && 0 == mp.nBoundary())
                mp.computeTopology();
        }
        else
        {
            std::vector< memory::unique_ptr< gsGeometry<real_t> > > geos =
                fd.template getAll< gsGeometry<real_t> >();
            GISMO_ENSURE(!geos.empty(),
                "'"<<o.geoFile<<"' contains neither a MultiPatch nor any "
                "Geometry object -- nothing to compute on.");
            for (size_t i = 0; i != geos.size(); ++i)
                mp.addPatch(give(geos[i]));
            mp.computeTopology();
        }
        GISMO_ENSURE(mp.nPatches() > 0, "'"<<o.geoFile<<"' yielded 0 patches.");
    }
    else
        mp = makeBoxGrid(static_cast<short_t>(o.dim), o.npx, o.npy, o.npz);

    for (index_t i = 0; i != o.nsplit; ++i)
        mp = mp.uniformSplit();

    return mp;
}

/// @brief The exterior boundary sides lying on the two extreme faces of the
/// domain along a coordinate axis: \c lo at the minimum, \c hi at the maximum.
struct BoxFaces
{
    std::vector<patchSide> lo, hi;
};

/// @brief Classify exterior boundary sides by where their centre falls along
/// \a axis. Used to pick a "clamp this face, load the opposite one" load case
/// without hard-coding patch indices, so it works for the generated box grid
/// and for a multipatch read from XML alike.
///
/// The extremes are taken from the evaluated side centres themselves rather
/// than from gsMultiPatch::boundingBox(): the latter is a control-point
/// (convex-hull) box, which for curved patches is strictly larger than the
/// geometry and would leave the tolerance test matching nothing.
///
/// A side is classified by its CENTRE only, so on a curved geometry a side
/// that merely touches an extreme is not selected. Both drivers report
/// |lo| and |hi| so a surprising classification is visible.
inline BoxFaces classifyFaces(const gsMultiPatch<real_t>& mp, index_t axis)
{
    const std::vector<patchSide>& bnd = mp.boundaries();
    const size_t n = bnd.size();

    std::vector<real_t> coord(n, 0);
    gsMatrix<real_t> pt, phys;
    for (size_t k = 0; k != n; ++k)
    {
        const patchSide& ps = bnd[k];
        const gsMatrix<real_t> sup = mp.patch(ps.patch).support();
        pt = 0.5 * (sup.col(0) + sup.col(1));            // parametric centre
        const index_t d = ps.side().direction();
        pt(d, 0) = ps.side().parameter() ? sup(d, 1) : sup(d, 0); // pinned to the face
        mp.patch(ps.patch).eval_into(pt, phys);
        coord[k] = phys(axis, 0);
    }

    BoxFaces out;
    if (0 == n) return out;

    real_t lo = coord[0], hi = coord[0];
    for (size_t k = 1; k != n; ++k)
    {
        lo = math::min(lo, coord[k]);
        hi = math::max(hi, coord[k]);
    }
    const real_t tol = (real_t)1e-6 * math::max((real_t)1, hi - lo);

    for (size_t k = 0; k != n; ++k)
    {
        if      (math::abs(coord[k] - lo) < tol) out.lo.push_back(bnd[k]);
        else if (math::abs(coord[k] - hi) < tol) out.hi.push_back(bnd[k]);
        // everything else stays natural (traction-free)
    }
    return out;
}

// Forward declarations: PhaseTimer's memory sampling needs these, but their
// definitions live further down (near the rest of the RSS/VmHWM helpers) so
// that gsPetscLocalToGlobal.h's petsc_rssPeakBytes() (which rssPeakBytes()
// forwards to, see below) only needs to be visible by the time of the call,
// not before this class.
inline double rssBytes();
inline double rssPeakBytes();

// ===========================================================================
// Phase timing
// ===========================================================================

/// @brief Barrier-synchronised named wall-clock phases.
///
/// tic() barriers first so that a phase's measured time is not polluted by
/// imbalance carried over from the previous phase; the barrier cost itself is
/// then attributed to the *previous* phase's wait, which is what makes the
/// per-phase max/min spread meaningful as a load-imbalance indicator.
///
/// When \a memReport is on, two additional per-phase samples (VmHWM and
/// instantaneous RSS, both in bytes) are recorded alongside each timed phase
/// -- see the class comment on --memReport in the task context. Sampling and
/// its collectives are entirely skipped when the flag is off: this must stay
/// a strictly zero-cost no-op for every run that does not ask for it.
class PhaseTimer
{
public:
    explicit PhaseTimer(MPI_Comm comm, bool memReport = false)
    : m_comm(comm), m_memReport(memReport)
    {
        if (m_memReport)
        {
            // Baseline, sampled once at construction. Not itself a row in
            // m_names/m_local (that alignment is timing-only and must stay
            // untouched) -- kept aside and used by the printer as the "start"
            // reference for the first phase's delta.
            m_startHwmBytes = rssPeakBytes();
            m_startRssBytes = rssBytes();
        }
    }

    void tic(const std::string& name)
    {
        MPI_Barrier(m_comm);
        m_current = name;
        m_watch.restart();
    }

    /// Stops the current phase and records it. Returns the local elapsed time.
    double toc()
    {
        const double t = m_watch.stop();
        m_names.push_back(m_current);
        m_local.push_back(t);
        sampleMem();
        return t;
    }

    /// Records \a name with zero time, for a phase this run did not execute
    /// (--nosolve, --noRBM, no --check). Keeping the phase list identical
    /// across every run of a driver is what keeps its CSV schema fixed, so a
    /// sweep mixing flag combinations still produces one aligned table.
    /// Also pushes a memory sample when --memReport is on, for the same
    /// reason: the memory trace's rows must stay index-aligned with m_names
    /// too, in every flag combination.
    void skip(const std::string& name)
    {
        m_names.push_back(name);
        m_local.push_back(0.0);
        sampleMem();
    }

    /// Sum of every recorded phase (local).
    double localTotal() const
    {
        double s = 0;
        for (double t : m_local) s += t;
        return s;
    }

    /// Reduces all recorded phases; \a maxT / \a minT are valid on rank 0.
    /// Every rank must have recorded the same phases in the same order.
    void reduce(std::vector<double>& maxT, std::vector<double>& minT) const
    {
        const int n = static_cast<int>(m_local.size());
        maxT.assign(m_local.size(), 0.0);
        minT.assign(m_local.size(), 0.0);
        if (0 == n) return;
        MPI_Reduce(m_local.data(), maxT.data(), n, MPI_DOUBLE, MPI_MAX, 0, m_comm);
        MPI_Reduce(m_local.data(), minT.data(), n, MPI_DOUBLE, MPI_MIN, 0, m_comm);
    }

    /// @brief MAX-reduces the per-phase VmHWM/RSS samples (block A of
    /// --memReport). \a hwmMaxMB / \a rssMaxMB come back with one MORE entry
    /// than names(): index 0 is the "start" baseline captured at
    /// construction, index i+1 corresponds to names()[i]. Both are in MB and
    /// valid on rank 0 only.
    ///
    /// Returns immediately -- no MPI collective at all -- when --memReport is
    /// off. Every rank shares the same flag from the same command line, so
    /// this early return is collectively consistent (every rank skips the
    /// same way).
    void reduceMem(std::vector<double>& hwmMaxMB, std::vector<double>& rssMaxMB) const
    {
        hwmMaxMB.clear();
        rssMaxMB.clear();
        if (!m_memReport) return;

        const int n = static_cast<int>(m_names.size());
        std::vector<double> localHwm(n + 1), localRss(n + 1);
        localHwm[0] = m_startHwmBytes;
        localRss[0] = m_startRssBytes;
        for (int i = 0; i != n; ++i)
        {
            localHwm[i + 1] = m_hwmBytes[static_cast<size_t>(i)];
            localRss[i + 1] = m_rssBytesVec[static_cast<size_t>(i)];
        }

        std::vector<double> mxHwm(static_cast<size_t>(n + 1), 0.0);
        std::vector<double> mxRss(static_cast<size_t>(n + 1), 0.0);
        MPI_Reduce(localHwm.data(), mxHwm.data(), n + 1, MPI_DOUBLE, MPI_MAX, 0, m_comm);
        MPI_Reduce(localRss.data(), mxRss.data(), n + 1, MPI_DOUBLE, MPI_MAX, 0, m_comm);

        hwmMaxMB.resize(static_cast<size_t>(n + 1));
        rssMaxMB.resize(static_cast<size_t>(n + 1));
        for (int i = 0; i != n + 1; ++i)
        {
            hwmMaxMB[static_cast<size_t>(i)] = mxHwm[static_cast<size_t>(i)] / (1024.0*1024.0);
            rssMaxMB[static_cast<size_t>(i)] = mxRss[static_cast<size_t>(i)] / (1024.0*1024.0);
        }
    }

    const std::vector<std::string>& names() const { return m_names; }
    const std::vector<double>&      local() const { return m_local; }

    /// Local time of the phase named \a name (0 if never recorded).
    double localOf(const std::string& name) const
    {
        for (size_t i = 0; i != m_names.size(); ++i)
            if (m_names[i] == name) return m_local[i];
        return 0.0;
    }

private:
    /// Pushes one VmHWM/RSS sample, keeping the vectors index-aligned with
    /// m_names. No-op (no /proc read, no PetscMemoryGetCurrentUsage call)
    /// when --memReport is off.
    void sampleMem()
    {
        if (!m_memReport) return;
        m_hwmBytes.push_back(rssPeakBytes());
        m_rssBytesVec.push_back(rssBytes());
    }

    MPI_Comm                 m_comm;
    gsStopwatch              m_watch;
    std::string              m_current;
    std::vector<std::string> m_names;
    std::vector<double>      m_local;

    bool                     m_memReport = false;
    double                   m_startHwmBytes = 0.0;
    double                   m_startRssBytes = 0.0;
    std::vector<double>      m_hwmBytes;     ///< index-aligned with m_names
    std::vector<double>      m_rssBytesVec;  ///< index-aligned with m_names
};

/// @brief Prints block (A) of --memReport: a per-phase VmHWM/RSS trace, with
/// deltas taken against the previous row. Call on rank 0 only, after
/// PhaseTimer::reduceMem(). \a hwmMaxMB / \a rssMaxMB must have one more
/// entry than \a names (index 0 = the "start" baseline, index i+1 = phase
/// \a names[i]) -- exactly PhaseTimer::reduceMem()'s output layout.
///
/// Both columns are printed side by side because VmHWM is monotone: a phase
/// that allocates under the running peak reports dVmHWM_MB == 0 even though
/// it demonstrably allocated (dRSS_MB != 0). The instantaneous RSS trace is
/// the only way to tell "this phase allocated nothing" apart from "this
/// phase was masked by an earlier, larger peak".
inline void printMemTrace(std::ostream& os,
                          const std::vector<std::string>& names,
                          const std::vector<double>&      hwmMaxMB,
                          const std::vector<double>&      rssMaxMB)
{
    GISMO_ENSURE(hwmMaxMB.size() == names.size() + 1 && rssMaxMB.size() == names.size() + 1,
        "printMemTrace: hwmMaxMB/rssMaxMB must have one more entry than names "
        "(the 'start' baseline row) -- did you forget PhaseTimer::reduceMem()?");

    os << "\n--- memReport block (A): per-phase memory trace ---\n"
          "  Values are per-rank MAXIMA over ranks. RSS_MB samples "
          "PetscMemoryGetCurrentUsage (instantaneous); VmHWM_MB parses "
          "VmHWM from /proc/self/status (monotone high-water mark). Deltas "
          "are taken against the previous row ('start' = the baseline "
          "sampled at PhaseTimer construction).\n";
    os << "  " << std::left << std::setw(16) << "phase"
       << std::right << std::setw(12) << "VmHWM_MB"
       << std::setw(13) << "dVmHWM_MB"
       << std::setw(12) << "RSS_MB"
       << std::setw(11) << "dRSS_MB" << "\n";

    for (size_t i = 0; i != hwmMaxMB.size(); ++i)
    {
        const std::string label = (0 == i) ? std::string("start") : names[i - 1];
        const double dHwm = (0 == i) ? 0.0 : hwmMaxMB[i] - hwmMaxMB[i-1];
        const double dRss = (0 == i) ? 0.0 : rssMaxMB[i] - rssMaxMB[i-1];
        os << "  " << std::left << std::setw(16) << label
           << std::right << std::fixed << std::setprecision(2)
           << std::setw(12) << hwmMaxMB[i]
           << std::setw(13) << dHwm
           << std::setw(12) << rssMaxMB[i]
           << std::setw(11) << dRss << "\n";
    }
    os << std::defaultfloat;
}

// ===========================================================================
// Result record (pretty print + CSV)
// ===========================================================================

/// @brief An ordered key/value row. Written both to stdout (aligned) and,
/// with --csv, appended to a CSV file whose header is emitted only when the
/// file does not yet exist or is empty -- so a SLURM sweep can point every
/// job at the same file and end up with one directly plottable table.
class Record
{
public:
    Record& add(const std::string& key, const std::string& value)
    {
        m_fields.push_back(std::make_pair(key, value));
        return *this;
    }

    Record& add(const std::string& key, index_t value)
    {
        std::ostringstream ss;
        ss << value;
        return add(key, ss.str());
    }

    Record& add(const std::string& key, double value, int prec = 6)
    {
        std::ostringstream ss;
        ss << std::setprecision(prec) << std::scientific << value;
        return add(key, ss.str());
    }

    void print(std::ostream& os) const
    {
        size_t w = 0;
        for (const auto& f : m_fields) w = math::max(w, f.first.size());
        for (const auto& f : m_fields)
            os << "  " << std::left << std::setw(static_cast<int>(w)) << f.first
               << std::right << " : " << f.second << "\n";
    }

    /// The header line this record would write.
    std::string header() const
    {
        std::string h;
        for (size_t i = 0; i != m_fields.size(); ++i)
        {
            if (i) h += ",";
            h += m_fields[i].first;
        }
        return h;
    }

    /// Appends this row to \a path, writing the header line first if the file
    /// is new or empty. Not MPI-collective: call from rank 0 only.
    ///
    /// If the file already has a header, it must match this record's exactly.
    /// Appending a row with a different column set would silently shift every
    /// value one column relative to the header -- the reader still parses it,
    /// and the plot is wrong in a way that is very hard to spot. That happens
    /// as soon as two DIFFERENT drivers (or the same driver at a different
    /// PETSc version, where the COO column disappears) point --csv at one
    /// file, so it is refused rather than warned about.
    void writeCsv(const std::string& path) const
    {
        const std::string myHeader = header();
        std::string existing;
        {
            std::ifstream probe(path.c_str());
            if (probe.good()) std::getline(probe, existing);
        }
        if (!existing.empty())
            GISMO_ENSURE(existing == myHeader,
                "CSV schema mismatch: '"<<path<<"' was written with a "
                "different column set. Use a separate --csv file per driver / "
                "per build configuration.\n  in file : "<<existing<<
                "\n  this run: "<<myHeader);

        std::ofstream out(path.c_str(), std::ios::app);
        GISMO_ENSURE(out.good(), "Could not open CSV file '"<<path<<"' for appending.");
        if (existing.empty()) out << myHeader << "\n";
        for (size_t i = 0; i != m_fields.size(); ++i)
            out << (i ? "," : "") << m_fields[i].second;
        out << "\n";
    }

private:
    std::vector<std::pair<std::string, std::string> > m_fields;
};

/// Current process resident set size in bytes (reads /proc on Linux).
inline double rssBytes()
{
    PetscLogDouble rss = 0;
    PetscMemoryGetCurrentUsage(&rss);
    return static_cast<double>(rss);
}

/// @brief Peak resident set size ever reached by this process, in bytes
/// (VmHWM from /proc/self/status; 0 if unavailable).
///
/// rssBytes() is an instantaneous sample, so it only sees whatever happens to
/// be live at that moment -- it misses transient peaks such as the COO triplet
/// arrays during insertion and, more importantly, the AMG hierarchy built
/// inside KSPSetUp. The high-water mark is the number to size a batch
/// scheduler's --mem-per-cpu from.
///
/// Forwards to gsPetscLocalToGlobal.h's petsc_rssPeakBytes() so there is
/// exactly one implementation of the VmHWM /proc parse: that header cannot
/// include this one (optional/gsPetsc/src/ must not depend on
/// optional/gsPetsc/examples/), so the single implementation lives there and
/// this driver-facing name is kept as a thin alias.
inline double rssPeakBytes()
{
    return petsc_rssPeakBytes();
}

/// @brief Allocated bytes of a gsSparseMatrix: values + inner indices +
/// outer pointers, plus the per-column nonzero counters when the matrix is
/// not compressed (uncompressed storage keeps a separate innerNonZeroPtr
/// array alongside the compressed-form outer pointers).
template<class T, int _Options, class _Index>
inline double sparseMatrixBytes(const gsSparseMatrix<T,_Options,_Index>& m)
{
    typedef typename gsSparseMatrix<T,_Options,_Index>::StorageIndex SI;
    double b = (double)m.nonZeros() * (double)(sizeof(T) + sizeof(SI))
             + (double)(m.outerSize() + 1) * (double)sizeof(SI);
    if (!m.isCompressed() && NULL != m.innerNonZeroPtr())
        b += (double)m.outerSize() * (double)sizeof(SI);
    return b;
}

/// @brief Number of columns of \a m holding at least one stored entry.
///
/// A stand-in for the lazy fiber matrix's touched-fiber count: task 01's
/// fiberPointerBytes()/fiberDataBytes() do not expose that count directly,
/// and gsFiberMatrix.h is out of scope for this task. \a m is the CSC copy
/// gsExprAssembler::matrix() makes of the fiber matrix, so a column with a
/// stored entry corresponds (up to the rare all-explicit-zero column) to a
/// touched fiber. Diagnostic only -- used to bound the per-allocation
/// overhead (~16-32 B per `new Fiber`) that none of the byte accessors count.
template<class T, int _Options, class _Index>
inline index_t touchedFiberColumns(const gsSparseMatrix<T,_Options,_Index>& m)
{
    index_t n = 0;
    for (index_t k = 0; k < m.outerSize(); ++k)
    {
        typename gsSparseMatrix<T,_Options,_Index>::InnerIterator it(m, k);
        if (it) ++n;
    }
    return n;
}

/// Max over ranks of the per-rank peak RSS (VmHWM), in MB. Valid on rank 0.
inline double reduceRssPeak(MPI_Comm comm)
{
    const double local = rssPeakBytes();
    double mx = 0;
    MPI_Reduce(&local, &mx, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    return mx / (1024.0*1024.0);
}

/// Min/max of a per-rank integer quantity (load-balance diagnostics).
/// Reduced as long long -- MPI reduction operators are not defined for
/// MPI_BYTE, so index_t cannot be reduced by its byte width the way
/// MPI_Bcast allows. Valid on rank 0.
inline void reduceMinMax(index_t local, index_t& mn, index_t& mx, MPI_Comm comm)
{
    const long long l = static_cast<long long>(local);
    long long a = 0, b = 0;
    MPI_Reduce(&l, &a, 1, MPI_LONG_LONG, MPI_MIN, 0, comm);
    MPI_Reduce(&l, &b, 1, MPI_LONG_LONG, MPI_MAX, 0, comm);
    mn = static_cast<index_t>(a);
    mx = static_cast<index_t>(b);
}

/// Max / sum of the per-rank RSS, in MB. Valid on rank 0.
inline void reduceRss(MPI_Comm comm, double& maxMB, double& sumMB)
{
    const double local = rssBytes();
    double mx = 0, sm = 0;
    MPI_Reduce(&local, &mx, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local, &sm, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    maxMB = mx / (1024.0*1024.0);
    sumMB = sm / (1024.0*1024.0);
}

// ===========================================================================
// --memReport block (B): byte budget
// ===========================================================================

/// @brief An ordered label/bytes budget table with a live/transient split,
/// printed as block (B) of --memReport.
///
/// LIVE lines (add()) are the structures actually resident at the boundary
/// this table is printed for; they are what sum_named counts. TRANSIENT
/// lines (addTransient()) are printed too, clearly labelled, but EXCLUDED
/// from sum_named -- they were observed peaks (e.g. the COO triplet arrays)
/// that are no longer resident by the time this table prints, so folding
/// them into sum_named would silently mask a real unnamed structure and let
/// residual go negative for a reason that has nothing to do with an actual
/// bug.
///
/// Collective: call print() on every rank (it MAX-reduces internally) and it
/// writes to \a os on rank 0 only. Not guarded on --memReport itself -- the
/// caller decides whether to call it at all, but every rank must agree,
/// since every rank shares the same flag from the same command line.
///
/// INVARIANT (enforced by print(), see the GISMO_ENSURE at its top): every
/// rank must call add()/addTransient() the SAME number of times, in the SAME
/// order, producing the SAME label at the SAME index -- print() lays the
/// live and transient lines out as one flat per-rank vector and MAX-reduces
/// it index-by-index, so any rank-dependent branch on whether/where a line
/// is added (e.g. a conditional based on a per-rank MatInfo value reading 0
/// only on ranks that own no rows) silently pairs one rank's line against a
/// different rank's *different* line. The fix is to make any such branch
/// collective (MAX/MIN-reduce the deciding scalar first) so every rank takes
/// the same path -- see addPetscMatBudget() below for the pattern.
class MemBudget
{
public:
    /// A live line: bytes resident at this boundary, counted in sum_named.
    void add(const std::string& label, double bytes)
    {
        m_live.push_back(std::make_pair(label, bytes));
    }

    /// A transient line: an observed peak, NOT resident at this boundary,
    /// printed below the live lines and excluded from sum_named.
    void addTransient(const std::string& label, double bytes)
    {
        m_transient.push_back(std::make_pair(label, bytes));
    }

    /// @brief Prints the table. \a nDofs is the (already global, identical on
    /// every rank) DoF count; \a localNnz is THIS rank's local nonzero count,
    /// MAX-reduced internally for the B/nnz column, consistent with the rest
    /// of the reductions here.
    void print(std::ostream& os, const std::string& title,
              index_t nDofs, index_t localNnz, MPI_Comm comm) const
    {
        int rank = 0;
        MPI_Comm_rank(comm, &rank);

        // 0. Cheap collective self-check enforcing the class-comment
        //    invariant: every rank must have pushed the same number of live
        //    and transient lines. Without this, a future conditional add()/
        //    addTransient() silently re-breaks the alignment fixed for
        //    addPetscMatBudget() -- see the review that caught it.
        {
            int localSizes[2] = { static_cast<int>(m_live.size()),
                                   static_cast<int>(m_transient.size()) };
            int minSizes[2] = {0, 0}, maxSizes[2] = {0, 0};
            MPI_Allreduce(localSizes, minSizes, 2, MPI_INT, MPI_MIN, comm);
            MPI_Allreduce(localSizes, maxSizes, 2, MPI_INT, MPI_MAX, comm);
            GISMO_ENSURE(minSizes[0] == maxSizes[0] && minSizes[1] == maxSizes[1],
                "MemBudget::print: every rank must add()/addTransient() the "
                "same number of lines, in the same order (live=["
                << minSizes[0] << "," << maxSizes[0] << "] transient=["
                << minSizes[1] << "," << maxSizes[1] << "] across ranks) -- "
                "a rank-local branch on add() vs addTransient() breaks the "
                "index alignment that the per-line MAX-reduce below relies on");
        }

        // 1. This rank's total of the LIVE lines only, then MAX-reduce that
        //    single scalar -- the exact worst-rank total, reported as
        //    sum_named. Deliberately NOT the sum of the (separately
        //    MAX-reduced) per-line column below: that column sums to >= the
        //    true per-rank total whenever different lines peak on different
        //    ranks, so summing it would overstate the worst rank.
        double localTotal = 0.0;
        for (size_t i = 0; i != m_live.size(); ++i) localTotal += m_live[i].second;
        double totalMaxBytes = 0.0;
        MPI_Reduce(&localTotal, &totalMaxBytes, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

        // 2. Per-line MAX reduce (live, then transient), for composition only.
        const size_t nLive = m_live.size(), nTrans = m_transient.size();
        std::vector<double> localLines(nLive + nTrans, 0.0);
        for (size_t i = 0; i != nLive;  ++i) localLines[i]         = m_live[i].second;
        for (size_t i = 0; i != nTrans; ++i) localLines[nLive + i] = m_transient[i].second;
        std::vector<double> maxLines(nLive + nTrans, 0.0);
        if (!localLines.empty())
            MPI_Reduce(localLines.data(), maxLines.data(),
                      static_cast<int>(localLines.size()), MPI_DOUBLE, MPI_MAX, 0, comm);

        // 3. RSS (instantaneous) and VmHWM (peak), both MAX-reduced -- reuse
        //    the existing collectives, no new /proc parsing.
        double rssMaxMB = 0, rssSumMB = 0;
        reduceRss(comm, rssMaxMB, rssSumMB);
        const double hwmMaxMB = reduceRssPeak(comm);

        // 4. Local nnz, MAX-reduced the same way as reduceMinMax's MAX half.
        const long long localNnzLL = static_cast<long long>(localNnz);
        long long nnzMaxLL = 0;
        MPI_Reduce(&localNnzLL, &nnzMaxLL, 1, MPI_LONG_LONG, MPI_MAX, 0, comm);

        if (0 != rank) return;

        const double MB = 1024.0*1024.0;
        const double totalMaxMB = totalMaxBytes / MB;
        const double residualMB = rssMaxMB - totalMaxMB;
        const index_t nnzMax = static_cast<index_t>(nnzMaxLL);

        auto bPerDof = [&](double bytes) -> std::string
        {
            if (0 == nDofs) return std::string("-");
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(2) << (bytes / (double)nDofs);
            return ss.str();
        };
        auto bPerNnz = [&](double bytes) -> std::string
        {
            if (0 == nnzMax) return std::string("-");
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(3) << (bytes / (double)nnzMax);
            return ss.str();
        };
        auto printRow = [&](const std::string& label, double bytes)
        {
            os << "  " << std::left << std::setw(20) << label
               << std::right << std::fixed << std::setprecision(2)
               << std::setw(12) << (bytes / MB) << " MB"
               << std::setw(10) << bPerDof(bytes) << " B/DoF"
               << std::setw(10) << bPerNnz(bytes) << " B/nnz\n";
        };

        os << "\n--- memReport block (B): byte budget (" << title << ") ---\n"
              "  Per-line values are per-rank MAXIMA and therefore sum to >= "
              "the reported total. sum_named is the MAX-reduced per-rank "
              "TOTAL of the LIVE lines only; lines marked (transient) are "
              "observed peaks, not resident here, and are excluded from "
              "sum_named. residual = RSS - sum_named includes allocator "
              "overhead (~16-32 B per allocation), the replicated basis/"
              "geometry and anything unnamed -- it is an honest unknown, not "
              "an error term.\n";

        for (size_t i = 0; i != nLive; ++i)
            printRow(m_live[i].first, maxLines[i]);
        for (size_t i = 0; i != nTrans; ++i)
            printRow(m_transient[i].first + " (transient)", maxLines[nLive + i]);

        printRow("sum_named", totalMaxBytes);
        printRow("RSS",       rssMaxMB * MB);
        printRow("residual",  residualMB * MB); // may be negative -- printed as-is, see header text
        printRow("VmHWM", hwmMaxMB * MB);
        os << "  local nnz (MAX over ranks): " << nnzMax << "\n";
        os << std::defaultfloat;
    }

private:
    std::vector<std::pair<std::string, double> > m_live;
    std::vector<std::pair<std::string, double> > m_transient;
};

/// @brief Adds the petsc_mat line to \a budget for the distributed matrix
/// \a K (this rank owns \a nOwned rows of it).
///
/// Prefers a COMPUTED LOWER BOUND from MatInfo.nz_allocated (AIJ layout:
/// values + column indices for every allocated nonzero, plus one row-pointer
/// entry per owned row) over MatInfo.memory: PETSc's own malloc accounting
/// (memory) was observed to read 0 in this build (RelWithDebInfo, OpenMPI +
/// PETSc built locally) -- gsMetisPetscAssembly_example.cpp already carries
/// the same caveat ("MatInfo.memory is unreliable in this build"). If BOTH
/// read 0, the line is added as an explicitly UNMEASURED transient (excluded
/// from sum_named) rather than a bare 0.00 that would otherwise silently
/// understate sum_named by the largest structure in the whole budget --
/// exactly the "table that can only look complete" failure mode the
/// partitioner lower bound already guards against.
///
/// The add()-vs-addTransient() branch below MUST be collective (review
/// finding, task 03): MatInfo.nz_allocated/memory are rank-LOCAL, and a rank
/// that owns zero rows (e.g. more ranks than partitions) reads 0 on both
/// while other ranks read >0 -- branching on the rank-local value alone
/// would then put "petsc_mat" into a different bucket (add vs addTransient)
/// on different ranks, corrupting MemBudget::print()'s flat index alignment
/// for every line after it. MAX-reducing the two candidate scalars first
/// makes every rank take the same branch; each rank still contributes its
/// own (possibly 0) value into that shared bucket, which is correct -- the
/// per-line MAX-reduce inside print() recovers the true worst-rank value.
inline void addPetscMatBudget(MemBudget& budget, Mat K, index_t nOwned, MPI_Comm comm)
{
    MatInfo minfo;
    MatGetInfo(K, MAT_LOCAL, &minfo);

    double localCand[2] = { (double)minfo.nz_allocated, (double)minfo.memory };
    double maxCand[2]   = { 0.0, 0.0 };
    MPI_Allreduce(localCand, maxCand, 2, MPI_DOUBLE, MPI_MAX, comm);

    if (maxCand[0] > 0)
    {
        const double bytes = minfo.nz_allocated * (double)(sizeof(PetscScalar) + sizeof(PetscInt))
                            + (double)(nOwned + 1) * (double)sizeof(PetscInt);
        budget.add("petsc_mat (computed lower bound, nz_allocated)", bytes);
    }
    else if (maxCand[1] > 0)
    {
        budget.add("petsc_mat", (double)minfo.memory);
    }
    else
    {
        budget.addTransient(
            "petsc_mat (UNMEASURED: MatInfo.memory and nz_allocated both 0 in this build)", 0.0);
    }
}

// ===========================================================================
// Partitioning
// ===========================================================================

/// @brief Run METIS once per job (rank 0) and broadcast the labels.
///
/// Every other rank only builds the element graph -- which
/// makeDofMapper()/setPartLabels() need -- instead of redundantly re-running
/// METIS. Afterwards every rank holds an identical, fully-partitioned
/// partitioner, so DOF ownership and the global permutation are computable
/// without further communication.
inline void partitionAndBroadcast(gsMetisPartitioner<real_t>& part,
                                  index_t numElements,
                                  int rank, MPI_Comm comm)
{
    if (0 == rank) part.partition();
    else           part.buildGraph();

    std::vector<idx_t> labels(static_cast<size_t>(numElements));
    if (0 == rank)
        labels.assign(part.partLabels().begin(), part.partLabels().end());
    index_t edgeCut = (0 == rank) ? part.edgeCut() : 0;

    MPI_Bcast(labels.data(), static_cast<int>(labels.size()*sizeof(idx_t)),
              MPI_BYTE, 0, comm);
    MPI_Bcast(&edgeCut, static_cast<int>(sizeof(index_t)), MPI_BYTE, 0, comm);

    part.setPartLabels(give(labels), edgeCut);
}

/// Everything the drivers need out of a partitioning run.
struct PartitionResult
{
    gsPartitionedDofMapper         dofMap;      ///< DOF ownership + global permutation
    typename gsDomain<real_t>::Ptr rankDomain;  ///< this rank's integration subdomain
    index_t                        edgeCut = -1;///< METIS edge cut; -1 for graph-free
    std::string                    partWeight;  ///< the weighting actually used
    /// @brief Lower bound on the partitioner's own footprint: the label
    /// array only (part->labels().capacity() * sizeof(index_t)). The
    /// partitioner object itself is destroyed at the end of buildPartition(),
    /// so it cannot be sized at any later budget boundary -- this is what
    /// survives. A LOWER bound, not the full figure: element centroids,
    /// weights and (on --partitioner metis) the element dual graph are not
    /// included. Always computed (one multiplication); only printed under
    /// --memReport.
    double                         partitionerBytesLB = 0.0;
};

/// @brief Build the partition selected by --partitioner and derive DOF
/// ownership from it. Collective: every rank must call it (PhaseTimer::tic()
/// barriers), and it records exactly two phases, "partition" and "ownership".
///
/// metis   : rank 0 runs METIS, the labels are broadcast (partitionAndBroadcast).
/// rcb / hilbert / morton : NO BROADCAST. gsGeometricPartitioner's orderings are
///   TOTAL (every comparison ties-breaks on the element id), so every rank
///   recomputes bit-identical labels from replicated data. That is what makes
///   the broadcast unnecessary rather than merely skipped -- it also assumes
///   one binary on identical nodes, see plan "Out of scope: heterogeneous nodes".
inline PartitionResult buildPartition(const ScalingOptions&       o,
                                      const gsMultiPatch<real_t>& mp,
                                      const gsMultiBasis<real_t>& mb,
                                      const gsDofMapper&          mapper,
                                      index_t                     nparts,
                                      index_t                     nElems,
                                      int                         rank,
                                      index_t                     nranks,
                                      PhaseTimer&                 timer,
                                      MPI_Comm                    comm)
{
    GISMO_ENSURE("metis" == o.partitioner || "rcb" == o.partitioner ||
                 "hilbert" == o.partitioner || "morton" == o.partitioner,
        "--partitioner must be one of rcb, hilbert, morton, metis (got '"
        <<o.partitioner<<"').");
    GISMO_ENSURE("dofs" == o.partWeight || "elements" == o.partWeight,
        "--partWeight must be 'dofs' or 'elements' (got '"<<o.partWeight<<"').");

    PartitionResult res;
    memory::unique_ptr<gsPartitionerBase<real_t> > part;

    timer.tic("partition");
    if ("metis" == o.partitioner)
    {
        gsMetisPartitioner<real_t>::Options partOpts;
        partOpts.storeElementDofs = true;   // required by makeDofMapper()
        partOpts.contiguous       = o.metisContig;
        partOpts.weightByDofs     = o.metisWeightDofs;
        partOpts.imbalance        = o.metisImbalance;

        gsMetisPartitioner<real_t>* mpart =
            new gsMetisPartitioner<real_t>(mb, mapper, nparts, partOpts);
        part.reset(mpart);
        partitionAndBroadcast(*mpart, nElems, rank, comm);
        res.edgeCut    = mpart->edgeCut();
        // METIS keeps its own switch, so that --partitioner metis reproduces
        // the pre-existing baseline exactly. --partWeight does not touch it.
        res.partWeight = o.metisWeightDofs ? std::string("dofs")
                                           : std::string("elements");
    }
    else
    {
        gsGeometricPartitioner<real_t>::Options partOpts;
        partOpts.strategy     = gsGeometricPartitioner<real_t>::strategyFromString(o.partitioner);
        partOpts.weightByDofs = ("dofs" == o.partWeight);

        gsGeometricPartitioner<real_t>* gpart =
            new gsGeometricPartitioner<real_t>(mp, mb, mapper, nparts, partOpts);
        part.reset(gpart);
        gpart->partition();               // every rank, identically; no MPI here
        res.edgeCut    = gpart->edgeCut();  // -1 by design (no graph, no edge cut)
        res.partWeight = o.partWeight;
    }
    timer.toc();

    timer.tic("ownership");
    res.dofMap     = part->makeDofMapper(nranks);
    res.rankDomain = part->subdomainForRank(rank, nranks);
    timer.toc();

    // Lower-bound footprint of the partitioner object, captured before it is
    // destroyed on scope exit below -- one multiplication, unconditional (see
    // the PartitionResult comment for why it can only ever be a lower bound).
    res.partitionerBytesLB = (double)part->labels().capacity() * (double)sizeof(index_t);

    return res;
}

// ===========================================================================
// DOF geometry + rigid-body near-null-space
// ===========================================================================

/// @brief Physical coordinate and component index of every free global DOF.
///
/// The coordinate of a DOF is the image, under the geometry map, of the
/// Greville anchor of the corresponding basis function. Coupled DOFs are
/// written more than once with (up to round-off) the same value, which is
/// harmless: the near-null-space only needs a smooth coordinate field, not an
/// exact one.
struct DofGeometry
{
    gsMatrix<real_t>     coords; ///< dim x freeSize
    std::vector<index_t> comp;   ///< component index per free DOF
};

/// @param nComp Number of components of the vector space (1 for Poisson).
inline DofGeometry computeDofGeometry(const gsMultiPatch<real_t>& mp,
                                      const gsMultiBasis<real_t>& mb,
                                      const gsDofMapper&          mapper,
                                      index_t                     nComp)
{
    DofGeometry dg;
    const index_t nDofs = mapper.freeSize();
    const short_t gdim  = mp.geoDim();

    dg.coords.setZero(gdim, nDofs);
    dg.comp.assign(static_cast<size_t>(nDofs), 0);

    gsMatrix<real_t> anchors, phys;
    for (size_t p = 0; p != mb.nBases(); ++p)
    {
        mb.basis(p).anchors_into(anchors);
        mp.patch(p).eval_into(anchors, phys); // gdim x nBasisFunctions

        const index_t sz = mb.basis(p).size();
        for (index_t c = 0; c != nComp; ++c)
            for (index_t i = 0; i != sz; ++i)
            {
                const index_t g = mapper.index(i, static_cast<index_t>(p), c);
                if (!mapper.is_free_index(g)) continue;
                dg.coords.col(g) = phys.col(i);
                dg.comp[g]       = c;
            }
    }
    return dg;
}

/// @brief Build the rigid-body modes of \a A and attach them as its near-null
/// space, so that PCGAMG builds a prolongator that reproduces them.
///
/// Not MatNullSpaceCreateRigidBody(): that expects one interlaced coordinate
/// Vec with block size dim, i.e. a node-major DOF ordering. gsDofMapper
/// numbers a vector space COMPONENT-major (and with a per-component free
/// count, since each component has its own Dirichlet elimination), and the
/// PETSc rows are additionally permuted by gsPartitionedDofMapper. The modes
/// are therefore assembled directly from (coordinate, component) per DOF.
///
/// Modes: dim translations plus 1 (2D) or 3 (3D) rotations, orthonormalised
/// by modified Gram-Schmidt using distributed PETSc dot products (no
/// replicated dense storage: each Vec is filled only over this rank's
/// ownership range).
///
/// @param invPerm invPerm(i) = the global DOF occupying PETSc row i, i.e. the
///                inverse of gsPartitionedDofMapper::permutation().
inline int attachRigidBodyNullSpace(Mat&                     A,
                                    const DofGeometry&       dg,
                                    short_t                  dim,
                                    const gsVector<index_t>& invPerm,
                                    MPI_Comm                 comm)
{
    const int nModes = (2 == dim) ? 3 : 6;

    std::vector<Vec> modes(static_cast<size_t>(nModes), PETSC_NULLPTR);
    for (int m = 0; m != nModes; ++m)
        PetscCall( MatCreateVecs(A, PETSC_NULLPTR, &modes[m]) );

    PetscInt lo = 0, hi = 0;
    PetscCall( VecGetOwnershipRange(modes[0], &lo, &hi) );

    // Rows [lo,hi) are local, so write straight into the local array.
    for (int m = 0; m != nModes; ++m)
    {
        PetscScalar* a = PETSC_NULLPTR;
        PetscCall( VecGetArray(modes[m], &a) );
        for (PetscInt i = lo; i != hi; ++i)
        {
            const index_t g = invPerm(static_cast<index_t>(i));
            const index_t c = dg.comp[static_cast<size_t>(g)];
            const real_t  x = dg.coords(0, g);
            const real_t  y = dg.coords(1, g);
            const real_t  z = (3 == dim) ? dg.coords(2, g) : (real_t)0;

            real_t v = 0;
            if (m < dim)                       v = (c == m) ? (real_t)1 : (real_t)0;
            else if (m == dim)                 v = (0 == c) ? -y : ((1 == c) ?  x : (real_t)0); // rot z
            else if (m == dim + 1)             v = (1 == c) ? -z : ((2 == c) ?  y : (real_t)0); // rot x
            else                               v = (2 == c) ? -x : ((0 == c) ?  z : (real_t)0); // rot y

            a[i - lo] = static_cast<PetscScalar>(v);
        }
        PetscCall( VecRestoreArray(modes[m], &a) );
    }

    // Modified Gram-Schmidt. A mode that collapses (norm ~ 0) is dropped
    // rather than normalised into noise -- MatNullSpaceCreate requires an
    // orthonormal set, and a near-zero vector would violate that.
    int kept = 0;
    for (int m = 0; m != nModes; ++m)
    {
        for (int k = 0; k != kept; ++k)
        {
            PetscScalar d;
            PetscCall( VecDot(modes[m], modes[k], &d) );
            PetscCall( VecAXPY(modes[m], -d, modes[k]) );
        }
        PetscReal nrm = 0;
        PetscCall( VecNorm(modes[m], NORM_2, &nrm) );
        if (nrm < 1e-10) continue;
        PetscCall( VecScale(modes[m], (PetscScalar)(1.0/nrm)) );
        if (kept != m) { Vec tmp = modes[kept]; modes[kept] = modes[m]; modes[m] = tmp; }
        ++kept;
    }

    MatNullSpace sp;
    PetscCall( MatNullSpaceCreate(comm, PETSC_FALSE, kept, modes.data(), &sp) );
    PetscCall( MatSetNearNullSpace(A, sp) );
    PetscCall( MatNullSpaceDestroy(&sp) );
    for (int m = 0; m != nModes; ++m)
        PetscCall( VecDestroy(&modes[m]) );

    return 0;
}

// ===========================================================================
// Solve
// ===========================================================================

struct SolveStats
{
    PetscInt           its    = 0;
    PetscReal          relRes = 0;               ///< true ||b-Ax||/||b||
    KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
    double             tSetup = 0;               ///< KSPSetUp (PC construction)
    double             tSolve = 0;               ///< KSPSolve only
};

/// @brief CG + GAMG by default, fully overridable from the PETSc command line.
///
/// KSPSetFromOptions() is called AFTER the defaults, so -ksp_type / -pc_type /
/// -ksp_rtol from the SLURM script always win. KSPSetUp is timed separately
/// from KSPSolve because on an AMG run the two answer different scaling
/// questions (setup cost vs. iteration cost). The residual is the TRUE
/// ||b-Ax||/||b||, computed with distributed PETSc operations only -- the
/// preconditioned residual KSP reports internally can be misleading.
inline int solveSystem(Mat& A, Vec& b, Vec& x,
                       real_t rtol, index_t maxIts,
                       SolveStats& st, PhaseTimer& timer, MPI_Comm comm)
{
    KSP ksp;
    PC  pc;
    PetscCall( KSPCreate(comm, &ksp) );
    PetscCall( KSPSetOperators(ksp, A, A) );
    PetscCall( KSPSetType(ksp, KSPCG) );
    PetscCall( KSPGetPC(ksp, &pc) );
    PetscCall( PCSetType(pc, PCGAMG) );
    PetscCall( KSPSetTolerances(ksp, static_cast<PetscReal>(rtol), PETSC_DEFAULT,
                                PETSC_DEFAULT, static_cast<PetscInt>(maxIts)) );
    PetscCall( KSPSetFromOptions(ksp) ); // user options win over the defaults above

    timer.tic("ksp_setup");
    PetscCall( KSPSetUp(ksp) );
    st.tSetup = timer.toc();

    timer.tic("ksp_solve");
    PetscCall( KSPSolve(ksp, b, x) );
    st.tSolve = timer.toc();

    PetscCall( KSPGetIterationNumber(ksp, &st.its) );
    PetscCall( KSPGetConvergedReason(ksp, &st.reason) );

    // True residual, distributed: r = A*x - b.
    Vec r;
    PetscCall( VecDuplicate(b, &r) );
    PetscCall( MatMult(A, x, r) );
    PetscCall( VecAXPY(r, -1.0, b) );
    PetscReal rn = 0, bn = 0;
    PetscCall( VecNorm(r, NORM_2, &rn) );
    PetscCall( VecNorm(b, NORM_2, &bn) );
    st.relRes = (bn > 0) ? rn/bn : rn;
    PetscCall( VecDestroy(&r) );

    PetscCall( KSPDestroy(&ksp) );
    return 0;
}

// ===========================================================================
// Reporting helper shared by both drivers
// ===========================================================================

/// Adds the phase table (max and min across ranks) to \a rec and prints it.
/// Call on rank 0 only, after PhaseTimer::reduce().
inline void addPhases(Record& rec,
                      const std::vector<std::string>& names,
                      const std::vector<double>&      maxT,
                      const std::vector<double>&      minT)
{
    double total = 0;
    for (size_t i = 0; i != names.size(); ++i)
    {
        rec.add("t_" + names[i], maxT[i]);
        rec.add("t_" + names[i] + "_min", minT[i]);
        total += maxT[i];
    }
    rec.add("t_total", total);
}

/// Handles the -h/--help/--version fast path BEFORE PetscInitialize().
///
/// A binary launched directly (not under mpirun) has no MPI runtime, and on
/// hosts where the singleton runtime cannot start, PetscInitialize() would
/// hang "-h" before a single line is printed. The options are registered
/// exactly once (by the caller) and rendered here on a synthetic argv, so
/// unrelated PETSc options such as -ksp_rtol can never fail this parse.
/// Returns true if the caller should exit immediately.
inline bool handleHelpBeforeMPI(gsCmdLine& cmd, int argc, char* argv[], int& exitCode)
{
    exitCode = 0;
    for (int i = 1; i != argc; ++i)
    {
        const std::string arg(argv[i]);
        if ("--" == arg) break;
        if ("-h" == arg || "--help" == arg || "--version" == arg)
        {
            char* hargv[] = { argv[0], argv[i], NULL };
            try { cmd.getValues(2, hargv); }
            catch (int ret) { exitCode = ret; }
            return true;
        }
    }
    return false;
}

} // namespace scaling
} // namespace gismo
