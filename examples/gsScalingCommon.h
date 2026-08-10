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
    architecture the geometry, basis, DOF mapper, element graph and METIS
    labelling are REPLICATED on every rank; only the PETSc Mat/Vec and the
    per-rank share of the assembled entries are distributed. Measured
    consequences:

      * STRONG scaling (fixed problem, more ranks): per-rank memory falls, but
        only sublinearly -- 517 -> 289 MB over 4x the ranks -- because the
        replicated part does not move. The SMALLEST rank count is therefore
        the memory-binding case of a strong-scaling sweep.
      * WEAK scaling (DoFs/rank fixed, more ranks): per-rank memory GROWS --
        80 -> 177 MB over 8x the ranks -- because the replicated structures
        track the GLOBAL problem. This, not communication, is the wall.

    The dominant replicated structure is the element dual graph: for a 3D
    degree-2 basis two elements are adjacent whenever their supports overlap,
    which is ~(2p+1)^3 - 1 = 124 neighbours per element.

    The drivers report rss_hwm_mb (VmHWM, the true peak, covering the AMG
    hierarchy and the transient COO arrays) precisely so a batch scheduler's
    --mem-per-cpu can be sized from a real number: the instantaneous
    rss_max_mb sample is measured 1.5-1.9x LOWER. --lazyMatrix (default ON
    here) keeps the assembler from reserving all nDofs columns up front and is
    the single biggest lever on the distributed part.

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

// ===========================================================================
// Phase timing
// ===========================================================================

/// @brief Barrier-synchronised named wall-clock phases.
///
/// tic() barriers first so that a phase's measured time is not polluted by
/// imbalance carried over from the previous phase; the barrier cost itself is
/// then attributed to the *previous* phase's wait, which is what makes the
/// per-phase max/min spread meaningful as a load-imbalance indicator.
class PhaseTimer
{
public:
    explicit PhaseTimer(MPI_Comm comm) : m_comm(comm) { }

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
        return t;
    }

    /// Records \a name with zero time, for a phase this run did not execute
    /// (--nosolve, --noRBM, no --check). Keeping the phase list identical
    /// across every run of a driver is what keeps its CSV schema fixed, so a
    /// sweep mixing flag combinations still produces one aligned table.
    void skip(const std::string& name)
    {
        m_names.push_back(name);
        m_local.push_back(0.0);
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
    MPI_Comm                 m_comm;
    gsStopwatch              m_watch;
    std::string              m_current;
    std::vector<std::string> m_names;
    std::vector<double>      m_local;
};

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
inline double rssPeakBytes()
{
    std::ifstream f("/proc/self/status");
    if (!f.good()) return 0.0;
    std::string key;
    while (f >> key)
    {
        if ("VmHWM:" == key)
        {
            double kb = 0;
            f >> kb;
            return kb * 1024.0;
        }
        std::getline(f, key); // skip the rest of the line
    }
    return 0.0;
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
