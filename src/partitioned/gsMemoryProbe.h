/** @file gsMemoryProbe.h

    @brief Per-rank heap/RSS probes and helpers that report the memory
    footprint of G+Smo objects across MPI ranks.

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <gsCore/gsDofMapper.h>
#include <gsCore/gsMultiPatch.h>
#include <gsCore/gsMultiBasis.h>
#include <gsMatrix/gsFiberMatrix.h>
#include <mpi.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace gismo
{

namespace memprobe
{

/// Reads a "VmXXX:  1234 kB" entry of /proc/self/status (bytes, -1 if unavailable)
inline long long procStatus(const char * key)
{
    std::ifstream f("/proc/self/status");
    std::string line;
    const size_t n = std::strlen(key);
    while (std::getline(f, line))
        if (0 == line.compare(0, n, key))
        {
            std::istringstream is(line.substr(n + 1));
            long long kb = -1;
            is >> kb;
            return kb * 1024;
        }
    return -1;
}

inline long long rssBytes()     { return procStatus("VmRSS"); }
inline long long peakRssBytes() { return procStatus("VmHWM"); }

/// The three parts of VmRSS (Linux >= 4.5): RssAnon is private memory (heap,
/// stacks, anonymous mmap); RssFile is mapped files, mostly shared libraries,
/// counted in full by every process that maps them although physically
/// shared; RssShmem is shared memory, e.g. the MPI shared-memory transport.
inline long long rssAnonBytes()  { return procStatus("RssAnon"); }
inline long long rssFileBytes()  { return procStatus("RssFile"); }
inline long long rssShmemBytes() { return procStatus("RssShmem"); }

/// Resets VmHWM to the current VmRSS (writes "5" to /proc/self/clear_refs,
/// Linux >= 4.0). Returns false if the kernel refused, in which case VmHWM
/// keeps growing from process start.
inline bool resetPeakRss()
{
    std::ofstream f("/proc/self/clear_refs");
    if (!f) return false;
    f << "5";
    f.close();
    return !f.fail();
}

/// Bytes currently allocated on the heap by this process (malloc + mmap).
/// Falls back to the resident set size if mallinfo2 is unavailable.
inline long long heapBytes()
{
#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 33))
    const struct mallinfo2 mi = mallinfo2();
    return static_cast<long long>(mi.uordblks + mi.hblkhd);
#else
    return rssBytes();
#endif
}


// ---------------------------------------------------------------------------
// Size estimators: bytes held by the dynamic storage of G+Smo objects.
// Only the dominant (size-dependent) storage is counted.
// ---------------------------------------------------------------------------

template<class Derived>
inline long long bytesOf(const gsEigen::DenseBase<Derived> & m)
{ return static_cast<long long>(m.size()) * sizeof(typename Derived::Scalar); }

template<class T, int Major>
inline long long bytesOf(const gsFiberMatrix<T,Major> & m)
{
    // one heap-allocated sparse vector object per fiber + its reserved storage
    // one pointer per fiber, plus a heap-allocated sparse vector and its
    // storage for every allocated fiber
    long long b = m.fibers() * sizeof(void*);
    for (index_t i = 0; i != m.fibers(); ++i)
        if (m.isAllocated(i))
            b += sizeof(typename gsFiberMatrix<T,Major>::Fiber) +
                 m.fiber(i).data().allocatedSize() * (sizeof(T) + sizeof(typename gsFiberMatrix<T,Major>::Fiber::StorageIndex));
    return b;
}

template<class T, int Major>
inline long long bytesOf(const gsSparseMatrix<T,Major> & m)
{
    long long b = (m.outerSize() + 1) * sizeof(typename gsSparseMatrix<T,Major>::StorageIndex);
    b += m.data().allocatedSize() * (sizeof(T) + sizeof(typename gsSparseMatrix<T,Major>::StorageIndex));
    return b;
}

/// Capacity-based byte count of gsDofMapper::nBytes(), valid in both storage
/// modes.
inline long long bytesOf(const gsDofMapper & m)
{
    return static_cast<long long>(m.nBytes());
}

template<class T>
inline long long bytesOf(const gsMultiPatch<T> & mp)
{
    long long b = 0;
    for (size_t p = 0; p != mp.nPatches(); ++p)
    {
        b += bytesOf(mp.patch(p).coefs());
    }
    b += (mp.nBoundary() + mp.nInterfaces()) * sizeof(boundaryInterface);
    return b;
}

// ---------------------------------------------------------------------------
// Ledger: named measurements, reduced over all ranks and printed by rank 0
// ---------------------------------------------------------------------------

struct Entry
{
    enum Kind { stage, object, count };
    std::string name;
    long long   value;   // stage: heap delta [B]; object: size estimate [B]; count: a number
    Kind        kind;
    double      seconds; // stage: wall time (stages are separated by barriers)
    long long   rss;     // stage: VmRSS at the end of the stage [B]
    long long   peakRss; // stage: VmHWM since the previous stage [B] (see Ledger::peakResets)
};

/**
   Ledger of named per-rank measurements, reduced over all ranks (min / max /
   sum) and printed by rank 0.

   The constructor records the process state at that point as objects (VmRSS
   and its RssAnon / RssFile / RssShmem parts, heap, and VmHWM so far);
   construct the ledger right after MPI and PETSc initialization so that these
   are the start-up baseline that no stage accounts for.

   Every stage() records three memory figures: the heap delta (mallinfo2,
   what the stage still holds at its end), VmRSS at its end, and VmHWM over
   the stage. VmHWM is reset after every stage, so the per-stage peak also
   sees transient allocations that are freed before the stage ends. If the
   kernel refuses the reset, VmHWM is cumulative from process start;
   peakResets() then returns false and the report says so.
*/
class Ledger
{
public:
    explicit Ledger(MPI_Comm comm) : m_comm(comm), m_last(heapBytes()), m_peakMax(peakRssBytes())
    {
        MPI_Comm_rank(comm, &m_rank);
        MPI_Comm_size(comm, &m_size);
        m_base = m_last;
        object("RSS at ledger start (after MPI + PETSc init)", rssBytes());
        object("  RssAnon at ledger start", rssAnonBytes());
        object("  RssFile at ledger start (libraries)", rssFileBytes());
        object("  RssShmem at ledger start (shared memory)", rssShmemBytes());
        object("  heap at ledger start", m_base);
        object("peak RSS before ledger start (init)", m_peakMax);
        m_resetOk = resetPeakRss();
        m_lastTime = MPI_Wtime();
    }

    /// Records the heap growth, the RSS and the peak RSS since the previous
    /// stage (or construction). The /proc reads and the peak reset happen
    /// between the two clock readings and are not charged to any stage.
    void stage(const std::string & name)
    {
        MPI_Barrier(m_comm);
        const double t = MPI_Wtime();
        const long long now = heapBytes(), rss = rssBytes(), peak = peakRssBytes();
        m_peakMax = std::max(m_peakMax, peak);
        m_entries.push_back({name, now - m_last, Entry::stage, t - m_lastTime, rss, peak});
        m_last = now;
        m_resetOk = resetPeakRss() && m_resetOk;
        m_lastTime = MPI_Wtime();
    }

    /// Records an object size estimate [B]
    void object(const std::string & name, long long bytes)
    { m_entries.push_back({name, bytes, Entry::object, 0.0, 0, 0}); }

    /// Records a plain number, e.g. local elements or owned rows of this rank
    void count(const std::string & name, long long value)
    { m_entries.push_back({name, value, Entry::count, 0.0, 0, 0}); }

    long long heapSinceStart() const { return heapBytes() - m_base; }

    /// True if every VmHWM reset so far succeeded on this rank
    bool peakResets() const { return m_resetOk; }

    /// Prints min / max / sum over ranks for every entry (rank 0 only).
    /// Collective: every rank must call it.
    void report(std::ostream & os, const std::string & tag) const
    {
        const Reduced r = reduce();
        if (0 != m_rank) return;

        const auto mb = [](long long b) { return static_cast<double>(b) / (1024.*1024.); };
        os << "\n" << std::left << std::setw(46) << ("[" + tag + "] (MiB, over "
              + std::to_string(m_size) + " ranks)")
           << std::right << std::setw(10) << "min" << std::setw(10) << "max"
           << std::setw(10) << "sum" << std::setw(8) << "max/sum" << std::setw(9) << "time[s]"
           << std::setw(10) << "rss max" << std::setw(10) << "hwm max" << "\n";
        const size_t n = m_entries.size();
        for (size_t i = 0; i != n; ++i)
        {
            const Entry & e = m_entries[i];
            const double ratio = r.sum[i] != 0 ? static_cast<double>(r.max[i]) / r.sum[i] : 0.;
            const char * label = Entry::stage == e.kind ? "  stage  "
                               : Entry::object == e.kind ? "  object " : "  count  ";
            os << std::left << std::setw(46) << (label + e.name) << std::right;
            if (Entry::count == e.kind)
                os << std::setw(10) << r.min[i] << std::setw(10) << r.max[i]
                   << std::setw(10) << r.sum[i];
            else
                os << std::fixed << std::setprecision(2)
                   << std::setw(10) << mb(r.min[i]) << std::setw(10) << mb(r.max[i])
                   << std::setw(10) << mb(r.sum[i]);
            os << std::fixed << std::setprecision(2) << std::setw(8) << ratio;
            if (Entry::stage == e.kind)
                os << std::setw(9) << std::setprecision(3) << e.seconds
                   << std::setprecision(2) << std::setw(10) << mb(r.rssMax[i])
                   << std::setw(10) << mb(r.peakMax[i]);
            os << "\n";
        }
        os << "  peak RSS (max over ranks): " << mb(r.peakOverall) << " MiB\n";
        if (!r.resetOk)
            os << "  note: VmHWM reset refused on some rank; \"hwm max\" is cumulative there\n";
        os.unsetf(std::ios::fixed);
    }

    /**
       CSV lines (rank 0 only), each `tag,name,kind,min,max,sum`. Per stage:
       kind `stage` (heap delta [B]), `time` (wall time [us], same value in
       all three fields), `rss` (VmRSS at the stage end [B]) and `peakrss`
       (VmHWM over the stage [B]). Objects have kind `object` [B], counts
       kind `count`. A last line `peak RSS (whole run),peakrss,...` holds the
       run's peak and, if any VmHWM reset failed, a line `VmHWM reset
       failed,count,...` follows. Collective: every rank must call it.
    */
    void csv(std::ostream & os, const std::string & tag) const
    {
        const Reduced r = reduce();
        if (0 != m_rank) return;
        const size_t n = m_entries.size();
        for (size_t i = 0; i != n; ++i)
        {
            const Entry & e = m_entries[i];
            const char * kind = Entry::stage == e.kind ? "stage"
                              : Entry::object == e.kind ? "object" : "count";
            os << tag << "," << e.name << "," << kind << ","
               << r.min[i] << "," << r.max[i] << "," << r.sum[i] << "\n";
            if (Entry::stage == e.kind)
            {
                const long long us = static_cast<long long>(1e6 * e.seconds);
                os << tag << "," << e.name << ",time,"
                   << us << "," << us << "," << us << "\n";
                os << tag << "," << e.name << ",rss,"
                   << r.rssMin[i] << "," << r.rssMax[i] << "," << r.rssSum[i] << "\n";
                os << tag << "," << e.name << ",peakrss,"
                   << r.peakMin[i] << "," << r.peakMax[i] << "," << r.peakSum[i] << "\n";
            }
        }
        os << tag << ",peak RSS (whole run),peakrss," << r.peakOverallMin << ","
           << r.peakOverall << "," << r.peakOverallSum << "\n";
        if (!r.resetOk)
            os << tag << ",VmHWM reset failed,count,1,1,1\n";
    }

private:
    /// Min / max / sum over ranks of every entry value, and of the stage RSS
    /// and peak RSS; valid on rank 0.
    struct Reduced
    {
        std::vector<long long> min, max, sum, rssMin, rssMax, rssSum, peakMin, peakMax, peakSum;
        long long peakOverallMin, peakOverall, peakOverallSum;
        bool resetOk;
    };

    Reduced reduce() const
    {
        const int n = static_cast<int>(m_entries.size());
        // block layout: [value | rss | peakRss | run peak, reset ok] per rank
        std::vector<long long> v(3 * n + 2), vmin(3 * n + 2), vmax(3 * n + 2), vsum(3 * n + 2);
        for (int i = 0; i != n; ++i)
        {
            v[i]         = m_entries[i].value;
            v[n + i]     = m_entries[i].rss;
            v[2 * n + i] = m_entries[i].peakRss;
        }
        v[3 * n]     = std::max(m_peakMax, peakRssBytes());
        v[3 * n + 1] = m_resetOk ? 1 : 0;
        MPI_Reduce(v.data(), vmin.data(), 3 * n + 2, MPI_LONG_LONG, MPI_MIN, 0, m_comm);
        MPI_Reduce(v.data(), vmax.data(), 3 * n + 2, MPI_LONG_LONG, MPI_MAX, 0, m_comm);
        MPI_Reduce(v.data(), vsum.data(), 3 * n + 2, MPI_LONG_LONG, MPI_SUM, 0, m_comm);

        Reduced r;
        const auto part = [n](const std::vector<long long> & x, int k)
        { return std::vector<long long>(x.begin() + k * n, x.begin() + (k + 1) * n); };
        r.min = part(vmin, 0);     r.max = part(vmax, 0);     r.sum = part(vsum, 0);
        r.rssMin = part(vmin, 1);  r.rssMax = part(vmax, 1);  r.rssSum = part(vsum, 1);
        r.peakMin = part(vmin, 2); r.peakMax = part(vmax, 2); r.peakSum = part(vsum, 2);
        r.peakOverallMin = vmin[3 * n];
        r.peakOverall    = vmax[3 * n];
        r.peakOverallSum = vsum[3 * n];
        r.resetOk        = 1 == vmin[3 * n + 1];
        return r;
    }

    MPI_Comm m_comm;
    int m_rank, m_size;
    long long m_base, m_last;
    long long m_peakMax; // largest VmHWM seen so far (VmHWM itself is reset per stage)
    bool m_resetOk;
    double m_lastTime;
    std::vector<Entry> m_entries;
};

} // namespace memprobe
} // namespace gismo
