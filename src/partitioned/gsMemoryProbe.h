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

/// The dof mapper stores one index per (basis function, component),
/// plus a few per-patch/per-component vectors
inline long long bytesOf(const gsDofMapper & m)
{
    return static_cast<long long>(m.mapSize()) * sizeof(index_t) +
           (m.numPatches() + m.componentsSize()) * 4 * sizeof(index_t);
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
    std::string name;
    long long   value; // bytes (or a count)
    bool        isStage; // true: heap delta of a stage, false: object estimate
};

class Ledger
{
public:
    explicit Ledger(MPI_Comm comm) : m_comm(comm), m_last(heapBytes())
    {
        MPI_Comm_rank(comm, &m_rank);
        MPI_Comm_size(comm, &m_size);
        m_base = m_last;
    }

    /// Records the heap growth since the previous stage (or construction)
    void stage(const std::string & name)
    {
        MPI_Barrier(m_comm);
        const long long now = heapBytes();
        m_entries.push_back({name, now - m_last, true});
        m_last = now;
    }

    /// Records an object size estimate
    void object(const std::string & name, long long bytes)
    { m_entries.push_back({name, bytes, false}); }

    long long heapSinceStart() const { return heapBytes() - m_base; }

    /// Prints min / max / sum over ranks for every entry (rank 0 only)
    void report(std::ostream & os, const std::string & tag) const
    {
        const int n = static_cast<int>(m_entries.size());
        std::vector<long long> v(n), vmin(n), vmax(n), vsum(n);
        for (int i = 0; i != n; ++i) v[i] = m_entries[i].value;
        MPI_Reduce(v.data(), vmin.data(), n, MPI_LONG_LONG, MPI_MIN, 0, m_comm);
        MPI_Reduce(v.data(), vmax.data(), n, MPI_LONG_LONG, MPI_MAX, 0, m_comm);
        MPI_Reduce(v.data(), vsum.data(), n, MPI_LONG_LONG, MPI_SUM, 0, m_comm);

        long long peak = peakRssBytes(), peakMax = 0;
        MPI_Reduce(&peak, &peakMax, 1, MPI_LONG_LONG, MPI_MAX, 0, m_comm);
        if (0 != m_rank) return;

        const auto mb = [](long long b) { return static_cast<double>(b) / (1024.*1024.); };
        os << "\n" << std::left << std::setw(46) << ("[" + tag + "] (MiB, over "
              + std::to_string(m_size) + " ranks)")
           << std::right << std::setw(10) << "min" << std::setw(10) << "max"
           << std::setw(10) << "sum" << std::setw(8) << "max/sum" << "\n";
        for (int i = 0; i != n; ++i)
        {
            const double ratio = vsum[i] != 0 ? static_cast<double>(vmax[i]) / vsum[i] : 0.;
            os << std::left << std::setw(46)
               << ((m_entries[i].isStage ? "  stage  " : "  object ") + m_entries[i].name)
               << std::right << std::fixed << std::setprecision(2)
               << std::setw(10) << mb(vmin[i]) << std::setw(10) << mb(vmax[i])
               << std::setw(10) << mb(vsum[i]) << std::setw(8) << ratio << "\n";
        }
        os << "  peak RSS (max over ranks): " << mb(peakMax) << " MiB\n";
        os.unsetf(std::ios::fixed);
    }

    /// One CSV line per entry (rank 0 only): tag,name,kind,min,max,sum
    void csv(std::ostream & os, const std::string & tag) const
    {
        const int n = static_cast<int>(m_entries.size());
        std::vector<long long> v(n), vmin(n), vmax(n), vsum(n);
        for (int i = 0; i != n; ++i) v[i] = m_entries[i].value;
        MPI_Reduce(v.data(), vmin.data(), n, MPI_LONG_LONG, MPI_MIN, 0, m_comm);
        MPI_Reduce(v.data(), vmax.data(), n, MPI_LONG_LONG, MPI_MAX, 0, m_comm);
        MPI_Reduce(v.data(), vsum.data(), n, MPI_LONG_LONG, MPI_SUM, 0, m_comm);
        if (0 != m_rank) return;
        for (int i = 0; i != n; ++i)
            os << tag << "," << m_entries[i].name << ","
               << (m_entries[i].isStage ? "stage" : "object") << ","
               << vmin[i] << "," << vmax[i] << "," << vsum[i] << "\n";
    }

private:
    MPI_Comm m_comm;
    int m_rank, m_size;
    long long m_base, m_last;
    std::vector<Entry> m_entries;
};

} // namespace memprobe
} // namespace gismo
