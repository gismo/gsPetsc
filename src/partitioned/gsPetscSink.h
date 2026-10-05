/** @file gsPetscSink.h

    @brief Sinks for gsExprAssembler::computePattern_into() and
    gsExprAssembler::assemble_into() that write element blocks directly
    into distributed PETSc objects. No gismo-side matrix or right-hand
    side is allocated.

    Indices delivered by the assembler are global indices of its dof
    mappers, -1 for dofs that are not free. They are optionally mapped
    through a permutation (e.g. gsPartitionedDofMapper::permutation()),
    which defines the PETSc row ownership. Negative indices are ignored
    by PETSc.

    The sinks are called from all OpenMP threads of the assembler. The
    system sink buffers per thread and passes the buffers to PETSc in a
    critical section; the pattern sink serializes every call.

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <petscmat.h>
#include <algorithm>
#include <climits>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace gismo
{

namespace petsc_sink_detail
{
inline void mapIndices(const gsVector<index_t> & idx, const gsVector<index_t> * perm,
                       std::vector<PetscInt> & out)
{
    out.resize(idx.size());
    for (index_t i = 0; i != idx.size(); ++i)
        out[i] = (idx[i] < 0) ? -1 : static_cast<PetscInt>(perm ? (*perm)[idx[i]] : idx[i]);
}
}

/// @brief Collects the sparsity pattern of the assembled blocks and
/// preallocates an AIJ matrix exactly with it.
///
/// The index blocks of all addPattern() calls are stored (E calls with
/// blocks of size k use O(E k) indices). createMatrix() counts the
/// distinct columns of every owned row, split into the diagonal block
/// (columns owned by this rank) and the off-diagonal block, and sends the
/// (row, column) pairs of rows owned by other ranks to their owners. The
/// matrix is then preallocated with MatXAIJSetPreallocation(), so the
/// number of allocated slots equals the number of nonzeros and no
/// reallocation (malloc) occurs during assembly. The layout is square and
/// is the PETSc default layout (PetscSplitOwnership) for \a nLocal.
class gsPetscPatternSink
{
public:
    /// @param comm communicator of the matrix
    /// @param N global number of rows (=cols)
    /// @param nLocal number of rows (=cols) owned by this rank, or PETSC_DECIDE
    /// @param perm optional map from assembler dof index to PETSc row
    gsPetscPatternSink(MPI_Comm comm, PetscInt N, PetscInt nLocal = PETSC_DECIDE,
                       const gsVector<index_t> * perm = nullptr)
    : m_comm(comm), m_N(N), m_nLocal(nLocal), m_perm(perm), m_created(false)
    {
        m_rowOff.push_back(0);
        m_colOff.push_back(0);
    }

    /// Stores the index blocks of one element (thread-safe). Negative
    /// indices (non-free dofs) are dropped.
    void addPattern(const gsVector<index_t> & rows, const gsVector<index_t> & cols)
    {
#       pragma omp critical (gsPetscSink)
        {
            petsc_sink_detail::mapIndices(rows, m_perm, m_r);
            petsc_sink_detail::mapIndices(cols, m_perm, m_c);
            m_r.erase(std::remove(m_r.begin(), m_r.end(), -1), m_r.end());
            m_c.erase(std::remove(m_c.begin(), m_c.end(), -1), m_c.end());
            if (!m_r.empty() && !m_c.empty())
            {
                m_rowIdx.insert(m_rowIdx.end(), m_r.begin(), m_r.end());
                m_colIdx.insert(m_colIdx.end(), m_c.begin(), m_c.end());
                m_rowOff.push_back(m_rowIdx.size());
                m_colOff.push_back(m_colIdx.size());
            }
        }
    }

    /// Bytes reserved by the stored index blocks (indices and offsets).
    /// They are released by createMatrix().
    std::size_t storedBytes() const
    {
        return (m_rowIdx.capacity() + m_colIdx.capacity()) * sizeof(PetscInt)
             + (m_rowOff.capacity() + m_colOff.capacity()) * sizeof(std::size_t);
    }

    /// Creates the AIJ matrix \a A with the same layout, preallocated
    /// exactly with the collected pattern. Collective; the sink can be
    /// used once.
    ///
    /// The checks on rank-local data (stored indices in [0, N), MPI count
    /// range, received rows owned, matrix layout) are agreed on by all
    /// ranks before the next collective call: if one fails anywhere, every
    /// rank throws, instead of the healthy ranks waiting in that call.
    ///
    /// Time O(E k^2 log k) plus O(number of received pairs); memory
    /// O(E k) stored indices plus O(r k) pairs for the r stored row indices
    /// owned by other ranks (worst case O(E k^2)), released before the
    /// matrix is allocated.
    PetscErrorCode createMatrix(Mat & A)
    {
        PetscFunctionBeginUser;
        GISMO_ENSURE(!m_created, "gsPetscPatternSink::createMatrix: the sink is single-use");
        m_created = true;

        // Ownership: contiguous ranges [rs,re), the layout of MatSetSizes
        PetscInt n = m_nLocal, Nglob = m_N;
        PetscCall( PetscSplitOwnership(m_comm, &n, &Nglob) );
        PetscMPIInt size;
        PetscCallMPI( MPI_Comm_size(m_comm, &size) );
        PetscInt re = 0;
        PetscCallMPI( MPI_Scan(&n, &re, 1, MPIU_INT, MPI_SUM, m_comm) );
        const PetscInt rs = re - n;
        std::vector<PetscInt> ends(size);
        PetscCallMPI( MPI_Allgather(&re, 1, MPIU_INT, ends.data(), 1, MPIU_INT, m_comm) );

        const std::size_t numCalls = m_rowOff.size() - 1;
        const bool callsOk = numCalls < static_cast<std::size_t>(INT_MAX);
        // An index outside [0, N) can only come from a broken perm; a row
        // index >= N would select the owner ends.size(), out of bounds.
        bool indicesOk = true;
        for (std::size_t k = 0; k != m_rowIdx.size(); ++k)
            indicesOk = indicesOk && m_rowIdx[k] >= 0 && m_rowIdx[k] < m_N;
        for (std::size_t k = 0; k != m_colIdx.size(); ++k)
            indicesOk = indicesOk && m_colIdx[k] >= 0 && m_colIdx[k] < m_N;

        // Owned rows: calls touching each row (counting sort)
        std::vector<std::size_t> rowPtr;
        rowPtr.reserve(n + 1);
        rowPtr.assign(n + 1, 0);
        for (std::size_t k = 0; k != m_rowIdx.size(); ++k)
            if (m_rowIdx[k] >= rs && m_rowIdx[k] < re)
                ++rowPtr[m_rowIdx[k] - rs + 1];
        for (PetscInt i = 0; i != n; ++i)
            rowPtr[i + 1] += rowPtr[i];
        std::vector<PetscInt> rowCalls;
        rowCalls.reserve(rowPtr[n]);
        rowCalls.resize(rowPtr[n]);
        {
            std::vector<std::size_t> cur(rowPtr.begin(), rowPtr.end() - 1);
            for (std::size_t j = 0; j != numCalls; ++j)
                for (std::size_t k = m_rowOff[j]; k != m_rowOff[j + 1]; ++k)
                    if (m_rowIdx[k] >= rs && m_rowIdx[k] < re)
                        rowCalls[cur[m_rowIdx[k] - rs]++] = static_cast<PetscInt>(j);
        }

        // Rows owned by other ranks: unique (row, col) pairs, sorted by
        // row and therefore grouped by owner (owner ranges are increasing)
        typedef std::pair<PetscInt,PetscInt> Pair;
        std::vector<Pair> pairs;
        {
            std::size_t cnt = 0;
            for (std::size_t j = 0; j != numCalls; ++j)
                for (std::size_t k = m_rowOff[j]; k != m_rowOff[j + 1]; ++k)
                    if (m_rowIdx[k] < rs || m_rowIdx[k] >= re)
                        cnt += m_colOff[j + 1] - m_colOff[j];
            pairs.reserve(cnt);
            for (std::size_t j = 0; j != numCalls; ++j)
                for (std::size_t k = m_rowOff[j]; k != m_rowOff[j + 1]; ++k)
                    if (m_rowIdx[k] < rs || m_rowIdx[k] >= re)
                        for (std::size_t l = m_colOff[j]; l != m_colOff[j + 1]; ++l)
                            pairs.push_back(Pair(m_rowIdx[k], m_colIdx[l]));
        }
        std::sort(pairs.begin(), pairs.end());
        pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());

        // MPI counts are int: count in size_t, narrow only after the check
        std::vector<int> sendCnt(size, 0), sendDsp(size, 0), recvCnt(size), recvDsp(size, 0);
        std::vector<PetscInt> sendBuf;
        bool sendOk = true;
        {
            std::vector<std::size_t> cnt(size, 0);
            sendBuf.reserve(2 * pairs.size());
            for (std::size_t q = 0; q != pairs.size(); ++q)
            {
                if (pairs[q].first < 0 || pairs[q].first >= m_N)
                    continue; // already flagged by indicesOk
                const PetscMPIInt dest = static_cast<PetscMPIInt>(
                    std::upper_bound(ends.begin(), ends.end(), pairs[q].first) - ends.begin());
                cnt[dest] += 2;
                sendBuf.push_back(pairs[q].first);
                sendBuf.push_back(pairs[q].second);
            }
            std::size_t tot = 0;
            for (PetscMPIInt p = 0; p != size; ++p)
            {
                sendDsp[p] = static_cast<int>(tot);
                tot += cnt[p];
                sendOk = sendOk && tot <= static_cast<std::size_t>(INT_MAX);
                sendCnt[p] = static_cast<int>(cnt[p]);
            }
        }
        std::vector<Pair>().swap(pairs);
        ensureOnAllRanks(!callsOk   ? "too many stored blocks (> INT_MAX)"
                       : !indicesOk ? "a stored row or column index is outside [0, N): broken perm?"
                       : !sendOk    ? "pair exchange exceeds the MPI count range (send)"
                       : nullptr);

        PetscCallMPI( MPI_Alltoall(sendCnt.data(), 1, MPI_INT, recvCnt.data(), 1, MPI_INT, m_comm) );
        {
            std::size_t tot = 0;
            bool recvOk = true;
            for (PetscMPIInt p = 0; p != size; ++p)
            {
                recvDsp[p] = static_cast<int>(tot);
                tot += recvCnt[p];
                recvOk = recvOk && tot <= static_cast<std::size_t>(INT_MAX);
            }
            ensureOnAllRanks(recvOk ? nullptr : "pair exchange exceeds the MPI count range (receive)");
        }
        std::size_t totRecv = static_cast<std::size_t>(recvDsp[size - 1]) + recvCnt[size - 1];
        std::vector<PetscInt> recvBuf(totRecv);
        PetscCallMPI( MPI_Alltoallv(sendBuf.data(), sendCnt.data(), sendDsp.data(), MPIU_INT,
                                    recvBuf.data(), recvCnt.data(), recvDsp.data(), MPIU_INT, m_comm) );
        std::vector<PetscInt>().swap(sendBuf);

        // Received pairs by owned row (counting sort)
        std::vector<std::size_t> recvPtr;
        recvPtr.reserve(n + 1);
        recvPtr.assign(n + 1, 0);
        bool ownedOk = true;
        for (std::size_t q = 0; q != totRecv; q += 2)
        {
            if (recvBuf[q] >= rs && recvBuf[q] < re)
                ++recvPtr[recvBuf[q] - rs + 1];
            else
                ownedOk = false;
        }
        ensureOnAllRanks(ownedOk ? nullptr : "received a row that is not owned");
        for (PetscInt i = 0; i != n; ++i)
            recvPtr[i + 1] += recvPtr[i];
        std::vector<PetscInt> recvCols;
        recvCols.reserve(totRecv / 2);
        recvCols.resize(totRecv / 2);
        {
            std::vector<std::size_t> cur(recvPtr.begin(), recvPtr.end() - 1);
            for (std::size_t q = 0; q != totRecv; q += 2)
                recvCols[cur[recvBuf[q] - rs]++] = recvBuf[q + 1];
        }
        std::vector<PetscInt>().swap(recvBuf);

        // Distinct columns of every owned row: diagonal / off-diagonal counts
        std::vector<PetscInt> d_nnz, o_nnz, scratch;
        d_nnz.reserve(n);
        o_nnz.reserve(n);
        d_nnz.assign(n, 0);
        o_nnz.assign(n, 0);
        for (PetscInt i = 0; i != n; ++i)
        {
            scratch.clear();
            for (std::size_t t = rowPtr[i]; t != rowPtr[i + 1]; ++t)
            {
                const std::size_t j = rowCalls[t];
                scratch.insert(scratch.end(), m_colIdx.begin() + m_colOff[j],
                                              m_colIdx.begin() + m_colOff[j + 1]);
            }
            scratch.insert(scratch.end(), recvCols.begin() + recvPtr[i],
                                          recvCols.begin() + recvPtr[i + 1]);
            std::sort(scratch.begin(), scratch.end());
            scratch.erase(std::unique(scratch.begin(), scratch.end()), scratch.end());
            for (std::size_t t = 0; t != scratch.size(); ++t)
                (scratch[t] >= rs && scratch[t] < re ? d_nnz[i] : o_nnz[i])++;
        }

        // Release the stored blocks before the matrix is allocated
        std::vector<PetscInt>().swap(m_rowIdx);
        std::vector<PetscInt>().swap(m_colIdx);
        std::vector<std::size_t>().swap(m_rowOff);
        std::vector<std::size_t>().swap(m_colOff);
        std::vector<PetscInt>().swap(m_r);
        std::vector<PetscInt>().swap(m_c);
        std::vector<std::size_t>().swap(rowPtr);
        std::vector<PetscInt>().swap(rowCalls);
        std::vector<std::size_t>().swap(recvPtr);
        std::vector<PetscInt>().swap(recvCols);
        std::vector<PetscInt>().swap(scratch);

        PetscCall( MatCreate(m_comm, &A) );
        PetscCall( MatSetSizes(A, n, n, m_N, m_N) );
        PetscCall( MatSetType(A, MATAIJ) );
        PetscCall( MatSetFromOptions(A) );
        PetscBool isAIJ;
        PetscCall( PetscObjectTypeCompareAny((PetscObject)A, &isAIJ, MATSEQAIJ, MATMPIAIJ, "") );
        GISMO_ENSURE(isAIJ, "gsPetscPatternSink: d/o counts are scalar AIJ counts; "
                            "-mat_type other than aij is not supported");
        PetscCall( MatXAIJSetPreallocation(A, 1, d_nnz.data(), o_nnz.data(), NULL, NULL) );
        PetscCall( MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE) );
        PetscInt rs2, re2;
        PetscCall( MatGetOwnershipRange(A, &rs2, &re2) );
        ensureOnAllRanks(rs2 == rs && re2 == re ? nullptr
                         : "matrix layout differs from the sink layout");
        PetscFunctionReturn(PETSC_SUCCESS);
    }

private:
    /// Makes a rank-local check collective: \a failure is null on the ranks
    /// where the check passed. If it is non-null on any rank, that rank
    /// names it on gsWarn and every rank throws. A plain GISMO_ENSURE would
    /// throw on the failing ranks only and leave the others waiting in the
    /// next collective call (under srun they are not killed: the job hangs
    /// until its time limit). Collective; one MPI_Allreduce of one int.
    void ensureOnAllRanks(const char * failure) const
    {
        int bad = failure ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, m_comm);
        if (failure)
        {
            int r = 0;
            MPI_Comm_rank(m_comm, &r);
            gsWarn << "gsPetscPatternSink::createMatrix, rank " << r << ": " << failure << "\n";
        }
        GISMO_ENSURE(0 == bad, "gsPetscPatternSink::createMatrix: a check failed on at least "
                               "one rank, see the warning of that rank");
    }

    MPI_Comm m_comm;
    PetscInt m_N, m_nLocal;
    const gsVector<index_t> * m_perm;
    bool m_created;
    // Stored blocks (CSR of calls): call j has rows m_rowIdx[m_rowOff[j]..m_rowOff[j+1])
    // and cols m_colIdx[m_colOff[j]..m_colOff[j+1]); offsets are size_t as E k
    // can exceed the 32-bit PetscInt range
    std::vector<PetscInt> m_rowIdx, m_colIdx;
    std::vector<std::size_t> m_rowOff, m_colOff;
    std::vector<PetscInt> m_r, m_c;
};

/// @brief Adds element blocks to a PETSc matrix and right-hand side vector
/// (ADD_VALUES; entries of rows owned by other ranks are communicated at
/// assembly()).
///
/// Every OpenMP thread collects its element blocks in its own buffer; a
/// buffer is passed to PETSc (under a critical section, PETSc is not
/// thread-safe) when it exceeds flushBytes and in assembly().
class gsPetscSystemSink
{
    struct Buffer
    {
        std::vector<PetscInt>    mIdx;  // per block: nr, nc, rows, cols
        std::vector<PetscScalar> mVal;  // per block: nr*nc values (column-major)
        std::vector<PetscInt>    vIdx;  // rhs rows
        std::vector<PetscScalar> vVal;  // rhs values
        PetscCount offRank = 0;
        size_t bytes() const
        {
            return (mIdx.size() + vIdx.size()) * sizeof(PetscInt) +
                   (mVal.size() + vVal.size()) * sizeof(PetscScalar);
        }
    };

public:
    gsPetscSystemSink(Mat A, Vec b, const gsVector<index_t> * perm = nullptr,
                      size_t flushBytes = size_t(1) << 22)
    : m_A(A), m_b(b), m_perm(perm), m_flushBytes(flushBytes)
    {
        // gsMatrix blocks are column-major
        PetscCallAbort(PETSC_COMM_SELF, MatSetOption(m_A, MAT_ROW_ORIENTED, PETSC_FALSE));
        // non-free dofs are passed as -1 (AIJ ignores them by default, Vec does not)
        PetscCallAbort(PETSC_COMM_SELF, VecSetOption(m_b, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE));
        PetscCallAbort(PETSC_COMM_SELF, MatGetOwnershipRange(m_A, &m_rs, &m_re));
#ifdef _OPENMP
        m_buf.resize(omp_get_max_threads());
#else
        m_buf.resize(1);
#endif
    }

    /// Number of matrix entries passed so far in rows owned by other ranks
    /// (these go through the PETSc stash)
    PetscCount offRankEntries() const
    {
        PetscCount n = 0;
        for (const Buffer & b : m_buf) n += b.offRank;
        return n;
    }

    template<class T>
    void addMatrix(const gsVector<index_t> & rows, const gsVector<index_t> & cols,
                   const gsMatrix<T> & block)
    {
        Buffer & B = buffer();
        const index_t nr = rows.size(), nc = cols.size();
        B.mIdx.push_back(nr);
        B.mIdx.push_back(nc);
        for (index_t i = 0; i != nr; ++i)
        {
            const PetscInt r = map(rows[i]);
            B.mIdx.push_back(r);
            if (r >= 0 && (r < m_rs || r >= m_re)) B.offRank += nc;
        }
        for (index_t j = 0; j != nc; ++j) B.mIdx.push_back(map(cols[j]));
        B.mVal.insert(B.mVal.end(), block.data(), block.data() + nr * nc);
        if (B.bytes() > m_flushBytes) flush(B);
    }

    template<class T>
    void addRhs(const gsVector<index_t> & rows, const gsMatrix<T> & block)
    {
        GISMO_ASSERT(1 == block.cols(), "Only a single right-hand side is supported.");
        Buffer & B = buffer();
        for (index_t i = 0; i != rows.size(); ++i)
        {
            B.vIdx.push_back(map(rows[i]));
            B.vVal.push_back(block(i, 0));
        }
        if (B.bytes() > m_flushBytes) flush(B);
    }

    /// Passes the remaining buffers to PETSc and communicates the off-rank
    /// contributions (collective; call outside of parallel regions)
    PetscErrorCode assembly()
    {
        PetscFunctionBeginUser;
        for (Buffer & B : m_buf) flush(B);
        PetscCall( MatAssemblyBegin(m_A, MAT_FINAL_ASSEMBLY) );
        PetscCall( VecAssemblyBegin(m_b) );
        PetscCall( MatAssemblyEnd  (m_A, MAT_FINAL_ASSEMBLY) );
        PetscCall( VecAssemblyEnd  (m_b) );
        PetscFunctionReturn(PETSC_SUCCESS);
    }

private:
    Buffer & buffer()
    {
#ifdef _OPENMP
        return m_buf[omp_get_thread_num()];
#else
        return m_buf[0];
#endif
    }

    PetscInt map(index_t i) const
    { return (i < 0) ? -1 : static_cast<PetscInt>(m_perm ? (*m_perm)[i] : i); }

    void flush(Buffer & B)
    {
#       pragma omp critical (gsPetscSink)
        {
            const PetscScalar * v = B.mVal.data();
            for (size_t k = 0; k < B.mIdx.size(); )
            {
                const PetscInt nr = B.mIdx[k], nc = B.mIdx[k+1];
                const PetscInt * r = &B.mIdx[k+2];
                PetscCallAbort(PETSC_COMM_SELF,
                    MatSetValues(m_A, nr, r, nc, r + nr, v, ADD_VALUES));
                v += nr * nc;
                k += 2 + nr + nc;
            }
            if (!B.vIdx.empty())
                PetscCallAbort(PETSC_COMM_SELF,
                    VecSetValues(m_b, B.vIdx.size(), B.vIdx.data(), B.vVal.data(), ADD_VALUES));
        }
        B.mIdx.clear(); B.mVal.clear(); B.vIdx.clear(); B.vVal.clear();
    }

    Mat m_A;
    Vec m_b;
    const gsVector<index_t> * m_perm;
    size_t m_flushBytes;
    PetscInt m_rs, m_re;
    std::vector<Buffer> m_buf;
};

} // namespace gismo
