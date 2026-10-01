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

/// @brief Collects the sparsity pattern in a MATPREALLOCATOR matrix and
/// preallocates the target matrix with it.
class gsPetscPatternSink
{
public:
    /// @param nLocal number of rows (=cols) owned by this rank, or PETSC_DECIDE
    gsPetscPatternSink(MPI_Comm comm, PetscInt N, PetscInt nLocal = PETSC_DECIDE,
                       const gsVector<index_t> * perm = nullptr)
    : m_perm(perm)
    {
        PetscCallAbort(comm, MatCreate(comm, &m_pre));
        PetscCallAbort(comm, MatSetSizes(m_pre, nLocal, nLocal, N, N));
        PetscCallAbort(comm, MatSetType(m_pre, MATPREALLOCATOR));
        PetscCallAbort(comm, MatSetUp(m_pre));
    }

    ~gsPetscPatternSink() { MatDestroy(&m_pre); }

    void addPattern(const gsVector<index_t> & rows, const gsVector<index_t> & cols)
    {
#       pragma omp critical (gsPetscSink)
        {
            // MATPREALLOCATOR does not accept negative (ignored) indices
            petsc_sink_detail::mapIndices(rows, m_perm, m_r);
            petsc_sink_detail::mapIndices(cols, m_perm, m_c);
            m_r.erase(std::remove(m_r.begin(), m_r.end(), -1), m_r.end());
            m_c.erase(std::remove(m_c.begin(), m_c.end(), -1), m_c.end());
            m_zeros.assign(m_r.size() * m_c.size(), 0.0);
            PetscCallAbort(PETSC_COMM_SELF,
                MatSetValues(m_pre, m_r.size(), m_r.data(), m_c.size(), m_c.data(),
                             m_zeros.data(), INSERT_VALUES));
        }
    }

    /// Creates the AIJ matrix \a A with the same layout, preallocated
    /// exactly with the collected pattern
    PetscErrorCode createMatrix(Mat & A)
    {
        PetscFunctionBeginUser;
        MPI_Comm comm;
        PetscInt m, n, M, N;
        PetscCall( PetscObjectGetComm((PetscObject)m_pre, &comm) );
        PetscCall( MatAssemblyBegin(m_pre, MAT_FINAL_ASSEMBLY) );
        PetscCall( MatAssemblyEnd  (m_pre, MAT_FINAL_ASSEMBLY) );
        PetscCall( MatGetLocalSize(m_pre, &m, &n) );
        PetscCall( MatGetSize(m_pre, &M, &N) );
        PetscCall( MatCreate(comm, &A) );
        PetscCall( MatSetSizes(A, m, n, M, N) );
        PetscCall( MatSetType(A, MATAIJ) );
        PetscCall( MatSetFromOptions(A) );
        PetscCall( MatPreallocatorPreallocate(m_pre, PETSC_TRUE, A) );
        PetscCall( MatDestroy(&m_pre) );
        PetscFunctionReturn(PETSC_SUCCESS);
    }

private:
    Mat m_pre;
    const gsVector<index_t> * m_perm;
    std::vector<PetscInt> m_r, m_c;
    std::vector<PetscScalar> m_zeros;
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
