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

    The sinks are called from all OpenMP threads of the assembler and
    serialize the PETSc calls with a critical section.

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <petscmat.h>
#include <algorithm>

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
class gsPetscSystemSink
{
public:
    gsPetscSystemSink(Mat A, Vec b, const gsVector<index_t> * perm = nullptr)
    : m_A(A), m_b(b), m_perm(perm), m_offRank(0)
    {
        // gsMatrix blocks are column-major
        PetscCallAbort(PETSC_COMM_SELF, MatSetOption(m_A, MAT_ROW_ORIENTED, PETSC_FALSE));
        // non-free dofs are passed as -1 (AIJ ignores them by default, Vec does not)
        PetscCallAbort(PETSC_COMM_SELF, VecSetOption(m_b, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE));
        PetscCallAbort(PETSC_COMM_SELF, MatGetOwnershipRange(m_A, &m_rs, &m_re));
    }

    /// Number of matrix entries passed so far in rows owned by other ranks
    /// (these go through the PETSc stash)
    PetscCount offRankEntries() const { return m_offRank; }

    template<class T>
    void addMatrix(const gsVector<index_t> & rows, const gsVector<index_t> & cols,
                   const gsMatrix<T> & block)
    {
#       pragma omp critical (gsPetscSink)
        {
            petsc_sink_detail::mapIndices(rows, m_perm, m_r);
            petsc_sink_detail::mapIndices(cols, m_perm, m_c);
            for (PetscInt r : m_r)
                if (r >= 0 && (r < m_rs || r >= m_re)) m_offRank += m_c.size();
            PetscCallAbort(PETSC_COMM_SELF,
                MatSetValues(m_A, m_r.size(), m_r.data(), m_c.size(), m_c.data(),
                             block.data(), ADD_VALUES));
        }
    }

    template<class T>
    void addRhs(const gsVector<index_t> & rows, const gsMatrix<T> & block)
    {
        GISMO_ASSERT(1 == block.cols(), "Only a single right-hand side is supported.");
#       pragma omp critical (gsPetscSink)
        {
            petsc_sink_detail::mapIndices(rows, m_perm, m_r);
            PetscCallAbort(PETSC_COMM_SELF,
                VecSetValues(m_b, m_r.size(), m_r.data(), block.data(), ADD_VALUES));
        }
    }

    /// Communicates the off-rank contributions (collective)
    PetscErrorCode assembly()
    {
        PetscFunctionBeginUser;
        PetscCall( MatAssemblyBegin(m_A, MAT_FINAL_ASSEMBLY) );
        PetscCall( VecAssemblyBegin(m_b) );
        PetscCall( MatAssemblyEnd  (m_A, MAT_FINAL_ASSEMBLY) );
        PetscCall( VecAssemblyEnd  (m_b) );
        PetscFunctionReturn(PETSC_SUCCESS);
    }

private:
    Mat m_A;
    Vec m_b;
    const gsVector<index_t> * m_perm;
    PetscInt m_rs, m_re;
    PetscCount m_offRank;
    std::vector<PetscInt> m_r, m_c;
};

} // namespace gismo
