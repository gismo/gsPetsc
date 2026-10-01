/** @file gsPetscCOO.h

    @brief Transfer of rank-local (element-partitioned) assembly results
    into distributed PETSc objects, using PETSc's COO interface.

    Each rank holds the contributions of its own elements, with global
    row/column indices. Contributions of different ranks to the same
    entry are summed by PETSc, and rows not owned by the rank are sent
    to their owner. No global-size gismo matrix is formed.

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <gsMatrix/gsFiberMatrix.h>
#include <petscmat.h>

namespace gismo
{

/// @brief Creates the distributed square matrix \a A from the rank-local
/// contributions stored in \a fm.
/// @param[out] nOffRank number of local entries in rows owned by other ranks
/// @param perm     optional global permutation of rows/columns
///                 (e.g. gsPartitionedDofMapper::permutation())
/// @param nLocal   rows owned by this rank (PETSC_DECIDE: equal split)
/// @param nGlobal  global number of rows (default: the size of \a fm)
template<class T, int Major>
PetscErrorCode petsc_matFromLocalFibers(const gsFiberMatrix<T,Major> & fm,
                                        MPI_Comm comm, Mat & A,
                                        PetscCount * nOffRank = nullptr,
                                        const gsVector<index_t> * perm = nullptr,
                                        PetscInt nLocal = PETSC_DECIDE,
                                        PetscInt nGlobal = PETSC_DETERMINE)
{
    PetscFunctionBeginUser;
    typedef typename gsFiberMatrix<T,Major>::Fiber Fiber;
    const bool rowMajor = (Major == RowMajor);

    PetscCall( MatCreate(comm, &A) );
    // with a rank-local numbering, fm is local and the global size must be given
    if (nGlobal < 0) nGlobal = fm.rows();
    PetscCall( MatSetSizes(A, nLocal, nLocal, nGlobal, nGlobal) );
    PetscCall( MatSetType(A, MATAIJ) );
    PetscCall( MatSetFromOptions(A) );
    PetscCall( MatSetUp(A) );

    const PetscCount nnz = fm.nonZeros();
    std::vector<PetscInt> ci, cj;
    std::vector<PetscScalar> cv;
    ci.reserve(nnz); cj.reserve(nnz); cv.reserve(nnz);
    const auto P = [perm](index_t i) { return static_cast<PetscInt>(perm ? (*perm)[i] : i); };
    for (index_t f = 0; f != fm.fibers(); ++f)
    {
        if (!fm.isAllocated(f)) continue;
        for (typename Fiber::InnerIterator it(fm.fiber(f)); it; ++it)
        {
            ci.push_back(P(rowMajor ? f : it.index()));
            cj.push_back(P(rowMajor ? it.index() : f));
            cv.push_back(it.value());
        }
    }

    if (nOffRank)
    {
        PetscInt rs, re;
        PetscCall( MatGetOwnershipRange(A, &rs, &re) );
        *nOffRank = 0;
        for (PetscInt i : ci) *nOffRank += (i < rs || i >= re);
    }

    PetscCall( MatSetPreallocationCOO(A, nnz, ci.data(), cj.data()) );
    PetscCall( MatSetValuesCOO(A, cv.data(), ADD_VALUES) );
    PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Creates a distributed vector compatible with the rows of \a A and
/// adds the rank-local contributions \a localRhs (global indexing,
/// optionally permuted, only nonzero entries are communicated).
template<class Derived>
PetscErrorCode petsc_vecFromLocalContributions(const gsEigen::MatrixBase<Derived> & localRhs,
                                               Mat A, Vec & b,
                                               const gsVector<index_t> * perm = nullptr)
{
    PetscFunctionBeginUser;
    PetscCall( MatCreateVecs(A, nullptr, &b) );
    PetscCall( VecSet(b, 0.0) );
    for (index_t i = 0; i != localRhs.rows(); ++i)
        if (0 != localRhs(i, 0))
            PetscCall( VecSetValue(b, perm ? (*perm)[i] : i, localRhs(i, 0), ADD_VALUES) );
    PetscCall( VecAssemblyBegin(b) );
    PetscCall( VecAssemblyEnd(b) );
    PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Gathers the entries \a idx (global, sorted) of the distributed
/// vector \a x into \a out (out[k] = x[idx[k]]).
template<class T>
PetscErrorCode petsc_gatherEntries(Vec x, const std::vector<PetscInt> & idx,
                                   gsMatrix<T> & out)
{
    PetscFunctionBeginUser;
    MPI_Comm comm;
    PetscCall( PetscObjectGetComm((PetscObject)x, &comm) );
    IS from;
    Vec loc;
    VecScatter sc;
    const PetscInt n = static_cast<PetscInt>(idx.size());
    PetscCall( ISCreateGeneral(PETSC_COMM_SELF, n, idx.data(), PETSC_USE_POINTER, &from) );
    PetscCall( VecCreateSeq(PETSC_COMM_SELF, n, &loc) );
    PetscCall( VecScatterCreate(x, from, loc, nullptr, &sc) );
    PetscCall( VecScatterBegin(sc, x, loc, INSERT_VALUES, SCATTER_FORWARD) );
    PetscCall( VecScatterEnd  (sc, x, loc, INSERT_VALUES, SCATTER_FORWARD) );
    const PetscScalar * a;
    out.resize(n, 1);
    PetscCall( VecGetArrayRead(loc, &a) );
    std::copy(a, a + n, out.data());
    PetscCall( VecRestoreArrayRead(loc, &a) );
    PetscCall( VecScatterDestroy(&sc) );
    PetscCall( VecDestroy(&loc) );
    PetscCall( ISDestroy(&from) );
    PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Gathers the full distributed vector \a x on every rank. This is
/// what gsFeSolution currently requires; its size is the global number
/// of dofs on every rank.
template<class T>
PetscErrorCode petsc_gatherAll(Vec x, gsMatrix<T> & out)
{
    PetscFunctionBeginUser;
    Vec all;
    VecScatter sc;
    PetscInt n;
    PetscCall( VecScatterCreateToAll(x, &sc, &all) );
    PetscCall( VecScatterBegin(sc, x, all, INSERT_VALUES, SCATTER_FORWARD) );
    PetscCall( VecScatterEnd  (sc, x, all, INSERT_VALUES, SCATTER_FORWARD) );
    PetscCall( VecGetSize(all, &n) );
    const PetscScalar * a;
    out.resize(n, 1);
    PetscCall( VecGetArrayRead(all, &a) );
    std::copy(a, a + n, out.data());
    PetscCall( VecRestoreArrayRead(all, &a) );
    PetscCall( VecScatterDestroy(&sc) );
    PetscCall( VecDestroy(&all) );
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace gismo
