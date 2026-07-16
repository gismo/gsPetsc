/** @file gsPetscLocalToGlobal.h

    @brief Insert a locally assembled gsSparseMatrix into a distributed PETSc Mat.

    gsExprAssembler::setIntegrationDomain() restricts which elements are looped
    during assembly but leaves the matrix sized at freeSize x freeSize with
    global DOF indices unchanged.  Partition matrices assembled this way can
    therefore be inserted directly into the PETSc matrix without any index
    translation — their row/column indices are already global.

    With ADD_VALUES mode each rank inserts its partition's non-zeros and PETSc
    accumulates them during MatAssemblyBegin/End.

    IMPORTANT: correct only for non-overlapping partitions.  If halo-expanded
    element sets are used, shared elements contribute to both partitions' local
    matrices and their entries would be double-counted.

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): H.M. Verhelst
*/

#pragma once

#include <petsc.h>
#include <gismo.h>

#include <vector>

namespace gismo {

/**
   @brief Insert a gsSparseMatrix into a PETSc distributed Mat.

   Iterates all stored non-zeros of @a mat and calls MatSetValue for each
   triplet.  The caller must invoke MatAssemblyBegin/MatAssemblyEnd after all
   partitions have been inserted.

   @param petscMat  Destination PETSc matrix (must already be created and sized).
   @param mat       Source sparse matrix.  Row/column indices are global DOF
                    indices (as produced by gsExprAssembler on the full space).
   @param mode      ADD_VALUES for parallel multi-partition assembly;
                    INSERT_VALUES when a single rank inserts the full matrix.

   @warning Assumes non-overlapping element partitions.  Halo-expanded sets
            introduce shared elements that are assembled by multiple partitions,
            so their contributions would be counted more than once.
*/
template<class T>
int petsc_insertSparseMatrix(Mat&                    petscMat,
                              const gsSparseMatrix<T>& mat,
                              InsertMode               mode = ADD_VALUES)
{
    // gsSparseMatrix<T> is column-major (CSC): outer index = column.
    for (index_t k = 0; k < mat.outerSize(); ++k)
        for (typename gsSparseMatrix<T>::InnerIterator it(mat, k); it; ++it)
            PetscCall( MatSetValue(petscMat,
                                   static_cast<PetscInt>(it.row()),
                                   static_cast<PetscInt>(it.col()),
                                   static_cast<PetscScalar>(it.value()),
                                   mode) );
    return 0;
}

/**
   @brief Insert a gsMatrix (single column) into a PETSc distributed Vec.

   Calls VecSetValue for each row.  The caller must invoke
   VecAssemblyBegin/VecAssemblyEnd after all partitions have been inserted.

   @param petscVec  Destination PETSc vector (must already be created and sized).
   @param vec       Source vector.  Row indices are global DOF indices (as
                    produced by gsExprAssembler::rhs() on the full space).
   @param mode      ADD_VALUES for parallel multi-partition assembly;
                    INSERT_VALUES when a single rank inserts the full vector.

   @warning Assumes non-overlapping element partitions, same as
            petsc_insertSparseMatrix.
*/
template<class T>
int petsc_insertVector(Vec&                petscVec,
                        const gsMatrix<T>& vec,
                        InsertMode          mode = ADD_VALUES)
{
    for (index_t i = 0; i < vec.rows(); ++i)
    {
        // Skip zeros under ADD_VALUES only -- a no-op there, but under
        // INSERT_VALUES a zero must still be written (it may be overwriting
        // a previously-inserted nonzero from a different rank/partition).
        if (mode == ADD_VALUES && vec(i, 0) == T(0)) continue;
        PetscCall( VecSetValue(petscVec,
                               static_cast<PetscInt>(i),
                               static_cast<PetscScalar>(vec(i, 0)),
                               mode) );
    }
    return 0;
}

/**
   @brief Insert a gsSparseMatrix into a distributed PETSc Mat through a
   global row/column permutation.

   Same as petsc_insertSparseMatrix, except every (row,col) pair is mapped
   through \a perm before insertion. Intended for the partitioned-layout
   case: gismo's assembler output stays global-indexed
   (row = free global DOF), while \a perm reorders rows so they are grouped
   contiguously by owning rank (see gsPartitionedDofMapper::permutation()),
   matching the row layout petsc_setupMatrixPartitioned() gave the matrix.

   @param petscMat  Destination PETSc matrix (already created and sized via
                    petsc_setupMatrixPartitioned()).
   @param mat       Source sparse matrix, global-indexed (gismo's native
                    assembler output — no local renumbering).
   @param perm      Global permutation, perm(g) = row/col \a g occupies in
                    \a petscMat.
   @param mode      ADD_VALUES for parallel multi-partition assembly;
                    INSERT_VALUES when a single rank inserts the full matrix.

   @warning Assumes non-overlapping element partitions, same as
            petsc_insertSparseMatrix. Skips exact-zero stored entries
            (sparsity-pattern leftovers) — inserting them via ADD_VALUES
            would be a no-op anyway, and skipping avoids the MatSetValue
            call on the (typically unpreallocated) fallback path.

   Batches one MatSetValues call per row instead of one MatSetValue call per
   stored nonzero: converts \a mat to a RowMajor view once (Eigen's
   conversion-assignment -- NOT gsFiberMatrix's toSparseMatrix_into<RowMajor>,
   which inserts out of row order), then gathers each row's permuted column
   indices/values into a buffer before a single MatSetValues call. The
   permutation itself is still applied per entry (perm() isn't monotonic in
   general); only the PETSc call is batched.
*/
template<class T>
int petsc_insertSparseMatrixPermuted(Mat&                      petscMat,
                                      const gsSparseMatrix<T>&  mat,
                                      const gsVector<index_t>&  perm,
                                      InsertMode                mode = ADD_VALUES)
{
    const gsSparseMatrix<T, RowMajor> matRow = mat;

    std::vector<PetscInt>    pcols;
    std::vector<PetscScalar> pvals;
    for (index_t r = 0; r < matRow.outerSize(); ++r)
    {
        pcols.clear();
        pvals.clear();
        for (typename gsSparseMatrix<T, RowMajor>::InnerIterator it(matRow, r); it; ++it)
        {
            if (it.value() == T(0)) continue; // explicit-zero pattern leftover
            pcols.push_back(static_cast<PetscInt>(perm(it.col())));
            pvals.push_back(static_cast<PetscScalar>(it.value()));
        }
        if (pcols.empty()) continue;
        const PetscInt prow = static_cast<PetscInt>(perm(r));
        PetscCall( MatSetValues(petscMat, 1, &prow,
                                static_cast<PetscInt>(pcols.size()), pcols.data(),
                                pvals.data(), mode) );
    }
    return 0;
}

/**
   @brief Insert a gsMatrix (single column) into a distributed PETSc Vec
   through a global row permutation.

   Same as petsc_insertVector, except every row \a i is mapped through
   \a perm before insertion. See petsc_insertSparseMatrixPermuted() for the
   rationale.

   @warning Assumes non-overlapping element partitions, same as
            petsc_insertVector.
*/
template<class T>
int petsc_insertVectorPermuted(Vec&                       petscVec,
                                const gsMatrix<T>&         vec,
                                const gsVector<index_t>&   perm,
                                InsertMode                 mode = ADD_VALUES)
{
    for (index_t i = 0; i < vec.rows(); ++i)
    {
        // See petsc_insertVector() -- zero-skip is only safe under ADD_VALUES.
        if (mode == ADD_VALUES && vec(i, 0) == T(0)) continue;
        PetscCall( VecSetValue(petscVec,
                               static_cast<PetscInt>(perm(i)),
                               static_cast<PetscScalar>(vec(i, 0)),
                               mode) );
    }
    return 0;
}

#if PETSC_VERSION_GE(3,18,0)
/**
   @brief Insert a gsSparseMatrix into a distributed PETSc Mat through a
   global row/column permutation, using PETSc's COO API
   (MatSetPreallocationCOO + MatSetValuesCOO, PETSc >= 3.18).

   Builds the permuted (row,col,value) triplet arrays once from the
   stored non-zeros of \a mat and hands them to PETSc, which handles
   off-process triplets and preallocation exactly (unlike the
   MatSetValue fallback, which needs
   MAT_NEW_NONZERO_ALLOCATION_ERR = PETSC_FALSE and pays for
   unpreallocated inserts). Call once per matrix per assembly pass —
   MatSetPreallocationCOO fixes the sparsity layout; a repeated assembly
   with the *same* layout may call MatSetValuesCOO again directly.

   @warning Assumes non-overlapping element partitions, same as
            petsc_insertSparseMatrix.

   Skips explicit-zero stored entries (sparsity-pattern leftovers), same as
   petsc_insertSparseMatrixPermuted() -- safe unconditionally here since this
   function always inserts under ADD_VALUES (no INSERT_VALUES mode exists
   for the COO path). This shrinks the triplet count handed to PETSc; the
   *contract* is unchanged -- still exactly one MatSetPreallocationCOO call
   per assembly pass, just over a (possibly smaller) fixed sparsity layout.
*/
template<class T>
int petsc_insertSparseMatrixCOO(Mat&                      petscMat,
                                 const gsSparseMatrix<T>&  mat,
                                 const gsVector<index_t>&  perm)
{
    std::vector<PetscInt> coo_i, coo_j;
    std::vector<PetscScalar> coo_v;
    const size_t nnz = static_cast<size_t>(mat.nonZeros());
    coo_i.reserve(nnz);
    coo_j.reserve(nnz);
    coo_v.reserve(nnz);

    for (index_t k = 0; k < mat.outerSize(); ++k)
        for (typename gsSparseMatrix<T>::InnerIterator it(mat, k); it; ++it)
        {
            if (it.value() == T(0)) continue; // explicit-zero pattern leftover
            coo_i.push_back(static_cast<PetscInt>(perm(it.row())));
            coo_j.push_back(static_cast<PetscInt>(perm(it.col())));
            coo_v.push_back(static_cast<PetscScalar>(it.value()));
        }

    PetscCall( MatSetPreallocationCOO(petscMat,
                                      static_cast<PetscCount>(coo_i.size()),
                                      coo_i.data(), coo_j.data()) );
    PetscCall( MatSetValuesCOO(petscMat, coo_v.data(), ADD_VALUES) );
    return 0;
}
#endif

} // namespace gismo
