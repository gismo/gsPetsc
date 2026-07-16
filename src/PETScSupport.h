
#pragma once

#include <petsc.h>
#include <vector>

#include <gsCore/gsDebug.h> // GISMO_STATIC_ASSERT
#include <gsParallel/gsMpiTraits.h> // MPITraits<index_t>::getType()

#if defined(PETSC_USE_COMPLEX)
#error "PETScSupport.h assumes a real PetscScalar (gismo real_t is real); this file does not handle complex-scalar PETSc builds."
#endif

// Pin gismo real_t/index_t widths to PETSc runtime types. Without these,
// a 64-bit-index PETSc build (--with-64-bit-indices) or a complex-scalar build
// would silently truncate at the static_casts sprinkled through this file.
GISMO_STATIC_ASSERT(sizeof(real_t) == sizeof(PetscScalar),
                    PETScSupport_real_t_width_must_match_PetscScalar);
GISMO_STATIC_ASSERT(sizeof(index_t) == sizeof(PetscInt),
                    PETScSupport_index_t_width_must_match_PetscInt);

// PetscCallVoid swallows PETSc error codes (returns void). In PetscImpl we
// need to surface those failures to callers via info(), so we capture the
// code into a member (m_lastError) instead. This macro does that and
// short-circuits the rest of the enclosing method on error, mirroring
// PetscCall's "return on error" semantics for void-returning methods.
// Use only inside PetscImpl-derived classes that expose m_lastError.
#define GISMO_PETSC_CAPTURE(call) \
    do { \
        PetscErrorCode _ierr_gismo_ = (call); \
        if (_ierr_gismo_ != PETSC_SUCCESS) { this->m_lastError = _ierr_gismo_; return; } \
    } while (0)

// ------------------------------------ PETSc auxiliary functions ------------------------------------
namespace gismo
{

// forward declaration
template<typename Derived> int petsc_copyVecToGismo(const Vec& petscVec, gsEigen::MatrixBase<Derived>& gismoVec, MPI_Comm comm, index_t nBlocks = 1);
inline int petsc_createRankInfoVectors(const std::pair<index_t, index_t>& locInfo, gsVector<index_t>& locSizes, gsVector<index_t>& offsets, MPI_Comm comm);

/// @brief Compute layout for parallel distribution.
/// @param[in]  globalDofs  global number of DOFs to be distributed
/// @param[out] locInfo   number of local DOFs, offset of the first local DOF (for the current rank))
/// @param[in]  comm        MPI communicator
/// @return error code
inline int petsc_computeMatLayout(index_t globalDofs, std::pair<index_t, index_t>& locInfo, MPI_Comm comm)
{
    // locInfo.first = number of local DOFs
    // locInfo.second = offset of the first local row (i.e. its global index)

    int nProc = -1;
    int rank = -1;
    MPI_Comm_size( comm, &nProc );
    MPI_Comm_rank( comm, &rank );

    PetscInt localDofs = PETSC_DECIDE;
    PetscInt globalDofs_petsc = static_cast<PetscInt>(globalDofs);
    PetscCall( PetscSplitOwnershipEqual(comm, &localDofs, &globalDofs_petsc) );

    locInfo.first = static_cast<index_t>(localDofs);

    if (nProc > 1 && rank == nProc-1)
        locInfo.second = globalDofs - static_cast<index_t>(localDofs);
    else
        locInfo.second = static_cast<index_t>(rank) * static_cast<index_t>(localDofs);

    GISMO_ENSURE(static_cast<PetscInt>(static_cast<index_t>(localDofs)) == localDofs,
                 "PETScSupport: PetscInt/index_t width mismatch for localDofs " << localDofs);

    return 0;
}

/// @brief Create vector of ranks owning the individual DOFs.
/// The same vector is created on each rank, i.e., this function requires MPI communication.
/// @param[in]  N           global number of DOFs
/// @param[in]  locInfo     parallel layout info (obtained from petsc_computeMatLayout(...))
/// @param[out] result      resulting ownership vector of length \a N
/// @param[in]  comm        MPI communicator
/// @return error code
inline int petsc_createOwnershipVector(index_t N, const std::pair<index_t, index_t>& locInfo, gsVector<index_t>& result, MPI_Comm comm)
{
    int nProc = -1;
    MPI_Comm_size( comm, &nProc );

    // Direct integer MPI_Allgather of each rank's (count, offset), then a
    // local O(N) fill -- replaces a VECMPI-of-PetscScalar round-trip through
    // petsc_copyVecToGismo's VecScatterCreateToAll (which pays for an O(N)
    // scatter just to move nProc-many integers, and casts them through
    // PetscScalar for no reason: rank ids are plain MPI data, not a PETSc
    // quantity).
    gsVector<index_t> counts, offsets;
    petsc_createRankInfoVectors(locInfo, counts, offsets, comm);

    result.resize(N);
    for (int r = 0; r < nProc; ++r)
        for (index_t i = 0; i < counts[r]; ++i)
            result(offsets[r] + i) = static_cast<index_t>(r);

    return 0;
}

/// @brief Create vectors containing information about the global parallel layout.
/// The same vectors are created on each rank, i.e., this function requires MPI communication.
/// @param[in]  locInfo parallel layout info (number of local DOFs and offset for the current rank)
/// @param[out] locSizes  vector of local DOF counts for all ranks
/// @param[out] offsets   vector of offsets for all ranks
/// @param[in]  comm      MPI communicator
/// @return error code
inline int petsc_createRankInfoVectors(const std::pair<index_t, index_t>& locInfo, gsVector<index_t>& locSizes, gsVector<index_t>& offsets, MPI_Comm comm)
{
    int nProc = -1;
    MPI_Comm_size( comm, &nProc );

    // Direct integer MPI_Allgather, replacing a VECMPI-of-PetscScalar
    // round-trip through petsc_copyVecToGismo: rank sizes/offsets are plain
    // MPI data (not a PETSc quantity), so there is no reason to create a
    // PETSc Vec, cast through PetscScalar, and scatter-to-all just to move
    // nProc-many integers between ranks.
    locSizes.resize(nProc);
    offsets.resize(nProc);

    const index_t localSize   = locInfo.first;
    const index_t localOffset = locInfo.second;
    MPI_Allgather(&localSize,   1, MPITraits<index_t>::getType(),
                  locSizes.data(), 1, MPITraits<index_t>::getType(), comm);
    MPI_Allgather(&localOffset, 1, MPITraits<index_t>::getType(),
                  offsets.data(),  1, MPITraits<index_t>::getType(), comm);

    return 0;
}

/// @brief Create mapping vector for block matrix reordering (index in block ordering -> index in interlaced ordering).
/// Block matrix with blocks of equal size is assumed. For example, blocks can correspond to different components of a vector variable.
/// Block ordering means that the matrix is ordered per component and each component is distributed between the given number of processes.
/// Interlaced ordering means that the matrix is ordered per rank.
/// @param[in] N        global number of DOFs
/// @param[in] nBlocks  number of blocks (in one direction)
/// @param[in] locSizes vector of local DOF counts for all ranks
/// @param[in] offsets  vector of offsets for all ranks
/// @param[in] comm     MPI communicator
/// @return mapping vector
inline gsVector<index_t> petsc_mapping_block2interlaced(index_t N, index_t nBlocks, const gsVector<index_t>& locSizes, const gsVector<index_t>& offsets, MPI_Comm comm)
{
    GISMO_ASSERT(N % nBlocks == 0, "Assuming blocks of equal size!");
    index_t blockSize = N / nBlocks;
    
    int nProc = -1;
    MPI_Comm_size( comm, &nProc );

    gsVector<index_t> result(N);
    index_t ii = 0;

    for (index_t r = 0; r < nProc; r++)
    {
        for (index_t b = 0; b < nBlocks; b++)
        {
            for (index_t i = 0; i < locSizes(r); i++)
            {
                result(b*blockSize + offsets(r) + i) = ii;
                ii++;
            }
        }
    }

    return result;
}

/// @brief /// @brief Create mapping vector for block matrix reordering (index in interlaced ordering -> index in block ordering).
/// @param[in] N         global number of DOFs
/// @param[in] nBlocks   number of blocks (in one direction)
/// @param[in] locInfo parallel layout info (number of local DOFs and offset for the current rank)
/// @param[in] rankVec   vector of ranks owning the individual DOFs (of one matrix block)
/// @param[in] locSizes  vector of local DOF counts for all ranks
/// @param[in] offsets   vector of offsets for all ranks
/// @param[in] comm      MPI communicator
/// @return mapping vector
inline gsVector<index_t> petsc_mapping_interlaced2block(index_t N, index_t nBlocks, const gsVector<index_t>& rankVec, const gsVector<index_t>& locSizes, const gsVector<index_t>& offsets, MPI_Comm comm)
{
    index_t blockSize = N / nBlocks;
    GISMO_ASSERT(N % nBlocks == 0, "Assuming blocks of equal size!");
    GISMO_ASSERT(rankVec.rows() == blockSize, "Wrong size of rankVec (should be equal to N/nBlocks).");

    gsVector<index_t> result(N);
    for (index_t b = 0; b < nBlocks; b++)
        for (index_t i = 0; i < blockSize; i++)
            result( (nBlocks-1)*offsets(rankVec(i)) + b*locSizes(rankVec(i)) + i ) = i + b*blockSize;

    return result;
}

/// distributes the matrix
inline int petsc_setupMatrix(Mat& petscMat, const index_t globalRows, const index_t globalCols, MPI_Comm comm, index_t nRowBlocks = 1, index_t nColBlocks = 1)
{
    GISMO_ASSERT((globalRows % nRowBlocks == 0) && (globalCols % nColBlocks == 0), "Assuming blocks of equal size!");

    PetscCall( MatCreate(comm, &petscMat) );

    index_t nRowsPerBlock = globalRows / nRowBlocks;
    index_t nColsPerBlock = globalCols / nColBlocks;

    std::pair<index_t, index_t> rLocInfo, cLocInfo;
    petsc_computeMatLayout(nRowsPerBlock, rLocInfo, comm);
    petsc_computeMatLayout(nColsPerBlock, cLocInfo, comm);

    int nProc = -1;
    MPI_Comm_size( comm, &nProc );
    PetscCall( MatSetType(petscMat, 1 == nProc ? MATSEQAIJ : MATMPIAIJ) );
    PetscCall( MatSetSizes(petscMat, static_cast<PetscInt>(nRowBlocks*rLocInfo.first), static_cast<PetscInt>(nColBlocks*cLocInfo.first), static_cast<PetscInt>(globalRows), static_cast<PetscInt>(globalCols)) );

    return 0;
}

/// @brief Create a square distributed matrix with an explicit (non-uniform)
/// row/column layout, one row block per rank.
///
/// Unlike petsc_setupMatrix (which splits rows evenly across ranks via
/// PetscSplitOwnershipEqual), this gives PETSc the exact local row count
/// for THIS rank -- e.g. gsPartitionedDofMapper::numOwnedDofs(rank), so
/// PETSc's row ownership matches a METIS-partition-derived DOF assignment
/// instead of an arbitrary contiguous split. Takes no gsMetis type: the
/// caller passes the already-computed local count.
///
/// @param[out] petscMat    Created, sized (not yet preallocated) matrix.
/// @param[in]  nGlobal     Global number of rows == columns.
/// @param[in]  nOwnedLocal Number of rows owned by the calling rank.
/// @param[in]  comm        MPI communicator.
inline int petsc_setupMatrixPartitioned(Mat& petscMat, index_t nGlobal, index_t nOwnedLocal, MPI_Comm comm)
{
    PetscCall( MatCreate(comm, &petscMat) );

    int nProc = -1;
    MPI_Comm_size( comm, &nProc );
    PetscCall( MatSetType(petscMat, 1 == nProc ? MATSEQAIJ : MATMPIAIJ) );
    PetscCall( MatSetSizes(petscMat,
                           static_cast<PetscInt>(nOwnedLocal), static_cast<PetscInt>(nOwnedLocal),
                           static_cast<PetscInt>(nGlobal), static_cast<PetscInt>(nGlobal)) );

    return 0;
}

/// @brief Compute the number of nonzeros in matrix (for doing the allocation of PETSc matrix).
/// If \a mat is a block matrix ( \a nRowBlocks or \a nColBlocks > 1), assumes that each block of \a mat is distributed
/// according to the given row and column layout and the matrix will be reordered such that all local rows/cols for rank 0
/// come first, then all local rows/cols for rank 1, etc. All blocks are assumed to be of the same size.
/// @tparam T real number type
/// @param[in]  mat              gismo matrix
/// @param[in]  rLocInfo    parallel layout info for rows (number of local rows, offset of the first local row)
/// @param[in]  cLocInfo    parallel layout info for columns (number of local cols, offset of the first local col)
/// @param[out] nnzRowsDiag      nonzeros per row in the diagonal "block" of the parallel layout 
/// @param[out] nnzRowsOffdiag   nonzeros per row in the rest of off-diagonal "blocks" of the parallel layout 
/// @param[in]  nRowBlocks       number of row blocks
/// @param[in]  nColBlocks       number of column blocks
template<class T>
void petsc_getNonzeroCounts(const gsSparseMatrix<T, RowMajor>& mat, const std::pair<index_t, index_t>& rLocInfo, const std::pair<index_t, index_t>& cLocInfo,
                            std::vector<index_t>& nnzRowsDiag, std::vector<index_t>& nnzRowsOffdiag, index_t nRowBlocks = 1, index_t nColBlocks = 1)
{
    index_t nLocRows = rLocInfo.first; // number of local rows per one block

    nnzRowsDiag.resize(nRowBlocks * nLocRows, 0);
    nnzRowsOffdiag.resize(nRowBlocks * nLocRows, 0);

    index_t nRowsPerBlock = mat.rows() / nRowBlocks; // number of rows of one matrix block
    index_t nColsPerBlock = mat.cols() / nColBlocks; // number of columns of one matrix block

    for (index_t rb = 0; rb < nRowBlocks; rb++)
    {
        index_t rowOffset = rb * nRowsPerBlock + rLocInfo.second;

        for (index_t row = 0; row < nLocRows; ++row)
        {
            index_t ii = rb * nLocRows + row;

            for (index_t cb = 0; cb < nColBlocks; cb++)
            {
                index_t globStartCol = cb * nColsPerBlock + cLocInfo.second; // global index of the first "local" column
                index_t globEndCol = globStartCol + cLocInfo.first; // first global index after the last "local" column

                for (typename gsSparseMatrix<real_t, RowMajor>::InnerIterator it(mat, rowOffset + row); it; ++it)
                    if ( it.col() >= globStartCol && it.col() < globEndCol )    // inside the diagonal block
                        nnzRowsDiag[ii]++;
            }

            nnzRowsOffdiag[ii] = mat.row(rowOffset + row).nonZeros() - nnzRowsDiag[ii];
        }
    }
}

/// Copy an already distributed gsSparseMatrix (only local rows of \a gismoMat have nonzeros)  to distributed PETSc matrix
/// Also, \a gismoMat is assumed to be of full size 
template<class T>
int petsc_copySparseMat(const gsSparseMatrix<T, RowMajor>& gismoMat, Mat& petscMat, const std::pair<index_t, index_t>& rLocInfo,
                        const std::pair<index_t, index_t>& cLocInfo, MPI_Comm comm, index_t nRowBlocks = 1, index_t nColBlocks = 1)
{
    PetscInt M = 0; // global number of rows
    PetscInt N = 0; // global number of columns
    PetscCall( MatGetSize(petscMat, &M, &N) );
    GISMO_ENSURE(static_cast<PetscInt>(static_cast<index_t>(M)) == M,
                 "PETScSupport: PetscInt/index_t width mismatch for global rows " << M);
    GISMO_ENSURE(static_cast<PetscInt>(static_cast<index_t>(N)) == N,
                 "PETScSupport: PetscInt/index_t width mismatch for global cols " << N);
    GISMO_ASSERT(M*N > 0, "petsc_copySparseMat: PETSc matrix with zero rows and/or columns, the global and local sizes of the matrix must be set before (e.g. in function petsc_setupMatrix): ");
    GISMO_ASSERT(static_cast<index_t>(M) == gismoMat.rows() && static_cast<index_t>(N) == gismoMat.cols(), "petsc_copySparseMat: Incompatible petscMat and gismoMat sizes.");

    int nProc = -1;
    MPI_Comm_size( comm, &nProc );

    // Single-field case (the common PetscImpl::compute path): interlaced and
    // block orderings are identical when there is only one block per side,
    // so skip building the O(N) mapRow/mapCol reordering vectors entirely
    // and insert with un-remapped indices below. Mirrors the nBlocks==1
    // special-casing already done in petsc_copyVec/petsc_copyVecToGismo.
    const bool identityMapping = (nRowBlocks == 1 && nColBlocks == 1);

    gsVector<index_t> mapRow, mapCol; // left empty when identityMapping
    if (!identityMapping)
    {
        gsVector<index_t> rLocSizes, rOffsets, cLocSizes, cOffsets;
        petsc_createRankInfoVectors(rLocInfo, rLocSizes, rOffsets, comm);
        petsc_createRankInfoVectors(cLocInfo, cLocSizes, cOffsets, comm);
        mapRow = petsc_mapping_block2interlaced(static_cast<index_t>(M), nRowBlocks, rLocSizes, rOffsets, comm);
        mapCol = petsc_mapping_block2interlaced(static_cast<index_t>(N), nColBlocks, cLocSizes, cOffsets, comm);
    }

    // preallocate PETSc matrix

    std::vector<index_t> nnzRowsDiag_idx, nnzRowsOffdiag_idx;
    petsc_getNonzeroCounts(gismoMat, rLocInfo, cLocInfo, nnzRowsDiag_idx, nnzRowsOffdiag_idx, nRowBlocks, nColBlocks);
    std::vector<PetscInt> nnzRowsDiag(nnzRowsDiag_idx.begin(), nnzRowsDiag_idx.end());
    std::vector<PetscInt> nnzRowsOffdiag(nnzRowsOffdiag_idx.begin(), nnzRowsOffdiag_idx.end());

    if (nProc == 1)
        PetscCall( MatSeqAIJSetPreallocation( petscMat, 0, nnzRowsDiag.data()) );
    else
        PetscCall( MatMPIAIJSetPreallocation( petscMat, 0, nnzRowsDiag.data(), 0, nnzRowsOffdiag.data()) );

    int rank = -1;
    MPI_Comm_rank( comm, &rank );
    
    // copy values

    // const int* outerIndex = gismoMat.outerIndexPtr();
    // const int* innerIndex = gismoMat.innerIndexPtr();
    // const double* values = gismoMat.valuePtr();

    index_t rBlockSize = static_cast<index_t>(M) / nRowBlocks;
    for (index_t b = 0; b < nRowBlocks; b++)
    {
        for (index_t i = 0; i < rLocInfo.first; i++)
        {
            index_t ii = b * rBlockSize + rLocInfo.second + i;

            // int ii = rLocInfo.second + i;
            // int indi[1];
            // indi[0] = ii;
            // int j =  outerIndex[ii];
            // PetscCall( MatSetValues(petscMat, 1, indi, outerIndex[ii+1] - outerIndex[ii], &innerIndex[j], &values[j], INSERT_VALUES) );

            const PetscInt prow = identityMapping ? static_cast<PetscInt>(ii) : static_cast<PetscInt>(mapRow(ii));
            for (typename gsSparseMatrix<real_t, RowMajor>::InnerIterator it(gismoMat, ii); it; ++it)
            {
                const PetscInt pcol = identityMapping ? static_cast<PetscInt>(it.col()) : static_cast<PetscInt>(mapCol(it.col()));
                PetscCall( MatSetValue(petscMat, prow, pcol, static_cast<PetscScalar>(it.value()), INSERT_VALUES) );
            }
        }
    }

    // PetscCopyMode
    // PETSC_COPY_VALUES , or PETSC_USE_POINTER 
    /*
        / Suppose you have m rows on this process and know the global size (M x N)
        MatCreate(comm, &A);
        MatSetSizes(A, m, n, M, N);
        MatSetType(A, MATMPIAIJ);
        // i: row pointers, j: column indices, a: values
        // These should be filled before the call
        MatMPIAIJSetPreallocationCSR(A, i, j, a);  // uses PETSC_COPY_VALUES by default
        // But you can use MatCreateMPIAIJWithArrays if you want PETSC_USE_POINTER behavior:
        MatCreateMPIAIJWithArrays(comm, m, n, M, N, i, j, a, &A);  // i, j, a must stay valid
        // PETSC_USE_POINTER is implied: PETSc won't copy data, just uses your pointers
     */

    PetscCall( MatAssemblyBegin( petscMat, MAT_FINAL_ASSEMBLY ) );
    PetscCall( MatAssemblyEnd( petscMat, MAT_FINAL_ASSEMBLY ) ); 

    return 0;
}

/// Copy an already distributed (dense) vector (only the local rows) to distributed PETSc vector
/// Note: \a gismoVec is assumed to be only the local part (number of rows = localRows)
/// If the vector is a block vector (e.g. with blocks corresponding to components of a vector quantity),
/// \a gismoVec should contain local parts individual blocks in its columns.
template<typename Derived>
int petsc_copyVec(const gsEigen::MatrixBase<Derived>& gismoVec, Vec& petscVec, MPI_Comm comm)
{
    PetscInt M = 0; // global number of rows
    PetscCall( VecGetSize(petscVec, &M) );
    GISMO_ENSURE(static_cast<PetscInt>(static_cast<index_t>(M)) == M,
                 "PETScSupport: PetscInt/index_t width mismatch for vector size " << M);
    GISMO_ASSERT(M > 0, "petsc_copyVec: PETSc vector with zero rows, the global and local sizes of the vector must be set before.");

    int nProc = -1;
    MPI_Comm_size( comm, &nProc );;

    index_t nrows = gismoVec.rows();
    index_t nBlocks = gismoVec.cols();
    index_t globalRowsPerBlock = static_cast<index_t>(M) / nBlocks;
    GISMO_ASSERT(M % static_cast<PetscInt>(nBlocks) == 0, "Assuming blocks of equal size!");

    PetscInt globalStart, globalEnd;
    PetscCall( VecGetOwnershipRange(petscVec, &globalStart, &globalEnd) );
    index_t localRows = static_cast<index_t>(globalEnd - globalStart);

    if (nProc == 1)
        GISMO_ASSERT(static_cast<index_t>(M) == nBlocks * nrows, "petsc_copyVec: Incompatible petscVec and gismoVec sizes.");
    else
        GISMO_ASSERT(localRows == nBlocks * nrows, "petsc_copyVec: Incompatible number of petscVec local rows and gismoVec rows.");


    std::pair<index_t, index_t> locInfo;
    if (nBlocks == 1)
        // petscVec already exists and its real ownership range was just
        // queried above (globalStart/localRows) -- use it directly instead
        // of re-deriving via petsc_computeMatLayout's rank*localDofs formula
        // (review #1: don't guess a layout PETSc already told us).
        locInfo = std::make_pair(localRows, static_cast<index_t>(globalStart));
    else
        // Multi-block case: locInfo here is the per-block (per-component)
        // local count/offset used by petsc_mapping_block2interlaced, which
        // is not simply the real (interlaced) ownership range above. This
        // re-derivation is only correct because petscVec is assumed to have
        // been sized block-consistently with this same petsc_computeMatLayout
        // split (as petsc_setupMatrix does); deriving the per-block offset
        // directly from the interlaced ownership range is not done here.
        petsc_computeMatLayout(globalRowsPerBlock, locInfo, comm);
    gsVector<index_t> locSizes, offsets;
    petsc_createRankInfoVectors(locInfo, locSizes, offsets, comm);
    gsVector<index_t> mapRow = petsc_mapping_block2interlaced(static_cast<index_t>(M), nBlocks, locSizes, offsets, comm);
    
    for (index_t b = 0; b < nBlocks; b++)
    {
        for (index_t i = 0; i < nrows; i++)
        {
            PetscInt ii = static_cast<PetscInt>(mapRow(b * globalRowsPerBlock + locInfo.second + i));
            PetscCall( VecSetValue(petscVec, ii, static_cast<PetscScalar>(gismoVec(i, b)), INSERT_VALUES) );
        }
    }

    PetscCall( VecAssemblyBegin(petscVec) );
    PetscCall( VecAssemblyEnd(petscVec) ); 

    return 0;
}

/// Copy a distributed PETSc vector to a global vector on each MPI node
/// Note: gismoVec is the same global vector on all processes
template<typename Derived>  
int petsc_copyVecToGismo(const Vec& petscVec, gsEigen::MatrixBase<Derived>& gismoVec, MPI_Comm comm, index_t nBlocks)
{
    PetscInt M = 0; // global number of rows
    PetscCall( VecGetSize(petscVec, &M) );
    GISMO_ENSURE(static_cast<PetscInt>(static_cast<index_t>(M)) == M,
                 "PETScSupport: PetscInt/index_t width mismatch for vector size " << M);
    gismoVec.derived().resize(static_cast<index_t>(M), 1);

    VecScatter scatterCtx;
    Vec globalVec;
    PetscCall( VecScatterCreateToAll(petscVec, &scatterCtx, &globalVec) );

    PetscCall( VecScatterBegin(scatterCtx, petscVec, globalVec, INSERT_VALUES, SCATTER_FORWARD) );
    PetscCall( VecScatterEnd(scatterCtx, petscVec, globalVec, INSERT_VALUES, SCATTER_FORWARD) );
    PetscCall( VecScatterDestroy(&scatterCtx) );

    // Heap-allocate these buffers: M is the global DOF count, which for refined
    // meshes overflows the stack if allocated as VLAs (segfault, see #issue).
    std::vector<PetscInt> rowIDs(static_cast<size_t>(M));
    for(PetscInt i = 0; i < M; i++)
        rowIDs[i] = i;

    std::vector<PetscScalar> vals(static_cast<size_t>(M));
    PetscCall( VecGetValues(globalVec, M, rowIDs.data(), vals.data()) );
    PetscCall( VecDestroy(&globalVec) );

    if (nBlocks == 1)
    {
        for(index_t i = 0; i < static_cast<index_t>(M); i++)
            gismoVec(i) = static_cast<real_t>(vals[static_cast<size_t>(i)]);
    }
    else
    {
        std::pair<index_t, index_t> locInfo;
        petsc_computeMatLayout(static_cast<index_t>(M) / nBlocks, locInfo, comm);
        gsVector<index_t> rankVec, locSizes, offsets;
        petsc_createOwnershipVector(static_cast<index_t>(M) / nBlocks, locInfo, rankVec, comm);
        petsc_createRankInfoVectors(locInfo, locSizes, offsets, comm);

        gsVector<index_t> mapRow = petsc_mapping_interlaced2block(static_cast<index_t>(M), nBlocks, rankVec, locSizes, offsets, comm);

        for(index_t i = 0; i < static_cast<index_t>(M); i++)
            gismoVec(mapRow(i)) = static_cast<real_t>(vals[static_cast<size_t>(i)]);
    }
    
    return 0;
}

/// @brief Print ordered output gathered from all ranks.
/// @param[in] outStr output string
/// @param[in] comm   MPI communicator
inline void printOrderedOutput(std::string outStr, gsMpiComm comm)
{
    int rank = comm.rank();
    int nProc = comm.size();

    int len = outStr.size();
    std::vector<int> lengths(nProc);
    comm.gather<int>(&len, lengths.data(), 1, 0);

    std::vector<char> recvbuf;
    std::vector<int> displs;
    if (rank == 0)
    {
        int total = 0;
        displs.resize(nProc);
        for (int i = 0; i < nProc; ++i)
        {
            displs[i] = total;
            total += lengths[i];
        }
        recvbuf.resize(total);
    }

    comm.gatherv<char>(const_cast<char*>(outStr.data()), len, recvbuf.data(), lengths.data(), displs.data(), 0);

    if (rank == 0)
    {
        for (int i = 0; i < nProc; ++i)
        {
            std::string s(recvbuf.begin() + displs[i], recvbuf.begin() + displs[i] + lengths[i]);
            gsInfo << s;
        }
    }
}

} // end namespace gismo

// ---------------------------------------------------------------------------------------


//extern "C"
// {
// } // extern "C"

namespace gsEigen
{

template<typename _MatrixType> struct petsc_traits;
template<typename _MatrixType> class PetscKSP;
template<typename _MatrixType> class PetscNestKSP;

template<typename _MatrixType>
struct petsc_traits< PetscKSP<_MatrixType> >
{
    typedef _MatrixType MatrixType;
    typedef typename _MatrixType::Scalar Scalar;
    typedef typename _MatrixType::RealScalar RealScalar;
    typedef typename _MatrixType::StorageIndex StorageIndex;
};

template<typename _MatrixType>
struct petsc_traits< PetscNestKSP<_MatrixType> >
{
    typedef _MatrixType MatrixType;
    typedef typename _MatrixType::Scalar Scalar;
    typedef typename _MatrixType::RealScalar RealScalar;
    typedef typename _MatrixType::StorageIndex StorageIndex;
};


/**
Base class for PETSc solvers
 */
template<class Derived>
class PetscImpl : public SparseSolverBase<Derived>
{
protected:
    typedef SparseSolverBase<Derived> Base;
    using Base::derived;
    using Base::m_isInitialized;
   
public:
    typedef petsc_traits<Derived> Traits;
    typedef typename Traits::MatrixType MatrixType;
    typedef typename Traits::Scalar Scalar;
    typedef typename Traits::RealScalar RealScalar;
    typedef typename Traits::StorageIndex StorageIndex;
    
    typedef Matrix<Scalar,Dynamic,1> VectorType;
    typedef Matrix<StorageIndex, 1, MatrixType::ColsAtCompileTime> IntRowVectorType;
    typedef Matrix<StorageIndex, MatrixType::RowsAtCompileTime, 1> IntColVectorType;
    typedef Array<StorageIndex,64,1,DontAlign> ParameterType;

    enum
    {
        ScalarIsComplex = NumTraits<Scalar>::IsComplex,
        ColsAtCompileTime = Dynamic,
        MaxColsAtCompileTime = Dynamic
    };

    gismo::gsOptionList m_options;     ///< Options

    mutable KSP m_ksp; ///< krylov solver
    mutable PC m_pc;   ///< preconditionner
    // Note: to use direct solver, then there is an option "use preconditionner only" so that KSP is not involved
    
    MPI_Comm m_comm; ///< communicator
    
    Mat m_pmatrix;      ///< PETSc matrix

    mutable Vec m_prhs, m_psol; ///< Solution vector and right-hand side vector

    mutable ComputationInfo m_info;
    mutable int m_lastError; ///< Most recent PETSc error code from a void-context call. Surfaces via info().
    Index m_size; ///< Local size of the matrix (number of local rows on this rank). For block solvers, this is the per-block local row count, matching b.rows() for a BlockVec b in _solve_impl.
    bool m_ownsPetscInit; ///< true if this instance called Initialize() (i.e. we are responsible for finalizing PETSc in the destructor)

    PetscImpl(MPI_Comm comm = PETSC_COMM_WORLD) : m_size(-1), m_ownsPetscInit(false)
    {
        m_info = Success;
        m_lastError = 0;
        initialize(comm);
        m_isInitialized = false;// Becomes true when the sparse matrix is given
    }

    ~PetscImpl()
    {
        if (m_isInitialized)
        {
            // m_isInitialized tracks the PETSc objects owned by this solver;
            // destroying them is the solver's responsibility even when the
            // caller owns PETSc finalization (see m_ownsPetscInit below).
            int ierr;
            ierr = MatDestroy(&m_pmatrix); if (ierr) m_lastError = ierr;
            ierr = VecDestroy(&m_psol);    if (ierr) m_lastError = ierr;
            ierr = VecDestroy(&m_prhs);    if (ierr) m_lastError = ierr;
            ierr = KSPDestroy(&m_ksp);     if (ierr) m_lastError = ierr;
        }
        if (m_ownsPetscInit)
        {
            // Finalize only if *we* called Initialize -- callers that
            // initialize PETSc themselves (e.g. the gsMetisPetscAssembly
            // example, which calls it before constructing a solver) own
            // the finalization. Forgetting this guard would double-finalize
            // against such a caller's explicit PetscFinalize().
            int ierr = PetscFinalize(); if (ierr) m_lastError = ierr;
        }
    }

    /// Initialize PETSc solver with the communicator \a comm
    void initialize(MPI_Comm comm = PETSC_COMM_WORLD)
    {
        m_comm = comm;

        PetscBool alreadyInit = PETSC_FALSE;
        GISMO_PETSC_CAPTURE( PetscInitialized(&alreadyInit) );
        m_ownsPetscInit = (alreadyInit == PETSC_FALSE);
        if (m_ownsPetscInit)
            GISMO_PETSC_CAPTURE( PetscInitializeNoArguments() );

        GISMO_PETSC_CAPTURE( KSPCreate(m_comm, &m_ksp) );
        GISMO_PETSC_CAPTURE( KSPGetPC(m_ksp, &m_pc) );
    }

    gismo::gsOptionList & options() {return m_options;}

    inline Index cols() const { return m_size; }
    inline Index rows() const { return m_size; }

    /** \brief Reports whether previous computation was successful.
      *
      * \returns \c Success if computation was successful,
      *          \c NumericalIssue if the matrix appears to be negative.
      */
    ComputationInfo info() const
    {
      if (m_lastError != 0)
        return InvalidInput; // a void-context PETSc call failed after compute()
      return m_info;
    }

    /// Prints details about this solver
    void print() const
    {
        PetscOptionsView(NULL, PETSC_VIEWER_STDOUT_(m_comm));
    }
    
    /// Copies the \a matrix to a PETSc matrix
    Derived& compute(const MatrixType& matrix);

    /// Computes local offset and number of rows for this node
    /// \param[in] nRows : number of global rows of the matrix
    /// \return result.first  : Number of local rows on this node
    /// \return result.second : Global index of the first local row on this node
    std::pair<index_t, index_t> computeLayout(index_t nRows)
    {
        std::pair<index_t, index_t> result;
        PetscCallAbort(m_comm, gismo::petsc_computeMatLayout(nRows, result, m_comm));
        return result;
    }
    
    template<typename Rhs,typename Dest>
    void _solve_impl(const MatrixBase<Rhs> &b, MatrixBase<Dest> &dest) const;

protected:

    void applyOptions() const
    {
        GISMO_PETSC_CAPTURE( PetscOptionsClear(NULL) );
        for ( auto & opt : m_options.getAllEntries() )
            GISMO_PETSC_CAPTURE( PetscOptionsSetValue(NULL, opt.label.c_str(), opt.val.c_str()) );

        GISMO_PETSC_CAPTURE( KSPSetFromOptions(this->m_ksp) );
        GISMO_PETSC_CAPTURE( PCSetFromOptions(this->m_pc) );

        // this is for systems with two fields...
        // PetscCallVoid( PetscOptionsSetValue(NULL, "-pc_type", "fieldsplit") );
        // PetscCallVoid( PetscOptionsSetValue(NULL, "-pc_fieldsplit_detect_saddle_point", NULL) );
        // PetscCallVoid( PetscOptionsSetValue(NULL, "-pc_fieldsplit_type", "schur") );
        // PetscCallVoid( PetscOptionsSetValue(NULL, "-pc_fieldsplit_schur_fact_type", "upper") );
        // PetscCallVoid( PetscOptionsSetValue(NULL, "-pc_fieldsplit_schur_precondition", "self") );

        // PetscCallVoid( PetscOptionsSetValue(NULL, "-fieldsplit_0_ksp_type", "preonly") );
        // PetscCallVoid( PetscOptionsSetValue(NULL, "-fieldsplit_0_pc_type", "lu") );
        // PetscCallVoid( PetscOptionsSetValue(NULL, "-fieldsplit_0_pc_factor_mat_solver_type", "mumps") );

        // PetscCallVoid( PetscOptionsSetValue(NULL, "-fieldsplit_1_pc_type", "lsc") );
        // PetscCallVoid( PetscOptionsSetValue(NULL, "-fieldsplit_1_pc_lsc_scale_diag", NULL) );
        // PetscCallVoid( PetscOptionsSetValue(NULL, "-fieldsplit_1_lsc_ksp_type", "preonly") );
        // PetscCallVoid( PetscOptionsSetValue(NULL, "-fieldsplit_1_lsc_pc_type", "lu") );
        // PetscCallVoid( PetscOptionsSetValue(NULL, "-fieldsplit_1_lsc_pc_factor_mat_solver_type", "mumps") );
    }

    void manageErrorCode(Index error) const
    {
      switch(error)
      {
        case 0:
          m_info = Success;
          break;
        case -4:
        case -7:
          m_info = NumericalIssue;
          break;
        default:
          m_info = InvalidInput;
      }
    }
};


template<class Derived>
Derived& PetscImpl<Derived>::compute(const MatrixType& matrix)
{
    m_lastError = 0; // clear any stale error from a previous compute()/solve() cycle

    if (m_isInitialized) // did we call compute before ?
    {
        PetscCallAbort(m_comm, MatDestroy(&m_pmatrix));
        PetscCallAbort(m_comm, VecDestroy(&m_psol));
        PetscCallAbort(m_comm, VecDestroy(&m_prhs));
    }

    int nProc = -1;
    MPI_Comm_size( m_comm, &nProc );

    int rank  = -1;
    MPI_Comm_rank( m_comm, &rank );

    index_t nRows = matrix.rows();
    index_t nCols = matrix.cols();
    assert(nRows==nCols && "expecting square mat");

    PetscCallAbort(m_comm, gismo::petsc_setupMatrix(m_pmatrix, nRows, nCols, m_comm));

    PetscCallAbort(m_comm, MatCreateVecs(m_pmatrix, &m_psol, &m_prhs));

    // Ask PETSc for the row range it actually assigned instead of
    // re-deriving it via petsc_computeMatLayout's rank*localDofs formula:
    // the Mat already exists at this point, so there is no need to guess,
    // and this stays correct regardless of PETSc's internal splitting logic.
    PetscInt rowStart = 0, rowEnd = 0;
    PetscCallAbort(m_comm, MatGetOwnershipRange(m_pmatrix, &rowStart, &rowEnd));
    std::pair<index_t, index_t> locInfo(static_cast<index_t>(rowEnd - rowStart),
                                         static_cast<index_t>(rowStart));

    m_size = locInfo.first;

    // Copy matrix [ASSUMES square matrix, same cols/rows layout]
    // Case: Matrix already distributed
    PetscCallAbort(m_comm, gismo::petsc_copySparseMat(matrix, m_pmatrix, locInfo, locInfo, m_comm));

    //.. else
    // Assumes matrix is non-empty and fully polulated on rank 0 only !
    //petsc_distributeSparseMat(matrix, m_pmatrix, ...)
    

    m_isInitialized = true;
    return this->derived();
}

template<class Derived>
template<typename BDerived,typename XDerived>
void PetscImpl<Derived>::_solve_impl(const MatrixBase<BDerived> &b, MatrixBase<XDerived>& x) const
{
    m_lastError = 0; // clear any stale error from a previous solve

    Index nrhs = Index(b.cols());
    assert(m_size==b.rows());
    assert(((MatrixBase<BDerived>::Flags & RowMajorBit) == 0 || nrhs == 1) && "Row-major right hand sides are not supported");
    assert(((MatrixBase<XDerived>::Flags & RowMajorBit) == 0 || nrhs == 1) && "Row-major matrices of unknowns are not supported");
    assert(((nrhs == 1) || b.outerStride() == b.rows()));

    // Copy right-hand side vector to PETSc
    GISMO_PETSC_CAPTURE( gismo::petsc_copyVec(b, m_prhs, m_comm) );

    this->applyOptions();

    // KSP set operators:
    // first: operator m_pmatrix, second: preconditionner build from the same matrix
    GISMO_PETSC_CAPTURE( KSPSetOperators(this->m_ksp, m_pmatrix, m_pmatrix) );

    // Solve the system
    GISMO_PETSC_CAPTURE( KSPSolve(this->m_ksp, m_prhs, m_psol) );

    // Get statistics
    PetscInt nIter = 0;
    GISMO_PETSC_CAPTURE( KSPGetIterationNumber(this->m_ksp, &nIter) );

    // Copy the solution back to \a x
    GISMO_PETSC_CAPTURE( gismo::petsc_copyVecToGismo(m_psol, x, m_comm) );

    // Clear petsc vector
    GISMO_PETSC_CAPTURE( VecZeroEntries(m_prhs) );
}

//Note: KSP has all linear solvers, but PETSc provides also nonlinear solvers and optimizers ...
template<typename MatrixType>
class PetscKSP : public PetscImpl< PetscKSP<MatrixType> >
{
  protected:
    typedef PetscImpl<PetscKSP> Base;

    using Base::m_options;
    using Base::m_pmatrix;
    using Base::m_prhs;
    using Base::m_psol;

    friend class PetscImpl< PetscKSP<MatrixType> >;

  public:

    typedef typename Base::Scalar Scalar;
    typedef typename Base::RealScalar RealScalar;

    using Base::compute;

    PetscKSP(MPI_Comm comm = PETSC_COMM_WORLD) : Base(comm) { }

    explicit PetscKSP(const MatrixType& matrix, MPI_Comm comm) : Base(comm)
    { compute(matrix); }

};

namespace internal {

typedef gismo::gsVector<gismo::gsMatrix<real_t>, 2>                     BlockVec;

// this solve_traits class permits to determine the evaluation type with respect to storage kind (Dense vs Sparse)
template<typename MatrixType>
struct solve_traits<PetscNestKSP<MatrixType>,BlockVec,Dense>
{
    typedef BlockVec PlainObject;
    typedef index_t StorageIndex;
    typedef traits<PlainObject> BaseTraits;
   
    enum {
        Flags = 0,//BaseTraits::Flags & RowMajorBit,
        CoeffReadCost = HugeCost
    };
};

template<> struct traits<BlockVec>
{
    typedef gsEigen::Dense StorageKind;
    typedef gismo::gsMatrix<real_t> Scalar;
    typedef BlockVec NestedExpression;
    enum {
        Flags = 0,//BaseTraits::Flags & RowMajorBit,
        CoeffReadCost = HugeCost,
        RowsAtCompileTime = Dynamic,
        ColsAtCompileTime = Dynamic,
        MaxRowsAtCompileTime = 10,
        MaxColsAtCompileTime = 10
    };
    typedef MatrixXpr XprKind;
    typedef MatrixXpr XprType;
};

// /*
template<typename MatrixType>
struct traits<Solve<PetscNestKSP<MatrixType>, BlockVec> >
  : traits<typename solve_traits<PetscNestKSP<MatrixType>,BlockVec,Dense>::PlainObject>
{
    typedef BlockVec PlainObject;
    typedef index_t StorageIndex;
    typedef traits<PlainObject> BaseTraits;
    enum {
        Flags = 0,//BaseTraits::Flags & RowMajorBit,
        CoeffReadCost = HugeCost
    };
    typedef MatrixXpr XprKind;
    typedef MatrixXpr XprType;
};
//*/


template <>
struct evaluator<BlockVec> : evaluator_base<BlockVec>
{
    typedef BlockVec XprType;
    typedef BlockVec ArgTypeNested;
    typedef ArgTypeNested ArgTypeNestedCleaned;
    typedef typename XprType::CoeffReturnType CoeffReturnType;
 
    enum { CoeffReadCost = HugeCost, Flags = gsEigen::ColMajor };

    evaluator() {}

    evaluator(const XprType& xpr)
    //: m_vec(&xpr)
    { }

    index_t cols() const { return m_vec->cols(); }
    index_t rows() const { return m_vec->rows(); }
    CoeffReturnType coeff(Index row, Index col) const
    {
        return (*m_vec)(row, col);
    }

    BlockVec * m_vec;
};

}  // namespace internal

template<typename MatrixType>
class PetscNestKSP : public PetscImpl< PetscNestKSP<MatrixType> >
{
  protected:
    typedef PetscImpl<PetscNestKSP> Base;

    using Base::m_options;
    using Base::m_pmatrix;
    using Base::m_prhs;
    using Base::m_psol;

    using Base::m_size;
    using Base::m_lastError;
    using Base::m_comm;
    using Base::m_isInitialized;
    
    friend class PetscImpl< PetscNestKSP<MatrixType> >;

    // Block matrices
  public:

    typedef typename Base::Scalar Scalar;
    typedef typename Base::RealScalar RealScalar;

    typedef gismo::gsMatrix<gismo::gsSparseMatrix<real_t, RowMajor>, 2, 2> BlockMat;
    typedef gismo::gsVector<gismo::gsMatrix<real_t>, 2>                     BlockVec;

    PetscNestKSP(MPI_Comm comm = PETSC_COMM_WORLD) : Base(comm) { }

    explicit PetscNestKSP(const MatrixType& matrix, MPI_Comm comm) : Base(comm)
    { compute(matrix); }

    /// Copies the \a matrix to a PETSc MATNEST
    PetscNestKSP & compute(const BlockMat& matrix);

    template<typename Rhs,typename Dest>
    void _solve_impl(const MatrixBase<Rhs> &b, MatrixBase<Dest> &dest) const;

    inline const Solve<PetscNestKSP, BlockVec>
    solve(const BlockVec& b) const
    {
        return Solve<PetscNestKSP, BlockVec>(*this, b);
    }

};

/// Input is Block system [A B; C D], wehere some blocks might be empty
template<typename MatrixType>
PetscNestKSP<MatrixType>& PetscNestKSP<MatrixType>::compute(const typename PetscNestKSP<MatrixType>::BlockMat& matrix)
{
    m_lastError = 0; // clear any stale error from a previous compute()/solve() cycle

    if (m_isInitialized) // did we call compute before ?
    {
        PetscCallAbort(m_comm, MatDestroy(&m_pmatrix));
        PetscCallAbort(m_comm, VecDestroy(&m_psol));
        PetscCallAbort(m_comm, VecDestroy(&m_prhs));
    }

    int nProc = -1;
    MPI_Comm_size( m_comm, &nProc );

    int rank  = -1;
    MPI_Comm_rank( m_comm, &rank );

    const index_t rBlocks = matrix.rows();
    const index_t cBlocks = matrix.cols();

    //how many rows belong to the process, and offset
    //locInfo.first  : number of local rows
    //locInfo.secind : offset for 1st local row
    std::vector<std::pair<index_t, index_t> > rlocInfo(rBlocks);
    std::vector<std::pair<index_t, index_t> > clocInfo(cBlocks);
    
    gismo::gsVector<index_t> rsz(rBlocks);
    for (index_t r = 0 ; r!=rBlocks; ++r)
    {
        // rsz[r] is the row count shared by every non-empty block in row r.
        // The previous code took the last non-empty block's row count, which
        // silently mismatched for non-square nests (1x3, 3x1). Take the first
        // non-empty block and assert the rest agree.
        bool rfound = false;
        for (index_t c = 0 ; c!=cBlocks; ++c)
            if (matrix(r,c).size() != 0)
            {
                if (!rfound) { rsz[r] = matrix(r,c).rows(); rfound = true; }
                else GISMO_ASSERT(matrix(r,c).rows() == rsz[r],
                                  "PetscNestKSP::compute: blocks in the same row must share row count");
            }
        GISMO_ASSERT(rfound, "PetscNestKSP::compute: every row must have at least one non-empty block");
        PetscCallAbort(m_comm, gismo::petsc_computeMatLayout(rsz[r], rlocInfo[r], m_comm));
    }

    gismo::gsVector<index_t> csz(cBlocks);
    for (index_t c = 0 ; c!=cBlocks; ++c)
    {
        bool cfound = false;
        for (index_t r = 0 ; r!=rBlocks; ++r)
            if (matrix(r,c).size() != 0)
            {
                if (!cfound) { csz[c] = matrix(r,c).cols(); cfound = true; }
                else GISMO_ASSERT(matrix(r,c).cols() == csz[c],
                                  "PetscNestKSP::compute: blocks in the same column must share col count");
            }
        GISMO_ASSERT(cfound, "PetscNestKSP::compute: every column must have at least one non-empty block");
        PetscCallAbort(m_comm, gismo::petsc_computeMatLayout(csz[c], clocInfo[c], m_comm));
    }

    // m_size is kept consistent with PetscImpl::compute's contract (local
    // row count on this rank), here taken as the common local row count
    // shared by every row-block (asserted below). Note BlockVec is fixed at
    // 2 blocks (gsVector<gsMatrix<real_t>,2>), so b.rows() in
    // PetscNestKSP::_solve_impl is the block count, not a DOF count --
    // that override never reads m_size (each sub-block's local size is
    // validated independently by petsc_copyVec). m_size/rows()/cols() are
    // set here for API consistency with the base class, not because the
    // current solve path consults them.
    GISMO_ASSERT(rBlocks > 0, "PetscNestKSP::compute: rBlocks must be > 0");
    m_size = rlocInfo[0].first;
    for (index_t r = 1 ; r != rBlocks; ++r)
        GISMO_ASSERT(rlocInfo[r].first == m_size,
                     "PetscNestKSP::compute: row blocks must have the same local row count on every rank");
        
    std::vector<Mat> bmatrix;
    bmatrix.reserve(rBlocks*cBlocks);
    for (index_t r = 0 ; r!=rBlocks; ++r)
        for (index_t c = 0 ; c!=cBlocks; ++c)
        {
            if (matrix(r,c).size() != 0)
            {
                bmatrix.push_back(Mat());
                PetscCallAbort(m_comm, MatCreate(m_comm, &bmatrix.back()));
                PetscCallAbort(m_comm, MatSetType(bmatrix.back(), 1 == nProc ? MATSEQAIJ : MATMPIAIJ));
                PetscCallAbort(m_comm, MatSetSizes(bmatrix.back(), static_cast<PetscInt>(rlocInfo[r].first), static_cast<PetscInt>(clocInfo[c].first), static_cast<PetscInt>(rsz[r]), static_cast<PetscInt>(csz[c])));
            }
            else
                bmatrix.push_back(NULL);
        }

    MatCreateNest(m_comm, static_cast<PetscInt>(rBlocks), nullptr, static_cast<PetscInt>(cBlocks), nullptr,  bmatrix.data(),  &m_pmatrix);
    MatNestSetVecType(m_pmatrix, VECNEST);
    PetscCallAbort(m_comm, MatCreateVecs(m_pmatrix, &m_psol, NULL));

    for (index_t r = 0 ; r!=rBlocks; ++r)
        for (index_t c = 0 ; c!=cBlocks; ++c)
        {
            if (matrix(r,c).size() != 0)
            {
                Mat tmp;
                PetscCallAbort(m_comm, MatNestGetSubMat(m_pmatrix, static_cast<PetscInt>(r), static_cast<PetscInt>(c), &tmp));
                PetscCallAbort(m_comm, gismo::petsc_copySparseMat(matrix(r,c), tmp, rlocInfo[r], clocInfo[c], m_comm));
            }
        }

    // NOTE:
    // not sure what is happening here
    // individual blocks are already assembled, but since they are filled "in place", the matnest stays in an unassembled state
    // KSPSolve then fails with error "Not for unassembled matrix"
    // hope that the assembly of blocks is not performed twice
    PetscCallAbort(m_comm, MatAssemblyBegin( m_pmatrix, MAT_FINAL_ASSEMBLY ));
    PetscCallAbort(m_comm, MatAssemblyEnd( m_pmatrix, MAT_FINAL_ASSEMBLY ));
    
    std::vector<Vec> brhs;
    brhs.reserve(rBlocks);
    for (index_t r = 0 ; r!=rBlocks; ++r)
    {
        brhs.push_back(Vec());
        for (index_t c = 0 ; c!=cBlocks; ++c)
            if (matrix(r,c).size() != 0)
            {
                PetscCallAbort(m_comm, MatCreateVecs(bmatrix[r*cBlocks+c], NULL, &brhs.back()));
                break;
            }
    }

    PetscCallAbort(m_comm, VecCreateNest(m_comm, static_cast<PetscInt>(rBlocks), NULL, brhs.data(), &m_prhs));

    m_isInitialized = true;
    return *this;
}

/*
template<class Derived>
inline const Solve<PetscBlockImpl<Derived>, Rhs>
PetscBlockImpl<Derived>::solveBlock(const std::vector<MatrixType>& b)
{

    return Solve<Derived, Rhs>(derived(), b.derived());
}
*/

template<class MatrixType>
template<typename BDerived,typename XDerived>
void PetscNestKSP<MatrixType>::_solve_impl(const MatrixBase<BDerived> &b, MatrixBase<XDerived>& x) const
{
    m_lastError = 0; // clear any stale error from a previous solve

    gsDebugVar( "Solving nest..");

    // Copy right-hand side vector to PETSc
    for (index_t c = 0 ; c!=b.rows(); ++c)
    {
        Vec tmp;
        VecNestGetSubVec(m_prhs, static_cast<PetscInt>(c), &tmp);
        GISMO_PETSC_CAPTURE( gismo::petsc_copyVec(b(c,0), tmp, m_comm) );
    }

    this->applyOptions();

    // KSP set operators:
    // first: operator m_pmatrix, second: preconditionner build from the same matrix
    GISMO_PETSC_CAPTURE( KSPSetOperators(this->m_ksp, m_pmatrix, m_pmatrix) );

    // Solve the system
    GISMO_PETSC_CAPTURE( KSPSolve(this->m_ksp, m_prhs, m_psol) );

    // Get statistics
    PetscInt nIter = 0;
    GISMO_PETSC_CAPTURE( KSPGetIterationNumber(this->m_ksp, &nIter) );

    // Copy the solution back to \a x
    for (index_t c = 0 ; c!=b.rows(); ++c)
    {
        Vec tmp;
        VecNestGetSubVec(m_psol, static_cast<PetscInt>(c), &tmp);
        GISMO_PETSC_CAPTURE( gismo::petsc_copyVecToGismo(tmp, x(c,0), m_comm) );
    }

    // Clear petsc vector
    GISMO_PETSC_CAPTURE( VecZeroEntries(m_prhs) );
}

} // end namespace Eigen
