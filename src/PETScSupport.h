
#pragma once

#include <petsc.h>

// ------------------------------------ PETSc auxiliary functions ------------------------------------
namespace gismo
{

/// @brief Compute layout for parallel distribution.
/// @param[in]  globalDofs  global number of DOFs to be distributed
/// @param[out] locInfo   number of local DOFs, offset of the first local DOF (for the current rank))
/// @param[in]  comm        MPI communicator
/// @return error code
int petsc_computeMatLayout(index_t globalDofs, std::pair<index_t, index_t>& locInfo, MPI_Comm comm)
{
    // locInfo.first = number of local DOFs
    // locInfo.second = offset of the first local row (i.e. its global index)

    int nProc = -1;
    int rank = -1;
    MPI_Comm_size( comm, &nProc );
    MPI_Comm_rank( comm, &rank );

    index_t localDofs = PETSC_DECIDE;
    PetscCall( PetscSplitOwnershipEqual(comm, &localDofs, &globalDofs) );

    locInfo.first = localDofs;

    if (nProc > 1 && rank == nProc-1) // last rank (localDofs can be different than for other ranks)
        locInfo.second = globalDofs - localDofs;
    else
        locInfo.second = rank * localDofs;

    return 0;
}

/// Copy a distributed PETSc vector to a global vector on each MPI node
/// Note: gismoVec is the same global vector on all processes
template<typename Derived>  
int petsc_copyVecToGismo(const Vec& petscVec, gsEigen::MatrixBase<Derived>& gismoVec, MPI_Comm comm)
{
    int M = 0; // global number of rows
    PetscCall( VecGetSize(petscVec, &M) );
    gismoVec.derived().resize(M, 1);

    VecScatter scatterCtx;
    Vec globalVec;
    PetscCall( VecScatterCreateToAll(petscVec, &scatterCtx, &globalVec) );

    PetscCall( VecScatterBegin(scatterCtx, petscVec, globalVec, INSERT_VALUES, SCATTER_FORWARD) );
    PetscCall( VecScatterEnd(scatterCtx, petscVec, globalVec, INSERT_VALUES, SCATTER_FORWARD) );
    PetscCall( VecScatterDestroy(&scatterCtx) );

    index_t rowIDs[M];
    for(index_t i = 0; i < M; i++)
        rowIDs[i] = i;

    real_t vals[M];
    PetscCall( VecGetValues(globalVec, M, rowIDs, vals) );
    PetscCall( VecDestroy(&globalVec) );

    for(index_t i = 0; i < M; i++)
        gismoVec(i) = vals[i];

    return 0;
}

/// @brief Create vector of ranks owning the individual DOFs.
/// The same vector is created on each rank, i.e., this function requires MPI communication.
/// @param[in]  N           global number of DOFs
/// @param[in]  locInfo   parallel layout info (obtained from petsc_computeMatLayout(...))
/// @param[out] result      resulting ownership vector of length \a N
/// @param[in]  comm        MPI communicator
/// @return error code
int petsc_createOwnershipVector(index_t N, const std::pair<index_t, index_t>& locInfo, gsVector<index_t>& result, MPI_Comm comm)
{
    int rank = -1;
    MPI_Comm_rank( comm, &rank );

    Vec rankVec;
    PetscCall( VecCreate(comm, &rankVec) );
    PetscCall( VecSetType(rankVec, VECMPI) );
    PetscCall( VecSetSizes(rankVec, locInfo.first, N) );

    for (index_t i = locInfo.second; i < locInfo.second + locInfo.first; i++)
        PetscCall( VecSetValue(rankVec, i, rank, INSERT_VALUES) );

    petsc_copyVecToGismo(rankVec, result, comm);

    return 0;
}

/// @brief Create vectors containing information about the global parallel layout.
/// The same vectors are created on each rank, i.e., this function requires MPI communication.
/// @param[in]  locInfo parallel layout info (number of local DOFs and offset for the current rank)
/// @param[out] locSizes  vector of local DOF counts for all ranks
/// @param[out] offsets   vector of offsets for all ranks
/// @param[in]  comm      MPI communicator
/// @return error code
int petsc_createRankInfoVectors(const std::pair<index_t, index_t>& locInfo, gsVector<index_t>& locSizes, gsVector<index_t>& offsets, MPI_Comm comm)
{
    int nProc = -1;
    int rank = -1;
    MPI_Comm_size( comm, &nProc );
    MPI_Comm_rank( comm, &rank );

    Vec sizesVec, offsetVec;
    PetscCall( VecCreate(comm, &sizesVec) );
    PetscCall( VecSetType(sizesVec, VECMPI) );
    PetscCall( VecSetSizes(sizesVec, 1, nProc) );
    PetscCall( VecCreate(comm, &offsetVec) );
    PetscCall( VecSetType(offsetVec, VECMPI) );
    PetscCall( VecSetSizes(offsetVec, 1, nProc) );

    for (index_t i = 0; i < nProc; i++)
    {
        PetscCall( VecSetValue(sizesVec, i, locInfo.first, INSERT_VALUES) );
        PetscCall( VecSetValue(offsetVec, i, locInfo.second, INSERT_VALUES) );
    }

    petsc_copyVecToGismo(sizesVec, locSizes, comm);
    petsc_copyVecToGismo(offsetVec, offsets, comm);

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
gsVector<index_t> petsc_mapping_block2interlaced(index_t N, index_t nBlocks, const gsVector<index_t>& locSizes, const gsVector<index_t>& offsets, MPI_Comm comm)
{
    GISMO_ASSERT(N % nBlocks == 0, "Assuming N divisible by nBlocks!");
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
gsVector<index_t> petsc_mapping_interlaced2block(index_t N, index_t nBlocks, const gsVector<index_t>& rankVec, const gsVector<index_t>& locSizes, const gsVector<index_t>& offsets, MPI_Comm comm)
{
    index_t blockSize = N / nBlocks;
    GISMO_ASSERT(N % nBlocks == 0, "Assuming N divisible by nBlocks!");
    GISMO_ASSERT(rankVec.rows() == blockSize, "Wrong size of rankVec (should be equal to N/nBlocks).");

    gsVector<index_t> result(N);
    for (index_t b = 0; b < nBlocks; b++)
        for (index_t i = 0; i < blockSize; i++)
            result( (nBlocks-1)*offsets(rankVec(i)) + b*locSizes(rankVec(i)) + i ) = i + b*blockSize;

    return result;
}

/// distributes the matrix
int petsc_setupMatrix(Mat& petscMat, const index_t globalRows, const index_t globalCols, MPI_Comm comm)
{
    PetscCall( MatCreate(comm, &petscMat) );

    std::pair<index_t, index_t> rLocInfo, cLocInfo;
    petsc_computeMatLayout(globalRows, rLocInfo, comm);
    petsc_computeMatLayout(globalCols, cLocInfo, comm);

    int nProc = -1;
    MPI_Comm_size( comm, &nProc );
    PetscCall( MatSetType(petscMat, 1 == nProc ? MATSEQAIJ : MATMPIAIJ) );
    PetscCall( MatSetSizes(petscMat, rLocInfo.first, cLocInfo.first, globalRows, globalCols) );

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
    int M = 0; // global number of rows
    int N = 0; // global number of columns
    PetscCall( MatGetSize(petscMat, &M, &N) );
    GISMO_ASSERT(M > 0 && N>0, "petsc_copySparseMat: PETSc matrix with zero rows and/or columns, the global and local sizes of the matrix must be set before (e.g. in function petsc_setupMatrix).");
    GISMO_ASSERT(M == gismoMat.rows() && N == gismoMat.cols(), "petsc_copySparseMat: Incompatible petscMat and gismoMat sizes.");

    int nProc = -1;
    MPI_Comm_size( comm, &nProc );

    // preallocate PETSc matrix

    std::vector<index_t> nnzRowsDiag(rLocInfo.first, 0);
    std::vector<index_t> nnzRowsOffdiag(rLocInfo.first, 0);    
    petsc_getNonzeroCounts(gismoMat, rLocInfo, cLocInfo, nnzRowsDiag, nnzRowsOffdiag, nRowBlocks, nColBlocks);

    if (nProc == 1)
        PetscCall( MatSeqAIJSetPreallocation( petscMat, 0, &(nnzRowsDiag[0])) );
    else
        PetscCall( MatMPIAIJSetPreallocation( petscMat, 0, &(nnzRowsDiag[0]), 0, &(nnzRowsOffdiag[0])) );

    // copy values

    gsVector<index_t> rLocSizes, rOffsets, cLocSizes, cOffsets;
    petsc_createRankInfoVectors(rLocInfo, rLocSizes, rOffsets, comm);
    petsc_createRankInfoVectors(cLocInfo, cLocSizes, cOffsets, comm);
    gsVector<index_t> mapRow = petsc_mapping_block2interlaced(M, nRowBlocks, rLocSizes, rOffsets, comm);
    gsVector<index_t> mapCol = petsc_mapping_block2interlaced(N, nColBlocks, cLocSizes, cOffsets, comm);

    // const int* outerIndex = gismoMat.outerIndexPtr();
    // const int* innerIndex = gismoMat.innerIndexPtr();
    // const double* values = gismoMat.valuePtr();

    index_t rBlockSize = M / nRowBlocks;
    for (index_t b = 0; b < nRowBlocks; b++)
    {
        for (int i = 0; i < rLocInfo.first; i++)
        {
            int ii = b * rBlockSize + rLocInfo.second + i;

            // int ii = rLocInfo.second + i;
            // int indi[1];
            // indi[0] = ii;
            // int j =  outerIndex[ii];
            // PetscCall( MatSetValues(petscMat, 1, indi, outerIndex[ii+1] - outerIndex[ii], &innerIndex[j], &values[j], INSERT_VALUES) );
        
            for (typename gsSparseMatrix<real_t, RowMajor>::InnerIterator it(gismoMat, ii); it; ++it)
                PetscCall( MatSetValue(petscMat, mapRow(ii), mapCol(it.col()), it.value(), INSERT_VALUES) );
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
template<typename Derived>
int petsc_copyVec(const gsEigen::MatrixBase<Derived>& gismoVec, Vec& petscVec, MPI_Comm comm)
{
    int M = 0; // global number of rows
    PetscCall( VecGetSize(petscVec, &M) );
    GISMO_ASSERT(M > 0, "petsc_copyVec: PETSc vector with zero rows, the global and local sizes of the vector must be set before.");

    int nProc = -1;
    MPI_Comm_size( comm, &nProc );;

    index_t nrows = gismoVec.rows();

    index_t globalStart, globalEnd;
    PetscCall( VecGetOwnershipRange(petscVec, &globalStart, &globalEnd) );

    if (nProc == 1)
        GISMO_ASSERT(M == nrows, "petsc_copyVec: Incompatible petscVec and gismoVec sizes.");
    else
    {
        index_t localRows = globalEnd - globalStart;
        GISMO_ASSERT(localRows == nrows, "petsc_copyVec: Incompatible number of petscVec local rows and gismoVec rows.");
    }

    for (index_t i = 0; i < nrows; i++)
    {
        int indi[1];
        indi[0] = globalStart + i;

        PetscCall( VecSetValues(petscVec, 1, indi, &(gismoVec(i)), INSERT_VALUES) );
    }

    PetscCall( VecAssemblyBegin(petscVec) );
    PetscCall( VecAssemblyEnd(petscVec) ); 

    return 0;
}

/// @brief Print ordered output gathered from all ranks.
/// @param[in] outStr output string
/// @param[in] comm   MPI communicator
void printOrderedOutput(std::string outStr, gsMpiComm comm)
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

template<typename _MatrixType>
struct petsc_traits< PetscKSP<_MatrixType> >
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

    mutable PetscErrorCode m_error; ///< Error code from PETSc
    mutable ComputationInfo m_info;
    Index m_size; ///< Global size of the matrix

    PetscImpl(MPI_Comm comm = PETSC_COMM_WORLD) : m_size(-1)
    {
        initialize(comm);
        m_isInitialized = false;// Becomes true when the sparse matrix is given
    }

    ~PetscImpl()
    {
        m_error = MatDestroy(&m_pmatrix);
        assert(0==m_error);
        m_error = VecDestroy(&m_psol);
        assert(0==m_error);
        m_error = VecDestroy(&m_prhs);
        assert(0==m_error);
        m_error = KSPDestroy(&m_ksp);
        assert(0==m_error);
        //m_error = PCDestroy(&m_pc); //managed by m_ksp
        //assert(0==m_error);
        PetscFinalize();
    }

    /// Initialize PETSc solver with the communicator \a comm
    void initialize(MPI_Comm comm = PETSC_COMM_WORLD)
    {
        m_comm = comm;

        PETSC_COMM_WORLD = comm;
        m_error = PetscInitializeNoArguments();
        assert(0==m_error);
        // Note: initialization with command line arguments is done as follows:
        // PetscInitialize(&argc, &argv, (char *)0, "");

        // Create KSP (Krylov Subspace Preconditioned) solver
        m_error = KSPCreate(m_comm, &m_ksp);
        assert(0==m_error);

        // Get the preconditionner
        m_error = KSPGetPC(m_ksp, &m_pc);
        assert(0==m_error);
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
        m_error = gismo::petsc_computeMatLayout(nRows, result, m_comm);
        assert(0==m_error);
        
        return result;
    }
    
    template<typename Rhs,typename Dest>
    void _solve_impl(const MatrixBase<Rhs> &b, MatrixBase<Dest> &dest) const;

protected:

    void applyOptions() const
    {
        m_error = PetscOptionsClear(NULL);
        for ( auto & opt : m_options.getAllEntries() )
        {
            m_error = PetscOptionsSetValue(NULL, opt.label.c_str(), opt.val.c_str());
            assert(0==m_error);
        }

        m_error = KSPSetFromOptions(this->m_ksp);
        assert(0==m_error);
        m_error = PCSetFromOptions(this->m_pc);
        assert(0==m_error);

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
    int nProc = -1;
    MPI_Comm_size( m_comm, &nProc );

    int rank  = -1;
    MPI_Comm_rank( m_comm, &rank );

    index_t nRows = matrix.rows();
    index_t nCols = matrix.cols();
    assert(nRows==nCols && "expecting square mat");

    std::pair<index_t, index_t> locInfo; 
    m_error = gismo::petsc_computeMatLayout(nRows, locInfo, m_comm);
    assert(0==m_error);

    m_error = gismo::petsc_setupMatrix(m_pmatrix, nRows, nCols, m_comm);
    assert(0==m_error);

    m_error = MatCreateVecs(m_pmatrix, &m_psol, &m_prhs);
    assert(0==m_error);

    m_size = locInfo.first;

    // Copy matrix [ASSUMES square matrix, same cols/rows layout]
    // Case: Matrix already distributed
    m_error = gismo::petsc_copySparseMat(matrix, m_pmatrix, locInfo, locInfo, m_comm);

    //.. else
    // Assumes matrix is non-empty and fully polulated on rank 0 only !
    //petsc_distributeSparseMat(matrix, m_pmatrix, ...)
    
    assert(0==m_error);

    m_isInitialized = true;
    return this->derived();
}

template<class Derived>
template<typename BDerived,typename XDerived>
void PetscImpl<Derived>::_solve_impl(const MatrixBase<BDerived> &b, MatrixBase<XDerived>& x) const
{
    Index nrhs = Index(b.cols());
    assert(m_size==b.rows());
    assert(((MatrixBase<BDerived>::Flags & RowMajorBit) == 0 || nrhs == 1) && "Row-major right hand sides are not supported");
    assert(((MatrixBase<XDerived>::Flags & RowMajorBit) == 0 || nrhs == 1) && "Row-major matrices of unknowns are not supported");
    assert(((nrhs == 1) || b.outerStride() == b.rows()));

    // Copy right-hand side vector to PETSc
    m_error = gismo::petsc_copyVec(b, m_prhs, m_comm);
    assert(0==m_error);

    this->applyOptions();

    // KSP set operators:
    // first: operator m_pmatrix, second: preconditionner build from the same matrix
    m_error = KSPSetOperators(this->m_ksp, m_pmatrix, m_pmatrix);
    assert(0==m_error);

    // Solve the system
    m_error = KSPSolve(this->m_ksp, m_prhs, m_psol);
    assert(0==m_error);

    // Get statistics
    int nIter = 0;
    m_error = KSPGetIterationNumber(this->m_ksp, &nIter);
    assert(0==m_error);

    // Copy the solution back to \a x
    m_error = gismo::petsc_copyVecToGismo(m_psol, x, m_comm);
    assert(0==m_error);

    // Clear petsc vector
    m_error = VecZeroEntries(m_prhs);
    assert(0==m_error);
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
    using Base::solve;

    PetscKSP(MPI_Comm comm = PETSC_COMM_WORLD) : Base(comm) { }

    explicit PetscKSP(const MatrixType& matrix, MPI_Comm comm) : Base(comm)
    { compute(matrix); }

};

} // end namespace Eigen
