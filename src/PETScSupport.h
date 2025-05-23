
#pragma once

#include <petsc.h>

// ------------------------------------ PETSc auxiliary functions ------------------------------------
namespace gismo
{

/// layout for parallel distribution
inline int petsc_computeMatLayout( index_t globalDofs, index_t& localDofs, index_t& globalStart, MPI_Comm comm)
{
    int nProc = -1;
    int rank = -1;
    MPI_Comm_size( comm, &nProc );
    MPI_Comm_rank( comm, &rank );

    localDofs = PETSC_DECIDE;
    PetscCall( PetscSplitOwnershipEqual(comm, &localDofs, &globalDofs) );

    if (nProc > 1 && rank == nProc-1) // last rank (localDofs can be different than for other ranks)
        globalStart = globalDofs - localDofs;
    else
        globalStart = rank * localDofs;

    return 0;
}

/// distributes the matrix
inline void petsc_setupMatrix(Mat& petscMat, const index_t globalRows, const index_t globalCols, MPI_Comm comm)
{
    PetscCallVoid( MatCreate(comm, &petscMat) );

    index_t localRows, globStartRow;
    index_t localCols, globStartCol;
    petsc_computeMatLayout(globalRows, localRows, globStartRow, comm);
    petsc_computeMatLayout(globalCols, localCols, globStartCol, comm);

    int nProc = -1;
    MPI_Comm_size( comm, &nProc );
    PetscCallVoid( MatSetType(petscMat, 1 == nProc ? MATSEQAIJ : MATMPIAIJ) );
    PetscCallVoid( MatSetSizes(petscMat, localRows, localCols, globalRows, globalCols) );
}


/// preallocates a sequential PETSCs matrix
template<class T>
void petsc_preallocSeq(const gsSparseMatrix<T, RowMajor>& gismoMat, Mat& petscMat)
{
    index_t nrows = gismoMat.rows();

    std::vector<index_t> nnzRows(nrows, 0);

    // compute number of nonzeros per row
    for (index_t i = 0; i < nrows; i++)
        for (typename gsSparseMatrix<real_t, RowMajor>::InnerIterator it(gismoMat, i); it; ++it)
            nnzRows[i]++;
    
    PetscCallVoid( MatSeqAIJSetPreallocation( petscMat, 0, &(nnzRows[0])) );
}

/// Computes the number of nonzeros in matrix (for doing the allocation of PETSc matrix)
/// \param[out] nnzRowsDiag     Nonzeros per row in the diagonal "block" of the parallel layout 
/// \param[out] nnzRowsOffdiag  Nonzeros per row in the rest of off-diagonal "blocks" of the parallel layout 
/// Used to preallocate PETSc matrix
template<class T>
static void petsc_getNonzeroCounts( const gsSparseMatrix<T, RowMajor>& mat, const index_t localRows, const index_t globStartRow, const index_t localCols, const index_t globStartCol,
                        std::vector<index_t>& nnzRowsDiag, std::vector<index_t>& nnzRowsOffdiag) 
{
    index_t globEndCol = globStartCol + localCols;

    for (index_t row = 0; row < localRows; ++row)
    {
        for (typename gsSparseMatrix<real_t, RowMajor>::InnerIterator it(mat, globStartRow + row); it; ++it)
        {
            index_t col = it.col();

            if ( col >= globStartCol && col < globEndCol )    // inside the diagonal block
                nnzRowsDiag[row]++;
            else                                            // outside the diagonal block
                nnzRowsOffdiag[row]++;

        }
    }
}

/// preallocates a parallel PETSc matrix
template<class T>
static void petsc_preallocMPI(const gsSparseMatrix<T, RowMajor>& gismoMat, Mat& petscMat, const index_t localRows, const index_t globStartRow, const index_t localCols, const index_t globStartCol)
{
    std::vector<index_t> nnzRowsDiag(localRows, 0);
    std::vector<index_t> nnzRowsOffdiag(localRows, 0);    

    petsc_getNonzeroCounts(gismoMat, localRows, globStartRow, localCols, globStartCol, nnzRowsDiag, nnzRowsOffdiag);
    PetscCallVoid( MatMPIAIJSetPreallocation(petscMat, 0, &(nnzRowsDiag[0]), 0, &(nnzRowsOffdiag[0])) );
}


/// Copy an already distributed gsSparseMatrix (only local rows of \a gismoMat have nonzeros)  to distributed PETSc matrix
/// Also, \a gismoMat is assumed to be of full size 
template<class T>
void petsc_copySparseMat(const gsSparseMatrix<T, RowMajor>& gismoMat, Mat& petscMat,
                                const index_t localRows, const index_t localCols,
                         const index_t globStartRow, const index_t globStartCol, MPI_Comm comm)
{
    int M = 0; // global number of rows
    int N = 0; // global number of columns
    PetscCallVoid( MatGetSize(petscMat, &M, &N) );
    GISMO_ASSERT(M*N > 0, "petsc_copySparseMat: PETSc matrix with zero rows and/or columns, the global and local sizes of the matrix must be set before (e.g. in function petsc_setupMatrix).");

    int nProc = -1;
    MPI_Comm_size( comm, &nProc );

    // preallocate PETSc matrix
    if (nProc == 1)
    {
        GISMO_ASSERT(M == gismoMat.rows() && N == gismoMat.cols(), "petsc_copySparseMat: Incompatible petscMat and gismoMat sizes.");
        petsc_preallocSeq(gismoMat, petscMat);
    }
    else
    {
        GISMO_ASSERT(M == gismoMat.rows(), "petsc_copySparseMat: Incompatible number of petscMat and gismoMat rows.");
        petsc_preallocMPI(gismoMat, petscMat, localRows, globStartRow, localCols, globStartCol);
    }

    // copy values

    const int* outerIndex = gismoMat.outerIndexPtr();
    const int* innerIndex = gismoMat.innerIndexPtr();
    const double* values = gismoMat.valuePtr();

    for (int i = 0; i < localRows; i++)
    {
        int ii = globStartRow + i;

        int indi[1];
        indi[0] = ii;

        int j =  outerIndex[ii];

        PetscCallVoid( MatSetValues(petscMat, 1, indi, outerIndex[ii+1] - outerIndex[ii], &innerIndex[j], &values[j], INSERT_VALUES) );
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

    PetscCallVoid( MatAssemblyBegin( petscMat, MAT_FINAL_ASSEMBLY ) );
    PetscCallVoid( MatAssemblyEnd( petscMat, MAT_FINAL_ASSEMBLY ) ); 
}

/// Copy an already distributed (dense) vector (only the local rows) to distributed PETSc vector
/// Note: \a gismoVec is assumed to be only the local part (number of rows = localRows)
template<typename Derived>
void petsc_copyVec(const gsEigen::MatrixBase<Derived>& gismoVec, Vec& petscVec, MPI_Comm comm)
{
    int M = 0; // global number of rows
    PetscCallVoid( VecGetSize(petscVec, &M) );
    GISMO_ASSERT(M > 0, "petsc_copyVec: PETSc vector with zero rows, the global and local sizes of the vector must be set before.");

    int nProc = -1;
    MPI_Comm_size( comm, &nProc );;

    index_t nrows = gismoVec.rows();

    index_t globalStart, globalEnd;
    PetscCallVoid( VecGetOwnershipRange(petscVec, &globalStart, &globalEnd) );

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

        PetscCallVoid( VecSetValues(petscVec, 1, indi, &(gismoVec(i)), INSERT_VALUES) );
    }

    PetscCallVoid( VecAssemblyBegin(petscVec) );
    PetscCallVoid( VecAssemblyEnd(petscVec) ); 
}

/// Copies a distributed PETSc vector to a global vector on each MPI node
/// Note: gismoVec is the same global vector on all processes
template<typename Derived>  
void petsc_copyVecToGismo(const Vec& petscVec, gsEigen::MatrixBase<Derived>& gismoVec, MPI_Comm comm)
{
    int M = 0; // global number of rows
    PetscCallVoid( VecGetSize(petscVec, &M) );
    gismoVec.derived().resize(M, 1);

    VecScatter scatterCtx;
    Vec globalVec;
    PetscCallVoid( VecScatterCreateToAll(petscVec, &scatterCtx, &globalVec) );

    PetscCallVoid( VecScatterBegin(scatterCtx, petscVec, globalVec, INSERT_VALUES, SCATTER_FORWARD) );
    PetscCallVoid( VecScatterEnd(scatterCtx, petscVec, globalVec, INSERT_VALUES, SCATTER_FORWARD) );
    PetscCallVoid( VecScatterDestroy(&scatterCtx) );

    index_t rowIDs[M];
    for(index_t i = 0; i < M; i++)
        rowIDs[i] = i;

    // real_t* vals;
    // VecGetArray(globalVec, &vals);

    real_t vals[M];

    PetscCallVoid( VecGetValues(globalVec, M, rowIDs, vals) );
    PetscCallVoid( VecDestroy(&globalVec) );

    for(index_t i = 0; i < M; i++)
        gismoVec(i) = vals[i];
}

}
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
        index_t localDofs, globalStart;
        gismo::petsc_computeMatLayout(nRows, localDofs, globalStart, m_comm);
        
        return std::make_pair(localDofs, globalStart);
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

    index_t localSize; // number of local rows
    index_t globalStart; // index of the first local row in the global matrix
    
    gismo::petsc_computeMatLayout(nRows, localSize, globalStart, m_comm);

    gismo::petsc_setupMatrix(m_pmatrix, nRows, nCols, m_comm);
    m_error = MatCreateVecs(m_pmatrix, &m_psol, &m_prhs);
    assert(0==m_error);

    m_size = localSize;

    // Copy matrix [ASSUMES square matrix, same cols/rows layout]
    gismo::petsc_copySparseMat(matrix, m_pmatrix, localSize, localSize, globalStart, globalStart, m_comm);

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
    gismo::petsc_copyVec(b, m_prhs, m_comm);

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
    gismo::petsc_copyVecToGismo(m_psol, x, m_comm);

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
