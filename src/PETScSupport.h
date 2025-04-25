/*
 Copyright (c) 2011, Intel Corporation. All rights reserved.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of Intel Corporation nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 ********************************************************************************
 *   Content : Eigen bindings to Intel(R) MKL PARDISO
 ********************************************************************************
*/

#pragma once

#include <petscksp.h>

// ---------------------------------------- PETSc start ----------------------------------------
namespace gismo
{

// layout for parallel distribution
inline int petsc_computeMatLayout( index_t globalDofs, index_t* localDofs, index_t* globalStart, MPI_Comm comm)
{
    int nProc = -1;
    int rank = -1;
    MPI_Comm_size( comm, &nProc );
    MPI_Comm_rank( comm, &rank );

    *localDofs = PETSC_DECIDE;
    PetscSplitOwnershipEqual(comm, localDofs, &globalDofs);

    if (globalStart) // globalStart is required
    {
        if (nProc > 1 && rank == nProc-1) // last rank
            *globalStart = globalDofs - *localDofs;
        else
            *globalStart = rank * (*localDofs);
    }

    return 0;
}

// distributes the matrix
inline void petsc_setupMatrix(Mat& petscMat, const index_t globalRows, const index_t globalCols, MPI_Comm comm)
{
    PetscCallVoid( MatCreate(comm, &petscMat) );

    index_t localRows, globStartRow;
    index_t localCols, globStartCol;
    petsc_computeMatLayout(globalRows, &localRows, &globStartRow, comm);
    petsc_computeMatLayout(globalCols, &localCols, &globStartCol, comm);

    int nProc = -1;
    MPI_Comm_size( comm, &nProc );
    PetscCallVoid( MatSetType(petscMat, 1 == nProc ? MATSEQAIJ : MATMPIAIJ) );
    PetscCallVoid( MatSetSizes(petscMat, localRows, localCols, globalRows, globalCols) );
}


// preallocates a parallel matrix (sequantial)
template<class T>
void petsc_preallocSeq(const gsSparseMatrix<T, RowMajor>& gismoMat, const Mat& petscMat)
{
    index_t nrows = gismoMat.rows();

    std::vector<index_t> nnzRows(nrows, 0);

    // compute number of nonzeros per row
    for (index_t i = 0; i < nrows; i++)
        for (typename gsSparseMatrix<real_t, RowMajor>::InnerIterator it(gismoMat, i); it; ++it)
            nnzRows[i]++;
    
    PetscCallVoid( MatSeqAIJSetPreallocation( petscMat, 0, &(nnzRows[0])) );
}


// reports number of nonzeros in matrix (for doing the allocation of PETSc matrix)
template<class T>
static void petsc_getNonzeroCounts( const gsSparseMatrix<T, RowMajor>& mat, const index_t localCols, const index_t globStartCol,
                        std::vector<index_t>& nnzRowsDiag, std::vector<index_t>& nnzRowsOffdiag) 
{
    index_t globEndCol = globStartCol + localCols;

    for (index_t row = 0; row < mat.rows(); ++row)
    {
        for (typename gsSparseMatrix<real_t, RowMajor>::InnerIterator it(mat, row); it; ++it)
        {
            index_t col = it.col();

            if ( col >= globStartCol && col < globEndCol )    // inside the diagonal block
                nnzRowsDiag[row]++;
            else                                            // outside the diagonal block
                nnzRowsOffdiag[row]++;

        }
    }
}

// preallocates a parallel matrix (parallel)
template<class T>
static void petsc_preallocMPI(const gsSparseMatrix<T, RowMajor>& gismoMat, Mat& petscMat, const index_t localCols, const index_t globStartCol)
{
    std::vector<index_t> nnzRowsDiag(gismoMat.rows(), 0);
    std::vector<index_t> nnzRowsOffdiag(gismoMat.rows(), 0);    

    petsc_getNonzeroCounts(gismoMat, localCols, globStartCol, nnzRowsDiag, nnzRowsOffdiag);
    PetscCallVoid( MatMPIAIJSetPreallocation(petscMat, 0, &(nnzRowsDiag[0]), 0, &(nnzRowsOffdiag[0])) );
}


// gismoMat is assumed to be only the local part (number of rows = localRows)

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
        GISMO_ASSERT(localRows == gismoMat.rows(), "petsc_copySparseMat: Incompatible number of petscMat local rows and gismoMat rows.");
        petsc_preallocMPI(gismoMat, petscMat, localCols, globStartCol);
    }

    // copy values

    const int* outerIndex = gismoMat.outerIndexPtr();
    const int* innerIndex = gismoMat.innerIndexPtr();
    const double* values = gismoMat.valuePtr();

    for (int i = 0; i < gismoMat.rows(); i++)
    {
        int indi[1];
        indi[0] = globStartRow + i;

        int j =  outerIndex[i];

        // PetscCopyMode
        // PETSC_COPY_VALUES , or : ETSC_USE_POINTER 
        PetscCallVoid( MatSetValues(petscMat, 1, indi, outerIndex[i+1] - outerIndex[i], &innerIndex[j], &values[j], INSERT_VALUES) );
    }

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


// gismoVec is assumed to be only the local part (number of rows = localRows)
template<typename Derived>
void petsc_copyVec(const gsEigen::MatrixBase<Derived>& gismoVec, Vec& petscVec, MPI_Comm comm)
{
    int M = 0; // global number of rows
    PetscCallVoid( VecGetSize(petscVec, &M) );
    GISMO_ASSERT(M > 0, "petsc_copyVec: PETSc vector with zero rows, the global and local sizes of the vector must be set before.");

    int nProc = -1;
    MPI_Comm_size( comm, &nProc );
    //int nProc = 1;

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


// gismoVec is assumed to be only the local part (number of rows = localRows)
template<typename Derived>  
void petsc_copyVecToGismo(const Vec& petscVec, gsEigen::MatrixBase<Derived>& gismoVec, MPI_Comm comm)
{
    index_t nrows = gismoVec.rows();
    int M = 0; // global number of rows
    PetscCallVoid( VecGetSize(petscVec, &M) );
    GISMO_ASSERT(M == nrows, "petsc_copyVecToGismo: Incompatible petscVec and gismoVec sizes.");

    int nProc = -1;
    MPI_Comm_size( comm, &nProc );

    VecScatter scatterCtx;
    Vec globalVec;
    PetscCallVoid( VecScatterCreateToAll(petscVec, &scatterCtx, &globalVec) );

    PetscCallVoid( VecScatterBegin(scatterCtx, petscVec, globalVec, INSERT_VALUES, SCATTER_FORWARD) );
    PetscCallVoid( VecScatterEnd(scatterCtx, petscVec, globalVec, INSERT_VALUES, SCATTER_FORWARD) );

    PetscCallVoid( VecScatterDestroy(&scatterCtx) );

    index_t rowIDs[nrows];
    for(index_t i = 0; i < nrows; i++)
        rowIDs[i] = i;

    real_t vals[nrows];

    PetscCallVoid( VecGetValues(globalVec, nrows, rowIDs, vals) );
    PetscCallVoid( VecDestroy(&globalVec) );

    for(index_t i = 0; i < nrows; i++)
        gismoVec(i) = vals[i];
}

}
// ---------------------------------------- PETSc end ----------------------------------------


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

    mutable KSP ksp; // krylov solver
    mutable PC pc; // preconditionner

    MPI_Comm m_comm;
    
// ----------- start PETSc
    Mat m_pmatrix;      ///< PETSc matrix
    mutable Vec m_prhs, m_psol; ///< Solution vector and right-hand side vector

    mutable ComputationInfo m_info;
    Index m_size;

    PetscImpl() : m_size(-1) { m_isInitialized = false; }

    ~PetscImpl()
    {
        PetscErrorCode err;
        err = MatDestroy(&m_pmatrix);
        assert(0==err);
        err = VecDestroy(&m_psol);
        assert(0==err);
        err = VecDestroy(&m_prhs);
        assert(0==err);
    }

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

    Derived& compute(const MatrixType& matrix, MPI_Comm comm = PETSC_COMM_WORLD);
    
    template<typename Rhs,typename Dest>
    void _solve_impl(const MatrixBase<Rhs> &b, MatrixBase<Dest> &dest) const;

protected:

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
Derived& PetscImpl<Derived>::compute(const MatrixType& matrix, MPI_Comm comm)
{
    m_comm = comm;
    PetscErrorCode err;

    PETSC_COMM_WORLD = comm;
    err = PetscInitializeNoArguments();
    assert(0==err);
    
    m_size = matrix.rows();
    int nProc = -1;
    int rank  = -1;

    MPI_Comm_size( m_comm, &nProc );
    MPI_Comm_rank( m_comm, &rank );

    index_t nRows = matrix.rows();
    index_t nCols = matrix.cols();
    assert(nRows==nCols && "expecting square mat");

    gismo::petsc_setupMatrix(m_pmatrix, nRows, nCols, m_comm);
    err = MatCreateVecs(m_pmatrix, &m_psol, &m_prhs);
    assert(0==err);

    index_t localDofs, globalStart;
    gismo::petsc_computeMatLayout(nRows, &localDofs, &globalStart, m_comm);

    // Copy matrix [ASSUMES square matrix, same cols/rows layout]
    gismo::petsc_copySparseMat(matrix, m_pmatrix, localDofs, localDofs, globalStart, globalStart, m_comm);

    //err = PetscFinalize(); //segfault!
    //assert(0==err);

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

    gismo::petsc_copyVec(b, m_prhs, m_comm);

    PetscErrorCode err;
    err = KSPCreate(m_comm, &ksp);
    assert(0==err);
    // setupSolver(ksp);
    err = KSPGetPC(ksp, &pc);
    assert(0==err);

    // Reset options
    static_cast<const Derived&>(*this).setDefaultOptions();

    err = KSPSetOperators(this->ksp, m_pmatrix, m_pmatrix); // operator m_pmatrix, preconditionner build from the same matrix
    assert(0==err);
    err = KSPSolve(this->ksp, m_prhs, m_psol);
    assert(0==err);

    int nIter = 0;
    err = KSPGetIterationNumber(this->ksp, &nIter);
    assert(0==err);
    gismo::petsc_copyVecToGismo(m_psol, x, m_comm);

    // clearing petsc matrices and vectors
    //MPI_Barrier(m_comm); // probably barrier not needed
    err = VecZeroEntries(m_prhs);
    assert(0==err);
}


template<typename MatrixType>
class PetscKSP : public PetscImpl< PetscKSP<MatrixType> >
{
  protected:
    typedef PetscImpl<PetscKSP> Base;

    using Base::m_pmatrix;
    using Base::m_prhs;
    using Base::m_psol;

    friend class PetscImpl< PetscKSP<MatrixType> >;

  public:
    
    void setDefaultOptions() const
    {
        PetscErrorCode err;
        //(!) clear all CMD given options and reset to the ones that appear here
        err = PetscOptionsClear(NULL);
        err = PetscOptionsSetValue(NULL, "-ksp_type", "fgmres");
        err = PetscOptionsSetValue(NULL, "-ksp_initial_guess_nonzero", "true");
        err = PetscOptionsSetValue(NULL, "-pc_type", "jacobi");

        err = KSPSetFromOptions(this->ksp);
        assert(0==err);
        err = PCSetFromOptions(this->pc);
        assert(0==err);

        //PetscOptionsView()

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

    
    typedef typename Base::Scalar Scalar;
    typedef typename Base::RealScalar RealScalar;

    using Base::compute;
    using Base::solve;

    PetscKSP(): Base() { }

    explicit PetscKSP(const MatrixType& matrix, MPI_Comm comm) : Base()
    { compute(matrix,comm); }

};

} // end namespace Eigen
