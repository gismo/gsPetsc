#ifndef GS_PETSC_LINEAR_SOLVER_H
#define GS_PETSC_LINEAR_SOLVER_H

#include <gismo.h>
#include <gsPetsc/PETScSupport.h>
#include <numeric>
#include <vector>

namespace gismo
{

// A helper class to encapsulate PETSc's verbose setup and solve logic
template <class T>
class PetscLinearSolver 
{
public:

    typedef gsSparseMatrix<T, RowMajor>    MatrixType;
    typedef gsVector<T>                    VectorType;

    // Constructor takes the MPI communicator
    PetscLinearSolver(MPI_Comm comm, gsOptionList options = defaultOptions())
    : 
    m_comm(comm), 
    m_A(NULL), 
    m_b(NULL), 
    m_x(NULL), 
    m_ksp(NULL), 
    m_options(options) 
    {
        MPI_Comm_rank(comm, &m_rank);
        MPI_Comm_size(comm, &m_nProc);
    }

    PetscLinearSolver(const MatrixType& A, const VectorType& b, MPI_Comm comm, gsOptionList options = defaultOptions())
    :
    m_comm(comm),
    m_A(NULL),
    m_b(NULL),
    m_x(NULL),
    m_ksp(NULL),
    m_options(options)
    {
        MPI_Comm_rank(comm, &m_rank);
        MPI_Comm_size(comm, &m_nProc);
    }

    // Delete copy constructor and assignment operator to prevent accidental copying
    PetscLinearSolver(const PetscLinearSolver&) = delete;
    PetscLinearSolver& operator=(const PetscLinearSolver&) = delete;

    ~PetscLinearSolver() 
    {
        // Intentionally left empty. Call destroy() for explicit cleanup.
    }

    void destroy() 
    {
        // Destroy PETSc objects. Order can matter.
        if (m_ksp) { KSPDestroy(&m_ksp); }
        if (m_A) { MatDestroy(&m_A); }
        if (m_b) { VecDestroy(&m_b); }
        if (m_x) { VecDestroy(&m_x); }
        m_ksp = NULL;
        m_A = NULL;
        m_b = NULL;
        m_x = NULL;
    }

    void reset  ()
    {
        destroy();
    }

    void compute(   const MatrixType& localA)
    {
        index_t rows = localA.rows();
        index_t cols = localA.cols();
        compute(localA, rows, cols);
    }

    void compute(   const MatrixType& localA,
                    index_t rows,   
                    index_t cols)
    {
        collect(localA, rows, cols);
    }

    void setupSolver()
    {
        // Initialize the solver
        if (m_ksp) { KSPDestroy(&m_ksp); m_ksp = NULL; } // Destroy existing KSP if re-setting up
        PetscCallVoid(KSPCreate(m_comm, &m_ksp));

        PetscCallVoid(KSPSetOperators(m_ksp, m_A, m_A));
        // Set KSP type from options
        std::string ksp_type = m_options.askString("ksp_type", "cg");
        PetscCallVoid(KSPSetType(m_ksp, ksp_type.c_str()));
        
        PC pc;
        PetscCallVoid(KSPGetPC(m_ksp, &pc));
        // Set PC type from options
        std::string pc_type = m_options.askString("pc_type", "jacobi");
        PetscCallVoid(PCSetType(pc, pc_type.c_str()));
        
        PetscCallVoid(KSPSetFromOptions(m_ksp)); // Allow overriding from command line
        T rtol = m_options.askReal("ksp_rtol", 1e-6);
        PetscCallVoid(KSPSetTolerances(m_ksp, rtol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
    }

    PetscErrorCode solve(const VectorType& localB, gsVector<T>& solution)
    {
        // Collect options
        bool verbose = m_options.askInt("verbose", 0);

        // Set the RHS
        index_t rows = localB.size();
        collect(localB, rows);
        
        // Solve the system
        gsStopwatch timer;
        int rank;
        MPI_Comm_rank(m_comm, &rank);

        if (m_x) { VecDestroy(&m_x); m_x = NULL; } // Destroy existing solution vector if re-solving
        // Create solution vector
        PetscCall(VecDuplicate(m_b, &m_x));
        PetscCall(VecSet(m_x, 0.0));

        if (rank == 0 && verbose>0) gsInfo << "Solving with PETSc...\n";
        timer.restart();

        PetscCall(KSPSolve(m_ksp, m_b, m_x));
        
        T solveTime = timer.stop();

        // Get and print convergence info
        KSPConvergedReason reason;
        PetscInt niter;
        PetscCall(KSPGetConvergedReason(m_ksp, &reason));
        PetscCall(KSPGetIterationNumber(m_ksp, &niter));
        
        if (rank == 0 && verbose>0) 
        {
            gsInfo << "PETSc solve time: " << solveTime << " seconds\n";
            gsInfo << "  Iterations: " << niter << "\n";
            gsInfo << "  Converged: " << (reason >= 0 ? "Yes" : "No") << "\n";
            gsInfo << "  Convergence reason: " << (int)reason << "\n";
        }

        // Extract solution to Gismo vector
        solution.resize(rows);
        gismo::petsc_copyVecToGismo(m_x, solution, m_comm);
        return 0;
    }

    static gsOptionList defaultOptions() 
    {
        gsOptionList opts;
        opts.addInt("verbose", "Verbosity level", 0);
        opts.addString("ksp_type", "KSP solver type", "cg");
        opts.addString("pc_type", "Preconditioner type", "jacobi");
        opts.addReal("ksp_rtol", "KSP relative tolerance", 1e-6);
        // TODO: Add more options as needed
        return opts;
    }

    T getSolutionNorm() {
        if (!m_x) return -1.0;
        T norm;
        PetscCall(VecNorm(m_x, NORM_2, &norm));
        return norm;
    }

    T getRHSNorm() {
        if (!m_b) return -1.0;
        T norm;
        PetscCall(VecNorm(m_b, NORM_2, &norm));
        return norm;
    }

    T getMatrixNorm() {
        if (!m_A) return -1.0;
        T norm;
        PetscCall(MatNorm(m_A, NORM_FROBENIUS, &norm));
        return norm;
    }

private:

    void collect(   const MatrixType& localA,
                    index_t rows, index_t cols)
    {
        // Destroy existing matrix if re-assembling
        if (m_A) { MatDestroy(&m_A); m_A = NULL; }
        // Create parallel matrix and set options
        PetscCallVoid(MatCreate(m_comm, &m_A));
        PetscCallVoid(MatSetSizes(m_A, PETSC_DECIDE, PETSC_DECIDE, rows, cols));
        PetscCallVoid(MatSetType(m_A, MATMPIAIJ));

        // Assemble Matrix using a robust, correct parallel assembly routine
        assembleMatrix(localA, m_rank, m_nProc);
    }

    void collect(   const VectorType& localB,
                    index_t rows)
    {
        // Destroy existing RHS vector if re-assembling
        if (m_b) { VecDestroy(&m_b); m_b = NULL; }
        // Create RHS vector
        PetscCallVoid(VecCreate(m_comm, &m_b));
        PetscCallVoid(VecSetSizes(m_b, PETSC_DECIDE, rows));
        PetscCallVoid(VecSetType(m_b, VECMPI));

        assembleVector(localB, m_rank, m_nProc);
    }


    void assembleMatrix(const gsSparseMatrix<T, RowMajor>& localA, int rank, int nProc)
    {
        // The previous COO-based assembly was faulty because it gathered all contributions to all processes,
        // resulting in each matrix value being added 'nProc' times.
        // This new implementation uses a standard MatSetValues loop, which is a robust
        // and correct way to perform parallel assembly. Each process adds only its local contributions.

        // 1. Preallocate the matrix.
        // We can estimate the number of non-zeros per row. For a tensor-product basis of degree p,
        // a row has roughly (2p+1)^d non-zeros. A simpler, more general approach is to find the
        // max non-zeros in any row of the local matrix and use that as a hint.
        PetscInt max_nnz_per_row = 0;
        if (localA.rows() > 0) {
            std::vector<PetscInt> nnz(localA.rows());
            for (index_t i = 0; i < localA.rows(); ++i) {
                nnz[i] = localA.innerVector(i).nonZeros();
            }
            max_nnz_per_row = *std::max_element(nnz.begin(), nnz.end());
        }
        // A broadcast is needed to ensure all processes have a safe upper bound
        PetscCallVoid(MPI_Allreduce(MPI_IN_PLACE, &max_nnz_per_row, 1, MPIU_INT, MPI_MAX, m_comm));

        PetscCallVoid(MatMPIAIJSetPreallocation(m_A, max_nnz_per_row, NULL, max_nnz_per_row, NULL));
        // Allow adding new non-zeros just in case our estimate was wrong for some rows
        PetscCallVoid(MatSetOption(m_A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

        // 2. Set values by looping over the local sparse matrix
        for (index_t i = 0; i < localA.rows(); ++i) {
            for (typename gsSparseMatrix<T, RowMajor>::InnerIterator it(localA, i); it; ++it) {
                PetscInt row = i;
                PetscInt col = it.col();
                PetscScalar val = it.value();
                PetscCallVoid(MatSetValues(m_A, 1, &row, 1, &col, &val, ADD_VALUES));
            }
        }

        // 3. Finalize assembly
        PetscCallVoid(MatAssemblyBegin(m_A, MAT_FINAL_ASSEMBLY));
        PetscCallVoid(MatAssemblyEnd(m_A, MAT_FINAL_ASSEMBLY));
    }

    void assembleVector(const gsVector<T>& localB, int rank, int nProc)
    {
        // Iterate through localB and set values in m_b using global indices
        // Only add non-zero values to avoid unnecessary communication
        for (index_t i = 0; i < localB.size(); ++i) {
            if (localB[i] != 0.0) { // Only add non-zero contributions
                PetscInt idx = i;
                PetscScalar val = localB[i];
                PetscCallVoid(VecSetValues(m_b, 1, &idx, &val, ADD_VALUES));
            }
        }
        
        PetscCallVoid(VecAssemblyBegin(m_b));
        PetscCallVoid(VecAssemblyEnd(m_b));
    }

private:
    MPI_Comm                          m_comm;
    Mat                               m_A;
    Vec                               m_b;
    Vec                               m_x;
    KSP                               m_ksp;
    gsOptionList                      m_options;
    int                               m_rank;    
    int                               m_nProc;
};

} // namespace gismo

#endif // GS_PETSC_LINEAR_SOLVER_H
