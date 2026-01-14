/// Parallel nonlinear elasticity 3D example
/// Uses gsIterative with MPI support for Newton iteration
/// Linear solves use PETSc
///
/// Run: mpirun -np 4 ./bin/test_petsc_nonlinear_elasticity_3D -r 2
///
/// Authors: O. Weeger (2012-2015, TU Kaiserslautern),
///          A.Shamanskiy (2016 - ...., TU Kaiserslautern)
///          Parallel PETSc modifications: 2025

#include <gismo.h>
#include <gsElasticity/src/gsElasticityAssembler.h>
#include <gsElasticity/src/gsIterative.h>
#include <gsElasticity/src/gsWriteParaviewMultiPhysics.h>
#include <gsElasticity/src/gsMaterialBase.h>
#include <gsElasticity/src/gsLinearMaterial.h>
#include <gsPetsc/PETScSupport.h>
#include <gsPetsc/gsPetscLinearSolver.h>
#include <gsDomain/gsCompositeDomain.h>
#include <gsDomain/gsSubDomain.h>

using namespace gismo;

/// @brief Custom function that returns rank color based on element location
/// This enables per-element coloring in visualization
template<class T>
class gsPartitionFunction : public gsFunction<T>
{
public:
    /// Constructor - precomputes element bounding boxes for fast lookup
    /// @param basis The basis used to determine element boundaries
    /// @param elementToRank Map from (patchId, elementId) to rank
    /// @param patchId The patch this function is defined on
    gsPartitionFunction(const gsBasis<T>& basis,
                        const std::map<std::pair<index_t, index_t>, index_t>& elementToRank,
                        index_t patchId)
        : m_patchId(patchId), m_dim(basis.dim())
    {
        // Precompute element bounding boxes and their ranks
        auto domain = basis.domain();
        for (auto domIt = domain->beginAll(); domIt != domain->endAll(); ++domIt) {
            gsVector<T> lower = domIt.lowerCorner();
            gsVector<T> upper = domIt.upperCorner();
            index_t elemId = domIt.localId();
            
            // Look up rank for this element
            auto it = elementToRank.find({patchId, elemId});
            index_t rank = (it != elementToRank.end()) ? it->second : -1;
            
            m_elements.push_back({lower, upper, rank});
        }
    }

    short_t domainDim() const override { return m_dim; }
    short_t targetDim() const override { return 1; }

    void eval_into(const gsMatrix<T>& u, gsMatrix<T>& result) const override
    {
        result.resize(1, u.cols());
        
        for (index_t i = 0; i < u.cols(); ++i) {
            gsVector<T> pt = u.col(i);
            
            // Find which element contains this point
            index_t rank = 0;
            for (const auto& elem : m_elements) {
                bool inside = true;
                for (short_t d = 0; d < m_dim; ++d) {
                    // Use <= for upper bound to handle boundary points
                    if (pt(d) < elem.lower(d) - 1e-10 || pt(d) > elem.upper(d) + 1e-10) {
                        inside = false;
                        break;
                    }
                }
                if (inside) {
                    rank = elem.rank;
                    break;
                }
            }
            
            result(0, i) = static_cast<T>(rank + 1);  // 1-indexed color
        }
    }

    GISMO_CLONE_FUNCTION(gsPartitionFunction)

private:
    struct ElementInfo {
        gsVector<T> lower;
        gsVector<T> upper;
        index_t rank;
    };
    
    std::vector<ElementInfo> m_elements;
    index_t m_patchId;
    short_t m_dim;
};

int main(int argc, char* argv[])
{
    // Initialize PETSc and MPI
    PetscInitialize(&argc, &argv, NULL,
        "Parallel Nonlinear Elasticity Solver with G+Smo and PETSc\n");
    MPI_Comm comm = PETSC_COMM_WORLD;

    int rank, nProc;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nProc);

    if (rank == 0) {
        gsInfo << "\n";
        gsInfo << "==============================================\n";
        gsInfo << "  Parallel Nonlinear Elasticity with PETSc\n";
        gsInfo << "  Problem: 3D cantilever beam under load\n";
        gsInfo << "==============================================\n";
        gsInfo << "Running on " << nProc << " processes\n\n";
    }

    // ============================================================
    // STEP 1: Parse command line and set up geometry
    // ============================================================

    real_t youngsModulus = 74e9;
    real_t poissonsRatio = 0.33;
    index_t materialLaw = material_law::saint_venant_kirchhoff;
    index_t numUniRef = 0;
    index_t numDegElev = 0;
    index_t numPlotPoints = 10000;

    gsCmdLine cmd("Parallel nonlinear elasticity solver in 3D with PETSc");
    cmd.addInt("l","law","Material law: 0 - St.V.-K., 1 - neoHookeLn, 2 - neoHookeQuad", materialLaw);
    cmd.addInt("r","refine","Number of uniform refinement applications", numUniRef);
    cmd.addInt("d","degelev","Number of degree elevation applications", numDegElev);
    cmd.addInt("p","points","Number of points to plot to Paraview", numPlotPoints);
    try { cmd.getValues(argc,argv); } catch (int rv) { PetscFinalize(); return rv; }

    // Create geometry: 3D cantilever beam
    gsMultiPatch<> patch, geometry;
    patch.addPatch(gsNurbsCreator<>::BSplineCube());
    patch.patch(0).coefs().col(0)*=10;  // Elongate in x-direction
    geometry = patch.uniformSplit(0);  // Split in x-direction (2 patches)
    geometry = geometry.uniformSplit(1);  // Split in y-direction (4 patches total = 2x2)
    geometry.computeTopology();

    // Create basis
    gsMultiBasis<> basis(geometry);
    for (index_t i = 0; i < numDegElev; ++i)
        basis.degreeElevate();
    for (index_t i = 0; i < numUniRef; ++i)
        basis.uniformRefine();

    if (rank == 0) {
        gsInfo << "Geometry created: " << geometry.nPatches() << " patch(es)\n";
        gsInfo << "Total DoFs: " << basis.totalSize() << "\n";
        gsInfo << "Refinement level: " << numUniRef << "\n\n";
    }

    // ============================================================
    // STEP 2: Define material properties and boundary conditions
    // ============================================================

    // Material and loads
    gsLinearMaterial<real_t> materialMat(youngsModulus, poissonsRatio, 3);
    gsConstantFunction<> bodyForce(0.0, 0.0, 0.0, 3);      // No body force
    gsConstantFunction<> surfaceLoad(0.0, 0.0, -1e8, 3);   // Load in z-direction

    // Boundary conditions
    gsBoundaryConditions<> bcInfo;
    // Fix bottom (west): Dirichlet BC
    for (index_t d = 0; d < 3; d++) {
        bcInfo.addCondition(0, boundary::west, condition_type::dirichlet, nullptr, d);
    }
    // Load on top (east): Neumann BC
    bcInfo.addCondition(1, boundary::east, condition_type::neumann, &surfaceLoad);

    // ============================================================
    // STEP 3: Domain decomposition for interior integration
    // ============================================================

    if (rank == 0)
        gsInfo << "Domain Decomposition:\n";

    // Create composite domain from the entire multiBasis (all patches)
    auto compositeDomain = memory::make_shared(new gsCompositeDomain<>(basis));
    
    // Decompose the entire domain across all processes
    auto decomposed = compositeDomain->decompose(nProc);

    index_t total_elements = compositeDomain->numElements();
    size_t numPieces = decomposed->nPieces();
    
    // Get number of elements for this rank (0 if rank >= numPieces)
    index_t num_rank_elements = (rank < (int)numPieces) ? 
        decomposed->subdomain(rank)->numElements() : 0;

    if (rank == 0) {
        gsInfo << "  Number of patches: " << geometry.nPatches() << "\n";
        gsInfo << "  Subdomains (= nProc): " << nProc << "\n";
        gsInfo << "  Total elements: " << total_elements << "\n";
        gsInfo << "  Decomposed domain has " << decomposed->nPieces() << " pieces\n";
        
        // Warning if nProc > total_elements
        if ((size_t)nProc > total_elements) {
            gsWarn << "\n  WARNING: More processes (" << nProc << ") than elements (" 
                   << total_elements << ")!\n";
            gsWarn << "  Some ranks will have no work. Consider using -r option to refine.\n\n";
        }
    }
    
    // Synchronize output
    MPI_Barrier(comm);
    
    // Print element distribution per rank
    for (int r = 0; r < nProc; ++r) {
        if (rank == r) {
            gsInfo << "  Rank " << rank << ": " << num_rank_elements << " elements\n";
        }
        MPI_Barrier(comm);
    }


    // ============================================================
    // STEP 4: Create assembler
    // ============================================================

    if (rank == 0)
        gsInfo << "\nCreating assembler...\n";

    gsElasticityAssembler<real_t> assembler(geometry, basis, bcInfo, bodyForce, &materialMat);
    assembler.options().setReal("YoungsModulus", youngsModulus);
    assembler.options().setReal("PoissonsRatio", poissonsRatio);
    assembler.options().setInt("MaterialLaw", materialLaw);

    // Set MPI info for metadata (domain decomposition info)
    assembler.setMPIInfo(rank, nProc);
    // Note: Don't set integration domain restriction here since Newton solver
    // on rank 0 needs the full assembled system. For distributed assembly/solving,
    // this would need to be handled differently (similar to linear elasticity example).

    if (rank == 0)
        gsInfo << "System: " << assembler.numDofs() << " DoFs\n\n";

    // ============================================================
    // STEP 5: Solve using Newton iteration with PETSc linear solver
    // ============================================================

    if (rank == 0)
        gsInfo << "Solving with Newton iteration (PETSc linear solver)...\n";

    gsStopwatch timer;
    timer.restart();

    // Solution variables
    gsMatrix<real_t> solVector;
    std::vector<gsMatrix<real_t> > fixedDoFs;
    
    if (rank == 0) {
        // Initialize solution vector
        solVector.setZero(assembler.numDofs(), 1);
        fixedDoFs = assembler.allFixedDofs();
        for (size_t d = 0; d < fixedDoFs.size(); ++d)
            fixedDoFs[d].setZero();
        
        // Create PETSc linear solver with options
        // Using PETSC_COMM_SELF for serial solve on rank 0
        // Use direct LU solver for robustness (elasticity problems can be difficult)
        gsOptionList petscOpts = PetscLinearSolver<real_t>::defaultOptions();
        petscOpts.setString("ksp_type", "preonly");  // Direct solver, no Krylov iterations
        petscOpts.setString("pc_type", "lu");        // LU factorization
        petscOpts.setReal("ksp_rtol", 1e-10);
        petscOpts.setInt("verbose", 1);
        
        PetscLinearSolver<real_t> petscSolver(PETSC_COMM_SELF, petscOpts);
        
        const int maxIter = 20;
        const real_t absTol = 1e-10;
        const real_t relTol = 1e-9;
        real_t initResNorm = 1.0;
        
        gsInfo << "\nNewton iteration with PETSc:\n";
        for (int iter = 0; iter < maxIter; ++iter) {
            // Update mode: homogenize Dirichlet BC after first iteration
            if (iter == 1)
                assembler.homogenizeFixedDofs(-1);
            
            // Assemble system at current solution
            assembler.assemble(solVector, fixedDoFs);
            
            // Get residual norm
            real_t residualNorm = assembler.rhs().norm();
            
            if (iter == 0)
                initResNorm = residualNorm;
            
            gsInfo << "  Iter " << iter << ": residual = " << residualNorm 
                   << " (rel: " << residualNorm/initResNorm << ")\n";
            
            // Check convergence
            if (residualNorm < absTol || residualNorm/initResNorm < relTol) {
                gsInfo << "  Converged!\n";
                break;
            }
            
            // Solve linear system with PETSc
            // Convert matrix to RowMajor format required by PETSc solver
            gsSparseMatrix<real_t, RowMajor> matRowMajor = assembler.matrix();
            petscSolver.compute(matRowMajor);
            petscSolver.setupSolver();
            
            // Convert rhs matrix to vector (assuming single column)
            gsVector<real_t> rhsVec = assembler.rhs().col(0);
            gsVector<real_t> delta;
            petscSolver.solve(rhsVec, delta);
            
            // Update solution
            solVector += delta;
            
            // Update fixed DoFs at first iteration only
            if (iter == 0) {
                for (size_t d = 0; d < fixedDoFs.size(); ++d)
                    fixedDoFs[d] += assembler.fixedDofs(d);
            }
        }
        
        petscSolver.destroy();
    }

    real_t solveTime = timer.stop();
    if (rank == 0) {
        gsInfo << "\nSolve time: " << solveTime << "s\n\n";
    }

    // ============================================================
    // STEP 6: Solution statistics
    // ============================================================

    if (rank == 0) {
        gsInfo << "Solution Statistics:\n";
        gsInfo << "  Solution norm: " << solVector.norm() << "\n";
        gsInfo << "  Solution min: " << solVector.minCoeff() << "\n";
        gsInfo << "  Solution max: " << solVector.maxCoeff() << "\n";
        gsInfo << "  Number of DoFs: " << solVector.size() << "\n\n";
    }

    // ============================================================
    // STEP 7: Output and visualization
    // ============================================================

    if (numPlotPoints > 0 && rank == 0) {
        // Reconstruct solution field
        gsMultiPatch<> solutionField;
        assembler.constructSolution(solVector, fixedDoFs, solutionField);

        // Compute stresses
        gsPiecewiseFunction<> stresses;
        assembler.constructCauchyStresses(solutionField, stresses, stress_components::von_mises);

        // Build partition field based on actual domain decomposition
        // First, create a mapping from each element to its subdomain (rank)
        // We iterate through all subdomains and record element ownership
        
        // For each patch, create a lookup that maps parameter coordinates to rank
        // Since patches may be split across multiple ranks, we need per-element info
        
        // Create partition functions for each patch based on actual decomposition
        std::vector<gsFunction<>*> rankColors;
        
        // Build element-to-rank mapping by iterating through decomposed subdomains
        // Note: decomposed->nPieces() may be less than nProc if there aren't enough elements
        const size_t actualPieces = decomposed->nPieces();
        std::map<std::pair<index_t, index_t>, index_t> elementToRank; // (patchId, localElementId) -> rank
        
        gsInfo << "  Actual number of pieces in decomposition: " << actualPieces << "\n";
        
        for (size_t r = 0; r < actualPieces; ++r) {
            auto subdomain = decomposed->subdomain(r);
            
            // Check if this subdomain is a composite domain (may contain multiple patches)
            auto* compSubdomain = dynamic_cast<gsCompositeDomain<real_t>*>(subdomain.get());
            if (compSubdomain) {
                // Iterate through all domains in this composite subdomain
                for (auto domIt = compSubdomain->beginAll(); domIt != compSubdomain->endAll(); ++domIt) {
                    index_t patchId = domIt.patch();
                    index_t localId = domIt.localId();
                    elementToRank[{patchId, localId}] = r;
                }
            } else {
                // Single subdomain - iterate through its elements
                for (auto domIt = subdomain->beginAll(); domIt != subdomain->endAll(); ++domIt) {
                    index_t patchId = domIt.patch();
                    index_t localId = domIt.localId();
                    elementToRank[{patchId, localId}] = r;
                }
            }
        }
        
        // Print decomposition summary
        gsInfo << "\nDomain Decomposition Summary (piece -> patches/elements):\n";
        for (size_t r = 0; r < actualPieces; ++r) {
            gsInfo << "  Piece " << r << ": ";
            std::map<index_t, index_t> patchElementCount;
            for (const auto& entry : elementToRank) {
                if (entry.second == static_cast<index_t>(r)) {
                    patchElementCount[entry.first.first]++;
                }
            }
            for (const auto& pc : patchElementCount) {
                gsInfo << "patch " << pc.first << " (" << pc.second << " elems) ";
            }
            gsInfo << "\n";
        }
        
        if (actualPieces < (size_t)nProc) {
            gsInfo << "  Note: Only " << actualPieces << " pieces created for " 
                   << nProc << " processes.\n";
            gsInfo << "  Processes " << actualPieces << "-" << (nProc-1) 
                   << " will have no work.\n";
        }
        
        // For visualization, create per-element coloring functions for each patch
        // This allows showing different colors within a single patch
        for (index_t p = 0; p < geometry.nPatches(); ++p) {
            // Create partition function that returns rank color based on element
            rankColors.push_back(new gsPartitionFunction<real_t>(
                basis.basis(p), elementToRank, p));
        }
        gsPiecewiseFunction<> rankField(rankColors);  // Constructor takes ownership

        // Output to Paraview
        gsField<> displField(geometry, solutionField);
        gsField<> stressField(assembler.patches(), stresses, true);
        gsField<> partitionField(assembler.patches(), rankField, true);

        std::map<std::string, const gsField<> *> fields;
        fields["Displacement"] = &displField;
        fields["von Mises"] = &stressField;
        fields["Partition"] = &partitionField;  // Domain decomposition coloring
        gsWriteParaviewMultiPhysics(fields, "nonlinear_elasticity_3D", numPlotPoints);
        gsInfo << "\nOpen \"nonlinear_elasticity_3D.pvd\" in Paraview for visualization.\n";
        gsInfo << "The 'Partition' field shows domain decomposition (colors 1-" << actualPieces << ").\n";
        if (actualPieces < (size_t)nProc) {
            gsInfo << "Note: Use -r <level> to increase refinement and get more elements for distribution.\n";
        }
    }

    // ============================================================
    // STEP 8: Cleanup
    // ============================================================

    MPI_Barrier(comm);

    if (rank == 0) {
        gsInfo << "\n==============================================\n";
        gsInfo << "  Nonlinear elasticity solver completed!\n";
        gsInfo << "==============================================\n\n";
    }

    PetscFinalize();
    return EXIT_SUCCESS;
}
