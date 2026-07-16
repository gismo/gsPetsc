# gsPetsc

PETSc integration for G+Smo: distributed matrices/vectors, KSP solvers, and
helpers for inserting gismo-assembled systems into PETSc objects.

- `src/PETScSupport.h` — layout helpers (`petsc_setupMatrix`,
  `petsc_setupMatrixPartitioned`, `petsc_computeMatLayout`), copy routines
  (`petsc_copySparseMat`, `petsc_copyVec`, `petsc_copyVecToGismo`), and the
  Eigen-style solver wrappers `PetscKSP` / `PetscNestKSP`.
- `src/gsPetscLocalToGlobal.h` — ADD_VALUES insertion of locally assembled
  (global-indexed) gismo matrices/vectors into distributed PETSc objects,
  optionally through a DOF permutation (`petsc_insertSparseMatrixPermuted`,
  `petsc_insertVectorPermuted`, `petsc_insertSparseMatrixCOO`).
- `examples/gsMetisPetscAssembly_example.cpp` — the end-to-end reference for
  everything described below.

Requires PETSc (>= 3.18 recommended, for the COO insertion API) and, for the
workflow below, the `gsMetis` module. Build gismo with
`-DGISMO_OPTIONAL="gsMetis;gsPetsc"`.

---

## How to MPI-Parallelize my gsExprAssembler code?

The model is **replicate-and-restrict**: every rank builds the *same*
geometry, basis, boundary conditions, and DOF mapper (cheap, communication-free),
but each rank *integrates only the elements it owns*. The `gsExprAssembler`
stays completely global-indexed — you do not renumber anything — and PETSc's
`ADD_VALUES` assembly sums the per-rank contributions into one distributed
matrix. A `gsPartitionedDofMapper` provides a permutation so that PETSc's row
ownership matches the METIS partition (most insertions stay on-process).

Working reference: `examples/gsMetisPetscAssembly_example.cpp`. Run it as

```bash
mpirun -np 4 ./bin/gsMetisPetscAssembly_example -n 4 -r 3
```

### Step 0 — initialize PETSc first

`PetscInitialize` also initializes MPI, so it must come before anything else:

```cpp
PetscInitialize(&argc, &argv, NULL, NULL);
int rank, nranks;
MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
MPI_Comm_size(PETSC_COMM_WORLD, &nranks);
```

### Step 1 — set up your problem identically on every rank

This is your existing serial `gsExprAssembler` code, unchanged:

```cpp
gsExprAssembler<real_t> A(1, 1);
A.setIntegrationElements(mb);          // mb: gsMultiBasis, replicated
auto u = A.getSpace(mb);
u.setup(bc, dirichlet::l2Projection, 0);
auto G  = A.getMap(mp);
auto ff = A.getCoeff(f, G);

const index_t nDofs = A.numDofs();     // reads the mapper only
```

Do **not** call `initSystem()` on the full domain here: `numDofs()` needs only
the finalized mapper, whereas a full-domain `initSystem()` reserves the entire
global sparsity pattern on every rank.

### Step 2 — partition with METIS and broadcast the labels

METIS runs serially and redundantly on every rank; broadcast rank 0's labels
so all ranks provably agree (heterogeneous nodes / different METIS builds
would otherwise silently diverge):

```cpp
gsMetisPartitioner<real_t>::Options opts;
opts.storeElementDofs = true;                     // needed by makeDofMapper()
gsMetisPartitioner<real_t> partitioner(mb, u.mapper(), nparts, opts);
partitioner.partition();

std::vector<idx_t> labels(partitioner.partLabels().begin(),
                          partitioner.partLabels().end());
MPI_Bcast(labels.data(), labels.size()*sizeof(idx_t), MPI_BYTE, 0, PETSC_COMM_WORLD);
partitioner.setPartLabels(give(labels));

const gsPartitionedDofMapper dofMap = partitioner.makeDofMapper(nranks);
const gsVector<index_t>&     perm   = dofMap.permutation();
```

`perm(g)` is the PETSc row that global free DOF `g` occupies in a layout where
rows are grouped contiguously by owning rank.

### Step 3 — build this rank's integration subdomain

Partitions are assigned to ranks **cyclically** (`part % nranks`), via
`gsPartitionedDofMapper::rankOfPart()`. `gsMetisPartitioner::subdomainForRank()`
applies that same convention internally, so element ownership and DOF
ownership (Step 2, `makeDofMapper()`) are always consistent with each other
by construction — no hand-rolled loop to keep in sync:

```cpp
gsDomain<real_t>::Ptr rankDomain = partitioner.subdomainForRank(rank, nranks);
```

One combined subdomain per rank (not one per partition) means one
`initSystem`/`assemble`/insert per rank. Equivalently, `ownedElements(rank,
nranks)` returns just the sorted global element ids, if you need the list
itself rather than a ready-made domain.

### Step 4 — restrict and assemble

```cpp
A.options().setSwitch("lazyMatrix", true);  // allocate matrix columns on first touch
A.setIntegrationDomain(rankDomain);
A.initSystem();
A.assemble(igrad(u,G) * igrad(u,G).tr() * meas(G),   // matrix
           u * ff * meas(G));                        // rhs
```

The result is a `freeSize × freeSize` sparse matrix with **global** row/column
indices, but only the entries touched by this rank's elements are nonzero.
With `lazyMatrix` on, untouched columns cost one null pointer each, so the
per-rank assembler footprint scales with the subdomain size, not the global
problem. `A.rhs()` contains this rank's load-term *and* Dirichlet-elimination
(`-K_ib·g`) contributions; both are summable across ranks.

`setIntegrationDomain()` invalidates the cached sparsity pattern, so you can
switch subdomains and re-assemble safely; `initSystem()` per switch is still
the cheapest, most explicit reset.

### Step 5 — create the distributed PETSc objects and insert

Size the PETSc matrix to the *partition-derived* ownership (not an even
split), and insert through the permutation:

```cpp
Mat petscMat;
petsc_setupMatrixPartitioned(petscMat, nDofs, dofMap.numOwnedDofs(rank), PETSC_COMM_WORLD);
Vec petscRhs, petscSol;
MatCreateVecs(petscMat, &petscSol, &petscRhs);

#if PETSC_VERSION_GE(3,18,0)
petsc_insertSparseMatrixCOO(petscMat, A.matrix(), perm);   // preallocates exactly
#else
MatSetOption(petscMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
petsc_insertSparseMatrixPermuted(petscMat, A.matrix(), perm);
#endif
petsc_insertVectorPermuted(petscRhs, A.rhs(), perm);

MatAssemblyBegin(petscMat, MAT_FINAL_ASSEMBLY);
MatAssemblyEnd  (petscMat, MAT_FINAL_ASSEMBLY);
VecAssemblyBegin(petscRhs);
VecAssemblyEnd  (petscRhs);
```

Because rows are grouped by owning rank and PETSc's layout was sized to match,
most inserted entries are on-process; the interface-DOF remainder is what
`MatAssemblyEnd` communicates.

### Step 6 — solve and map the solution back

```cpp
KSP ksp;
KSPCreate(PETSC_COMM_WORLD, &ksp);
KSPSetOperators(ksp, petscMat, petscMat);
KSPSetFromOptions(ksp);
KSPSolve(ksp, petscRhs, petscSol);

gsMatrix<real_t> solPermuted, solVector(nDofs, 1);
petsc_copyVecToGismo(petscSol, solPermuted, PETSC_COMM_WORLD); // gathers on every rank
for (index_t g = 0; g < nDofs; ++g)
    solVector(g, 0) = solPermuted(perm(g), 0);                 // back to global DOF order

auto uSol = A.getSolution(u, solVector);   // usable in gsExprEvaluator etc.
```

`petsc_copyVecToGismo` replicates the full solution on every rank — fine for
post-processing, but it is an `O(nDofs)` object per rank; avoid it in the
scaling-critical path.

### Boundary and interface terms

`assembleBdr(...)` and `assembleIfc(...)` are subdomain-aware too: a
`gsIndexSubDomain` filters boundary/interface elements down to those whose
adjacent *volume* element it owns, so each boundary/interface element is
integrated by exactly one rank and the `ADD_VALUES` sum is exact — the same
insert-and-sum flow as the volume term applies.

### Rules and pitfalls

1. **Partitions must not overlap.** Every insertion helper in
   `gsPetscLocalToGlobal.h` sums with `ADD_VALUES`; halo-expanded element sets
   (`gsHaloExpander`) would double-count shared elements. One element, one rank.
2. **Match the part→rank convention.** Always derive rank ownership with the
   same cyclic rule as `gsPartitionedDofMapper::rankOfPart(part, nranks)`.
3. **OpenMP + interface assembly.** `assembleBdr(bcRefList)` and
   `assembleIfc()` precompute their sparsity pattern inside an OpenMP parallel
   region. This is thread-safe: run them at any `OMP_NUM_THREADS`. (A former
   data race here — an unguarded lazy-init of the interface-mirror helper in
   `gsExprHelper` — was fixed; verified race-free under ThreadSanitizer and in
   a 50-run 8-thread stress test.)
4. **Collectives are collective.** `MatAssemblyBegin/End`, `VecAssembly*`,
   `KSPSolve`, and the `petsc_copy*` helpers must be called by every rank,
   even if a rank owns no partition (possible when `nparts < nranks`).
5. **What is still replicated:** geometry, basis, DOF mapper, METIS labels,
   the element graph, and `perm` are `O(global)` on every rank. The things
   that scale as `O(global/nranks)` are the assembler's matrix content
   (with `lazyMatrix`), the PETSc matrix/vector rows, and the assembly work.
   For memory-bound runs, avoid any full-matrix or full-vector gather
   (in the example: `--noverify`).
6. **Keep `nparts` ≥ `nranks`**, ideally an integer multiple, so every rank
   owns at least one partition and load stays balanced.
