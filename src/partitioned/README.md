# Partitioned objects for MPI assembly with G+Smo

This folder holds the building blocks for element-partitioned assembly
with `gsExprAssembler`, and the measurements that show which G+Smo objects
stop scaling when every rank holds them in full.

| File | Content |
|---|---|
| `gsElementRangeDomain.h` | A `gsDomain` that is a contiguous range `[first,last)` of the elements of another domain (e.g. `gsMultiBasis::domain()`), plus a block partition helper. Element ids are local, so OpenMP chunking in `gsDomain::allElements()` still works. Pass it to `gsExprAssembler::setIntegrationDomain()`. |
| `gsPetscCOO.h` | Rank-local assembly results → distributed PETSc `Mat`/`Vec` (COO interface, off-rank rows are summed by PETSc), plus gathers of selected entries (local + ghost) or of the full vector. |
| `gsMemoryProbe.h` | Heap probes (`mallinfo2`), peak RSS, size estimators for G+Smo objects, and a ledger that reduces min/max/sum over ranks. |
| `../../examples/mpi_memory_profile.cpp` | Toy problem: Poisson on a grid of B-spline patches, element-partitioned assembly, PETSc KSP solve, error check. |
| `../../scripts/memory_sweep.sh`, `memory_tables.py` | The sweep used below and the table generator. |

```
mpirun -np 4 ./bin/mpi_memory_profile -d 2 -r 9 [--legacy] [--noreserve] [--geo] [-s 5] [--csv]
```

The L2 error does not depend on the number of ranks (e.g. 1.573e-9 / 1.575e-9 /
1.575e-9 / 1.573e-9 for P = 1/2/4/8, and identical in 3D). This confirms that the
element partition and the COO summation are consistent.

## Measurements

Each number is the **maximum per-rank heap growth** in MiB, measured with
`mallinfo2` around each stage, or an object size estimate. All runs use
the default reservation of `gsExprAssembler` unless marked `noreserve`.
Machine: 4 cores, 15 GB, OpenMPI 4.1, PETSc 3.19, CG + GAMG.

### 2D, p = 2, 4 patches, N = 1 050 625 dofs, growing P

| object | P=1 | P=2 | P=4 | P=8 | behaviour |
|---|---:|---:|---:|---:|---|
| `gsFiberMatrix` after `initSystem` | 444.9 | 444.9 | 444.9 | 444.9 | **replicated** |
| rhs vector (`m_rhs`) | 8.0 | 8.0 | 8.0 | 8.0 | replicated |
| `gsDofMapper` | 4.0 | 4.0 | 4.0 | 4.0 | replicated |
| full solution vector (for `gsFeSolution`) | 8.0 | 8.0 | 8.0 | 8.0 | replicated |
| solution as `gsMultiPatch` (`extract`) | 8.1 | 8.1 | 8.1 | 8.1 | replicated |
| geometry `gsMultiPatch`, `gsMultiBasis`, BCs | 0.1 | 0.1 | 0.1 | 0.1 | replicated, but coarse |
| PETSc Mat/Vec via COO | 723.3 | 371.5 | 187.7 | 95.9 | distributed |
| KSP (GAMG) setup + solve | 132.6 | 167.7 | 85.2 | 44.1 | distributed |
| local + ghost solution entries (what is actually needed) | 12.0 | 6.0 | 3.0 | 1.5 | distributed |
| legacy path: global `gsSparseMatrix` CSC + CSR copy | 607.3 | 307.7 | 157.9 | 83.2 | local nnz, but global outer index |

### 3D, p = 2, 8 patches, N = 274 625 dofs, growing P

| object | P=1 | P=2 | P=4 | P=8 |
|---|---:|---:|---:|---:|
| `gsFiberMatrix` after `initSystem` | 534.3 | 534.3 | 534.3 | 534.3 |
| PETSc Mat/Vec via COO | 857.0 | 475.8 | 284.1 | 189.3 |
| KSP (GAMG) | 40.3 | 66.7 | 35.4 | 21.0 |
| rhs / solution / solution multipatch / mapper | 2.1 / 2.1 / 2.4 / 1.2 | same | same | same |

### Replicated cost per global dof, per rank

| object | bytes per global dof |
|---|---:|
| `gsFiberMatrix`, default reservation, 2D p=2 | 444 |
| `gsFiberMatrix`, default reservation, 2D p=4 | 1 332 |
| `gsFiberMatrix`, default reservation, 3D p=2 | 2 040 |
| `gsFiberMatrix`, `noreserve` (one heap `gsSparseVector` + pointer per column) | 48 |
| rhs vector | 8 × #rhs |
| full solution vector | 8 |
| solution `gsMultiPatch` | 8 × #components |
| `gsDofMapper` | 4 × #components |
| refined geometry (`--geo`) | 8 × d |
| OpenMP pattern locks (`std::vector<omp_lock_t>(numDofs())`, only with OpenMP) | 4 |

Even with the reservation fixed, roughly 76 B × N stays on every rank (2D
scalar). With 2 GB per rank, that caps N at about 25 M dofs, however many
ranks are used.

### Variants (P = 4)

* **N = 4.2 M (2D, `-r 10`) with the default reservation was OOM-killed** in a 15 GB
  container. With `--noreserve` it runs: fiber matrix 553 MiB per rank (192 MiB of it replicated
  column objects), peak RSS 2.6 GB per rank.
* `--noreserve`: fiber matrix 445 → 138 MiB per rank in 2D and 534 → 113 MiB in 3D.
  Assembly time does not get worse (5.1 s → 4.6 s in 2D, 11.1 s → 10.0 s in 3D), because
  `computePattern` sizes the fibers anyway.
* Refined geometry (`--geo`, 1 M dofs): geometry `gsMultiPatch` 16.2 MiB per rank, replicated.
* 1024 patches (`-s 5 -r 5`, 1.1 M dofs): geometry 1.0, multibasis 1.8, composite domain 0.2 MiB.
  The patch count itself is cheap; the dof-proportional objects still dominate.
* Off-rank matrix entries (rows owned by another rank under PETSc's equal split):
  0.4 % (2D, 4 patches) but 13 % (2D, 1024 patches) and 20 % (3D, P = 8). The global
  patch-wise dof numbering and the element block partition only line up by accident.
* PETSc COO keeps permutation arrays (`PetscCount`, 8 B per entry) next to the AIJ data.
  That is about 2.3× the CSR size (723 MiB vs. ~315 MiB of AIJ at P = 1). It is distributed,
  but it is the largest single item once the fiber matrix is fixed.

## Findings, ranked by impact

1. **`gsFiberMatrix` in `gsExprAssembler` is the dominant non-scaling object.**
   `clearMatrix()` creates one heap-allocated `gsSparseVector` per *global*
   column and calls `reservePerColumn(nz)`. Here `nz = Π(bdA·p_i + bdB)·(1+bdO)` (33
   entries for 2D p=2, 166 for 3D p=2). Only the columns touched by the rank's
   elements are ever used.
2. **Global dense vectors**: `m_rhs` (assembler), the full solution vector
   required by `gsFeSolution` (`_Sv->at(ii)` with global `ii`), and
   `gsFeSolution::extract(gsMultiPatch&)`. Each costs 8 B × N per rank.
3. **`gsDofMapper` is global**: `m_dofs` has one entry per basis function per
   component on every patch, 4 B × N × components. It is small next to (1) but
   still O(N).
4. **Geometry `gsMultiPatch`** costs O(N) only when the geometry is fine
   (8·d B/dof). Coarse CAD multipatches, even with 1024 patches, are cheap
   (topology is O(#patches)).
5. **Correctness trap**: an element subset restricts `allElements()` only.
   `assembleBdr` / `assembleIfc` / `computePatternBdr` iterate
   `domain().subdomain(patch)->beginBdr(side)`, i.e. all boundary elements. With
   an element-partitioned domain, every rank would add the full Neumann /
   interface terms, so they would be counted P times.
6. **Legacy helpers in `PETScSupport.h`**:
   * `petsc_copySparseMat` assumes that each rank already holds *complete* rows for
     its PETSc-owned range, in a global-size `gsSparseMatrix<RowMajor>`. With element
     partitioning, rows on partition boundaries are incomplete on every rank, so the
     copy would drop contributions. It also builds two global-size index maps.
   * `petsc_copyVecToGismo` declares `index_t rowIDs[M]; real_t vals[M];` on the
     stack. With the default 8 MB stack, that overflows at about M ≈ 700 k (found by
     reading the code, not by running it), and it always gathers the full vector.
7. `gsDirichletValues` (interpolation / L2 projection) is computed for the whole
   boundary on every rank. It is O(N^{(d-1)/d}) and minor.

## Recommendations

**Short term (small gismo patches)**

* Do not reserve per global column in `gsExprAssembler::clearMatrix()` when
  `computePattern` is going to run anyway (or reserve after the pattern is
  known). Allocate fibers lazily (`nullptr` until first touched) to remove the
  remaining 48 B per column.
* Restrict boundary/interface iteration in partitioned domains. Rule: a boundary or
  interface element belongs to the rank that owns the adjacent volume element (on
  interfaces, the left-hand one).
* Replace the VLAs in `petsc_copyVecToGismo` with `VecGetArrayRead`.

**Assembler back end (your templating question)**

Prefer member templates over making `gsExprAssembler` a class template:

```cpp
template<class Sink, class... expr> void assemble_into(Sink & sink, const expr &... args);
template<class Sink, class... expr> void computePattern_into(Sink & sink, const expr &... args);
// assemble(args...) == assemble_into(m_fiberSink, args...)
```

* `_eval::push` becomes `sink.add(rowIdx, colIdx, localMat)` per element block (with
  the elimination / rhs-correction logic kept in the assembler). This replaces the
  per-entry `coeffRef` + `omp atomic`. A PETSc sink can then call `MatSetValues` once
  per element, with negative indices for eliminated dofs (PETSc ignores them).
* `computePattern_into` with a sink backed by PETSc's `MATPREALLOCATOR`
  (`MatPreallocatorPreallocate`) gives exact preallocation without COO's ~2.3×
  overhead and without any gismo-side matrix.
* Advantages over `gsExprAssembler<T, Backend = gsFiberMatrix<T>>`: no new
  type that leaks into every module and the Python bindings; several back ends per
  assembler (e.g. a mass matrix in gismo and the stiffness matrix in PETSc); the rhs
  sink can be a PETSc `Vec` as well, which removes `m_rhs`. The existing
  `fiberMatrix()`, `matrix()` and `rhs()` API keeps working.
* OpenMP + MPI: `MatSetValues` is not thread-safe, so a sink needs per-thread
  staging buffers or a critical section. The concept should declare whether it is
  thread-safe.

**Partitioned objects (this folder, next steps)**

* `gsPartitionedDofMapper`: built only for the patches touched by local elements
  (+ interface matching with neighbouring patches). Ownership is a contiguous global
  range per rank (`MPI_Exscan`), and shared dofs go to the lowest touching rank. It
  exposes a local→global map (`ISLocalToGlobalMapping`, so the sink can use
  `MatSetValuesLocal`). Then the PETSc layout equals the ownership layout, and
  off-rank entries are limited to the partition interfaces.
* Ghosted solution: `gsFeSolution` should read coefficients through an index
  map instead of a raw global `gsMatrix*`, e.g. a `VecGhost` local form + the
  partitioned mapper. That needs only owned + ghost entries: 25 % of N at P = 4 here,
  instead of 100 %. Nonlinear assembly then calls `VecGhostUpdate` once per Newton step.
* `gsPartitionedMultiPatch`: owned + halo patches (via the replicated topology),
  only needed when geometries are fine or the patch count is very large. Output
  of the solution per rank (`.pvtu`) instead of `extract(gsMultiPatch&)`.

## Status with `features/petsc-support-partitioned-dm` + sinks

Variants of `mpi_memory_profile` (all give identical L2 errors):
`block` (element block partition, PETSc equal split), `lazy` (`lazyMatrix`),
`rcb` (`gsGeometricPartitioner` + `gsPartitionedDofMapper` ownership),
`sink` (`computePattern_into` with `MATPREALLOCATOR`, `assemble_into` with `MatSetValues`).

Peak RSS per rank, P = 4:

| case | block | block+lazy | rcb+lazy | block+sink | rcb+sink |
|---|---:|---:|---:|---:|---:|
| 2D p=2, N = 1.05 M | 922 MiB | 649 | 577 | 445 | 450 |
| 3D p=2, N = 275 k | 1106 MiB | 868 | 760 | 505 | 483 |

rcb+sink, 2D N = 1.05 M, peak RSS per rank for P = 1/2/4/8: 1467 / 846 / 450 / 251 MiB.
rcb+sink, 2D N = 4.2 M, P = 4: 1671 MiB per rank. With the default `initSystem` reservation, the same run was OOM-killed.

Matrix entries assembled in rows owned by another rank (3D, P = 4): block 52 % of element-block
entries, causing 161 MiB of PETSc stash per rank. With rcb it is 6 % (20 MiB of stash).

### Replicated cost per global (scalar) dof, rcb+sink

| object | B/dof | lifetime |
|---|---:|---|
| `gsDofMapper` | 4 | whole run |
| permutation (`gsPartitionedDofMapper::permutation()`) | 4 | whole run |
| partitioner labels/weights + ownership tables | ~20 | transient (partitioning) |
| full solution vector for `gsFeSolution` (+ unpermuted copy) | 8 (+8) | post-processing |
| solution `gsMultiPatch` (`extract`) | 8 | optional |
| `lazy` without sink: fiber pointers + `m_rhs` | 16 | assembly |
| fine (deformed) geometry | 8·d | if used |

### Distributed cost per owned dof (p = 2)

| | 2D | 3D |
|---|---:|---:|
| MATPREALLOCATOR (transient, peak) | ~520 B | ~2.2 kB |
| final AIJ matrix | ~340 B | ~1.6 kB |
| GAMG setup + CG | ~320 B | ~470 B |

## Weak scaling (P = 1, 2, 4; 4-core container)

About 262k dofs per rank (2D, p=2) and 275k (3D, p=2): the patch grid grows with `--aspect P`.
All variants give the same L2 error per size. GAMG needs 20–22 CG iterations in every run.

Peak RSS per rank [MiB]:

| variant | 2D P=1 | 2 | 4 | 3D P=1 | 2 | 4 |
|---|---:|---:|---:|---:|---:|---:|
| block (stable `initSystem`/`assemble`) | 518 | 681 | 929 | 2295 | 3097 | OOM-killed |
| rcb+sink | 397 | 428 | 450 | 1574 | 1823 | 1845 |
| rcb+local+rendezvous+sink | 417 | 445 | 448 | 1609 | 1717 | 1755 |
| block+local+rendezvous+sink | 416 | 444 | 447 | 1608 | 1812 | 1826 |

Stage wall time [s], rcb+local+rendezvous+sink:

| stage | 2D P=1 | 2 | 4 | 3D P=1 | 2 | 4 |
|---|---:|---:|---:|---:|---:|---:|
| computePattern_into | 2.05 | 2.15 | 2.07 | 18.98 | 19.06 | 19.40 |
| assemble_into | 2.89 | 3.03 | 3.11 | 29.08 | 30.56 | 31.99 |
| KSP setup + solve | 1.70 | 2.51 | 2.78 | 10.16 | 11.59 | 14.39 |
| error evaluation | 0.93 | 0.79 | 0.95 | 3.43 | 3.89 | 4.28 |
| RCB partition (replicated) | 0.10 | 0.20 | 0.44 | 0.15 | 0.36 | 0.73 |
| localize (scans the global mapper) | 0.35 | 0.42 | 0.43 | 0.63 | 0.79 | 0.91 |

What still grows with the global size on every rank:

- `gsDofMapper`: 1 → 2 → 4 MiB in 2D, i.e. 4 B per global dof.
- RCB labels and weights: 3 → 5 → 9 MiB while partitioning, plus O(N) time on every rank.
  The block partition with rendezvous ownership avoids the partitioner entirely.
  In 3D it has more ghost dofs and more stash than RCB.
- `localize`: one pass over the global mapper.

The growth in KSP time comes from the cost per iteration (communication, and the
memory bandwidth of 4 ranks on 4 cores); the iteration counts are constant.
